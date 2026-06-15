from datasets import load_dataset, concatenate_datasets
import os, gc, argparse, json
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    MBartTokenizer,  # MBART: tokenizer dedicato
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluation import get_tokenizer, prepare_compute_metrics
import sys
import warnings
import torch.distributed as dist
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')

os.environ["WANDB_PROJECT"] = "mbart-srl-finetuning"

MODEL_NAME = "facebook/mbart-large-cc25"

# MBART: mappa dai codici lingua usati nel dataset ai codici interni di mBART
LANG_MAP = {
    "EN": "en_XX",
    "ZH": "zh_CN",
    "ES": "es_XX",
    "FR": "fr_XX",
}


def make_preprocess(tokenizer_, max_length=256):
    def preprocess_(batch):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        langs = batch.get("lang", [None] * len(batch["input"]))
        for src, tgt, lang in zip(batch["input"], batch["output"], langs):
            mbart_lang = LANG_MAP.get(lang, "en_XX") if lang is not None else "en_XX"
            tokenizer_.src_lang = mbart_lang
            tokenizer_.tgt_lang = mbart_lang  # MBART: obbligatorio settare anche tgt_lang
            encoded = tokenizer_(src, truncation=True, padding=False, max_length=max_length)
            labels = tokenizer_(text_target=tgt, truncation=True, padding=False, max_length=max_length)
            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["attention_mask"].append(encoded["attention_mask"])
            model_inputs["labels"].append(labels["input_ids"])
        return model_inputs
    return preprocess_


def load_lang_dataset(srl_type, split, lang):
    """Carica un file TSV e aggiunge la colonna 'lang' con il codice lingua."""
    split_map = {"train": "Train", "tune": "FT", "val": "Val", "test": "Test"}
    data_file = f"data/linearizations_{srl_type}_{split_map[split]}_{lang}.tsv"
    ds = load_dataset("csv", data_files={split: data_file}, delimiter="\t", download_mode="force_redownload")[split]
    # MBART: aggiunge colonna lingua per il preprocessing
    if "lang" not in ds.column_names:
        ds = ds.add_column("lang", [lang] * len(ds))
    return ds, data_file


def tune(train_langs, srl_type):
    # MBART: usa MBartTokenizer invece di AutoTokenizer
    tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(tokenizer)
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_tuning"
    train_datasets = []
    val_datasets = []
    loaded_files = set()
    for lang in train_langs:
        is_tune = "-s" in lang
        base_lang = lang.replace("-s", "")
        val_ds, val_file = load_lang_dataset(srl_type, "val", base_lang)
        if val_file not in loaded_files:
            val_datasets.append(val_ds)
            loaded_files.add(val_file)
        split = "tune" if is_tune else "train"
        train_ds, data_file = load_lang_dataset(srl_type, split, base_lang)
        if data_file not in loaded_files:
            train_datasets.append(train_ds)
            loaded_files.add(data_file)
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42).select(range(1000))
    combined_val = concatenate_datasets(val_datasets).select(range(500))
    preprocess = make_preprocess(tokenizer)
    train_ds = combined_train.map(preprocess, batched=True, num_proc=1, load_from_cache_file=False)
    val_ds = combined_val.map(preprocess, batched=True, num_proc=1, load_from_cache_file=False)
    compute_metrics_val = prepare_compute_metrics(val_ds, srl_type, train_langs, tokenizer, run_name=f"{srl_type}_{train_name}_mbart")

    def model_init():
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.resize_token_embeddings(len(tokenizer))
        return model

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [4, 5]),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [4, 8])
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODELS_DIR, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=6,
        predict_with_generate=True,
        generation_max_length=256,
        report_to=["none"],
        run_name=run_name,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
    )
    trainer = Seq2SeqTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100),
        compute_metrics=compute_metrics_val
    )
    print(f"Launching Optuna Hyperparameter Search for {run_name}...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=5,
    )

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        base_run_name = f"{srl_type}_{train_name}_mbart"
        run_results_dir = os.path.join(RESULTS_DIR, base_run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        params_path = os.path.join(run_results_dir, f"{base_run_name}_best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_run.hyperparameters, f, indent=4)
        print(f"Saved best hyperparameters to {params_path}")


def train(train_langs, srl_type):
    # MBART: usa MBartTokenizer invece di AutoTokenizer
    tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_mbart"
    train_datasets = []
    val_datasets = []
    loaded_files = set()
    for lang in train_langs:
        is_tune = "-s" in lang
        base_lang = lang.replace("-s", "")
        val_ds, val_file = load_lang_dataset(srl_type, "val", base_lang)
        if val_file not in loaded_files:
            val_datasets.append(val_ds)
            loaded_files.add(val_file)
        split = "tune" if is_tune else "train"
        train_ds, data_file = load_lang_dataset(srl_type, split, base_lang)
        if data_file not in loaded_files:
            train_datasets.append(train_ds)
            loaded_files.add(data_file)
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val = concatenate_datasets(val_datasets)
    preprocess = make_preprocess(tokenizer)
    train_ds = combined_train.map(preprocess, batched=True, num_proc=1, load_from_cache_file=False)
    val_ds = combined_val.map(preprocess, batched=True, num_proc=1, load_from_cache_file=False)

    lr = 2e-4
    warmup_ratio = 0.02
    weight_decay = 0.01
    grad_accum = 4
    num_train_epochs = 4
    run_results_dir = os.path.join(RESULTS_DIR, run_name)
    params_path = os.path.join(run_results_dir, f"{run_name}_best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
        print(f"\n--- Loaded best parameters from {params_path} ---")
        print(json.dumps(best_params, indent=4))
        lr = best_params.get("learning_rate", lr)
        warmup_ratio = best_params.get("warmup_ratio", warmup_ratio)
        weight_decay = best_params.get("weight_decay", weight_decay)
        grad_accum = best_params.get("gradient_accumulation_steps", grad_accum)
        num_train_epochs = best_params.get("num_train_epochs", num_train_epochs)
    else:
        print(f"\nWARNING: {params_path} not found. Falling back to default parameters.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODELS_DIR, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=6,
        predict_with_generate=True,
        generation_max_length=256,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        optim="adamw_bnb_8bit",
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
    )
    compute_metrics_val = prepare_compute_metrics(val_ds, srl_type, train_langs, tokenizer, run_name=run_name)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100),
        compute_metrics=compute_metrics_val
    )
    print(f"Starting training for {run_name}...")
    trainer.train()
    best_model_dir = os.path.join(MODELS_DIR, f"{run_name}_best")
    trainer.save_model(best_model_dir)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tokenizer.save_pretrained(best_model_dir)
    if dist.is_initialized():
        dist.barrier()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_dir


def evaluate(train_langs, srl_type):
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_mbart"
    best_model_dir = os.path.join(MODELS_DIR, f"{run_name}_best")
    # MBART: usa MBartTokenizer invece di AutoTokenizer
    tokenizer = MBartTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir)
    preprocess = make_preprocess(tokenizer)
    all_results = []
    for test_lang in ['EN', 'ZH', 'ES', 'FR']:
        test_ds, _ = load_lang_dataset(srl_type, "test", test_lang)
        test_ds = test_ds.map(preprocess, batched=True)
        # MBART: forced_bos_token_id forza la lingua di output durante la generazione
        forced_bos_token_id = tokenizer.lang_code_to_id[LANG_MAP[test_lang]]
        compute_metrics_test = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer, run_name=run_name)
        eval_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(MODELS_DIR, "temp_eval"),
            per_device_eval_batch_size=6,
            predict_with_generate=True,
            generation_max_length=256,
            generation_config=None,
            report_to=["wandb"],
            run_name=f"{run_name}_eval_{test_lang}",
        )
        evaluator = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=test_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100),
            compute_metrics=compute_metrics_test
        )
        # MBART: passa forced_bos_token_id per generare nella lingua corretta
        print(f"Evaluating on {test_lang} test set...")
        test_results = evaluator.evaluate(
            metric_key_prefix="eval",
            forced_bos_token_id=forced_bos_token_id,
        )
        row = {
            "srl_type": srl_type,
            "train_lang": train_name,
            "test_lang": test_lang,
            **test_results
        }
        all_results.append(row)
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        df = pd.DataFrame(all_results)
        run_results_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        results_path = os.path.join(run_results_dir, f"{run_name}_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Evaluation completed. Results saved to {results_path}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mBART Fine-tuning for SRL")
    parser.add_argument("--action", type=str, required=True, choices=['train', 'eval', 'tune'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='+', required=True, help="List of languages (e.g., EN ZH)")
    args = parser.parse_args()
    MODELS_DIR = "mbart_models/"
    RESULTS_DIR = "mbart_results/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")
    if args.action == 'train':
        train(args.langs, args.srl_type)
    elif args.action == 'eval':
        evaluate(args.langs, args.srl_type)
    elif args.action == 'tune':
        tune(args.langs, args.srl_type)