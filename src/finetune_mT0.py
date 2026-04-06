import numpy as np
from datasets import load_dataset, concatenate_datasets
import os, gc, argparse
import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluation import get_tokenizer, prepare_compute_metrics
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
# Change working directory to project root if running from src
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')
# --- GLOBAL CONFIGURATION ---
MODEL_NAME = "bigscience/mt0-base"
MODELS_DIR = "mt0_models/"
RESULTS_DIR = "results/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# Tell Hugging Face Trainer to use this WandB project automatically
os.environ["WANDB_PROJECT"] = "mt0-srl-finetuning"

def make_preprocess_mT0(tokenizer_, max_length=1024):
    def preprocess_(batch):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for src, tgt in zip(batch["input"], batch["output"]):
            encoded = tokenizer_(src, truncation=True, padding="max_length", max_length=max_length)
            labels = tokenizer_(text_target=tgt, truncation=True, padding="max_length", max_length=max_length)
            pad = tokenizer_.pad_token_id
            labels_ids = [tok if tok != pad else -100 for tok in labels["input_ids"]]
            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["attention_mask"].append(encoded["attention_mask"])
            model_inputs["labels"].append(labels_ids)
        return model_inputs
    return preprocess_

def train_mT0(train_langs, srl_type):
    # 1. Load tokenizer and get custom vocabulary (VA roles)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(tokenizer)
    # 2. Load base model and resize embeddings
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_mt0"
    # 3. Load Datasets for the train_langs
    train_datasets = []
    val_datasets = []
    for lang in train_langs:
        data_files = {
            "train": f"data/linearizations_{srl_type}_Train_{lang}.tsv",
            "val": f"data/linearizations_{srl_type}_Val_{lang}.tsv"
        }
        raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_datasets.append(raw_datasets["train"])
        val_datasets.append(raw_datasets["val"])
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val = concatenate_datasets(val_datasets)
    preprocess = make_preprocess_mT0(tokenizer)
    train_ds = combined_train.map(preprocess, batched=True)
    val_ds = combined_val.map(preprocess, batched=True)
    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(MODELS_DIR, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1, # This gets multiplied by GPUs automatically
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=1024,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name, # This lets Trainer handle wandb safely across multiple GPUs
        num_train_epochs=3,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False, # Speeds up DDP training
        fp16=True, # Use mixed precision
        gradient_accumulation_steps=4, # Must be equal to train_batch_size, set 1 to that
        gradient_checkpointing=True, # Save memory at the cost of slower training
        # --- ADD THESE TWO LINES FOR FSDP ---
        # fsdp="full_shard auto_wrap",
        # fsdp_transformer_layer_cls_to_wrap="MT5Block", # Tells FSDP how to chop the model
    )
    compute_metrics_val = prepare_compute_metrics(val_ds, srl_type, train_langs, tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        compute_metrics=compute_metrics_val
    )
    # 5. Train
    print(f"Starting training for {run_name}...")
    trainer.train()
    # Save best model
    best_model_dir = os.path.join(MODELS_DIR, f"{run_name}_best")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_dir

def evaluate_mT0(train_langs, srl_type):
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_mt0"
    best_model_dir = os.path.join(MODELS_DIR, f"{run_name}_best")
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir)
    preprocess = make_preprocess_mT0(tokenizer)
    all_results = []
    # 6. Evaluation on ALL test languages
    for test_lang in ['ZH', 'ES', 'EN', 'FR']:
        test_data_files = {"test": f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"}
        raw_test = load_dataset("csv", data_files=test_data_files, delimiter="\t")
        test_ds = raw_test["test"].map(preprocess, batched=True)
        compute_metrics_test = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer)
        eval_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(MODELS_DIR, "temp_eval"),
            per_device_eval_batch_size=2,
            predict_with_generate=True,
            generation_max_length=1024,
            report_to=["wandb"],
            run_name=f"{run_name}_eval_{test_lang}"
        )
        evaluator = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=test_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model),
            compute_metrics=compute_metrics_test
        )
        print(f"Evaluating on {test_lang} test set...")
        test_results = evaluator.evaluate()
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
    # 7. Save final results
    # Only save on the main process to prevent multiple GPUs writing to the same file
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULTS_DIR, f"{run_name}_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Evaluation completed. Results saved to {results_path}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mT0 Fine-tuning for SRL")
    parser.add_argument("--action", type=str, required=True, choices=['train', 'eval'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='+', required=True, help="List of languages (e.g., EN ZH)")
    args = parser.parse_args()
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")
    if args.action == 'train':
        train_mT0(args.langs, args.srl_type)
    if args.action == 'eval':
        evaluate_mT0(args.langs, args.srl_type)