import os, gc, argparse, json, sys, warnings
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from evaluation import get_tokenizer, prepare_compute_metrics
from tqdm.auto import tqdm
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')

os.environ["WANDB_PROJECT"] = "gemma-srl-finetuning"


# Gemma instruction models should perform best when wrapped in their native conversational template.
def apply_chat_template(example, is_training=True):
    # Modify the system/user instruction prompt below to match your exact SRL task phrasing
    prompt = f"<start_of_turn>user\nPerform Semantic Role Labeling on this sentence:\n{example['input']}<end_of_turn>\n<start_of_turn>model\n"
    result = {"prompt": prompt}
    if is_training:
        result["completion"] = f"{example['output']}<end_of_turn>"
    return result


def train(train_langs, srl_type, model_name, run_name, models_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = get_tokenizer(tokenizer)
    # Gemma does not have a native pad token; it is standard to use the EOS token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right padding is optimal for causal LM training
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if not dist.is_initialized() else None
    )
    model.resize_token_embeddings(len(tokenizer))
    train_datasets = []
    val_datasets = []
    loaded_files = set()
    for lang in train_langs:
        if "-s" in lang:
            base_lang = lang.replace("-s", "")
            val_file = f"data/linearizations_{srl_type}_Val_{base_lang}.tsv"
            if val_file not in loaded_files:
                val_ds = load_dataset("csv", data_files={"val": val_file}, delimiter="\t")
                val_datasets.append(val_ds["val"])
                loaded_files.add(val_file)
            data_file = f"data/linearizations_{srl_type}_FT_{base_lang}.tsv"
            if data_file not in loaded_files:
                train_ds = load_dataset("csv", data_files={"tune": data_file}, delimiter="\t")
                train_datasets.append(train_ds["tune"])
                loaded_files.add(data_file)
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val = concatenate_datasets(val_datasets)
    train_ds = combined_train.map(lambda x: apply_chat_template(x, is_training=True))
    val_ds = combined_val.map(lambda x: apply_chat_template(x, is_training=True))

    training_args = SFTConfig(
        output_dir=os.path.join(models_dir, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=0.0009981416500547528,
        weight_decay=0.01,
        num_train_epochs=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True, # Gemma trains best in bf16
        optim="adamw_bnb_8bit",
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_seq_length=256,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5,
        loss_type="chunked_nll",
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # SFTTrainer handles the causal LM masking and formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer
    )

    print(f"Starting SFT for {run_name}...")
    trainer.train()
    best_model_dir = os.path.join(models_dir, f"{run_name}_best")
    trainer.save_model(best_model_dir)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tokenizer.save_pretrained(best_model_dir)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(train_langs, srl_type, base_model_name, run_name, models_dir, results_dir):
    is_zero_shot = len(train_langs) == 0
    best_model_dir = os.path.join(models_dir, f"{run_name}_best")
    # Resolve which model to load based on the flag or availability
    if is_zero_shot:
        print(f"--- Running ZERO-SHOT Evaluation on base model: {base_model_name} ---")
        model_to_load = base_model_name
    else:
        if os.path.exists(best_model_dir):
            print(f"--- Running Evaluation on fine-tuned model: {best_model_dir} ---")
            model_to_load = best_model_dir
        else:
            print(
                f"WARNING: Fine-tuned model not found at {best_model_dir}. Falling back to ZERO-SHOT on {base_model_name}.")
            model_to_load = base_model_name
            is_zero_shot = True
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    tokenizer = get_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    all_results = []
    test_langs = ['EN', 'ZH', 'ES', 'FR']
    for test_lang in test_langs:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            eval_mode_str = "zeroshot" if is_zero_shot else "_".join(train_langs)
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-srl-finetuning"),
                name=f"{run_name}_{eval_mode_str}_{test_lang}",
                reinit=True  # Required to start a new run in the same script
            )
        print(f"\nEvaluating on {test_lang} test set...")
        test_file = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
        raw_test = load_dataset("csv", data_files={"test": test_file}, delimiter="\t")
        # We need sequential dataset access to properly write CoNLL files
        test_ds = raw_test["test"]
        test_ds = test_ds.map(lambda x: apply_chat_template(x, is_training=False))
        compute_metrics = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer,
                                                  run_name=f"{run_name}_{test_lang}")
        all_preds = []
        all_labels = []
        batch_size = 8
        for i in tqdm(range(0, len(test_ds), batch_size), desc=f"Generating {test_lang}"):
            batch = test_ds[i:i + batch_size]
            inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # Greedy decoding for structured tasks
                    pad_token_id=tokenizer.eos_token_id
                )
            # Extract only the newly generated tokens
            prompt_lengths = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, prompt_lengths:].cpu().numpy()
            # Tokenize labels so we can feed them into compute_metrics
            labels = tokenizer(batch["output"], return_tensors="pt", padding=True, truncation=True,
                               max_length=128).input_ids.cpu().numpy()
            for p, l in zip(generated_tokens, labels):
                all_preds.append(p)
                all_labels.append(l)

        # Pad tokens arrays to their local max length using standard pad tokens
        max_pred_len = max(len(p) for p in all_preds)
        max_label_len = max(len(l) for l in all_labels)
        padded_preds = np.full((len(all_preds), max_pred_len), tokenizer.pad_token_id, dtype=np.int64)
        padded_labels = np.full((len(all_labels), max_label_len), -100,
                                dtype=np.int64)  # evaluation.py expects labels to have -100 padding
        for idx, (p, l) in enumerate(zip(all_preds, all_labels)):
            padded_preds[idx, :len(p)] = p
            padded_labels[idx, :len(l)] = l
        metrics_dict = compute_metrics((padded_preds, padded_labels))
        row = {
            "srl_type": srl_type,
            "mode": "zero_shot" if model_to_load == base_model_name else "eval",
            "test_lang": test_lang,
            "model": model_to_load,
            **metrics_dict
        }
        all_results.append(row)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            wandb.finish()
    # Output file handling isolated to Main GPU
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        df = pd.DataFrame(all_results)
        run_results_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        eval_mode_str = "zeroshot" if is_zero_shot else "_".join(train_langs)
        results_path = os.path.join(run_results_dir, f"{run_name}_{eval_mode_str}_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nEvaluation completed. Results saved to {results_path}")
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gemma 3 Fine-tuning for SRL")
    parser.add_argument("--action", type=str, required=True, choices=['train', 'eval'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='*', default=[], help="List of languages (e.g., EN-s ZH-s) Required for train, optional for eval (triggers zero-shot)")
    args = parser.parse_args()
    if args.action == 'train' and not args.langs:
        parser.error("--langs must be provided when action is 'train'")
    MODEL_NAME = "google/gemma-3-1b-it"
    if args.langs:
        train_name = "_".join(args.langs)
        RUN_NAME = f"{args.srl_type}_{train_name}_gemma3_1B"
    else:
        # Graceful handling for zero-shot run tracking
        RUN_NAME = f"{args.srl_type}_zeroshot_gemma3_1B"
    MODELS_DIR = "gemma_models/"
    RESULTS_DIR = "gemma_results/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")
    if args.action == 'train':
        train(args.langs, args.srl_type, MODEL_NAME, RUN_NAME, MODELS_DIR)
    elif args.action == 'eval':
        evaluate(args.langs, args.srl_type, MODEL_NAME, RUN_NAME, MODELS_DIR, RESULTS_DIR)