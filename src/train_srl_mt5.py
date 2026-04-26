import os
import csv
import gc
import argparse
import pandas as pd
import torch
import wandb
import numpy as np
import warnings
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from evaluation import get_tokenizer, prepare_compute_metrics
import torch.distributed as dist

# Silence the Hugging Face deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- GLOBAL CONFIGURATION ---
MODEL_NAME = "google/mt5-base"
VERBATLAS_PATH = 'data/verbatlas_worksheet_1.1 - Clustering.tsv'
MODELS_DIR = "models/"
RESULTS_DIR = "results/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set WandB Project Globally
os.environ["WANDB_PROJECT"] = "srl-mt5-project"

def make_preprocess_mt5(tokenizer_, max_length=1024):
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

def train_mt5(train_langs, srl_type):
    # 1. Setup Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Add roles and frames as special tokens
    tokenizer = get_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    
    train_tag = "_".join(train_langs)
    run_name = f"{srl_type}_{train_tag}_mt5"

    # 2. Load and merge datasets (Train + FT + Val)
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
    full_train_ds = concatenate_datasets(train_datasets).shuffle(seed=42)
    full_val_ds = concatenate_datasets(val_datasets)
    # 3. Preprocess datasets
    print("START PREPROCESS", flush=True)
    preprocess_fn = make_preprocess_mt5(tokenizer)
    train_data = full_train_ds.map(preprocess_fn, batched=True)
    val_data = full_val_ds.map(preprocess_fn, batched=True)
    print("FINISH PREPROCESS", flush=True)
    # 4. Training Arguments
    output_dir = os.path.join(MODELS_DIR, f"mt5_{srl_type}_{train_tag}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,
        predict_with_generate=True,
        ddp_find_unused_parameters=False,
        optim="adamw_bnb_8bit",
        load_best_model_at_end=True,
        report_to=["wandb"],
        logging_dir="logs",
        run_name=run_name,
        save_total_limit=1,
        # --- ADD THESE TWO LINES FOR FSDP ---
        # fsdp="full_shard auto_wrap",
        # fsdp_transformer_layer_cls_to_wrap="MT5Block", # Tells FSDP how to chop the model
    )
    print("START PREPARE METRICS", flush=True)
    compute_metrics_val = prepare_compute_metrics(val_data, srl_type, train_langs, tokenizer, run_name=run_name)
    print("FINISH PREPARE METRICS", flush=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics = compute_metrics_val
    )

    # 5. Train
    print(f"Starting training for {run_name}...", flush=True)
    trainer.train()

    # Save best model
    best_model_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_model_dir)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tokenizer.save_pretrained(best_model_dir)
    if dist.is_initialized():
        dist.barrier()
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_dir

def evaluate_mt5(train_langs, srl_type):
    train_tag = "_".join(train_langs)
    run_name = f"{srl_type}_{train_tag}_mt5"
    best_model_dir = os.path.join(MODELS_DIR, f"mt5_{srl_type}_{train_tag}", "best")
    
    # Crea una cartella specifica per la run: es. results/dependency_EN/
    run_results_dir = os.path.join(RESULTS_DIR, f"{srl_type}_{train_tag}")
    os.makedirs(run_results_dir, exist_ok=True)
    
    if not os.path.exists(best_model_dir):
        raise FileNotFoundError(f"Model directory not found: {best_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir)
    
    all_results = []
    test_langs = ["EN", "ZH", "ES", "FR"]

    # 6. Evaluation Loop
    for test_lang in test_langs:
        test_path = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
        if not os.path.exists(test_path):
            print(f"Skipping {test_lang}: file {test_path} not found.")
            continue

        print(f"Evaluating {run_name} on {test_lang}...")
        test_ds = load_dataset("csv", data_files=test_path, delimiter="\t")["train"]
        preprocess_fn = make_preprocess_mt5(tokenizer)
        test_data_processed = test_ds.map(preprocess_fn, batched=True)

        compute_metrics_test = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer, run_name=run_name)
        
        eval_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(MODELS_DIR, "temp_eval"),
            per_device_eval_batch_size=4,
            predict_with_generate=True,
            generation_max_length=1024,
            report_to=["wandb"],
            run_name=f"{run_name}_eval_{test_lang}",
            # --- ADD THESE TWO LINES FOR FSDP ---
            # fsdp="full_shard auto_wrap",
            # fsdp_transformer_layer_cls_to_wrap="MT5Block",
        )

        evaluator = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=test_data_processed,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics_test
        )

        # Generate predictions
        test_results = evaluator.evaluate(metric_key_prefix=f"eval_{test_lang}")

        # preds = predictions_output.predictions
        # # Decode and Save Predictions
        # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # pred_filename = os.path.join(RESULTS_DIR, f"preds_{srl_type}_{train_tag}_test_{test_lang}.tsv")

        # df_preds = pd.DataFrame({
        #     "input": test_ds["input"],
        #     "prediction": decoded_preds,
        #     "output": test_ds["output"]
        # })
        # df_preds.to_csv(pred_filename, sep="\t", index=False)

        row = {
            "srl_type": srl_type,
            "train_langs": train_tag,
            "test_lang": test_lang,
            # "pred_file": pred_filename,
            **test_results
        }
        all_results.append(row)

        del evaluator
        gc.collect()
        torch.cuda.empty_cache()

    # 7. Final summary
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        summary_path = os.path.join(run_results_dir, f"{run_name}_summary.csv")
        pd.DataFrame(all_results).to_csv(summary_path, index=False)
        print(f"Evaluation completed. Summary saved to {summary_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mT5 Fine-tuning for SRL")
    parser.add_argument("--action", type=str, required=True, choices=['train', 'eval'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='+', required=True, help="List of languages (e.g., EN ZH)")
    
    args = parser.parse_args()
    
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")
    
    if args.action == 'train':
        train_mt5(args.langs, args.srl_type)
    elif args.action == 'eval':
        evaluate_mt5(args.langs, args.srl_type)