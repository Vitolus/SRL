import os
import csv
import gc
import argparse
import pandas as pd
import torch
import numpy as np
import warnings
from datasets import load_dataset, concatenate_datasets
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from evaluation import get_tokenizer, prepare_compute_metrics

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

VA_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO-AGENT", "CO-PATIENT", "CO-THEME",
            "DESTINATION", "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION",
            "MATERIAL", "PATIENT", "PRODUCT", "PURPOSE", "RECIPIENT", "RESULT", "SOURCE",
            "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE"]

def get_special_tokens(verbatlas_path):
    """ Extracts frames and roles to create special tokens for the tokenizer """
    frames = []
    if os.path.exists(verbatlas_path):
        with open(verbatlas_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            frames = [line[1].upper().strip() for line in reader if line and line[0] != '']

    frames = list(set(frames))
    special_tags = []
    for i in range(10): # Support up to 10 predicates per sentence
        for role in VA_ROLES:
            special_tags.append(f"<P{i}:{role}>")
            special_tags.append(f"</P{i}:{role}>")
        for frame in frames:
            special_tags.append(f"<P{i}:{frame}>")
            special_tags.append(f"</P{i}:{frame}>")
    return special_tags

def preprocess_seq2seq(batch, tokenizer, max_len=1024):
    """ Standard mT5 tokenization for Seq2Seq """
    model_inputs = tokenizer(batch["input"], max_length=max_len, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["output"], max_length=max_len, truncation=True, padding="max_length")

    # Ignore padding in loss calculation
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_mt5(train_langs, srl_type):
    # 1. Setup Tokenizer and Model
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Add roles and frames as special tokens
    special_tokens = get_special_tokens(VERBATLAS_PATH)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    train_tag = "_".join(train_langs)
    run_name = f"{srl_type}_{train_tag}_mt5"

    # 2. Load and merge datasets (Train + FT + Val)
    train_sets, ft_sets, val_sets = [], [], []
    for lang in train_langs:
        if lang.endswith("-s"):
            ft_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_FT_{lang[:-2]}.tsv", delimiter="\t")["train"])
        else:
            train_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_Train_{lang}.tsv", delimiter="\t")["train"])
        val_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_Val_{lang}.tsv", delimiter="\t")["train"])
    
    full_train_ds = concatenate_datasets(train_sets + ft_sets).shuffle(seed=42)
    full_val_ds = concatenate_datasets(val_sets)

    # 3. Preprocess datasets
    train_data = full_train_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)
    val_data = full_val_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)

    # 4. Training Arguments
    output_dir = os.path.join(MODELS_DIR, f"mt5_{srl_type}_{train_tag}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        predict_with_generate=True, 
        fp16=torch.cuda.is_available(),
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        deepspeed=None,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        report_to=["wandb"],
        run_name=run_name,
        save_total_limit=1,
        # --- ADD THESE TWO LINES FOR FSDP ---
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="MT5Block", # Tells FSDP how to chop the model
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    # 5. Train
    print(f"Starting training for {run_name}...")
    trainer.train()

    # Save best model
    best_model_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_dir

def evaluate_mt5(train_langs, srl_type):
    train_tag = "_".join(train_langs)
    run_name = f"{srl_type}_{train_tag}_mt5"
    best_model_dir = os.path.join(MODELS_DIR, f"mt5_{srl_type}_{train_tag}", "best")
    
    if not os.path.exists(best_model_dir):
        raise FileNotFoundError(f"Model directory not found: {best_model_dir}")

    tokenizer = MT5Tokenizer.from_pretrained(best_model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(best_model_dir)
    
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
        test_data_processed = test_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)

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
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics_test
        )

        # Generate predictions
        predictions_output = evaluator.predict(test_data_processed)
        preds = predictions_output.predictions

        # Decode and Save Predictions
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        pred_filename = os.path.join(RESULTS_DIR, f"preds_{srl_type}_{train_tag}_test_{test_lang}.tsv")

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            df_preds = pd.DataFrame({
                "input": test_ds["input"],
                "prediction": decoded_preds,
                "output": test_ds["output"]
            })
            df_preds.to_csv(pred_filename, sep="\t", index=False)

            row = {
                "srl_type": srl_type,
                "train_langs": train_tag,
                "test_lang": test_lang,
                "pred_file": pred_filename,
                **predictions_output.metrics
            }
            all_results.append(row)

        del evaluator
        gc.collect()
        torch.cuda.empty_cache()

    # 7. Final summary
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        summary_path = os.path.join(RESULTS_DIR, f"{run_name}_summary.csv")
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