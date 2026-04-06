import os
import csv
import gc
import pandas as pd
import torch
import wandb
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

# Silence the Hugging Face deprecation warnings during the multi-GPU progress bar
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SPECIAL TOKENS SETUP ---
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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    MODEL_NAME = "google/mt5-base"
    VERBATLAS_PATH = 'data/verbatlas_worksheet_1.1 - Clustering.tsv'
    SRL_TYPES = ["dependency", "span"]
    # Training configurations: EN only, ZH only, and Multi (EN+ZH)
    TRAIN_CONFIGS = [["EN"], ["ZH"], ["EN", "ZH"]]
    # Languages to evaluate for each trained model
    TEST_LANGS = ["EN", "ZH", "ES", "FR"]

    # Set WandB Project Globally (Fixes WandB DDP crashes)
    os.environ["WANDB_PROJECT"] = "srl-mt5-project"

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    all_results = []

    for srl_type in SRL_TYPES:
        for train_langs in TRAIN_CONFIGS:
            # Create a tag for the current training setup (e.g., "EN_ZH")
            train_tag = "_".join(train_langs)
            run_name = f"{srl_type}_{train_tag}_mt5"

            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"\n>>> Starting: {srl_type.upper()} trained on {train_tag} <<<")

            torch.cuda.empty_cache()

            # 1. Setup Tokenizer and Model
            tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
            model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
            
            # Add roles and frames as special tokens
            special_tokens = get_special_tokens(VERBATLAS_PATH)
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))

            # 2. Load and merge datasets (Train + FT + Val)
            train_sets, ft_sets, val_sets = [], [], []
            for lang in train_langs:
                train_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_Train_{lang}.tsv", delimiter="\t")["train"])
                ft_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_FT_{lang}.tsv", delimiter="\t")["train"])
                val_sets.append(load_dataset("csv", data_files=f"data/linearizations_{srl_type}_Val_{lang}.tsv", delimiter="\t")["train"])
            
            # Concatenate all languages for the training phase
            full_train_ds = concatenate_datasets(train_sets + ft_sets).shuffle(seed=42)
            full_val_ds = concatenate_datasets(val_sets)

            # 3. Preprocess datasets
            train_data = full_train_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)
            val_data = full_val_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)

            # 4. Training Arguments (Memory & DDP Fixes Applied)
            output_dir = f"./models/mt5_{srl_type}_{train_tag}"
            args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                num_train_epochs=5,
                # --- MEMORY AND BATCHING FIXES ---
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                per_device_eval_batch_size=2,
                fp16=True,
                gradient_checkpointing=True,
                ddp_find_unused_parameters=False,
                # --- LARGE MODEL PARAMS ---
                fsdp="full_shard auto_wrap",
                fsdp_transformer_layer_cls_to_wrap="MT5Block",
                # --- LOGIC FIX: Don't generate text during validation since there's no metric function ---
                predict_with_generate=False,
                load_best_model_at_end=True,
                report_to=["wandb"],
                run_name=run_name
            )

            trainer = Seq2SeqTrainer(
                model=model,
                args=args,
                train_dataset=train_data,
                eval_dataset=val_data,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
            )

            # Execution of Fine-Tuning
            trainer.train()
            best_model_dir = f"{output_dir}/best"
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # 5. Evaluation Loop (Testing) - Fixed for DDP
            # This fills the rows of your table across all test languages
            for test_lang in TEST_LANGS:
                test_path = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
                if not os.path.exists(test_path):
                    continue

                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    print(f"Inference: {srl_type} {train_tag} -> Testing on {test_lang}")

                test_ds = load_dataset("csv", data_files=test_path, delimiter="\t")["train"]
                test_data_processed = test_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)

                compute_metrics_test = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer)
                # Create a temporary evaluator for safe DDP prediction gathering
                eval_args = Seq2SeqTrainingArguments(
                    output_dir=f"{output_dir}/temp_eval",
                    per_device_eval_batch_size=2,
                    predict_with_generate=True,  # Generate text only during inference
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

                # trainer.predict automatically handles the DDP sharding and gathering
                predictions_output = evaluator.predict(test_data_processed)
                preds = predictions_output.predictions

                # Decode the gathered predictions
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

                pred_filename = f"results/preds_{srl_type}_{train_tag}_test_{test_lang}.tsv"

                # --- CRITICAL: Save files ONLY on the Main GPU ---
                if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    df_preds = pd.DataFrame({
                        "input": test_ds["input"],
                        "prediction": decoded_preds,
                        "output": test_ds["output"]
                    })
                    df_preds.to_csv(pred_filename, sep="\t", index=False)

                    all_results.append({
                        "srl_type": srl_type,
                        "train_langs": train_tag,
                        "test_lang": test_lang,
                        "pred_file": pred_filename
                    })

                # Clean up memory after each test language
                del evaluator
                gc.collect()
                torch.cuda.empty_cache()

            # Clean up memory after each training config loop
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

    # Save a summary CSV of all generated files ONLY on the Main GPU
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        pd.DataFrame(all_results).to_csv("./results/experiment_summary.csv", index=False)
        print("All experiments completed.")