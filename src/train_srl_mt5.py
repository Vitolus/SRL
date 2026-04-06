import os
import csv
import pandas as pd
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import (
    MT5Tokenizer, 
    MT5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)


# --- SPECIAL TOKENS SETUP ---
VA_ROLES = ["AGENT","ASSET","ATTRIBUTE","BENEFICIARY","CAUSE","CO-AGENT","CO-PATIENT","CO-THEME", 
            "DESTINATION","EXPERIENCER","EXTENT","GOAL", "IDIOM","INSTRUMENT","LOCATION",
            "MATERIAL","PATIENT","PRODUCT","PURPOSE","RECIPIENT","RESULT","SOURCE",
            "STIMULUS","THEME","TIME","TOPIC","VALUE"]

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
            special_tags.append(f"<P{i}:{role}>"); special_tags.append(f"</P{i}:{role}>")
        for frame in frames:
            special_tags.append(f"<P{i}:{frame}>"); special_tags.append(f"</P{i}:{frame}>")
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
    MODEL_NAME = "google/mt5-small"
    VERBATLAS_PATH = 'data/verbatlas_worksheet_1.1 - Clustering.tsv'
    SRL_TYPES = ["dependency", "span"]
    # Training configurations: EN only, ZH only, and Multi (EN+ZH)
    TRAIN_CONFIGS = [["EN"], ["ZH"], ["EN", "ZH"]]
    # Languages to evaluate for each trained model
    TEST_LANGS = ["EN", "ZH", "ES", "FR"]
    
    all_results = []

    for srl_type in SRL_TYPES:
        for train_langs in TRAIN_CONFIGS:
            # Create a tag for the current training setup (e.g., "EN_ZH")
            train_tag = "_".join(train_langs)
            print(f"\n>>> Starting: {srl_type.upper()} trained on {train_tag} <<<")
            
            # Clear GPU memory and init tracking
            torch.cuda.empty_cache()
            wandb.init(project="srl-mt5-project", name=f"{srl_type}_{train_tag}")

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
            full_train_ds = concatenate_datasets(train_sets + ft_sets)
            full_val_ds = concatenate_datasets(val_sets)

            # 3. Preprocess datasets
            train_data = full_train_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)
            val_data = full_val_ds.map(lambda x: preprocess_seq2seq(x, tokenizer), batched=True)

            # 4. Training Arguments (Fine-Tuning)
            output_dir = f"./models/mt5_{srl_type}_{train_tag}"
            args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=4,
                num_train_epochs=5,
                predict_with_generate=True,
                load_best_model_at_end=True,
                report_to="wandb"
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
            trainer.save_model(f"{output_dir}/best")

            # 5. Evaluation Loop (Testing)
            # This fills the rows of your table across all test languages
            for test_lang in TEST_LANGS:
                test_path = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
                if not os.path.exists(test_path):
                    continue

                print(f"Inference: {srl_type} {train_tag} -> Testing on {test_lang}")
                test_ds = load_dataset("csv", data_files=test_path, delimiter="\t")["train"]
                
                # Generation function for inference
                def generate_srl(batch):
                    inputs = tokenizer(batch["input"], return_tensors="pt", padding=True, truncation=True).to(model.device)
                    outputs = model.generate(inputs.input_ids, max_length=1024)
                    batch["prediction"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    return batch

                # Execute prediction on test set
                results_ds = test_ds.map(generate_srl, batched=True, batch_size=8)
                
                # Save predictions to file (required by your scorer_united_*.py scripts)
                pred_filename = f"results/preds_{srl_type}_{train_tag}_test_{test_lang}.tsv"
                results_ds.to_csv(pred_filename, sep="\t", index=False)
                
                # Log progress
                all_results.append({
                    "srl_type": srl_type,
                    "train_langs": train_tag,
                    "test_lang": test_lang,
                    "pred_file": pred_filename
                })
            
            wandb.finish()
    
    # Save a summary CSV of all generated files
    pd.DataFrame(all_results).to_csv("./results/experiment_summary.csv", index=False)
    print("All experiments completed.")