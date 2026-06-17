from datasets import load_dataset, concatenate_datasets
import os, gc, argparse, json
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluation import prepare_compute_metrics
from finetune import make_preprocess, evaluate
import sys
import warnings
import torch.distributed as dist
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)
# Change working directory to project root if running from src
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')
# Tell Hugging Face Trainer to use this WandB project automatically
# TODO: change to correct project name
os.environ["WANDB_PROJECT"] = "mt0-srl-finetuning"

def sft(train_langs, srl_type):
    MODEL_DIR = f"{args.model}_models/{srl_type}_EN_ZH_{args.model}_best"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    train_name = "_".join(train_langs)
    run_name = f"{srl_type}_{train_name}_{args.model}"
    train_datasets = []
    val_datasets = []
    for lang in train_langs:
        if "-s" in lang:
            base_lang = lang.replace("-s", "")
            val_file = f"data/linearizations_{srl_type}_Val_{base_lang}.tsv"
            val_ds = load_dataset("csv", data_files={"val": val_file}, delimiter="\t")
            val_datasets.append(val_ds["val"])
            data_file = f"data/linearizations_{srl_type}_FT_{base_lang}.tsv"
            train_ds = load_dataset("csv", data_files={"tune": data_file}, delimiter="\t")
            train_datasets.append(train_ds["tune"])
    combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
    combined_val = concatenate_datasets(val_datasets)
    preprocess = make_preprocess(tokenizer)
    train_ds = combined_train.map(preprocess, batched=True)
    val_ds = combined_val.map(preprocess, batched=True)

    lr = 2e-4
    warmup_ratio = 0.02
    weight_decay = 0.01
    grad_accum = 4
    num_train_epochs = 4
    params_path = f"results/{srl_type}_EN_ZH_{args.model}/{srl_type}_EN_ZH_{args.model}_best_params.json"
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
        # TODO: if using adamw_torch set per_device_train_batch_size to 6, if using adamw_bnb_8bit set it to 8
        per_device_train_batch_size=8, # This gets multiplied by GPUs * accum step automatically
        per_device_eval_batch_size=6,
        predict_with_generate=True,
        generation_max_length=256,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name, # This lets Trainer handle wandb safely across multiple GPUs
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False, # False speeds up DDP training
        optim="adamw_bnb_8bit", # bed and breakfast is better
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True, # Save memory at the cost of slower training, activate only if fsdp is commented out
        # --- ADD THESE TWO LINES FOR FSDP ---
        # fsdp="full_shard auto_wrap",
        # fsdp_transformer_layer_cls_to_wrap="MT5Block", # Tells FSDP how to chop the model
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
    # Save best model
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mT0 Fine-tuning for SRL")
    parser.add_argument("--model", type=str, required=True, choices=["mt0", "mt5"], help="Model name")
    parser.add_argument("--action", type=str, required=True, choices=['train', 'eval'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='+', required=True, help="List of languages (e.g., EN ZH ES-s)")
    args = parser.parse_args()
    if args.model == 'mt5':
        MODEL_NAME = "google/mt5-base"
    else:
        MODEL_NAME = "bigscience/mt0-base"
    MODELS_DIR = f"FT/{args.model}_models/"
    RESULTS_DIR = "FT/results/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")
    if args.action == 'train':
        sft(args.langs, args.srl_type)
    elif args.action == 'eval':
        evaluate(args.langs, args.srl_type)