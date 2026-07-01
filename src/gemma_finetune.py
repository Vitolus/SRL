import os, gc, argparse, sys, warnings
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from evaluation import get_tokenizer, prepare_compute_metrics
from tqdm.auto import tqdm
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')

os.environ["WANDB_PROJECT"] = "gemma-srl-finetuning"


def make_chat_template(tokenizer):
    def apply_chat_formatter(example, is_training=True):
        # Use the dictionary format as recommended by Gemma 3 documentation
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a strict Semantic Role Labeling (SRL) system. "
                    "Output ONLY the sentence with the correct SRL tags applied. "
                    "Do not include any conversational text, explanations, or introductory phrases.\n\n"
                    f"Sentence: {example['input']}"
                )
            }
        ]
        # Generate the evaluation prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = {"prompt": prompt}
        # Generate the full conversational text for training
        if is_training:
            messages.append({"role": "model", "content": example['output']})
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            result["full_text"] = full_text
        return result
    return apply_chat_formatter


def train(train_langs, srl_type, model_name, run_name, models_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = get_tokenizer(tokenizer)
    # Gemma does not have a native pad token; it is standard to use the EOS token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right padding is optimal for causal LM training
    tokenizer.padding_side = "right"
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map={"": accelerator.local_process_index}
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
    train_ds = combined_train.map(lambda x: make_chat_template(tokenizer)(x, is_training=True))
    val_ds = combined_val.map(lambda x: make_chat_template(tokenizer)(x, is_training=True))

    # Because we use skip_prepare_dataset=True, we must manually tokenize the dataset before passing it to the trainer.
    def tokenize_function(example):
        return tokenizer(example["full_text"], truncation=True, max_length=256, add_special_tokens=False)

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=val_ds.column_names)

    training_args = SFTConfig(
        output_dir=os.path.join(models_dir, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        # TODO: apply pipeline parallelism (HF naive model) to slise the model to the gpus and salvage memory
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True, # Gemma trains best in bf16
        bf16=False,
        optim="adamw_bnb_8bit",
        use_liger_kernel=False,
        max_grad_norm=1.0,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_length=256,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5,
        # loss_type="nll", # TODO: only if defaults to chunked
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataset_kwargs={"skip_prepare_dataset": True}
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
    # Resolve which model to load (Fail fast if fine-tuned model is missing)
    if is_zero_shot:
        print(f"--- Running ZERO-SHOT Evaluation on base model: {base_model_name} ---")
        model_to_load = base_model_name
    else:
        if os.path.exists(best_model_dir):
            print(f"--- Running Evaluation on fine-tuned model: {best_model_dir} ---")
            model_to_load = best_model_dir
        else:
            raise FileNotFoundError(
                f"CRITICAL ERROR: Expected to evaluate fine-tuned model, but it was not found at {best_model_dir}. "
            )
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    tokenizer = get_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map={"": accelerator.local_process_index}
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    all_results = []
    test_langs = ['EN', 'ZH', 'ES', 'FR']
    for test_lang in test_langs:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-srl-finetuning"),
                name=f"{run_name}_eval_{test_lang}",
                reinit=True  # Required to start a new run in the same script
            )
            print(f"\nEvaluating on {test_lang} test set...")
            test_file = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
            raw_test = load_dataset("csv", data_files={"test": test_file}, delimiter="\t")
            # We need sequential dataset access to properly write CoNLL files
            test_ds = raw_test["test"]
            test_ds = test_ds.map(lambda x: make_chat_template(tokenizer)(x, is_training=False))
            compute_metrics = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer,
                                                      run_name=f"{run_name}_{test_lang}")
            all_preds = []
            all_labels = []
            batch_size = 1
            for i in tqdm(range(0, len(test_ds), batch_size), desc=f"Generating {test_lang}"):
                batch = test_ds[i:i + batch_size]
                inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=256,
                                   add_special_tokens=False).to(
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
            wandb.finish()
        else:
            pass
    # Output file handling isolated to Main GPU
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        df = pd.DataFrame(all_results)
        run_results_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        results_path = os.path.join(run_results_dir, f"{run_name}_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nEvaluation completed. Results saved to {results_path}")
    # Forces all GPUs to wait here until Rank 0 finishes generating and writing files.
    # Prevents torchrun from crashing the process group prematurely.
    accelerator.wait_for_everyone()


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
    else:
        train_name = "zeroshot"
    RUN_NAME = f"{args.srl_type}_{train_name}_gemma3"
    MODELS_DIR = "gemma3_models/"
    RESULTS_DIR = "results/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs if args.langs else 'ZERO-SHOT'} ---")
    if args.action == 'train':
        train(args.langs, args.srl_type, MODEL_NAME, RUN_NAME, MODELS_DIR)
    elif args.action == 'eval':
        evaluate(args.langs, args.srl_type, MODEL_NAME, RUN_NAME, MODELS_DIR, RESULTS_DIR)