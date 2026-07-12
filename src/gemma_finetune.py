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
from tqdm.auto import tqdm
from unsloth import FastModel
import wandb
from evaluation import get_tokenizer, prepare_compute_metrics, va_roles

warnings.filterwarnings("ignore", category=FutureWarning)

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')

os.environ["WANDB_PROJECT"] = "gemma-srl-finetuning"

def prompt_template(example, roles, srl_type):
    if srl_type == "span":
        format_instructions = (
            "3. Identify the arguments corresponding to that predicate. Wrap the ENTIRE argument phrase in a tag formatted exactly as: <P0:ROLE_NAME>full argument text</P0:ROLE_NAME>\n"
        )
    elif srl_type == "dependency":
        format_instructions = (
            "3. Identify the arguments corresponding to that predicate. Wrap ONLY the syntactic head word of the argument in a tag formatted exactly as: <P0:ROLE_NAME>head_word</P0:ROLE_NAME>\n"
        )
    else:
        raise ValueError(f"Unsupported SRL type: {srl_type}")
    return (
        f"You are a strict {srl_type} based Semantic Role Labeling (SRL) system.\n"
        "ALLOWED ROLES:\n"
        f"{roles}\n\n"
        "OUTPUT FORMAT INSTRUCTIONS:\n"
        "1. You must output ONLY the original sentence modified by XML-style tags wrapping predicates and arguments.\n"
        "2. Identify the main predicate. Wrap the predicate word in a tag formatted exactly as: <P0:PREDICATE_LEMMA>word</P0:PREDICATE_LEMMA>\n"
        f"{format_instructions}\n"
        "4. The 'P0' prefix must match across the predicate and its corresponding arguments in increasing number.\n"
        "5. Do not include any introductory text, conversational pleasantries, explanations, or trailing remarks. Output ONLY the tagged sentence.\n\n"
        f"Sentence: {example['input']}"
    )

def make_prompt_completion(srl_type):
    roles = ", ".join(va_roles)
    def formatter(example):
        return {
            "prompt": prompt_template(example, roles, srl_type),
            "completion": example['output']
        }
    return formatter


def train(train_langs, srl_type, model_name, run_name, models_dir):
    max_seq_length = 256

    # Unsloth Model Loading
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,  # Force FP16 for Tesla T4
        load_in_4bit=False,  # Strictly disable quantization
    )

    tokenizer = get_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right padding is optimal for causal LM training
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer))
    # Unsloth PEFT/LoRA Setup (Crucial for T4 FP16 stability)
    model = FastModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth optimizes dropout = 0
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's native VRAM saver
        random_state=42,
    )
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
    train_ds = combined_train.map(make_prompt_completion(srl_type), remove_columns=combined_train.column_names)
    val_ds = combined_val.map(make_prompt_completion(srl_type), remove_columns=combined_val.column_names)

    training_args = SFTConfig(
        output_dir=os.path.join(models_dir, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True, # Gemma trains best in bf16, unsloth resolve for old gpu
        bf16=False,
        optim="adamw_bnb_8bit",
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        completion_only_loss=True, # Automatically calculates loss ONLY on the completion
        dataset_text_field=None, # Explicitly set to None for prompt-completion columns
        packing=False, # Set to False to ensure clean token masking
        max_length=max_seq_length,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5,
        # gradient_checkpointing_kwargs={"use_reentrant": False}
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
    trainer.save_model(best_model_dir) # Saves the lightweight LoRA adapters safely
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
    # Unsloth Loading for Evaluation
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=256,
        dtype=torch.float16,
        load_in_4bit=False,
    )
    tokenizer = get_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left padding is required for batched causal inference
    tokenizer.padding_side = 'left'
    model.resize_token_embeddings(len(tokenizer))
    # Enable Native 2x Faster Inference
    FastModel.for_inference(model)
    accelerator = Accelerator()
    all_results = []
    test_langs = ['EN', 'ZH', 'ES', 'FR']

    def prepare_test_sample(example):
        # We handle zero-shot and finetuned identically via the model's native chat template configuration
        messages = [{"role": "user", "content": prompt_template(example, ", ".join(va_roles), srl_type)}]
        templated_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return {"prompt": templated_prompt, "output": example["output"]}

    # Output file handling and heavy metric evaluation isolated to Main GPU
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        for test_lang in test_langs:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "gemma-srl-finetuning"),
                name=f"{run_name}_eval_{test_lang}",
                reinit=True # Required to start a new run in the same script
            )
            print(f"\nEvaluating on {test_lang} test set...")
            test_file = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
            raw_test = load_dataset("csv", data_files={"test": test_file}, delimiter="\t")
            # We need sequential dataset access to properly write CoNLL files
            test_ds = raw_test["test"].map(prepare_test_sample)
            # Binding the test dataset context for evaluation.py
            compute_metrics = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer, suffix="_eval",
                                                      run_name=f"{run_name}_{test_lang}")
            all_preds = []
            all_labels = []
            batch_size = 4
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
            for i in tqdm(range(0, len(test_ds), batch_size), desc=f"Generating {test_lang}"):
                batch = test_ds[i:i + batch_size]
                inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=256,
                                   add_special_tokens=False).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False, # Greedy decoding for structured tasks
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=terminators,
                        repetition_penalty=1.15
                    )
                # Extract only the newly generated tokens
                prompt_lengths = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, prompt_lengths:].cpu().numpy()
                # Tokenize labels with add_special_tokens=False to avoid embedding <bos> tokens into Exact Match tests
                labels = tokenizer(batch["output"], return_tensors="pt", padding=True, truncation=True,
                                   max_length=128, add_special_tokens=False).input_ids.cpu().numpy()
                for p, l in zip(generated_tokens, labels):
                    # Strip left/right padding safely before saving
                    p_clean = [tok for tok in p if tok != tokenizer.pad_token_id]
                    l_clean = [tok for tok in l if tok != tokenizer.pad_token_id]
                    all_preds.append(p_clean)
                    all_labels.append(l_clean)
            # Pad tokens arrays globally to their maximum length
            max_pred_len = max(len(p) for p in all_preds) if all_preds else 0
            max_label_len = max(len(l) for l in all_labels) if all_labels else 0
            padded_preds = np.full((len(all_preds), max_pred_len), tokenizer.pad_token_id, dtype=np.int64)
            # Metrics computation expects labels to have -100 padding to skip decoding empty space
            padded_labels = np.full((len(all_labels), max_label_len), -100, dtype=np.int64)
            for idx, (p, l) in enumerate(zip(all_preds, all_labels)):
                padded_preds[idx, :len(p)] = p
                padded_labels[idx, :len(l)] = l
            # Fire the helper metrics engine with tuple matching Trainer format
            metrics_dict = compute_metrics((padded_preds, padded_labels))
            wandb.log(metrics_dict)
            row = {
                "srl_type": srl_type,
                "mode": "zero_shot" if is_zero_shot else "eval",
                "test_lang": test_lang,
                "model": model_to_load,
                **metrics_dict
            }
            all_results.append(row)
            wandb.finish()

        # Write files sequentially after the main loop
        df = pd.DataFrame(all_results)
        run_results_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        results_path = os.path.join(run_results_dir, f"{run_name}_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nEvaluation completed. Results saved to {results_path}")

    # Forces all GPUs to wait here until Rank 0 finishes generating and writing files.
    # Prevents torchrun from crashing the process group prematurely.
    accelerator.wait_for_everyone()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


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