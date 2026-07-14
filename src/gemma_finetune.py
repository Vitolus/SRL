import os, gc, argparse, sys, warnings
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth import FastModel, add_new_tokens
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from evaluation import prepare_compute_metrics, va_roles, get_VA_arg_struct
from tqdm.auto import tqdm
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)
if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')
os.environ["WANDB_PROJECT"] = "gemma-srl-finetuning"

def prompt_template(example, roles, frames, srl_type):
    if srl_type == "span":
        format_instructions = (
            "3. Identify the arguments corresponding to each predicate. Wrap the ENTIRE argument phrase in an XML tag formatted as: <Pi:ROLE_NAME>argument text</Pi:ROLE_NAME>, where 'i' matches the integer index of its corresponding predicate.\n"
        )
    elif srl_type == "dependency":
        format_instructions = (
            "3. Identify the arguments corresponding to each predicate. Wrap ONLY the syntactic head word of the argument in an XML tag formatted as: <Pi:ROLE_NAME>head_word</Pi:ROLE_NAME>, where 'i' matches the integer index of its corresponding predicate.\n"
        )
    else:
        raise ValueError(f"Unsupported SRL type: {srl_type}")
    return (
        f"You are a strict {srl_type} based Semantic Role Labeling (SRL) system.\n"
        "ALLOWED ROLES:\n"
        f"{roles}\n\n"
        "ALLOWED FRAMES:\n"
        f"{frames}\n\n"
        "OUTPUT FORMAT INSTRUCTIONS:\n"
        "1. Output ONLY the input sentence modified by XML tags wrapping the predicates and arguments.\n"
        "2. Identify every main predicate. Wrap each predicate word in a tag formatted as: <Pi:SEMANTIC_FRAME>word</Pi:SEMANTIC_FRAME>, where 'i' is the 0-indexed integer order of the predicate (P0 for the first predicate, P1 for the second, etc.).\n"
        f"{format_instructions}\n"
        "4. The index prefix (e.g., P0, P1) must match exactly between a specific predicate and all of its arguments.\n"
        "5. Do not include introductory text, conversational pleasantries, explanations, or trailing remarks. Output ONLY the tagged sentence.\n\n"
        f"Sentence: {example['input']}"
    )

def make_unsloth_chat_format(va_frames, srl_type, tokenizer):
    frames = ", ".join(va_frames)
    roles = ", ".join(va_roles)
    def formatter(example):
        # Convert text columns into standard Hugging Face conversation format
        messages = [
            {"role": "user", "content": prompt_template(example, roles, frames, srl_type)},
            {"role": "model", "content": example['output']}
        ]
        # Compile it using the patched template string
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    return formatter


def train(train_langs, srl_type, model_name, run_name, models_dir):
    # Unsloth Model Loading
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None, # Unsloth automatically resolves the best dtype for GPU
        load_in_4bit=False,  # Using full FP16, disable quantization
    )
    # Inject native Gemma 3 token structures into the tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    va_frames = [k for k in get_VA_arg_struct()]
    add_new_tokens(model, tokenizer, new_tokens= va_frames + va_roles)
    # Unsloth PEFT/LoRA Setup (Crucial for T4 FP16 stability)
    model = FastModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0, # Unsloth optimizes dropout = 0
        bias="none",
        use_gradient_checkpointing="unsloth", # Unsloth's native VRAM saver
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
    # Map into a standard "text" field containing the templated conversation
    formatter = make_unsloth_chat_format(va_frames, srl_type, tokenizer)
    train_ds = combined_train.map(formatter, remove_columns=combined_train.column_names)
    val_ds = combined_val.map(formatter, remove_columns=combined_val.column_names)

    training_args = SFTConfig(
        output_dir=os.path.join(models_dir, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=3,
        save_total_limit=1,
        load_best_model_at_end=True,
        optim="adamw_8bit",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=1024,
        packing=True,
        packing_strategy="bfd",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        neftune_noise_alpha=1,
        ddp_find_unused_parameters=False
    )

    # SFTTrainer handles the causal LM masking and formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer
    )

    # Apply Unsloth's native prompt masking engine
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
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
    accelerator = Accelerator()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
    # Load base model structure first to align tensor coordinates safely
    actual_model_name = base_model_name if not is_zero_shot else model_to_load
    # Unsloth Loading for Evaluation
    model, tokenizer = FastModel.from_pretrained(
        model_name=actual_model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=False,
        device_map={'': local_rank}
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    va_frames = [k for k in get_VA_arg_struct()]
    add_new_tokens(model, tokenizer, new_tokens=va_frames + va_roles)
    # If evaluating a fine-tuned run, manually overlay your saved LoRA adapters now
    if not is_zero_shot:
        print(f"Loading LoRA adapters from {best_model_dir} onto expanded base model...")
        model.load_adapter(best_model_dir)
    # Enable Native 2x Faster Inference
    FastModel.for_inference(model)
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

    for test_lang in test_langs:
        if accelerator.is_main_process:
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
        with accelerator.split_between_processes(list(range(len(test_ds)))) as local_indices:
            local_preds = []
            local_labels = []
            batch_size = 64
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]

            # Each GPU generates ONLY its local slice of data
            for i in tqdm(range(0, len(local_indices), batch_size), desc=f"Gen Rank {accelerator.process_index}",
                          disable=not accelerator.is_main_process):
                batch_idx = local_indices[i:i + batch_size]
                batch_prompts = [test_ds[int(idx)]["prompt"] for idx in batch_idx]
                batch_outputs = [test_ds[int(idx)]["output"] for idx in batch_idx]

                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                                   add_special_tokens=False).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=False, # Greedy decoding for structured tasks
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=terminators,
                        repetition_penalty=1.15
                    )
                # Extract only the newly generated tokens
                prompt_lengths = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, prompt_lengths:]
                # Tokenize labels with add_special_tokens=False to avoid embedding <bos> tokens into Exact Match tests
                labels = tokenizer(batch_outputs, return_tensors="pt", padding=True, truncation=True, max_length=128,
                                   add_special_tokens=False).input_ids.to(model.device)

                # Standardize tensor shapes across GPUs before gathering
                padded_preds = accelerator.pad_across_processes(generated_tokens, dim=1,
                                                                pad_index=tokenizer.pad_token_id)
                padded_labels = accelerator.pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id)

                # Gather all data back to GPU 0 safely
                gathered_preds = accelerator.gather_for_metrics(padded_preds)
                gathered_labels = accelerator.gather_for_metrics(padded_labels)

                if accelerator.is_main_process:
                    local_preds.extend(gathered_preds.cpu().numpy())
                    local_labels.extend(gathered_labels.cpu().numpy())

        # Compute metrics and write results exclusively on GPU 0
        if accelerator.is_main_process:
            compute_metrics = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer, suffix="_eval",
                                                      run_name=f"{run_name}_{test_lang}")

            all_preds = []
            all_labels = []
            for p, l in zip(local_preds, local_labels):
                all_preds.append([tok for tok in p if tok != tokenizer.pad_token_id])
                all_labels.append([tok for tok in l if tok != tokenizer.pad_token_id])
            # Pad tokens arrays globally to their maximum length
            max_pred_len = max((len(p) for p in all_preds), default=0)
            max_label_len = max((len(l) for l in all_labels), default=0)
            final_preds = np.full((len(all_preds), max_pred_len), tokenizer.pad_token_id, dtype=np.int64)
            # Metrics computation expects labels to have -100 padding to skip decoding empty space
            final_labels = np.full((len(all_labels), max_label_len), -100, dtype=np.int64)
            for idx, (p, l) in enumerate(zip(all_preds, all_labels)):
                final_preds[idx, :len(p)] = p
                final_labels[idx, :len(l)] = l
            metrics_dict = compute_metrics((final_preds, final_labels))
            decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
            prediction_table = wandb.Table(columns=["Ground Truth (Target)", "Model Prediction"])
            for gt, pred in zip(decoded_labels, decoded_preds):
                prediction_table.add_data(gt, pred)
            metrics_dict[f"predictions_table"] = prediction_table
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

    # Write sequential files natively on the main process
    if accelerator.is_main_process:
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