import os, gc, argparse, json
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from evaluation import get_tokenizer, prepare_compute_metrics
import sys
import warnings
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)

if os.path.basename(os.getcwd()) == 'src':
    os.chdir('..')
if 'src' not in sys.path:
    sys.path.append('src')

os.environ["WANDB_PROJECT"] = "gemma-srl-finetuning"


# [Inference] Gemma instruction models perform best when wrapped in their native conversational template.
def apply_chat_template(example, is_training=True):
    """Formats the input and output into Gemma's expected chat template."""
    # Note: Modify the system/user instruction prompt below to match your exact SRL task phrasing
    prompt = f"<start_of_turn>user\nPerform Semantic Role Labeling on this sentence:\n{example['input']}<end_of_turn>\n<start_of_turn>model\n"

    if is_training:
        prompt += f"{example['output']}<end_of_turn>"

    return {"text": prompt}


def finetune(train_langs, srl_type, model_name, run_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = get_tokenizer(tokenizer)

    # Gemma does not have a native pad token; it is standard to use the EOS token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model in bfloat16 as recommended for Gemma models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if not dist.is_initialized() else None
    )
    model.resize_token_embeddings(len(tokenizer))

    train_datasets = []
    val_datasets = []
    loaded_files = set()

    # Load Data using the '-s' isolated logic
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

    # Apply the chat template format
    train_ds = combined_train.map(lambda x: apply_chat_template(x, is_training=True))
    val_ds = combined_val.map(lambda x: apply_chat_template(x, is_training=True))

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, f"{run_name}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,  # Reduced for causal LM memory footprint
        per_device_eval_batch_size=4,
        logging_dir="logs",
        report_to=["wandb"],
        run_name=run_name,
        learning_rate=2e-5,  # Lower learning rate recommended for fine-tuning LLMs
        weight_decay=0.01,
        num_train_epochs=4,
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=True,  # Gemma trains best in bf16
        optim="adamw_bnb_8bit",
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
    )

    # Use DataCollatorForCompletionOnlyLM to only calculate loss on the model's response, not the user prompt
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # TRL's SFTTrainer handles the causal LM masking and formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=256
    )

    print(f"Starting SFT for {run_name}...")
    trainer.train()

    best_model_dir = os.path.join(MODELS_DIR, f"{run_name}_best")
    trainer.save_model(best_model_dir)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        tokenizer.save_pretrained(best_model_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def zero_shot(test_langs, srl_type, model_name, run_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    all_results = []

    for test_lang in test_langs:
        print(f"Running zero-shot inference on {test_lang}...")
        test_file = f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"
        raw_test = load_dataset("csv", data_files={"test": test_file}, delimiter="\t")

        # Apply prompt template without appending the actual output
        test_ds = raw_test["test"].map(lambda x: apply_chat_template(x, is_training=False))

        predictions = []
        references = raw_test["test"]["output"]

        # Basic inference loop
        for batch in test_ds.iter(batch_size=8):
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # Greedy decoding for structured tasks
                    pad_token_id=tokenizer.eos_token_id
                )

            # Slice the output to only extract the newly generated tokens (ignore the prompt)
            prompt_lengths = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, prompt_lengths:]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions.extend(decoded_preds)

        # Assuming prepare_compute_metrics returns a dict of standard metrics
        # You will need to adapt evaluation.py to accept raw lists of text instead of EvalPrediction objects for zero-shot
        metrics_dict = calculate_metrics(predictions, references)  # Placeholder for your custom metric execution

        row = {
            "srl_type": srl_type,
            "mode": "zero_shot",
            "test_lang": test_lang,
            **metrics_dict
        }
        all_results.append(row)

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        df = pd.DataFrame(all_results)
        run_results_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(run_results_dir, exist_ok=True)
        results_path = os.path.join(run_results_dir, f"{run_name}_zeroshot_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Zero-shot completed. Results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gemma 3 Fine-tuning for SRL")
    parser.add_argument("--action", type=str, required=True, choices=['finetune', 'zeroshot'], help="Execution mode")
    parser.add_argument("--srl-type", type=str, required=True, choices=['dependency', 'span'], help="Type of SRL task")
    parser.add_argument("--langs", nargs='+', required=True, help="List of languages (e.g., EN-s ZH-s)")
    args = parser.parse_args()

    # Define model identifier here
    MODEL_NAME = "google/gemma-3-1b-it"

    train_name = "_".join(args.langs)
    RUN_NAME = f"{args.srl_type}_{train_name}_gemma3_1B"

    MODELS_DIR = "gemma_models/"
    RESULTS_DIR = "gemma_results/"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"--- Starting Pipeline: {args.action.upper()} | {args.srl_type.upper()} SRL on {args.langs} ---")

    if args.action == 'finetune':
        finetune(args.langs, args.srl_type, MODEL_NAME, RUN_NAME)
    elif args.action == 'zeroshot':
        zero_shot(args.langs, args.srl_type, MODEL_NAME, RUN_NAME)