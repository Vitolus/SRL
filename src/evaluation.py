import numpy as np
from datasets import load_dataset
import os, re, csv
import pandas as pd
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
import wandb
import scorer_united_span as scorer_united_span
import scorer_united_dependency as scorer_united_dep

va_roles = ["AGENT","ASSET","ATTRIBUTE","BENEFICIARY","CAUSE","CO-AGENT","CO-PATIENT","CO-THEME", "DESTINATION","EXPERIENCER","EXTENT","GOAL", "IDIOM","INSTRUMENT","LOCATION","MATERIAL","PATIENT","PRODUCT","PURPOSE","RECIPIENT","RESULT","SOURCE","STIMULUS","THEME","TIME","TOPIC","VALUE"]

# ---------------------------
# Preprocessing
# ---------------------------

def get_VA_arg_struct():
    VA = {}
    with open(os.path.join('data', 'verbatlas_worksheet_1.1 - Clustering.tsv')) as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [line for line in reader]
    for line in lines:
        if line[0] != '':
            frame = line[1].upper()
            #if frame == 'CARRY-OUT-ACTION':
            #    print('CARRY-OUT-ACTION')
            arguments = [l.strip().lower() for l in line[2:] if l != '' and len(l.strip().split()) == 1]
            VA[frame] = {}
            VA[frame]['roles'] = arguments
            VA[frame]['synsets'] = []
        else:
            if line[3].strip().lower().startswith('bn:'):
                VA[frame]['synsets'].append(line[3].strip().lower())
    return VA

def get_tokenizer(tokenizer):
    frames = [k for k in get_VA_arg_struct()]
    tokenizer.add_special_tokens({'additional_special_tokens': frames + va_roles})
    return tokenizer

def make_preprocess(tokenizer_, max_length=1024):
    def preprocess_(batch):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "forced_bos_token_id" : []}
        map_langs_ = {"EN": "en_XX",
                     'ZH': 'zh_CN',
                     "FR": 'fr_XX',
                     "ES": "es_XX"
                      }
        for src, lang, tgt in zip(batch["input"], batch["lang"], batch["output"]):
            tokenizer_.src_lang = map_langs_[lang]
            tokenizer_.tgt_lang = map_langs_[lang]
            # Encode source
            encoded = tokenizer_(src, truncation=True, padding="max_length", max_length=max_length)
            # Encode target
            #with tokenizer.as_target_tokenizer():
            labels = tokenizer_(text_target=tgt, truncation=True, padding="max_length", max_length=max_length)
            pad = tokenizer_.pad_token_id
            labels = [tok if tok != pad else -100 for tok in labels["input_ids"]]
            model_inputs["input_ids"].append(encoded["input_ids"])
            model_inputs["attention_mask"].append(encoded["attention_mask"])
            model_inputs["labels"].append(labels)
            model_inputs["forced_bos_token_id"].append(tokenizer_.lang_code_to_id[map_langs_[lang]])
        return model_inputs
    return preprocess_

### ----

def clean_linearization_span(text: str) -> str:
    text = normalize_tags(text)
    VALID_TAG = re.compile(r"<(/?P\d+):([^>\s]+)>")
    tokens = re.split(r"(<[^>]+>)", text)
    cleaned_tokens = []

    for tok in tokens:
        if not tok.strip():
            continue

        if tok.startswith("<"):
            tok = tok.strip()
            tok = tok.replace(">>", ">")
            tok = tok.replace("</P", "</P")

            if not VALID_TAG.match(tok):
                m = re.match(r"<\/?P\d+:([A-Z0-9_]+)", tok)
                if m:
                    role = m.group(1)
                    tok = f"</P0:{role}>" if tok.startswith("</") else f"<P0:{role}>"

            if VALID_TAG.match(tok):
                cleaned_tokens.append(tok)
        else:
            cleaned_tokens.append(tok)

    return "".join(cleaned_tokens)


def normalize_tags(text: str) -> str:
    # Remove spaces after < and before >
    text = re.sub(r"<\s+", "<", text)
    text = re.sub(r"\s+>", ">", text)

    # Remove spaces after </ and before >
    text = re.sub(r"<\s*/\s*", "</", text)

    # Remove spaces around colon inside tags
    text = re.sub(r":\s+", ":", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

def clean_linearization_dep(text: str) -> str:
    """
    Clean a linearized SRL string before parsing.
    - Fix small tag errors (missing ':', misplaced slashes).
    - Enforce <Pn:ROLE> format.
    - Remove unbalanced tags entirely (drop opening if not matched).
    """
    #print(text)
    text = normalize_tags(text)
    #print(text)
    #print()
    tag_pattern = re.compile(r"</?P\d+:[A-Za-z0-9_\-]+>")
    parts = re.split(r"(<[^>]+>)", text)
    parts = [p for p in parts if p.strip()]

    output = []
    stack = []  # (tag, index_in_output)

    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            tag = part.strip()

            # Try to repair common issues
            if not tag_pattern.fullmatch(tag):
                tag = tag.replace(" ", "")
                tag = tag.replace("</", "<")  # fix broken opener
                if ":" not in tag and re.match(r"<P\d+[A-Za-z]+>", tag):
                    tag = re.sub(r"(<P\d+)([A-Za-z]+>)", r"\1:\2", tag)

            if tag_pattern.fullmatch(tag):
                if tag.startswith("</"):  # closing
                    opener = tag.replace("/", "", 1)
                    if stack and stack[-1][0] == opener:
                        # ✅ Balanced pair
                        output.append(tag)
                        stack.pop()
                    else:
                        # ❌ Stray closing, ignore
                        continue
                else:  # opening
                    stack.append((tag, len(output)))
                    output.append(tag)
            else:
                # ❌ Invalid tag
                continue
        else:
            # normal word
            output.append(part)

    # ❌ Remove any unmatched openings left in the stack
    for _, idx in reversed(stack):
        output[idx] = ""  # erase the unmatched opening

    return " ".join([o for o in output if o]).strip()



def back_to_dict_dep(text):
    """
    Parse a cleaned linearized annotation string into token structures.
    Assumes tags are balanced and in <Pn:ROLE> format (from clean_linearization).
    Returns a list of tokens with SRL roles.
    """

    parts = re.split(r"(\s+|<[^>]+>)", text)
    parts = [p for p in parts if p.strip()]

    active_roles = []  # stack of (pid, role)
    tokens = []

    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            tag = part[1:-1].strip()  # remove < >
            if tag.startswith("/"):
                # closing tag
                active_roles.pop()
            else:
                # opening tag
                pid, role = tag.split(":", 1)
                active_roles.append((pid, role))
        else:
            # normal word/token
            tok = {"form": part, "frame": None, "roles": {}}
            for pid, role in active_roles:
                if role.upper().strip() not in va_roles:  # convention: predicates are uppercase
                    tok["frame"] = role.upper().strip()
                    tok["roles"][pid] = "B-V"
                else:
                    tok["roles"][pid] = role.lower().strip()
            tokens.append(tok)

    # guarantee at least one token
    if not tokens:
        tokens = [{"form": "<EMPTY>", "frame": None, "roles": {}}]
    return tokens


def back_to_dict_span(text: str):
    """
    Parse a cleaned linearized SRL annotation string into tokens, frames, and roles.
    Robust against unbalanced or malformed tags.
    """
    parts = re.split(r"(\s+|<[^>]+>)", text)
    parts = [p for p in parts if p.strip()]

    tokens = []
    active_roles = []

    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            tag = part[1:-1].strip()  # remove < >
            if tag.startswith("/"):  # closing tag
                if active_roles:
                    active_roles.pop()
                else:
                    # nothing to close → skip (unbalanced closing tag)
                    continue
            else:  # opening tag
                if ":" not in tag:
                    # malformed tag, skip
                    continue
                pid, role = tag.split(":", 1)
                pid, role = pid.strip(), role.strip().upper()
                active_roles.append((pid, role))

        else:
            # normal word/token
            tok = {"form": part, "frame": None, "roles": {}}
            for pid, role in active_roles:
                if role.upper().strip() not in va_roles:  # convention: predicates are uppercase
                    tok["frame"] = role.upper().strip()
                    tok["roles"][pid] = "B-V"
                else:
                    tok["roles"][pid] = role.lower().strip()
            tokens.append(tok)
    if not tokens:
        tokens = [{"form": "<EMPTY>", "frame": None, "roles": {}}]
    return tokens


def to_connl_file_span(parsed, fw, united_id, doc_id, sent_id, domain="domain", placeholder_bn="bn:00000000v"):
    fw.write('# united_srl_id = ' + united_id + '\n')
    fw.write('# document_id = ' + doc_id + '\n')
    fw.write('# sentence_id = ' + sent_id + '\n')
    fw.write('# domain = ' + domain + '\n')
    fw.write('# text = ' + ' '.join([t['form'] for t in parsed ]) + '\n')

    pred_ids = sorted({pid for tok in parsed for pid in tok["roles"].keys()})

    lines = []
    for i, tok in enumerate(parsed):
        row = [
            str(i),  # ID
            tok["form"],  # FORM
            tok["form"],  # LEMMA (same as form)
            tok["frame"] or "_",  # FRAME
            placeholder_bn if tok["frame"] else '_'  # BABELNET placeholder
        ]
        for pid in pred_ids:
            row.append(tok["roles"].get(pid, "_"))
        lines.append("\t".join(row))
        fw.write('\t'.join(row) + '\n')
        #if srl_type == 'span':
        #    for i, r in enumerate(rols):
        #        if r != '_':
        #            check_roles[i].append(r)
    fw.write('\n')

    # if srl_type == 'span':
    #    print('')



def to_connl_file_dep(tokens, fw, united_id, doc_id, sent_id, domain="domain", placeholder_bn="bn:00000000v"):
    fw.write('# united_srl_id = ' + united_id + '\n')
    fw.write('# document_id = ' + doc_id + '\n')
    fw.write('# sentence_id = ' + sent_id + '\n')
    fw.write('# domain = ' + domain + '\n')
    fw.write('# text = ' + ' '.join([t['form'] for t in tokens ]) + '\n')
    pred_ids = sorted({pid for tok in tokens for pid in tok["roles"].keys()})

    lines = []
    for i, tok in enumerate(tokens):
        row = [
            str(i),  # ID
            tok["form"],  # FORM
            tok["form"],  # LEMMA (same as form)
            tok["frame"] or "_",  # FRAME
            placeholder_bn if tok["frame"] else '_'  # BABELNET placeholder
        ]
        for pid in pred_ids:
            row.append(tok["roles"].get(pid, "_"))
        lines.append("\t".join(row))
        fw.write('\t'.join(row) + '\n')
        #if srl_type == 'span':
        #    for i, r in enumerate(rols):
        #        if r != '_':
        #            check_roles[i].append(r)
    fw.write('\n')

    # if srl_type == 'span':
    #    print('')


def to_conll(actuals, predictions, srl_type, df, langs, run_dir='results'):
    langs_ = '_'.join(langs)
    with (open(os.path.join(run_dir, f'{srl_type}_{langs_}_predictions.conll'), 'w') as fw,
          open(os.path.join(run_dir, f'{srl_type}_{langs_}_actuals.conll'), 'w') as fw_actuals):
        if srl_type.startswith('dep'):
            for index, (id_, prediction) in enumerate(zip(df['id'], actuals)):
                clean_prediction = clean_linearization_dep(prediction)
                tokens = back_to_dict_dep(clean_prediction)
                doc_id = id_.split('_')[0]
                sent_id = ''.join(id_.split('_')[1:]) if len(id_.split('_')[1:]) > 0 else 'missing'
                to_connl_file_dep(tokens, fw_actuals, id_, doc_id, sent_id)
            for id_, prediction in zip(df['id'], predictions):
                clean_prediction = clean_linearization_dep(prediction)
                tokens = back_to_dict_dep(clean_prediction)
                doc_id = id_.split('_')[0]
                sent_id = ''.join(id_.split('_')[1:]) if len(id_.split('_')[1:]) > 0 else 'missing'
                to_connl_file_dep(tokens, fw, id_, doc_id, sent_id)
        if srl_type == 'span':
            for id_, prediction in zip(df['id'], actuals):
                clean_prediction = clean_linearization_span(prediction)
                tokens = back_to_dict_span(clean_prediction)
                doc_id = id_.split('_')[0]
                sent_id = ''.join(id_.split('_')[1:]) if len(id_.split('_')[1:]) > 0 else 'missing'
                to_connl_file_span(tokens, fw_actuals, id_, doc_id, sent_id)
            for id_, prediction in zip(df['id'], predictions):
                clean_prediction = clean_linearization_span(prediction)
                tokens = back_to_dict_span(clean_prediction)
                doc_id = id_.split('_')[0]
                sent_id = ''.join(id_.split('_')[1:]) if len(id_.split('_')[1:]) > 0 else 'missing'
                to_connl_file_span(tokens, fw, id_, doc_id, sent_id)


# ---------------------------
# Simple metrics
# ---------------------------
def prepare_compute_metrics(val_ds, srl_type, langs, tokenizer, run_name=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        #print('after')
        #predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        #print([tokenizer.decode([k]) if k > 0 else k for k in predictions[0]])
        #decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print(tokenizer.decode(predictions[0]))
        decoded_preds = [tokenizer.decode(prediction, skip_special_tokens=False) for prediction in predictions]
        decoded_labels = tokenizer.batch_decode(
            np.where(labels != -100, labels, tokenizer.pad_token_id),
            skip_special_tokens=False,
        )
        # Slice off any duplicate items added by the 4-GPU distributed sampler
        dataset_length = len(val_ds)
        decoded_preds = decoded_preds[:dataset_length]
        decoded_labels = decoded_labels[:dataset_length]
        # Strip the pad and eos tokens, but leave the AGENT/Frames alone
        pad = tokenizer.pad_token
        eos = tokenizer.eos_token
        decoded_preds = [p.replace(pad, "").replace(eos, "").strip() for p in decoded_preds]
        decoded_labels = [l.replace(pad, "").replace(eos, "").strip() for l in decoded_labels]
        # Remove spaces in between role tags
        decoded_preds = [normalize_tags(p) for p in decoded_preds]
        decoded_labels = [normalize_tags(l) for l in decoded_labels]
        # Calculate basic exact match on all GPUs
        exact = np.mean([p.strip() == r.strip() for p, r in zip(decoded_preds, decoded_labels)])
        result_metrics = {"exact_match": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        # ONLY the Main GPU (Rank 0) performs file writing, heavy scoring, and WandB logging
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            run_dir = os.path.join('results', run_name) if run_name else 'results'
            os.makedirs(run_dir, exist_ok=True)

            to_conll(decoded_labels, decoded_preds, srl_type, val_ds.to_pandas(), langs, run_dir)
            langs_ = '_'.join(langs)
            if srl_type == "span":
                gold_data = scorer_united_span.read_file(os.path.join(run_dir, f'{srl_type}_{langs_}_actuals.conll'))
                pred_data = scorer_united_span.read_file(os.path.join(run_dir, f'{srl_type}_{langs_}_predictions.conll'))
                metrics = scorer_united_span.evaluate(gold_data, pred_data)
            else:
                gold_data = scorer_united_dep.read_file(os.path.join(run_dir, f'{srl_type}_{langs_}_actuals.conll'))
                pred_data = scorer_united_dep.read_file(os.path.join(run_dir, f'{srl_type}_{langs_}_predictions.conll'))
                metrics = scorer_united_dep.evaluate(gold_data, pred_data)
            f1 = metrics['overall-semantics']['coarse-grained']['f1'] * 100
            precision = metrics['overall-semantics']['coarse-grained']['precision'] * 100
            recall = metrics['overall-semantics']['coarse-grained']['recall'] * 100
            print(f'Overall coarse-F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

        #wandb.log({"SCORES": f1})
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            final_df = pd.DataFrame(
                {'Input Text': val_ds.to_pandas()['input'], 'Generated Text': decoded_preds,
                 'Actual Text': decoded_labels})
            final_df.to_csv(os.path.join(run_dir, f"{srl_type}_{'_'.join(langs)}.tsv"), sep='\n')
            print('Output Files generated for review')
            if wandb.run is not None:
                tbl = wandb.Table(data=final_df)
                wandb.log({"Generated text": tbl})

            # Update the return dictionary with actual scores for the Main GPU
            result_metrics = {"exact_match": exact, "f1": f1, "precision": precision, "recall": recall}
        return result_metrics
    return compute_metrics

# ---------------------------
# Main
# ---------------------------
# if __name__ == "__main__":
#     checkpoint_dir = "./mbart_model/"
#     all_results = []
#     map_langs = {"EN": "en_XX",
#                  'ZH': 'zh_CN',
#                  "FR": 'fr_XX',
#                  "ES": "es_XX"
#                  }
#     for srl_type in ["dependency", 'span']:
#         for langs in [['EN'], ['ZH'], ['EN','ZH']]:
#             for test_lang in ['ZH', 'ES', 'EN', 'FR']:
#                 torch.cuda.empty_cache()
#                 wandb.init(project="mbart-testing",settings=wandb.Settings(x_disable_stats=True),name=f"{srl_type}_{'_'.join(langs)}_{test_lang}_FT")
#                 model = MBartForConditionalGeneration.from_pretrained(
#                     checkpoint_dir + '_'.join(langs) + "_" + srl_type + "_FT_best",local_files_only=True)
#                 tokenizer = MBart50Tokenizer.from_pretrained(checkpoint_dir + '_'.join(langs) + "_" + srl_type + "_FT_best",local_files_only=True)
#                 print("Tokenizer vocab size:", len(tokenizer))
#                 print("Model embedding size:", model.get_input_embeddings().weight.size(0))
#                 if len(tokenizer) != model.get_input_embeddings().weight.size(0):
#                     tokenizer = get_tokenizer(tokenizer)
#                     print("Tokenizer vocab size:", len(tokenizer))
#                     print("Model embedding size:", model.get_input_embeddings().weight.size(0))
#                 assert (len(tokenizer) == model.get_input_embeddings().weight.size(0))
#                 # Load dataset (TSV with 4 columns: id, input, output, lang)
#                 print("Tokenizer vocab size:", len(tokenizer))
#                 print("Model embedding size:", model.get_input_embeddings().weight.size(0))
#                 data_files = {
#                     "test": [f"data/linearizations_{srl_type}_Test_{test_lang}.tsv"]
#                 }
#                 #model.config.forced_bos_token_id = tokenizer.lang_code_to_id[map_langs[test_lang]]
#
#                 gen_config = GenerationConfig(**model.generation_config.to_dict())
#                 gen_config.forced_bos_token_id = tokenizer.lang_code_to_id[map_langs[test_lang]]
#                 model.generation_config = gen_config
#
#                 raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")
#                 preprocess = make_preprocess(tokenizer)
#                 test_ds = raw_datasets["test"].map(preprocess, batched=True).shuffle(seed=42)
#                 #test_ds = raw_datasets["test"].select(range(20)).map(preprocess, batched=True)
#                 compute_metrics = prepare_compute_metrics(test_ds, srl_type, [test_lang], tokenizer)
#                 # Training arguments
#                 args = Seq2SeqTrainingArguments(
#                     output_dir="./eval_results",
#                     per_device_eval_batch_size=10,
#                     predict_with_generate=True,
#                     generation_max_length=1024,
#                     logging_dir="./logs",
#                     report_to=["wandb"],   # <--- log to wandb
#                     no_cuda=False,
#                 )
#                 # Trainer
#                 trainer = Seq2SeqTrainer(
#                     model=model,
#                     args=args,
#                     eval_dataset=test_ds,
#                     processing_class=tokenizer,
#                     data_collator=DataCollatorForSeq2Seq(tokenizer, model),
#                     compute_metrics=compute_metrics
#                 )
#                 # Train & evaluate
#                 results = trainer.evaluate()
#                 row = {
#                     "srl_type": srl_type,
#                     "train_lang": '_'.join(langs),
#                     "test_lang": test_lang,
#                     **results
#                 }
#                 all_results.append(row)
#                 print(f"Results for {checkpoint_dir + '_'.join(langs+[srl_type, test_lang]) }: {results}")
#                 df = pd.DataFrame(all_results)
#                 df.to_csv("./results/FT_results.csv", index=False)
#                 wandb.finish()
