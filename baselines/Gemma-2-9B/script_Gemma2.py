"""
Gemma-2-9B-IT Baseline for PII Extraction 
====================================================================
Model: google/gemma-2-9b-it
Paradigm: Zero-shot prompting with task description (no examples)
Evaluation: 80/20 held-out split on PII data, entity-level F1
            (same protocol as GPT-4o, UniNER-7B, GLiNER, LLaMA-3)

This baseline tests Gemma-2's zero-shot ability — the simplest possible
prompting approach, representing the lower bound of what modern LLMs
can achieve on PII extraction without any task-specific guidance.

Usage:
    pip install torch transformers accelerate pandas seqeval
    python script_Gemma2.py

Hardware: NVIDIA RTX 3090 (24GB) — ~18GB VRAM in bfloat16

Note: You need to accept Google's license on HuggingFace first:
      https://huggingface.co/google/gemma-2-9b-it
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import pandas as pd
import numpy as np
import re
import time
import json
import torch
torch._dynamo.config.suppress_errors = True

from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_NAME = "google/gemma-2-9b-it"
DATA_PATH = "../../data/sample/sample_pii_tweets.csv"
SPLIT_RATIO = 0.8
RANDOM_SEED = 42
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # Deterministic

PII_CATEGORIES = ["Age", "Contact", "Date", "ID", "Location", "Name", "Profession"]


# ============================================================
# 2. DATA LOADING & SPLITTING (same as other baselines)
# ============================================================
def load_and_split_data(data_path, split_ratio=0.8, seed=42):
    df = pd.read_csv(data_path, encoding='cp1252')
    df = df.dropna(subset=['Tokens', 'Word_Level_BIOES']).reset_index(drop=True)
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * split_ratio)
    test_indices = indices[split_idx:]
    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"Dataset: {len(df)} total → {len(test_df)} test samples")
    return test_df


# ============================================================
# 3. GROUND TRUTH PARSING (same as other baselines)
# ============================================================
def parse_bioes_to_entities(tokens_str, bioes_str):
    tokens = tokens_str.strip().split()
    tags = bioes_str.strip().split()
    min_len = min(len(tokens), len(tags))
    tokens = tokens[:min_len]
    tags = tags[:min_len]

    entities = []
    current_tokens = []
    current_cat = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_tokens and current_cat:
                entities.append((' '.join(current_tokens), current_cat))
            current_cat = tag[2:]
            current_tokens = [token]
        elif tag.startswith('I-') and current_tokens:
            current_tokens.append(token)
        elif tag.startswith('E-') and current_tokens:
            current_tokens.append(token)
            entities.append((' '.join(current_tokens), current_cat))
            current_tokens = []
            current_cat = None
        elif tag.startswith('S-'):
            entities.append((token, tag[2:]))
            current_tokens = []
            current_cat = None
        else:
            if current_tokens and current_cat:
                entities.append((' '.join(current_tokens), current_cat))
            current_tokens = []
            current_cat = None

    if current_tokens and current_cat:
        entities.append((' '.join(current_tokens), current_cat))
    return entities


# ============================================================
# 4. ZERO-SHOT PROMPT
# ============================================================
def build_prompt(tweet_text):
    """
    Build zero-shot prompt for Gemma-2-9B-IT.
    No examples — just task description and output format.
    This represents the simplest prompting baseline.
    """
    prompt = (
        "Extract all Personally Identifiable Information (PII) from the following "
        "social media post. Classify each PII entity into exactly one of these "
        "categories: Age, Contact, Date, ID, Location, Name, Profession.\n\n"
        "Output format: one entity per line as \"- Category: \\\"entity text\\\"\"\n"
        "If no PII is found, respond with \"No PII found.\"\n"
        "Only extract text spans that actually appear in the post.\n\n"
        f"Post: \"{tweet_text}\"\n\n"
        "PII entities:"
    )
    return prompt


# ============================================================
# 5. RESPONSE PARSING (same as LLaMA-3 baseline)
# ============================================================
def parse_llm_response(response_text):
    entities = []
    if not response_text or "no pii" in response_text.lower():
        return entities

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Pattern: "- Category: "entity text""
        match = re.match(
            r'^[-*•]?\s*(\w+(?:\s+\w+)?)\s*:\s*["\"](.+?)["\"]',
            line
        )
        if match:
            cat_raw = match.group(1).strip()
            entity_text = match.group(2).strip()
        else:
            match = re.match(
                r'^[-*•]?\s*(\w+(?:\s+\w+)?)\s*:\s*(.+)',
                line
            )
            if match:
                cat_raw = match.group(1).strip()
                entity_text = match.group(2).strip().strip('"\'')
            else:
                continue

        cat_mapped = map_category(cat_raw)
        if cat_mapped and entity_text:
            entities.append((entity_text, cat_mapped))

    return entities


def map_category(raw_cat):
    raw = raw_cat.lower().strip()
    mapping = {
        'age': 'Age',
        'contact': 'Contact',
        'phone': 'Contact',
        'email': 'Contact',
        'phone number': 'Contact',
        'date': 'Date',
        'id': 'ID',
        'identification': 'ID',
        'identification number': 'ID',
        'location': 'Location',
        'address': 'Location',
        'place': 'Location',
        'name': 'Name',
        'person': 'Name',
        'person name': 'Name',
        'profession': 'Profession',
        'job': 'Profession',
        'occupation': 'Profession',
        'job title': 'Profession',
    }
    return mapping.get(raw, None)


# ============================================================
# 6. ENTITY-LEVEL EVALUATION (same as other baselines)
# ============================================================
def normalize_entity_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,;:!?')
    return text


def compute_entity_level_metrics(all_gold, all_pred):
    tp_total = fp_total = fn_total = 0
    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)

    for gold_entities, pred_entities in zip(all_gold, all_pred):
        gold_set = {(normalize_entity_text(t), c) for t, c in gold_entities}
        pred_set = {(normalize_entity_text(t), c) for t, c in pred_entities}

        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)

        for _, cat in tp: cat_tp[cat] += 1
        for _, cat in fp: cat_fp[cat] += 1
        for _, cat in fn: cat_fn[cat] += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    categories = sorted(set(list(cat_tp) + list(cat_fp) + list(cat_fn)))
    cat_metrics = {}
    for cat in categories:
        tp_c, fp_c, fn_c = cat_tp[cat], cat_fp[cat], cat_fn[cat]
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        cat_metrics[cat] = {
            'precision': p, 'recall': r, 'f1': f,
            'tp': tp_c, 'fp': fp_c, 'fn': fn_c,
            'support': tp_c + fn_c
        }

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
        'per_category': cat_metrics
    }


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Gemma-2-9B-IT BASELINE FOR PII EXTRACTION")
    print(f"Model: {MODEL_NAME}")
    print(f"Paradigm: Zero-shot (task description only, no examples)")
    print(f"Temperature: {TEMPERATURE} (deterministic)")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading Gemma-2-9B-IT...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # Gemma-2 works best with bfloat16
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load data
    print("\n[2/4] Loading and splitting data...")
    test_df = load_and_split_data(DATA_PATH, SPLIT_RATIO, RANDOM_SEED)

    # Run inference
    print(f"\n[3/4] Running zero-shot inference on {len(test_df)} test samples...")
    all_gold = []
    all_pred = []
    errors = 0
    t_start = time.time()

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        tokens_str = str(row['Tokens']).strip()
        bioes_str = str(row['Word_Level_BIOES']).strip()
        tweet_text = str(row['Tweet Content']).strip()

        # Ground truth
        gold_entities = parse_bioes_to_entities(tokens_str, bioes_str)
        all_gold.append(gold_entities)

        # Build prompt
        prompt = build_prompt(tweet_text)

        # Format for Gemma-2 chat template
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            if input_ids.shape[1] > 4096:
                all_pred.append([])
                errors += 1
                continue

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            pred_entities = parse_llm_response(response)
            all_pred.append(pred_entities)

        except Exception as e:
            print(f"  [ERROR] Sample {idx}: {e}")
            all_pred.append([])
            errors += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(test_df) - idx - 1) / rate
            print(f"  Processed {idx+1}/{len(test_df)} "
                  f"({rate:.1f} samples/sec, ETA: {eta:.0f}s)")

    t_total = time.time() - t_start
    print(f"  Done: {len(test_df)} samples in {t_total:.1f}s "
          f"({len(test_df)/t_total:.1f} samples/sec, {errors} errors)")

    # Evaluate
    print("\n[4/4] Computing metrics...")
    results = compute_entity_level_metrics(all_gold, all_pred)

    print("\n" + "=" * 70)
    print("ENTITY-LEVEL RESULTS")
    print("=" * 70)
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  Recall:    {results['recall']*100:.2f}%")
    print(f"  F1-score:  {results['f1']*100:.2f}%")
    print(f"  (TP={results['tp']}, FP={results['fp']}, FN={results['fn']})")

    print(f"\n  Per-Category Breakdown:")
    print(f"  {'Category':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-'*42}")
    for cat in PII_CATEGORIES:
        if cat in results['per_category']:
            m = results['per_category'][cat]
            print(f"  {cat:<12} {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% "
                  f"{m['f1']*100:>6.2f}% {m['support']:>7d}")
        else:
            print(f"  {cat:<12} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'0':>8}")

    # Save results
    output = {
        'model': MODEL_NAME,
        'paradigm': 'zero-shot (task description only)',
        'temperature': TEMPERATURE,
        'test_size': len(test_df),
        'errors': errors,
        'inference_time_seconds': round(t_total, 1),
        'entity_level': {
            'precision': round(results['precision'] * 100, 2),
            'recall': round(results['recall'] * 100, 2),
            'f1': round(results['f1'] * 100, 2),
        },
        'per_category': {
            cat: {k: round(v * 100, 2) if k not in ('support', 'tp', 'fp', 'fn') else v
                  for k, v in m.items()}
            for cat, m in results['per_category'].items()
        }
    }

    with open('gemma2_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to gemma2_baseline_results.json")

    # Save sample predictions
    samples = []
    for i in range(min(100, len(test_df))):
        samples.append({
            'text': str(test_df.iloc[i]['Tweet Content'])[:200],
            'gold': all_gold[i],
            'pred': all_pred[i],
        })
    with open('gemma2_baseline_predictions.json', 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"  Gemma-2-9B-IT + zero-shot")
    print(f"  Entity-level F1: {results['f1']*100:.2f}%")
    print(f"  Paradigm: Zero-shot prompting (open-source LLM)")
    print(f"  No training, no examples. Greedy decoding.")
    print(f"  Inference: {t_total:.0f}s for {len(test_df)} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()