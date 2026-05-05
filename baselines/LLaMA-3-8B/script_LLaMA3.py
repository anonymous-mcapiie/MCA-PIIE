"""
LLaMA-3-8B-Instruct Baseline for PII Extraction 
==========================================================================
Model: NousResearch/Meta-Llama-3-8B-Instruct (non-gated public mirror)
Paradigm: Few-shot in-context learning (3 examples from training set)
Evaluation: 80/20 held-out split on PII data, entity-level F1
            (same protocol as GPT-4o, UniNER-7B, GLiNER baselines)

This baseline tests LLaMA-3's native few-shot learning ability for PII
extraction WITHOUT GPT-NER style prompting — representing the standard
in-context learning paradigm.

Usage:
    pip install torch transformers accelerate pandas seqeval
    python script_LLaMA3.py

Hardware: NVIDIA RTX 3090 (24GB) — ~16GB VRAM in float16

Note: This script uses NousResearch/Meta-Llama-3-8B-Instruct, a public
      mirror of the official Meta model that does not require gated access.
      To use the official Meta model, accept the license at:
      https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
"""

import pandas as pd
import numpy as np
import re
import time
import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
DATA_PATH = "../../data/sample/sample_pii_tweets.csv"
SPLIT_RATIO = 0.8
RANDOM_SEED = 42
NUM_FEW_SHOT = 3  # Number of in-context examples
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # Deterministic (greedy decoding)

# PII categories in our dataset
PII_CATEGORIES = ["Age", "Contact", "Date", "ID", "Location", "Name", "Profession"]


# ============================================================
# 2. DATA LOADING & SPLITTING
# ============================================================
def load_and_split_data(data_path, split_ratio=0.8, seed=42):
    """Load PII_tweet.csv and create 80/20 split."""
    df = pd.read_csv(data_path, encoding='cp1252')
    df = df.dropna(subset=['Tokens', 'Word_Level_BIOES']).reset_index(drop=True)

    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * split_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"Dataset: {len(df)} total → {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


# ============================================================
# 3. GROUND TRUTH PARSING (same as other baselines)
# ============================================================
def parse_bioes_to_entities(tokens_str, bioes_str):
    """Convert BIOES tags to list of (entity_text, category) tuples."""
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
# 4. FEW-SHOT EXAMPLE SELECTION
# ============================================================
def select_few_shot_examples(train_df, n=3, seed=42):
    """
    Select diverse few-shot examples from training set.
    Strategy: pick examples covering different PII categories.
    """
    np.random.seed(seed)

    # Try to get examples with diverse PII types
    selected = []
    used_types = set()

    # Shuffle training data
    shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for _, row in shuffled.iterrows():
        pii_type = str(row.get('Detected_PII_Types', ''))
        tokens = str(row['Tokens']).strip()
        bioes = str(row['Word_Level_BIOES']).strip()

        # Skip very long tweets
        if len(tokens.split()) > 30:
            continue

        entities = parse_bioes_to_entities(tokens, bioes)
        if not entities:
            continue

        # Prefer examples with PII types we haven't covered yet
        new_types = set(cat for _, cat in entities) - used_types
        if new_types or len(selected) < n:
            selected.append({
                'text': str(row['Tweet Content']).strip(),
                'entities': entities
            })
            used_types.update(cat for _, cat in entities)

        if len(selected) >= n:
            break

    # If we still need more, just grab any valid examples
    if len(selected) < n:
        for _, row in shuffled.iterrows():
            tokens = str(row['Tokens']).strip()
            bioes = str(row['Word_Level_BIOES']).strip()
            entities = parse_bioes_to_entities(tokens, bioes)
            if entities and len(tokens.split()) <= 30:
                example = {
                    'text': str(row['Tweet Content']).strip(),
                    'entities': entities
                }
                if example not in selected:
                    selected.append(example)
            if len(selected) >= n:
                break

    return selected


def format_entities_for_prompt(entities):
    """Format entity list as readable string for prompt."""
    if not entities:
        return "No PII found."
    lines = []
    for text, cat in entities:
        lines.append(f"- {cat}: \"{text}\"")
    return "\n".join(lines)


# ============================================================
# 5. PROMPT CONSTRUCTION
# ============================================================
def build_prompt(tweet_text, few_shot_examples):
    """
    Build few-shot prompt for LLaMA-3-8B-Instruct.
    Uses LLaMA-3's chat template format.
    """
    system_msg = (
        "You are a PII (Personally Identifiable Information) extraction expert. "
        "Given a social media post, extract all PII entities and classify each into "
        "one of these categories: Age, Contact, Date, ID, Location, Name, Profession.\n\n"
        "Rules:\n"
        "- Only extract text spans that are actually present in the post.\n"
        "- Each entity should be classified into exactly one category.\n"
        "- If no PII is found, respond with \"No PII found.\"\n"
        "- Output format: one entity per line as \"- Category: \\\"entity text\\\"\""
    )

    # Build few-shot examples
    examples_text = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"\nExample {i}:\n"
        examples_text += f"Post: \"{ex['text']}\"\n"
        examples_text += f"PII entities:\n{format_entities_for_prompt(ex['entities'])}\n"

    user_msg = (
        f"{examples_text}\n"
        f"Now extract PII from this post:\n"
        f"Post: \"{tweet_text}\"\n"
        f"PII entities:"
    )

    return system_msg, user_msg


# ============================================================
# 6. RESPONSE PARSING
# ============================================================
def parse_llm_response(response_text):
    """
    Parse LLM output to extract (entity_text, category) tuples.
    Expected format: "- Category: \"entity text\""
    Also handles variations like "Category: entity text" etc.
    """
    entities = []

    if not response_text or "no pii" in response_text.lower():
        return entities

    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try pattern: "- Category: "entity text""
        match = re.match(
            r'^[-*•]?\s*(\w+(?:\s+\w+)?)\s*:\s*["\"](.+?)["\"]',
            line
        )
        if match:
            cat_raw = match.group(1).strip()
            entity_text = match.group(2).strip()
        else:
            # Try pattern: "- Category: entity text"
            match = re.match(
                r'^[-*•]?\s*(\w+(?:\s+\w+)?)\s*:\s*(.+)',
                line
            )
            if match:
                cat_raw = match.group(1).strip()
                entity_text = match.group(2).strip().strip('"\'')
            else:
                continue

        # Map to our PII categories
        cat_mapped = map_category(cat_raw)
        if cat_mapped and entity_text:
            entities.append((entity_text, cat_mapped))

    return entities


def map_category(raw_cat):
    """Map LLM output category to our standard PII categories."""
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
# 7. ENTITY-LEVEL EVALUATION (same as other baselines)
# ============================================================
def normalize_entity_text(text):
    """Normalize entity text for flexible matching."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,;:!?')
    return text


def compute_entity_level_metrics(all_gold, all_pred):
    """Compute entity-level P/R/F1 with per-category breakdown."""
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
# 8. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("LLaMA-3-8B-Instruct BASELINE FOR PII EXTRACTION")
    print(f"Model: {MODEL_NAME}")
    print(f"Paradigm: Few-shot in-context learning ({NUM_FEW_SHOT} examples)")
    print(f"Temperature: {TEMPERATURE} (deterministic)")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading LLaMA-3-8B-Instruct...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load data
    print("\n[2/5] Loading and splitting data...")
    train_df, test_df = load_and_split_data(DATA_PATH, SPLIT_RATIO, RANDOM_SEED)

    # Select few-shot examples from training set
    print(f"\n[3/5] Selecting {NUM_FEW_SHOT} few-shot examples...")
    few_shot_examples = select_few_shot_examples(train_df, n=NUM_FEW_SHOT, seed=RANDOM_SEED)
    print("  Selected examples:")
    for i, ex in enumerate(few_shot_examples):
        cats = [c for _, c in ex['entities']]
        print(f"    {i+1}. \"{ex['text'][:60]}...\" → {cats}")

    # Run inference
    print(f"\n[4/5] Running inference on {len(test_df)} test samples...")
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
        system_msg, user_msg = build_prompt(tweet_text, few_shot_examples)

        # Format for LLaMA-3 chat template
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            # Check length — skip if too long
            if input_ids.shape[1] > 4096:
                all_pred.append([])
                errors += 1
                continue

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # Greedy
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the generated part
            generated = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            # Parse response
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
    print("\n[5/5] Computing metrics...")
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
        'paradigm': f'few-shot ICL ({NUM_FEW_SHOT} examples)',
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
            cat: {k: round(v * 100, 2) if k != 'support' and k != 'tp' and k != 'fp' and k != 'fn' else v
                  for k, v in m.items()}
            for cat, m in results['per_category'].items()
        }
    }

    with open('llama3_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to llama3_baseline_results.json")

    # Save sample predictions
    samples = []
    for i in range(min(100, len(test_df))):
        samples.append({
            'text': str(test_df.iloc[i]['Tweet Content'])[:200],
            'gold': all_gold[i],
            'pred': all_pred[i],
        })
    with open('llama3_baseline_predictions.json', 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"  LLaMA-3-8B-Instruct + {NUM_FEW_SHOT}-shot ICL")
    print(f"  Entity-level F1: {results['f1']*100:.2f}%")
    print(f"  Paradigm: Few-shot in-context learning (open-source LLM)")
    print(f"  No training required. Greedy decoding.")
    print(f"  Inference: {t_total:.0f}s for {len(test_df)} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()