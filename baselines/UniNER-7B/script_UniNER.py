"""
UniNER-7B Baseline for PII Detection
======================================
Uses the UniNER-7B-all model (Zhou et al., 2023, "UniversalNER: Targeted
Distillation from Large Language Models for Open Named Entity Recognition")
for PII entity recognition on the PII_tweet target dataset.

This script serves as a new instruction-tuned NER baseline for the MCA-PIIE
paper, evaluated as an LLM-based reference baseline.

UniNER architecture:
  - Based on LLaMA-7B, instruction-tuned on ChatGPT-generated NER data
  - Uses a conversation template: asks "What describes {entity_type}?" per type
  - Returns JSON lists of entities per type
  - Runs locally on a single GPU (3090 24GB is sufficient)

Experimental setup:
  - 80/20 train/test split on PII_tweet.csv (default, consistent with GPT-NER baseline)
  - Metrics: Accuracy, Precision, Recall, F1, AUC, Duration
  - Results directly comparable to Table 8 (Target Domain)

Requirements:
  pip install torch transformers accelerate tqdm scikit-learn pandas

Usage:
  # Quick test (10 samples)
  python uniner_baseline.py --max_samples 10

  # Full run (auto-resumes if interrupted)
  python uniner_baseline.py

  # Use a specific model variant
  python uniner_baseline.py --model Universal-NER/UniNER-7B-type


"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q",
                           "--break-system-packages"])
    from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

PII_TYPES = ["Age", "Contact", "Date", "ID", "Location", "Name", "Profession"]

# UniNER conversation template (from official repo)
# The model expects this exact format:
#   "A virtual assistant answers questions from a user based on the provided text.
#    USER: Text: {text}
#    ASSISTANT: I've read this text.
#    USER: What describes {entity_type} in the text?
#    ASSISTANT: "
# Then the model generates a JSON list like ["entity1", "entity2"]

UNINER_PROMPT_TEMPLATE = (
    "A virtual assistant answers questions from a user based on the provided text.\n"
    "USER: Text: {text}\n"
    "ASSISTANT: I've read this text.\n"
    "USER: What describes {entity_type} in the text?\n"
    "ASSISTANT:"
)

# Map our PII types to natural language descriptions for UniNER queries
PII_TYPE_QUERIES = {
    "Age":        "age",
    "Contact":    "contact information such as phone number or email address",
    "Date":       "date",
    "ID":         "identification number",
    "Location":   "location",
    "Name":       "person name",
    "Profession": "profession or job title",
}


# ============================================================================
# Data Loading (shared with GPT-NER baseline)
# ============================================================================

def load_pii_data(filepath):
    """Load PII_tweet.csv and return processed data."""
    for enc in ['cp1252', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Loaded {len(df)} rows from {filepath} (encoding: {enc})")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Cannot load {filepath}")

    processed = []
    for idx, row in df.iterrows():
        tags_str = row.get('Word_Level_BIOES', '')
        if pd.isna(tags_str) or str(tags_str).strip() == '':
            continue
        tags = str(tags_str).split()
        tweet_content = str(row.get('Tweet Content', ''))
        if 'Tokens' in df.columns and not pd.isna(row.get('Tokens', None)):
            tokens = str(row['Tokens']).split()
        else:
            tokens = tweet_content.split()
        min_len = min(len(tokens), len(tags))
        if min_len == 0:
            continue
        tokens, tags = tokens[:min_len], tags[:min_len]
        normalized = [
            t if (t == 'O' or any(t.startswith(p) for p in ['B-','I-','E-','S-']))
            else 'O' for t in tags
        ]
        processed.append({
            'tweet_content': tweet_content,
            'tokens': tokens,
            'tags': normalized,
            'id': str(row.get('Tweet Id', f'row_{idx}')),
        })
    print(f"Processed {len(processed)} valid samples")
    return processed


# ============================================================================
# UniNER Model Loading
# ============================================================================

def load_uniner_model(model_path, device="cuda", load_in_8bit=False, load_in_4bit=False):
    """
    Load UniNER model and tokenizer using HuggingFace Transformers.
    Supports optional 8-bit or 4-bit quantization to save VRAM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading UniNER model from: {model_path}")
    print(f"  Device: {device}")
    print(f"  Quantization: {'8-bit' if load_in_8bit else '4-bit' if load_in_4bit else 'none (fp16)'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    # Check VRAM usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.1f} GB")

    return model, tokenizer


# ============================================================================
# UniNER Inference
# ============================================================================

def query_uniner(model, tokenizer, text, entity_type_query, device="cuda",
                 max_new_tokens=256):
    """
    Query UniNER for a single entity type on a given text.

    Args:
        text: The tweet text
        entity_type_query: Natural language description of the entity type

    Returns:
        List of entity strings found, e.g. ["John Smith", "NYC"]
    """
    import torch

    prompt = UNINER_PROMPT_TEMPLATE.format(
        text=text,
        entity_type=entity_type_query
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (after the prompt)
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Parse the response — UniNER returns JSON list like ["entity1", "entity2"]
    entities = parse_uniner_response(response)
    return entities


def parse_uniner_response(response_text):
    """
    Parse UniNER's output into a list of entity strings.
    UniNER typically returns: ["entity1", "entity2"] or []
    Sometimes it returns plain text or partial JSON.
    """
    response_text = response_text.strip()

    # Try direct JSON parse
    try:
        result = json.loads(response_text)
        if isinstance(result, list):
            return [str(e).strip() for e in result if e and str(e).strip()]
        return []
    except json.JSONDecodeError:
        pass

    # Try to find a JSON list in the response
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(e).strip() for e in result if e and str(e).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: if response looks like a plain entity (no brackets)
    # UniNER sometimes returns just the entity text without JSON
    if response_text and response_text not in ['[]', 'None', 'none', 'N/A', '']:
        # Check if it looks like a comma-separated list
        if ',' in response_text and not response_text.startswith('{'):
            parts = [p.strip().strip('"').strip("'") for p in response_text.split(',')]
            return [p for p in parts if p and p not in ['None', 'none', 'N/A']]

    return []


def run_uniner_on_sample(model, tokenizer, tweet_text, tokens, device="cuda"):
    """
    Run UniNER on a single tweet for all PII types.
    Returns BIOES tags aligned with tokens.
    """
    predicted_tags = ['O'] * len(tokens)
    tokens_lower = [t.lower() for t in tokens]
    all_entities = {}  # type -> list of entity strings

    for pii_type, query in PII_TYPE_QUERIES.items():
        entities = query_uniner(model, tokenizer, tweet_text, query, device=device)
        if entities:
            all_entities[pii_type] = entities

    # Map entities to BIOES tags using character-level alignment
    for pii_type, entities in all_entities.items():
        for entity_text in entities:
            if not entity_text.strip():
                continue

            matched = _match_entity_to_tokens(
                entity_text, tokens, tokens_lower, predicted_tags, pii_type
            )

    return predicted_tags, all_entities


def _match_entity_to_tokens(entity_text, tokens, tokens_lower, predicted_tags, pii_type):
    """
    Robustly match an entity string to token positions using multiple strategies.
    Handles tokenization mismatches (e.g., "No.:" vs "No", ".", ":").

    Returns True if a match was found and tags were applied.
    """
    entity_tokens_lower = [t.lower() for t in entity_text.split()]
    n = len(entity_tokens_lower)
    if n == 0:
        return False

    # Strategy 1: Exact token-sequence match (fast path)
    for start in range(len(tokens) - n + 1):
        if tokens_lower[start:start+n] == entity_tokens_lower:
            _apply_bioes(predicted_tags, start, n, pii_type)
            return True

    # Strategy 2: Character-level alignment
    # Build token boundary map on the raw joined string
    entity_lower = entity_text.lower().strip()

    # Build character-to-token index on the ORIGINAL joined text
    char_to_token = []
    for ti, tok in enumerate(tokens_lower):
        for ch in tok:
            char_to_token.append(ti)
        char_to_token.append(-1)  # space separator
    joined_lower = ' '.join(tokens_lower)

    # Also build a CLEANED version (punctuation → space, collapse spaces)
    # with a mapping from cleaned-char-position → original-char-position
    cleaned_chars = []
    clean_to_orig = []
    for oi, ch in enumerate(joined_lower):
        if ch.isalnum():
            cleaned_chars.append(ch)
            clean_to_orig.append(oi)
        elif ch == ' ' or not ch.isalnum():
            # Collapse into single space
            if cleaned_chars and cleaned_chars[-1] != ' ':
                cleaned_chars.append(' ')
                clean_to_orig.append(oi)
    joined_clean = ''.join(cleaned_chars).strip()

    # Clean the entity the same way
    entity_clean = ''.join(
        ch if ch.isalnum() else ' ' for ch in entity_lower
    )
    entity_clean = ' '.join(entity_clean.split())  # collapse spaces

    # Try matching in both raw and cleaned versions
    search_pairs = [
        (entity_lower, joined_lower, list(range(len(joined_lower)))),  # raw → raw
        (entity_clean, joined_clean, clean_to_orig),                    # clean → clean
    ]

    for variant, haystack, pos_map in search_pairs:
        if not variant:
            continue
        pos = haystack.find(variant)
        if pos >= 0:
            end_pos = pos + len(variant) - 1
            # Map back to original char positions, then to tokens
            orig_start = pos_map[pos] if pos < len(pos_map) else -1
            orig_end = pos_map[end_pos] if end_pos < len(pos_map) else -1
            if orig_start >= 0 and orig_end >= 0:
                if orig_start < len(char_to_token) and orig_end < len(char_to_token):
                    start_tok = char_to_token[orig_start]
                    end_tok = char_to_token[orig_end]
                    # Walk past spaces (-1)
                    while start_tok == -1 and orig_start < len(char_to_token) - 1:
                        orig_start += 1
                        start_tok = char_to_token[orig_start]
                    while end_tok == -1 and orig_end > 0:
                        orig_end -= 1
                        end_tok = char_to_token[orig_end]
                    if start_tok >= 0 and end_tok >= 0 and start_tok <= end_tok:
                        span_len = end_tok - start_tok + 1
                        if all(predicted_tags[start_tok + j] == 'O' for j in range(span_len)):
                            _apply_bioes(predicted_tags, start_tok, span_len, pii_type)
                            return True

    # Strategy 3: Single-token containment match (last resort)
    # For short entities (1-2 words), check if any token contains or equals the entity
    if len(entity_clean) >= 3:  # avoid matching very short strings
        for i, tok in enumerate(tokens_lower):
            tok_clean = re.sub(r'[^\w]', '', tok)
            if tok_clean and tok_clean == re.sub(r'[^\w]', '', entity_clean) and predicted_tags[i] == 'O':
                predicted_tags[i] = f'S-{pii_type}'
                return True

    return False


def _apply_bioes(predicted_tags, start, length, pii_type):
    """Apply BIOES tags to a span."""
    if length == 1:
        if predicted_tags[start] == 'O':
            predicted_tags[start] = f'S-{pii_type}'
    else:
        if all(predicted_tags[start + j] == 'O' for j in range(length)):
            predicted_tags[start] = f'B-{pii_type}'
            for j in range(1, length - 1):
                predicted_tags[start + j] = f'I-{pii_type}'
            predicted_tags[start + length - 1] = f'E-{pii_type}'


# ============================================================================
# Checkpoint Management
# ============================================================================

def get_checkpoint_path(output_dir, model_name):
    safe_name = model_name.replace("/", "_").replace("-", "_")
    return os.path.join(output_dir, f'ckpt_{safe_name}.jsonl')


def load_checkpoint(path):
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                completed[rec['sample_id']] = rec['pred_tags']
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def save_checkpoint_entry(path, sample_id, pred_tags, raw_entities=None):
    rec = {'sample_id': sample_id, 'pred_tags': pred_tags,
           'ts': datetime.now().isoformat()}
    if raw_entities:
        rec['entities'] = {k: v[:10] for k, v in raw_entities.items()}  # truncate
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ============================================================================
# Evaluation Metrics (aligned with Table 7/8)
# ============================================================================

def compute_metrics(all_gold_tags, all_pred_tags):
    assert len(all_gold_tags) == len(all_pred_tags)
    gold_bin = [0 if t == 'O' else 1 for t in all_gold_tags]
    pred_bin = [0 if t == 'O' else 1 for t in all_pred_tags]
    accuracy = accuracy_score(all_gold_tags, all_pred_tags)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_bin, pred_bin, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(gold_bin, pred_bin)
    except ValueError:
        auc = 0.0
    cat_f1 = {}
    for pt in PII_TYPES:
        cg = [1 if pt in t else 0 for t in all_gold_tags]
        cp = [1 if pt in t else 0 for t in all_pred_tags]
        if sum(cg) > 0:
            _, _, cf, _ = precision_recall_fscore_support(
                cg, cp, average='binary', zero_division=0)
            cat_f1[pt] = cf
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'category_f1': cat_f1}


# ============================================================================
# Data Splitting
# ============================================================================

def single_split(data, test_ratio=0.2, seed=42):
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - test_ratio))
    return [(indices[:split_point], indices[split_point:])]


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):
    print("=" * 70)
    print("UniNER-7B Baseline for PII Detection")
    print("(Zhou et al., 2023 - UniversalNER)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Data:        {args.data_path}")
    print(f"  Eval mode:   80/20 Split")
    print(f"  Quantize:    {'8-bit' if args.load_8bit else '4-bit' if args.load_4bit else 'fp16'}")
    print(f"  Max samples: {args.max_samples or 'ALL'}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_pii_data(args.data_path)
    if args.max_samples and args.max_samples < len(data):
        random.seed(args.seed)
        data = random.sample(data, args.max_samples)
        print(f"Subsampled to {len(data)} samples")

    folds = single_split(data, test_ratio=0.2, seed=args.seed)
    train_idx, test_idx = folds[0]
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Load checkpoint
    model_short = args.model.split("/")[-1]
    ckpt_path = get_checkpoint_path(args.output_dir, model_short)
    cached = load_checkpoint(ckpt_path)
    n_cached = sum(1 for j in test_idx if data[j]['id'] in cached)
    n_remaining = len(test_idx) - n_cached
    print(f"Checkpoint: {n_cached} cached, {n_remaining} remaining")

    # Load model (only if we have work to do)
    if n_remaining > 0:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: No GPU detected. UniNER-7B will be very slow on CPU.")

        model, tokenizer = load_uniner_model(
            args.model, device=device,
            load_in_8bit=args.load_8bit,
            load_in_4bit=args.load_4bit,
        )
        # Determine actual device (may differ with device_map="auto")
        if hasattr(model, 'device'):
            device = str(model.device)
        else:
            device = "cuda"
    else:
        print("All samples cached. Skipping model loading.")
        model = tokenizer = None
        device = "cuda"

    # Run inference
    exp_start = time.time()
    all_gold, all_pred = [], []
    new_calls = parse_fails = 0

    pbar = tqdm(total=len(test_idx), desc="UniNER", unit="tweet", ncols=110,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining} {rate_fmt}] {postfix}")
    live_f1 = 0.0

    for i, sample_idx in enumerate(test_idx):
        sample = data[sample_idx]
        tokens = sample['tokens']
        gold_tags = sample['tags']
        sid = sample['id']

        if sid in cached:
            pred_tags = cached[sid]
            if len(pred_tags) != len(tokens):
                pred_tags = ['O'] * len(tokens)
        else:
            try:
                pred_tags, raw_ents = run_uniner_on_sample(
                    model, tokenizer, sample['tweet_content'], tokens, device=device
                )
                new_calls += 1
                save_checkpoint_entry(ckpt_path, sid, pred_tags, raw_entities=raw_ents)
            except Exception as e:
                tqdm.write(f"  Error on sample {sid}: {e}")
                pred_tags = ['O'] * len(tokens)
                parse_fails += 1
                save_checkpoint_entry(ckpt_path, sid, pred_tags)

        all_gold.extend(gold_tags)
        all_pred.extend(pred_tags)

        # Live F1
        if (i + 1) % 10 == 0 or (i + 1) == len(test_idx):
            gb = [0 if t == 'O' else 1 for t in all_gold]
            pb = [0 if t == 'O' else 1 for t in all_pred]
            _, _, live_f1, _ = precision_recall_fscore_support(
                gb, pb, average='binary', zero_division=0)

        pbar.set_postfix_str(f"F1={live_f1*100:.1f}% new={new_calls}", refresh=False)
        pbar.update(1)

    pbar.close()

    # ========================================================================
    # Results
    # ========================================================================
    total_dur = time.time() - exp_start
    h, rem = divmod(int(total_dur), 3600)
    mi, se = divmod(rem, 60)

    metrics = compute_metrics(all_gold, all_pred)

    print(f"\n{'='*70}")
    print("RESULTS (80/20 Split)")
    print(f"{'='*70}")
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics['precision']*100:.2f}%")
    print(f"  Recall:      {metrics['recall']*100:.2f}%")
    print(f"  F1-score:    {metrics['f1']*100:.2f}%")
    print(f"  AUC:         {metrics['auc']*100:.2f}%")
    print(f"  Duration:    {h}:{mi:02d}:{se:02d}")
    print(f"  New inferences: {new_calls} (from cache: {n_cached})")
    if parse_fails:
        print(f"  Errors:      {parse_fails}")

    print(f"\n  Per-Category F1:")
    for pt in PII_TYPES:
        v = metrics['category_f1'].get(pt)
        if v is not None:
            print(f"    {pt:12s}: {v*100:.2f}%")

    # Save results
    row = {
        'Method': f'{model_short} (UniNER)',
        'Accuracy': f"{metrics['accuracy']*100:.2f}%",
        'Precision': f"{metrics['precision']*100:.2f}%",
        'Recall': f"{metrics['recall']*100:.2f}%",
        'F1-score': f"{metrics['f1']*100:.2f}%",
        'AUC': f"{metrics['auc']*100:.2f}%",
        'Duration': f"{h}:{mi:02d}:{se:02d}",
    }

    results = {
        'experiment': 'UniNER Baseline', 'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'config': {'eval_mode': '80/20 split', 'seed': args.seed,
                   'max_samples': args.max_samples, 'n_samples': len(data),
                   'n_test': len(test_idx),
                   'quantization': '8bit' if args.load_8bit else '4bit' if args.load_4bit else 'fp16'},
        'metrics': {k: float(v) for k, v in metrics.items() if k != 'category_f1'},
        'category_f1': {k: float(v) for k, v in metrics['category_f1'].items()},
        'duration': f"{h}:{mi:02d}:{se:02d}", 'duration_sec': total_dur,
        'stats': {'new_calls': new_calls, 'cached': n_cached, 'errors': parse_fails},
        'table_8_row': row,
    }

    out = os.path.join(args.output_dir, f'uniner_results_{model_short.replace("-","_")}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    csv_out = os.path.join(args.output_dir, f'uniner_table8_{model_short.replace("-","_")}.csv')
    pd.DataFrame([row]).to_csv(csv_out, index=False)
    print(f"\nResults: {out}")
    print(f"Table 8: {csv_out}")

    print(f"\n{'='*70}")
    print("TABLE 8 ROW (copy to manuscript):")
    print(f"{'='*70}")
    print(f"  Target | {row['Method']} | {row['Accuracy']} | {row['Precision']} | "
          f"{row['Recall']} | {row['F1-score']} | {row['AUC']} | {row['Duration']}")
    print(f"{'='*70}")
    print(f"\nCheckpoint: {ckpt_path} — re-run to resume.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UniNER-7B Baseline for PII Detection',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--model', type=str, default='Universal-NER/UniNER-7B-all',
                        help='HuggingFace model path (default: UniNER-7B-all, the best variant)')
    parser.add_argument('--data_path', type=str,
                        default='../../data/sample/sample_pii_tweets.csv',
                        help='Path to PII data CSV (default: anonymized sample)')
    parser.add_argument('--output_dir', type=str, default='./uniner_output',
                        help='Output & checkpoint directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples (for testing)')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load model in 8-bit quantization (saves VRAM, needs bitsandbytes)')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Load model in 4-bit quantization (saves more VRAM)')

    args = parser.parse_args()
    run_experiment(args)