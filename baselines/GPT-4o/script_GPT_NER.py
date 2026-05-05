"""
GPT-NER Baseline for PII Detection
====================================
Implements the GPT-NER prompting strategy (Wang et al., 2025, NAACL Findings)
for PII entity recognition on the PII_tweet target dataset.

Experimental setup:
  - 5-fold cross-validation on the target PII corpus
  - GPT-NER-style structured prompting with few-shot demonstrations
  - Metrics: Accuracy, Precision, Recall, F1, AUC, Duration
  - Results directly comparable to Table 8 (Target Domain)

Features:
  - tqdm progress bars with live F1 display
  - Per-sample checkpoint: saves after every API call, auto-resumes on restart
  - Cost tracking and estimation

Usage:
  # Quick test (20 samples, cheap model)
  python gpt_ner_baseline.py --model gpt-4o-mini --max_samples 20

  # Full run (auto-resumes if interrupted)
  python gpt_ner_baseline.py --model gpt-4o

  # Resume: just re-run the same command — cached results are reused automatically

Note on reproducibility:
  OpenAI may update the underlying `gpt-4o` snapshot over time without
  changing the model name. Results obtained today may therefore differ
  from those originally reported. To pin a specific snapshot, pass an
  exact model id (e.g., `--model gpt-4o-2024-08-06`).
"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q", "--break-system-packages"])
    from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

PII_TYPES = ["Age", "Contact", "Date", "ID", "Location", "Name", "Profession"]

MODEL_COSTS = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1":     {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":{"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":{"input": 0.10,  "output": 0.40},
    "o3-mini":     {"input": 1.10,  "output": 4.40},
}


# ============================================================================
# GPT-NER Prompt Design
# ============================================================================

SYSTEM_PROMPT = """You are a Named Entity Recognition (NER) system specialized in detecting \
Personally Identifiable Information (PII) in social media posts (tweets).

Your task: Given a tweet, identify ALL PII entities and their types.

PII categories to detect:
- Age: age-related expressions (e.g., "41 years old", "age 8", "turned 30")
- Contact: phone numbers, email addresses, physical addresses for contact
- Date: specific dates, birth dates, time references tied to personal events
- ID: identification numbers, SSN, account numbers, card numbers
- Location: geographic locations, cities, countries, area codes as locations
- Name: person names, usernames when used as real names, nicknames identifying individuals
- Profession: job titles, occupations, professional roles

CRITICAL RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no extra text.
2. For each entity found, provide the exact text span and its PII type.
3. If no PII is found, return {"entities": []}
4. Entity text must be an EXACT substring of the input tweet.
5. Each entity has: "text" (exact span), "type" (one of the 7 categories above).

Output format:
{"entities": [{"text": "exact span", "type": "PII_TYPE"}, ...]}"""


def build_few_shot_examples():
    """Few-shot demonstrations following GPT-NER strategy."""
    return [
        {"tweet": "He was 45 years old!",
         "output": '{"entities": [{"text": "45 years old", "type": "Age"}]}'},
        {"tweet": "reach me at 0179412404",
         "output": '{"entities": [{"text": "0179412404", "type": "Contact"}]}'},
        {"tweet": "Thank you from Lukas, age 8",
         "output": '{"entities": [{"text": "Lukas", "type": "Name"}, {"text": "age 8", "type": "Age"}]}'},
        {"tweet": "#NaijaPidgin @NaijaPidgin_ 0062698268 sterling bank",
         "output": '{"entities": [{"text": "#NaijaPidgin", "type": "Name"}, {"text": "0062698268", "type": "Contact"}]}'},
        {"tweet": "Missing home in Brooklyn, NYC so much today",
         "output": '{"entities": [{"text": "Brooklyn", "type": "Location"}, {"text": "NYC", "type": "Location"}]}'},
        {"tweet": "Born on March 15, 1990 and proud of it",
         "output": '{"entities": [{"text": "March 15, 1990", "type": "Date"}]}'},
        {"tweet": "Working as a software engineer at Google",
         "output": '{"entities": [{"text": "software engineer", "type": "Profession"}, {"text": "Google", "type": "Name"}]}'},
        {"tweet": "What a beautiful day! Love the sunshine #happy",
         "output": '{"entities": []}'},
    ]


def build_messages(tweet_text, few_shot_examples):
    """Build the full message sequence for OpenAI API."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": f"Tweet: {ex['tweet']}"})
        messages.append({"role": "assistant", "content": ex['output']})
    messages.append({"role": "user", "content": f"Tweet: {tweet_text}"})
    return messages


# ============================================================================
# Data Loading
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
        normalized = [t if (t == 'O' or any(t.startswith(p) for p in ['B-','I-','E-','S-'])) else 'O' for t in tags]
        processed.append({
            'tweet_content': tweet_content,
            'tokens': tokens,
            'tags': normalized,
            'id': str(row.get('Tweet Id', f'row_{idx}')),
        })
    print(f"Processed {len(processed)} valid samples")
    return processed


# ============================================================================
# GPT Response Parsing
# ============================================================================

def normalize_type(pii_type_str):
    """Normalize GPT's output type to our standard PII types."""
    type_map = {
        'age': 'Age', 'AGE': 'Age',
        'contact': 'Contact', 'CONTACT': 'Contact', 'phone': 'Contact', 'email': 'Contact',
        'date': 'Date', 'DATE': 'Date', 'birthday': 'Date', 'dob': 'Date',
        'id': 'ID', 'ID': 'ID', 'identification': 'ID', 'ssn': 'ID',
        'location': 'Location', 'LOCATION': 'Location', 'loc': 'Location',
        'place': 'Location', 'city': 'Location', 'country': 'Location',
        'name': 'Name', 'NAME': 'Name', 'person': 'Name', 'per': 'Name',
        'profession': 'Profession', 'PROFESSION': 'Profession',
        'job': 'Profession', 'occupation': 'Profession', 'title': 'Profession',
    }
    normalized = type_map.get(pii_type_str, type_map.get(pii_type_str.lower(), None))
    if normalized and normalized in PII_TYPES:
        return normalized
    if pii_type_str in PII_TYPES:
        return pii_type_str
    return None


def parse_gpt_response(response_text, tokens):
    """Parse GPT's JSON response → BIOES tags aligned with tokens."""
    predicted_tags = ['O'] * len(tokens)
    try:
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', response_text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                return predicted_tags
        else:
            return predicted_tags

    entities = result.get('entities', [])
    if not isinstance(entities, list):
        return predicted_tags
    tokens_lower = [t.lower() for t in tokens]

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_text = entity.get('text', '')
        entity_type = entity.get('type', '')
        if not entity_text or not entity_type:
            continue
        pii_type = normalize_type(entity_type)
        if pii_type is None:
            continue
        entity_tokens_lower = [t.lower() for t in entity_text.split()]
        n = len(entity_tokens_lower)

        matched = False
        for start in range(len(tokens) - n + 1):
            if tokens_lower[start:start+n] == entity_tokens_lower:
                if n == 1:
                    if predicted_tags[start] == 'O':
                        predicted_tags[start] = f'S-{pii_type}'
                else:
                    if all(predicted_tags[start+j] == 'O' for j in range(n)):
                        predicted_tags[start] = f'B-{pii_type}'
                        for j in range(1, n-1):
                            predicted_tags[start+j] = f'I-{pii_type}'
                        predicted_tags[start+n-1] = f'E-{pii_type}'
                matched = True
                break

        if not matched:
            el = entity_text.lower().strip()
            for i, tok in enumerate(tokens_lower):
                if el == tok and predicted_tags[i] == 'O':
                    predicted_tags[i] = f'S-{pii_type}'
                    break

    return predicted_tags


# ============================================================================
# OpenAI API
# ============================================================================

def call_openai_api(messages, api_key, model="gpt-4o", temperature=0.0,
                    max_tokens=512, max_retries=5):
    """Call OpenAI API with retry logic."""
    import requests
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                usage = data.get('usage', {})
                return {
                    'content': data['choices'][0]['message']['content'],
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                }
            elif resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                tqdm.write(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = min(2 ** attempt * 3, 30)
                tqdm.write(f"  ⚠️ Server error {resp.status_code}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                tqdm.write(f"  ❌ API error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.Timeout:
            wait = min(2 ** attempt * 3, 30)
            tqdm.write(f"  ⏳ Timeout. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            tqdm.write(f"  ❌ Exception: {e}")
            return None
    tqdm.write("  ❌ Max retries exceeded.")
    return None


# ============================================================================
# Checkpoint Management
# ============================================================================

def get_checkpoint_path(output_dir, model, fold_idx):
    return os.path.join(output_dir, f'ckpt_fold{fold_idx}_{model.replace("-","_")}.jsonl')


def load_checkpoint(path):
    """Load completed predictions: sample_id → pred_tags."""
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


def save_checkpoint_entry(path, sample_id, pred_tags, gpt_raw=None):
    """Append one prediction to checkpoint."""
    rec = {'sample_id': sample_id, 'pred_tags': pred_tags, 'ts': datetime.now().isoformat()}
    if gpt_raw is not None:
        rec['gpt_raw'] = gpt_raw[:500]
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
            _, _, cf, _ = precision_recall_fscore_support(cg, cp, average='binary', zero_division=0)
            cat_f1[pt] = cf
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'auc': auc, 'category_f1': cat_f1}


# ============================================================================
# Data Splitting
# ============================================================================

def single_split(data, test_ratio=0.2, seed=42):
    """Single 80/20 train/test split. Default for LLM baselines."""
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    return [(train_idx, test_idx)]  # returns list of 1 fold for unified loop


def kfold_split(data, k=5, seed=42):
    """K-fold CV split. Use --use_cv to enable."""
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    folds, sz = [], len(indices) // k
    for i in range(k):
        s, e = i * sz, (i * sz + sz if i < k - 1 else len(indices))
        test = set(indices[s:e])
        folds.append(([j for j in indices if j not in test], list(test)))
    return folds


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):
    eval_mode = "5-Fold CV" if args.use_cv else "80/20 Split"

    print("=" * 70)
    print("GPT-NER Baseline for PII Detection")
    print("(Wang et al., 2025 - GPT-NER Prompting Strategy)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Data:        {args.data_path}")
    print(f"  Eval mode:   {eval_mode}")
    print(f"  Max samples: {args.max_samples or 'ALL'}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_pii_data(args.data_path)

    if args.max_samples and args.max_samples < len(data):
        random.seed(args.seed)
        data = random.sample(data, args.max_samples)
        print(f"Subsampled to {len(data)} samples")

    few_shot = build_few_shot_examples()

    # Choose split strategy
    if args.use_cv:
        folds = kfold_split(data, k=args.k_folds, seed=args.seed)
        n_folds = args.k_folds
    else:
        folds = single_split(data, test_ratio=0.2, seed=args.seed)
        n_folds = 1

    start_fold = (args.resume_from_fold - 1) if args.resume_from_fold else 0

    all_fold_metrics = []
    tot_in_tok = tot_out_tok = tot_calls = tot_new_calls = tot_fails = 0
    exp_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if fold_idx < start_fold:
            print(f"\n[Fold {fold_idx+1}] Skipped (resuming from fold {start_fold+1})")
            continue

        ckpt_path = get_checkpoint_path(args.output_dir, args.model, fold_idx)
        cached = load_checkpoint(ckpt_path)
        n_cached = sum(1 for sid in [data[j]['id'] for j in test_idx] if sid in cached)
        n_remaining = len(test_idx) - n_cached

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{n_folds}  "
              f"(test: {len(test_idx)} | cached: {n_cached} | remaining: {n_remaining})")
        print(f"{'='*60}")

        fold_start = time.time()
        fold_gold, fold_pred = [], []
        fold_fails = fold_new = 0

        pbar = tqdm(total=len(test_idx), desc=f"Fold {fold_idx+1}",
                    unit="tweet", ncols=110,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining} {rate_fmt}] {postfix}")
        live_f1 = 0.0

        for i, sample_idx in enumerate(test_idx):
            sample = data[sample_idx]
            tokens = sample['tokens']
            gold_tags = sample['tags']
            sid = sample['id']

            # ---- Check cache ----
            if sid in cached:
                pred_tags = cached[sid]
                if len(pred_tags) != len(tokens):
                    pred_tags = ['O'] * len(tokens)
                pbar.set_postfix_str(f"F1={live_f1*100:.1f}% cached", refresh=False)

            elif args.dry_run:
                pred_tags = ['O'] * len(tokens)

            else:
                # ---- API call ----
                msgs = build_messages(sample['tweet_content'], few_shot)
                result = call_openai_api(msgs, api_key=args.api_key,
                                         model=args.model, temperature=args.temperature)
                if result is None:
                    pred_tags = ['O'] * len(tokens)
                    fold_fails += 1
                    gpt_raw = None
                else:
                    tot_in_tok += result['input_tokens']
                    tot_out_tok += result['output_tokens']
                    tot_calls += 1
                    fold_new += 1
                    gpt_raw = result['content']
                    pred_tags = parse_gpt_response(gpt_raw, tokens)

                # ---- Save immediately ----
                save_checkpoint_entry(ckpt_path, sid, pred_tags,
                                      gpt_raw=gpt_raw if result else None)

                # Rate limit pacing
                if fold_new % 50 == 0:
                    time.sleep(1)

                # Live cost estimate
                if args.model in MODEL_COSTS and fold_new % 100 == 0 and fold_new > 0:
                    ci = MODEL_COSTS[args.model]
                    cost = tot_in_tok/1e6*ci['input'] + tot_out_tok/1e6*ci['output']
                    tqdm.write(f"  💰 Running cost: ${cost:.2f} ({tot_calls} calls)")

            fold_gold.extend(gold_tags)
            fold_pred.extend(pred_tags)

            # Update live F1 every 20 samples
            if (i + 1) % 20 == 0 or (i + 1) == len(test_idx):
                gb = [0 if t == 'O' else 1 for t in fold_gold]
                pb = [0 if t == 'O' else 1 for t in fold_pred]
                _, _, live_f1, _ = precision_recall_fscore_support(
                    gb, pb, average='binary', zero_division=0)

            pbar.set_postfix_str(f"F1={live_f1*100:.1f}% new={fold_new}", refresh=False)
            pbar.update(1)

        pbar.close()
        tot_new_calls += fold_new
        tot_fails += fold_fails

        # ---- Fold results ----
        fm = compute_metrics(fold_gold, fold_pred)
        dur = time.time() - fold_start
        fm.update({'duration_sec': dur, 'n_test': len(test_idx),
                   'parse_failures': fold_fails, 'cached': n_cached,
                   'new_api_calls': fold_new})
        all_fold_metrics.append(fm)

        m, s = divmod(int(dur), 60)
        print(f"\n  Fold {fold_idx+1} Results:")
        print(f"    Accuracy:  {fm['accuracy']*100:.2f}%")
        print(f"    Precision: {fm['precision']*100:.2f}%")
        print(f"    Recall:    {fm['recall']*100:.2f}%")
        print(f"    F1-score:  {fm['f1']*100:.2f}%")
        print(f"    AUC:       {fm['auc']*100:.2f}%")
        print(f"    Duration:  {m}m{s:02d}s (new calls: {fold_new}, from cache: {n_cached})")
        if fold_fails:
            print(f"    Parse failures: {fold_fails}")
        for pt in PII_TYPES:
            v = fm['category_f1'].get(pt)
            if v is not None:
                print(f"    {pt:12s} F1: {v*100:.2f}%")

    # ========================================================================
    # Aggregate
    # ========================================================================
    if not all_fold_metrics:
        print("No folds run.")
        return

    total_dur = time.time() - exp_start
    h, rem = divmod(int(total_dur), 3600)
    mi, se = divmod(rem, 60)

    print("\n" + "=" * 70)
    print(f"AGGREGATE RESULTS ({eval_mode})")
    print("=" * 70)

    summary = {}
    for met in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [m[met] for m in all_fold_metrics]
        summary[met] = {'mean': np.mean(vals), 'std': np.std(vals)}
        print(f"  {met.capitalize():12s}: {np.mean(vals)*100:.2f}% (± {np.std(vals)*100:.2f}%)")

    print(f"  {'Duration':12s}: {h}:{mi:02d}:{se:02d}")
    print(f"  API calls (this run): {tot_new_calls}")
    print(f"  Total tokens: {tot_in_tok:,} in / {tot_out_tok:,} out")
    if args.model in MODEL_COSTS:
        ci = MODEL_COSTS[args.model]
        cost = tot_in_tok/1e6*ci['input'] + tot_out_tok/1e6*ci['output']
        print(f"  Estimated cost: ${cost:.2f}")

    print(f"\n  Per-Category F1:")
    for pt in PII_TYPES:
        vs = [m['category_f1'].get(pt, 0) for m in all_fold_metrics]
        if any(v > 0 for v in vs):
            print(f"    {pt:12s}: {np.mean(vs)*100:.2f}%")

    # ---- Save ----
    row = {
        'Method': f'{args.model} (GPT-NER prompt)',
        'Accuracy': f"{summary['accuracy']['mean']*100:.2f}%",
        'Precision': f"{summary['precision']['mean']*100:.2f}%",
        'Recall': f"{summary['recall']['mean']*100:.2f}%",
        'F1-score': f"{summary['f1']['mean']*100:.2f}%",
        'AUC': f"{summary['auc']['mean']*100:.2f}%",
        'Duration': f"{h}:{mi:02d}:{se:02d}",
    }

    results = {
        'experiment': 'GPT-NER Baseline', 'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'config': {'eval_mode': eval_mode, 'k_folds': args.k_folds if args.use_cv else 1,
                   'seed': args.seed,
                   'temperature': args.temperature, 'max_samples': args.max_samples,
                   'n_samples': len(data), 'n_few_shot': len(few_shot)},
        'aggregate': {k: {'mean': float(v['mean']), 'std': float(v['std']),
                          'pct': f"{v['mean']*100:.2f}%"} for k, v in summary.items()},
        'duration': f"{h}:{mi:02d}:{se:02d}", 'duration_sec': total_dur,
        'api_stats': {'total_calls': tot_calls, 'new_this_run': tot_new_calls,
                      'in_tokens': tot_in_tok, 'out_tokens': tot_out_tok,
                      'parse_failures': tot_fails},
        'per_fold': [{
            'fold': i+1, **{k: float(m[k]) for k in ['accuracy','precision','recall','f1','auc']},
            'duration_sec': m['duration_sec'], 'n_test': m['n_test'],
            'new_calls': m['new_api_calls'], 'cached': m['cached'],
            'category_f1': {k: float(v) for k, v in m['category_f1'].items()},
        } for i, m in enumerate(all_fold_metrics)],
        'table_8_row': row,
    }

    out = os.path.join(args.output_dir, f'gpt_ner_results_{args.model.replace("-","_")}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    csv_out = os.path.join(args.output_dir, f'gpt_ner_table8_{args.model.replace("-","_")}.csv')
    pd.DataFrame([row]).to_csv(csv_out, index=False)
    print(f"\nResults: {out}")
    print(f"Table 8: {csv_out}")

    print("\n" + "=" * 70)
    print("TABLE 8 ROW (copy to manuscript):")
    print("=" * 70)
    print(f"  Target | {row['Method']} | {row['Accuracy']} | {row['Precision']} | "
          f"{row['Recall']} | {row['F1-score']} | {row['AUC']} | {row['Duration']}")
    print("=" * 70)
    print(f"\nCheckpoint files in {args.output_dir}/ — re-run same command to resume.")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPT-NER Baseline for PII Detection',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--api_key', type=str,
                        default=os.environ.get('OPENAI_API_KEY', ''),
                        help='OpenAI API key (defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo',
                                 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3-mini'],
                        help='Model (default: gpt-4o)')
    parser.add_argument('--data_path', type=str,
                        default='../../data/sample/sample_pii_tweets.csv',
                        help='Path to PII data CSV (default: anonymized sample)')
    parser.add_argument('--output_dir', type=str, default='./gpt_ner_output',
                        help='Output & checkpoint directory')
    parser.add_argument('--use_cv', action='store_true',
                        help='Use 5-fold CV instead of single 80/20 split (costs 5x more)')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds if --use_cv is set (default: 5)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit samples (for testing)')
    parser.add_argument('--dry_run', action='store_true',
                        help='No API calls (predicts all O)')
    parser.add_argument('--resume_from_fold', type=int, default=None,
                        help='Skip to fold N (1-indexed). Within folds, auto-resumes.')

    args = parser.parse_args()
    if not args.api_key and not args.dry_run:
        raise SystemExit(
            "ERROR: No OpenAI API key provided.\n"
            "Either set the OPENAI_API_KEY environment variable, or pass --api_key.\n"
            "(Use --dry_run to test without API calls.)"
        )
    run_experiment(args)