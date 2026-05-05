"""
GLiNER Baseline for PII Extraction 
=============================================================
Model: urchade/gliner_medium-v2.1 (general-purpose, NAACL 2024)
Paradigm: Zero-shot span extraction via bidirectional Transformer encoder
Evaluation: 80/20 held-out split on PII data, entity-level F1
            (same protocol as GPT-4o and UniNER-7B baselines)

GLiNER predicts entity spans directly (start/end char offsets + label),
so we need to align these spans back to our tokenized BIOES ground truth.

Usage:
    pip install gliner pandas seqeval
    python script_GLiNER.py
"""

import pandas as pd
import numpy as np
import re
import time
import json
from collections import defaultdict

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_NAME = "urchade/gliner_medium-v2.1"  # General-purpose GLiNER (NAACL 2024)
DATA_PATH = "../../data/sample/sample_pii_tweets.csv"
THRESHOLD = 0.5  # GLiNER confidence threshold (default from paper)
SPLIT_RATIO = 0.8  # 80% train (unused), 20% test — consistent with other baselines
RANDOM_SEED = 42

# PII category labels for GLiNER
# These are the 7 PII categories in our dataset.
# GLiNER works best with natural language labels in lower/title case.
LABEL_MAP = {
    "person name": "Name",
    "age": "Age",
    "phone number or email": "Contact",
    "date": "Date",
    "location": "Location",
    "identification number": "ID",
    "profession or job title": "Profession",
}
# Reverse map for evaluation
LABEL_TO_PII = {v: v for v in LABEL_MAP.values()}

# GLiNER input labels (what we pass to predict_entities)
GLINER_LABELS = list(LABEL_MAP.keys())


# ============================================================
# 2. DATA LOADING & SPLITTING
# ============================================================
def load_and_split_data(data_path, split_ratio=0.8, seed=42):
    """Load PII_tweet.csv and create 80/20 split (same as other baselines)."""
    df = pd.read_csv(data_path, encoding='cp1252')

    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['Tokens', 'Word_Level_BIOES']).reset_index(drop=True)

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * split_ratio)
    test_indices = indices[split_idx:]

    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"Dataset: {len(df)} total → {len(test_df)} test samples")
    return test_df


# ============================================================
# 3. GROUND TRUTH PARSING
# ============================================================
def parse_bioes_to_entities(tokens_str, bioes_str):
    """
    Convert tokenized BIOES tags to a list of (entity_text, category) tuples.
    This is our ground truth format.

    Example:
        tokens: "He was 45 years old !"
        bioes:  "O O B-Age I-Age E-Age O"
        → [("45 years old", "Age")]
    """
    tokens = tokens_str.strip().split()
    tags = bioes_str.strip().split()

    # Align lengths (truncate to shorter)
    min_len = min(len(tokens), len(tags))
    tokens = tokens[:min_len]
    tags = tags[:min_len]

    entities = []
    current_entity_tokens = []
    current_category = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            # Start of multi-token entity
            if current_entity_tokens and current_category:
                # Flush previous incomplete entity
                entities.append((' '.join(current_entity_tokens), current_category))
            current_category = tag[2:]
            current_entity_tokens = [token]
        elif tag.startswith('I-') and current_entity_tokens:
            current_entity_tokens.append(token)
        elif tag.startswith('E-') and current_entity_tokens:
            current_entity_tokens.append(token)
            entities.append((' '.join(current_entity_tokens), current_category))
            current_entity_tokens = []
            current_category = None
        elif tag.startswith('S-'):
            # Single-token entity
            category = tag[2:]
            entities.append((token, category))
            current_entity_tokens = []
            current_category = None
        else:
            # O tag or mismatch — flush any incomplete entity
            if current_entity_tokens and current_category:
                entities.append((' '.join(current_entity_tokens), current_category))
            current_entity_tokens = []
            current_category = None

    # Flush remaining
    if current_entity_tokens and current_category:
        entities.append((' '.join(current_entity_tokens), current_category))

    return entities


# ============================================================
# 4. GLINER PREDICTION
# ============================================================
def gliner_predict_entities(model, text, threshold=0.5):
    """
    Run GLiNER on a single text and return predicted entities
    as (entity_text, PII_category) tuples.
    """
    try:
        raw_entities = model.predict_entities(text, GLINER_LABELS, threshold=threshold)
    except Exception as e:
        print(f"  [WARNING] GLiNER error on text: {text[:50]}... → {e}")
        return []

    predicted = []
    for ent in raw_entities:
        gliner_label = ent["label"]
        pii_category = LABEL_MAP.get(gliner_label, None)
        if pii_category:
            predicted.append((ent["text"].strip(), pii_category))

    return predicted


# ============================================================
# 5. ENTITY-LEVEL EVALUATION (same protocol as GPT-4o/UniNER)
# ============================================================
def normalize_entity_text(text):
    """Normalize entity text for flexible matching."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove trailing punctuation
    text = text.rstrip('.,;:!?')
    return text


def compute_entity_level_metrics(all_gold, all_pred):
    """
    Compute entity-level Precision, Recall, F1 (micro-averaged).
    Also compute per-category breakdown.

    Matching: normalized text + category must match.
    """
    # Overall counts
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # Per-category counts
    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)

    for gold_entities, pred_entities in zip(all_gold, all_pred):
        # Normalize
        gold_set = set()
        for text, cat in gold_entities:
            gold_set.add((normalize_entity_text(text), cat))

        pred_set = set()
        for text, cat in pred_entities:
            pred_set.add((normalize_entity_text(text), cat))

        # True positives
        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)

        for _, cat in tp:
            cat_tp[cat] += 1
        for _, cat in fp:
            cat_fp[cat] += 1
        for _, cat in fn:
            cat_fn[cat] += 1

    # Micro-averaged metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-category metrics
    categories = sorted(set(list(cat_tp.keys()) + list(cat_fp.keys()) + list(cat_fn.keys())))
    cat_metrics = {}
    for cat in categories:
        tp_c = cat_tp[cat]
        fp_c = cat_fp[cat]
        fn_c = cat_fn[cat]
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        cat_metrics[cat] = {
            'precision': p, 'recall': r, 'f1': f,
            'tp': tp_c, 'fp': fp_c, 'fn': fn_c,
            'support': tp_c + fn_c
        }

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'per_category': cat_metrics
    }


# ============================================================
# 6. ALSO COMPUTE TOKEN-LEVEL METRICS VIA SEQEVAL
# ============================================================
def align_gliner_to_bioes(tokens_str, pred_entities):
    """
    Convert GLiNER span predictions back to BIOES token-level tags
    for seqeval-compatible evaluation.

    Strategy: for each predicted entity span, find matching token
    subsequence in the tokenized text and assign BIOES tags.
    """
    tokens = tokens_str.strip().split()
    pred_tags = ['O'] * len(tokens)

    # Build character-to-token mapping from the tokenized text
    # Reconstruct text from tokens to get char offsets
    reconstructed = ' '.join(tokens)

    for ent_text, ent_cat in pred_entities:
        # Find the entity text in the reconstructed token string
        ent_text_normalized = ent_text.strip()

        # Try exact match first
        start_char = reconstructed.find(ent_text_normalized)
        if start_char == -1:
            # Try case-insensitive
            start_char = reconstructed.lower().find(ent_text_normalized.lower())

        if start_char == -1:
            continue  # Can't align this entity

        # Find which tokens this span covers
        # Count spaces before start_char to find token index
        char_pos = 0
        start_tok = None
        end_tok = None

        for i, tok in enumerate(tokens):
            tok_start = char_pos
            tok_end = char_pos + len(tok)

            if tok_start <= start_char < tok_end:
                start_tok = i
            if tok_start < start_char + len(ent_text_normalized) <= tok_end:
                end_tok = i
                break

            char_pos = tok_end + 1  # +1 for space

        if start_tok is not None and end_tok is not None:
            span_len = end_tok - start_tok + 1
            if span_len == 1:
                pred_tags[start_tok] = f'S-{ent_cat}'
            else:
                pred_tags[start_tok] = f'B-{ent_cat}'
                for j in range(start_tok + 1, end_tok):
                    pred_tags[j] = f'I-{ent_cat}'
                pred_tags[end_tok] = f'E-{ent_cat}'

    return pred_tags


# ============================================================
# 7. MAIN EXECUTION
# ============================================================
def main():
    print("=" * 70)
    print("GLiNER BASELINE FOR PII EXTRACTION")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Labels: {GLINER_LABELS}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading GLiNER model...")
    t0 = time.time()
    from gliner import GLiNER
    model = GLiNER.from_pretrained(MODEL_NAME)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load data
    print("\n[2/4] Loading and splitting data...")
    test_df = load_and_split_data(DATA_PATH, SPLIT_RATIO, RANDOM_SEED)

    # Run inference
    print(f"\n[3/4] Running GLiNER inference on {len(test_df)} test samples...")
    all_gold_entities = []
    all_pred_entities = []
    all_gold_tags = []
    all_pred_tags = []

    error_count = 0
    t_start = time.time()

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        tokens_str = str(row['Tokens']).strip()
        bioes_str = str(row['Word_Level_BIOES']).strip()
        text = str(row['Tweet Content']).strip()

        # Parse ground truth
        gold_entities = parse_bioes_to_entities(tokens_str, bioes_str)

        # GLiNER prediction (use Tweet Content, not tokenized form)
        pred_entities = gliner_predict_entities(model, text, THRESHOLD)

        all_gold_entities.append(gold_entities)
        all_pred_entities.append(pred_entities)

        # Also build token-level tags for seqeval
        gold_tags = bioes_str.strip().split()
        pred_tags = align_gliner_to_bioes(tokens_str, pred_entities)

        # Align lengths
        min_len = min(len(gold_tags), len(pred_tags))
        all_gold_tags.append(gold_tags[:min_len])
        all_pred_tags.append(pred_tags[:min_len])

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(test_df) - idx - 1) / rate
            print(f"  Processed {idx + 1}/{len(test_df)} "
                  f"({rate:.1f} samples/sec, ETA: {eta:.0f}s)")

    t_total = time.time() - t_start
    print(f"  Inference complete: {len(test_df)} samples in {t_total:.1f}s "
          f"({len(test_df) / t_total:.1f} samples/sec)")

    # ============================================================
    # 4. EVALUATE
    # ============================================================
    print("\n[4/4] Computing metrics...")

    # Entity-level metrics (primary)
    entity_results = compute_entity_level_metrics(all_gold_entities, all_pred_entities)

    print("\n" + "=" * 70)
    print("ENTITY-LEVEL RESULTS (Primary Metric)")
    print("=" * 70)
    print(f"  Precision: {entity_results['precision'] * 100:.2f}%")
    print(f"  Recall:    {entity_results['recall'] * 100:.2f}%")
    print(f"  F1-score:  {entity_results['f1'] * 100:.2f}%")
    print(f"  (TP={entity_results['tp']}, FP={entity_results['fp']}, FN={entity_results['fn']})")

    print("\n  Per-Category Breakdown:")
    print(f"  {'Category':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-' * 42}")
    for cat in ['Age', 'Contact', 'Date', 'ID', 'Location', 'Name', 'Profession']:
        if cat in entity_results['per_category']:
            m = entity_results['per_category'][cat]
            print(f"  {cat:<12} {m['precision'] * 100:>6.2f}% {m['recall'] * 100:>6.2f}% "
                  f"{m['f1'] * 100:>6.2f}% {m['support']:>7d}")
        else:
            print(f"  {cat:<12} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'0':>8}")

    # Token-level metrics via seqeval
    try:
        from seqeval.metrics import classification_report, f1_score
        # Convert BIOES to BIO for seqeval compatibility
        def bioes_to_bio(tags):
            bio = []
            for t in tags:
                if t.startswith('S-'):
                    bio.append('B-' + t[2:])
                elif t.startswith('E-'):
                    bio.append('I-' + t[2:])
                else:
                    bio.append(t)
            return bio

        gold_bio = [bioes_to_bio(tags) for tags in all_gold_tags]
        pred_bio = [bioes_to_bio(tags) for tags in all_pred_tags]

        token_f1 = f1_score(gold_bio, pred_bio)
        print(f"\n  Token-level F1 (seqeval, strict): {token_f1 * 100:.2f}%")
        print("\n  Token-level Classification Report:")
        print(classification_report(gold_bio, pred_bio))
    except ImportError:
        print("\n  [INFO] seqeval not installed; skipping token-level metrics.")
        print("  Install with: pip install seqeval")

    # ============================================================
    # 5. SAVE RESULTS
    # ============================================================
    results = {
        'model': MODEL_NAME,
        'threshold': THRESHOLD,
        'test_size': len(test_df),
        'inference_time_seconds': round(t_total, 1),
        'entity_level': {
            'precision': round(entity_results['precision'] * 100, 2),
            'recall': round(entity_results['recall'] * 100, 2),
            'f1': round(entity_results['f1'] * 100, 2),
        },
        'per_category': {
            cat: {
                'precision': round(m['precision'] * 100, 2),
                'recall': round(m['recall'] * 100, 2),
                'f1': round(m['f1'] * 100, 2),
                'support': m['support']
            }
            for cat, m in entity_results['per_category'].items()
        }
    }

    with open('gliner_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to gliner_baseline_results.json")

    # Save detailed predictions for error analysis
    pred_details = []
    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        pred_details.append({
            'text': str(row['Tweet Content'])[:200],
            'gold': all_gold_entities[idx],
            'pred': all_pred_entities[idx],
        })

    with open('gliner_baseline_predictions.json', 'w') as f:
        json.dump(pred_details[:100], f, indent=2, ensure_ascii=False)
    print(f"  First 100 predictions saved to gliner_baseline_predictions.json")

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"  GLiNER ({MODEL_NAME})")
    print(f"  Entity-level F1: {entity_results['f1'] * 100:.2f}%")
    print(f"  Paradigm: Zero-shot span extraction (bidirectional Transformer)")
    print(f"  No training required. Threshold={THRESHOLD}")
    print(f"  Inference: {t_total:.0f}s for {len(test_df)} samples "
          f"({len(test_df) / t_total:.1f} samples/sec)")
    print("=" * 70)


if __name__ == "__main__":
    main()