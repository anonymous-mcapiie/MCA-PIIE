"""
Evaluation: Entity-Level Precision / Recall / F1 for PII Detection
====================================================================

Usage:
    python evaluate.py --predictions ./baselines/GLiNER/gliner_predictions.csv

Computes per-category and overall (micro-averaged) entity-level metrics
following the standard CoNLL-2003 evaluation protocol (exact span match).
"""
import argparse
import pandas as pd
from collections import defaultdict


def bioes_to_entities(tags):
    """Convert a BIOES tag sequence to a set of (start, end, type) tuples."""
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or tag == "":
            i += 1
            continue
        prefix = tag.split("-")[0]
        etype = tag.split("-")[1] if "-" in tag else None
        if prefix == "S":
            entities.append((i, i, etype))
            i += 1
        elif prefix == "B":
            j = i + 1
            while j < len(tags) and tags[j].startswith("I-") and \
                    tags[j].split("-")[1] == etype:
                j += 1
            if j < len(tags) and tags[j] == f"E-{etype}":
                entities.append((i, j, etype))
                i = j + 1
            else:
                # Malformed — treat as B-only span
                entities.append((i, i, etype))
                i += 1
        else:
            # Stray I-/E- — skip
            i += 1
    return set(entities)


def compute_metrics(gold_tags_list, pred_tags_list):
    """Compute per-type and micro-averaged P/R/F1."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gold_tags, pred_tags in zip(gold_tags_list, pred_tags_list):
        gold_ents = bioes_to_entities(gold_tags)
        pred_ents = bioes_to_entities(pred_tags)
        for ent in gold_ents & pred_ents:
            tp[ent[2]] += 1
        for ent in pred_ents - gold_ents:
            fp[ent[2]] += 1
        for ent in gold_ents - pred_ents:
            fn[ent[2]] += 1

    types = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    rows = []
    total_tp = total_fp = total_fn = 0
    for t in types:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append((t, p, r, f, tp[t] + fn[t]))
        total_tp += tp[t]; total_fp += fp[t]; total_fn += fn[t]

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) \
        if (micro_p + micro_r) > 0 else 0.0
    return rows, (micro_p, micro_r, micro_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True,
                        help='CSV with Gold_BIOES and Pred_BIOES columns')
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    gold = [str(g).split() for g in df['Gold_BIOES']]
    pred = [str(p).split() for p in df['Pred_BIOES']]

    rows, (mp, mr, mf) = compute_metrics(gold, pred)

    print(f"\nEvaluation: {args.predictions}")
    print("=" * 65)
    print(f"{'Type':<14}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    print("-" * 65)
    for t, p, r, f, sup in rows:
        print(f"{t:<14}{p:>12.4f}{r:>12.4f}{f:>12.4f}{sup:>12d}")
    print("-" * 65)
    print(f"{'Micro-Avg':<14}{mp:>12.4f}{mr:>12.4f}{mf:>12.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
