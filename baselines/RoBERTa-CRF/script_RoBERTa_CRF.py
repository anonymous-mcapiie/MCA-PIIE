"""
PLM + CRF Baselines for PII Detection
=======================================
Fine-tunes pre-trained language models (RoBERTa / DeBERTa-v3) with a CRF
sequence labeling head on the PII_tweet target dataset.

These are fine-tuned baselines that use the SAME evaluation protocol as the
original Table 8 models (5-fold CV with training), unlike the inference-only
GPT-4o and UniNER baselines.

Supports:
  - roberta-base + CRF   (Liu et al., 2019)
  - deberta-v3-base + CRF (He et al., 2023)

Experimental setup:
  - 5-fold cross-validation on PII_tweet.csv (consistent with Table 8)
  - Fine-tune PLM + linear + CRF on BIOES tags
  - Metrics: Accuracy, Precision, Recall, F1, AUC, Duration

Requirements:
  pip install torch transformers tqdm scikit-learn pandas

Usage:
  # RoBERTa baseline
  python plm_crf_baseline.py --model roberta

  # DeBERTa-v3 baseline
  python plm_crf_baseline.py --model deberta

  # Quick test
  python plm_crf_baseline.py --model roberta --max_samples 100 --epochs 2

  # Resume (reuses saved fold checkpoints)
  python plm_crf_baseline.py --model roberta --resume


"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q", "--break-system-packages"])
    from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


# ============================================================================
# Configuration
# ============================================================================

PII_TYPES = ["age", "contact", "date", "ID", "location", "name", "profession"]

MODEL_CONFIGS = {
    "roberta": {
        "hf_name": "roberta-base",
        "display_name": "RoBERTa-base + CRF",
        "lr": 2e-5,
        "reference": "Liu et al. (2019)",
    },
    "deberta": {
        "hf_name": "microsoft/deberta-v3-base",
        "display_name": "DeBERTa-v3-base + CRF",
        "lr": 1e-5,
        "reference": "He et al. (2023)",
    },
}


def create_label_mappings():
    """Create BIOES label mappings (consistent with MCA-PIIE)."""
    labels = ["O"]
    for pii_type in PII_TYPES:
        for prefix in ["B", "I", "E", "S"]:
            labels.append(f"{prefix}-{pii_type}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label, labels


LABEL2ID, ID2LABEL, ALL_LABELS = create_label_mappings()
NUM_LABELS = len(ALL_LABELS)


# ============================================================================
# CRF Layer (same logic as MCA-PIIE for fair comparison)
# ============================================================================

class CRF(nn.Module):
    """Conditional Random Field for sequence labeling."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._init_constraints()

    def _init_constraints(self):
        """Initialize transition constraints for BIOES scheme."""
        with torch.no_grad():
            for i, li in ID2LABEL.items():
                for j, lj in ID2LABEL.items():
                    if li.startswith('O') and lj.startswith(('I-', 'E-')):
                        self.transitions[i, j] = -10000
                    if li.startswith('B-'):
                        ti = li[2:]
                        if lj.startswith(('I-', 'E-')):
                            tj = lj[2:]
                            if ti != tj:
                                self.transitions[i, j] = -10000
                    if li.startswith('S-') and lj.startswith(('I-', 'E-')):
                        self.transitions[i, j] = -10000

    def forward(self, emissions, tags, mask):
        """Compute negative log-likelihood."""
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        score = self._compute_score(emissions, tags, mask)
        partition = self._compute_log_partition(emissions, mask)
        return score - partition

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for i in range(1, seq_len):
            m = mask[:, i].float()
            emit = emissions[:, i].gather(1, tags[:, i:i+1]).squeeze(1)
            trans = self.transitions[tags[:, i-1], tags[:, i]]
            score = score + (emit + trans) * m
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            m = mask[:, i].float().unsqueeze(1).unsqueeze(2)
            emit = emissions[:, i].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            next_score = score.unsqueeze(2) + emit + trans
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions, mask):
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)
        score = score + self.end_transitions
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)
        best_tags_list.append(best_last_tag)
        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags_list.append(best_last_tag)
        best_tags_list.reverse()
        best_tags = torch.stack(best_tags_list, dim=1)
        return best_tags.cpu().numpy().tolist()


# ============================================================================
# PLM + CRF Model
# ============================================================================

class PLM_CRF(nn.Module):
    """Pre-trained Language Model + Linear + CRF for token classification."""

    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if label_mask is None:
            label_mask = attention_mask.bool()

        if labels is not None:
            loss = self.crf(emissions, labels, label_mask)
            predictions = self.crf.decode(emissions, label_mask)
            return {"loss": loss, "predictions": predictions}
        else:
            predictions = self.crf.decode(emissions, label_mask)
            return {"predictions": predictions}


# ============================================================================
# Data Loading & Tokenization
# ============================================================================

def normalize_tag(tag):
    """Normalize BIOES tag (consistent with MCA-PIIE)."""
    if tag == 'O':
        return tag
    if not any(tag.startswith(p) for p in ['B-', 'I-', 'E-', 'S-']):
        return 'O'
    prefix = tag[:2]
    pii_type = tag[2:]
    type_map = {
        'age': 'age', 'Age': 'age', 'AGE': 'age',
        'contact': 'contact', 'Contact': 'contact',
        'date': 'date', 'Date': 'date',
        'id': 'ID', 'Id': 'ID', 'ID': 'ID',
        'location': 'location', 'Location': 'location',
        'name': 'name', 'Name': 'name',
        'profession': 'profession', 'Profession': 'profession',
        'occupation': 'profession',
    }
    normalized = type_map.get(pii_type, pii_type)
    if normalized in PII_TYPES:
        return f"{prefix}{normalized}"
    return 'O'


def load_pii_data(filepath):
    """Load PII_tweet.csv."""
    for enc in ['cp1252', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"Loaded {len(df)} rows ({enc})")
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
        tweet = str(row.get('Tweet Content', ''))
        if 'Tokens' in df.columns and not pd.isna(row.get('Tokens', None)):
            tokens = str(row['Tokens']).split()
        else:
            tokens = tweet.split()
        min_len = min(len(tokens), len(tags))
        if min_len == 0:
            continue
        tokens, tags = tokens[:min_len], tags[:min_len]
        norm_tags = [normalize_tag(t) for t in tags]
        processed.append({
            'tokens': tokens,
            'tags': norm_tags,
            'id': str(row.get('Tweet Id', f'row_{idx}')),
        })
    print(f"Processed {len(processed)} samples")
    return processed


class PIITokenDataset(Dataset):
    """Dataset that aligns word-level BIOES tags with subword tokenization."""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['tags']

        # Tokenize with word-to-subword alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        word_ids = encoding.word_ids(batch_index=0)

        # Align tags to subword tokens
        label_ids = []
        label_mask = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(LABEL2ID['O'])
                label_mask.append(False)
            elif word_id != prev_word_id:
                # First subword of a word: use the word's tag
                if word_id < len(tags):
                    tag = tags[word_id]
                    label_ids.append(LABEL2ID.get(tag, LABEL2ID['O']))
                else:
                    label_ids.append(LABEL2ID['O'])
                label_mask.append(True)
            else:
                # Continuation subword: copy the tag (I- or same)
                if word_id < len(tags):
                    tag = tags[word_id]
                    # For B- tags on continuation, use I- instead
                    if tag.startswith('B-'):
                        tag = 'I-' + tag[2:]
                    label_ids.append(LABEL2ID.get(tag, LABEL2ID['O']))
                else:
                    label_ids.append(LABEL2ID['O'])
                label_mask.append(False)
            prev_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'label_mask': torch.tensor(label_mask, dtype=torch.bool) # for evaluation alignment
        }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, dataloader, device):
    """Evaluate and return metrics aligned with Table 8."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)

            result = model(input_ids, attn_mask, labels=labels, label_mask=label_mask)
            total_loss += result['loss'].item()
            n_batches += 1

            predictions = result['predictions']
            labels_np = labels.cpu().numpy()
            mask_np = label_mask.cpu().numpy()

            for pred_seq, label_seq, mask_seq in zip(
                    predictions, labels_np, mask_np
            ):
                for j in range(len(mask_seq)):
                    if mask_seq[j]:
                        all_preds.append(ID2LABEL.get(pred_seq[j], 'O'))
                        all_labels.append(ID2LABEL.get(label_seq[j], 'O'))

    gold_bin = [0 if t == 'O' else 1 for t in all_labels]
    pred_bin = [0 if t == 'O' else 1 for t in all_preds]

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_bin, pred_bin, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(gold_bin, pred_bin)
    except ValueError:
        auc = 0.0

    cat_f1 = {}
    for pt in PII_TYPES:
        cg = [1 if pt in t else 0 for t in all_labels]
        cp = [1 if pt in t else 0 for t in all_preds]
        if sum(cg) > 0:
            _, _, cf, _ = precision_recall_fscore_support(cg, cp, average='binary', zero_division=0)
            cat_f1[pt] = cf

    return {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc,
        'category_f1': cat_f1,
    }


# ============================================================================
# Training
# ============================================================================

def train_one_fold(model, train_loader, eval_loader, device, config, save_path):
    """Train model for one fold with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{config['epochs']}", leave=True, ncols=90)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)

            optimizer.zero_grad()
            result = model(input_ids, attn_mask, labels=labels, label_mask=label_mask)
            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

        # Evaluate
        metrics = evaluate(model, eval_loader, device)
        scheduler.step(metrics['f1'])

        print(f"    Eval — Loss: {metrics['loss']:.4f}  F1: {metrics['f1']*100:.2f}%  "
              f"P: {metrics['precision']*100:.1f}%  R: {metrics['recall']*100:.1f}%")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"    Early stopping at epoch {epoch+1}. Best F1: {best_f1*100:.2f}%")
                break

    # Load best
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, best_f1


# ============================================================================
# K-Fold CV
# ============================================================================

def kfold_split(data, k=5, seed=42):
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    folds, sz = [], len(indices) // k
    for i in range(k):
        s = i * sz
        e = s + sz if i < k - 1 else len(indices)
        test = set(indices[s:e])
        folds.append(([j for j in indices if j not in test], list(test)))
    return folds


# ============================================================================
# Main
# ============================================================================

def run_experiment(args):
    model_cfg = MODEL_CONFIGS[args.model]
    hf_name = model_cfg['hf_name']
    display_name = model_cfg['display_name']

    print("=" * 70)
    print(f"PLM + CRF Baseline: {display_name}")
    print(f"({model_cfg['reference']})")
    print("=" * 70)
    print(f"  HF Model:    {hf_name}")
    print(f"  Data:        {args.data_path}")
    print(f"  K-folds:     {args.k_folds}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {model_cfg['lr']}")
    print(f"  Max samples: {args.max_samples or 'ALL'}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = load_pii_data(args.data_path)
    if args.max_samples and args.max_samples < len(data):
        random.seed(args.seed)
        data = random.sample(data, args.max_samples)
        print(f"Subsampled to {len(data)} samples")

    # Tokenizer
    print(f"Loading tokenizer: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, add_prefix_space=True)

    # K-fold
    folds = kfold_split(data, k=args.k_folds, seed=args.seed)
    all_fold_metrics = []
    exp_start = time.time()

    config = {
        'lr': model_cfg['lr'],
        'epochs': args.epochs,
        'patience': args.patience,
    }

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_result_path = os.path.join(args.output_dir, f'fold{fold_idx}_results_{args.model}.json')

        # Check if fold already completed (resume support)
        if args.resume and os.path.exists(fold_result_path):
            print(f"\n[Fold {fold_idx+1}] Loading cached results...")
            with open(fold_result_path, 'r') as f:
                cached = json.load(f)
            all_fold_metrics.append(cached)
            print(f"  F1: {cached['f1']*100:.2f}%")
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{args.k_folds}  (train: {len(train_idx)}, test: {len(test_idx)})")
        print(f"{'='*60}")

        fold_start = time.time()

        # Build datasets
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        train_dataset = PIITokenDataset(train_data, tokenizer, max_length=args.max_length)
        test_dataset = PIITokenDataset(test_data, tokenizer, max_length=args.max_length)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Build model (fresh for each fold)
        model = PLM_CRF(hf_name, NUM_LABELS, dropout=0.1).to(device)
        save_path = os.path.join(args.output_dir, f'best_fold{fold_idx}_{args.model}.pt')

        # Train
        model, best_f1 = train_one_fold(model, train_loader, test_loader, device, config, save_path)

        # Final evaluation
        metrics = evaluate(model, test_loader, device)
        fold_dur = time.time() - fold_start
        metrics['duration_sec'] = fold_dur
        metrics['n_train'] = len(train_idx)
        metrics['n_test'] = len(test_idx)

        m, s = divmod(int(fold_dur), 60)
        print(f"\n  Fold {fold_idx+1} Final Results:")
        print(f"    Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"    Precision: {metrics['precision']*100:.2f}%")
        print(f"    Recall:    {metrics['recall']*100:.2f}%")
        print(f"    F1-score:  {metrics['f1']*100:.2f}%")
        print(f"    AUC:       {metrics['auc']*100:.2f}%")
        print(f"    Duration:  {m}m{s:02d}s")
        for pt in PII_TYPES:
            v = metrics['category_f1'].get(pt)
            if v is not None:
                print(f"    {pt:12s} F1: {v*100:.2f}%")

        # Save fold results
        with open(fold_result_path, 'w') as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in metrics.items()}, f, indent=2)

        all_fold_metrics.append(metrics)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ========================================================================
    # Aggregate
    # ========================================================================
    total_dur = time.time() - exp_start
    h, rem = divmod(int(total_dur), 3600)
    mi, se = divmod(rem, 60)

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({args.k_folds}-Fold CV) — {display_name}")
    print(f"{'='*70}")

    summary = {}
    for met in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [m[met] for m in all_fold_metrics]
        summary[met] = {'mean': np.mean(vals), 'std': np.std(vals)}
        print(f"  {met.capitalize():12s}: {np.mean(vals)*100:.2f}% (± {np.std(vals)*100:.2f}%)")

    print(f"  {'Duration':12s}: {h}:{mi:02d}:{se:02d}")

    print(f"\n  Per-Category F1:")
    for pt in PII_TYPES:
        vs = [m['category_f1'].get(pt, 0) for m in all_fold_metrics]
        if any(v > 0 for v in vs):
            print(f"    {pt:12s}: {np.mean(vs)*100:.2f}%")

    # Save
    row = {
        'Method': display_name,
        'Accuracy': f"{summary['accuracy']['mean']*100:.2f}%",
        'Precision': f"{summary['precision']['mean']*100:.2f}%",
        'Recall': f"{summary['recall']['mean']*100:.2f}%",
        'F1-score': f"{summary['f1']['mean']*100:.2f}%",
        'AUC': f"{summary['auc']['mean']*100:.2f}%",
        'Duration': f"{h}:{mi:02d}:{se:02d}",
    }

    results = {
        'experiment': f'{display_name} Baseline',
        'model': hf_name, 'display_name': display_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'k_folds': args.k_folds, 'seed': args.seed,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': model_cfg['lr'], 'patience': args.patience,
            'max_length': args.max_length,
            'max_samples': args.max_samples, 'n_samples': len(data),
        },
        'aggregate': {k: {'mean': float(v['mean']), 'std': float(v['std']),
                          'pct': f"{v['mean']*100:.2f}%"} for k, v in summary.items()},
        'duration': f"{h}:{mi:02d}:{se:02d}", 'duration_sec': total_dur,
        'per_fold': [{
            'fold': i+1,
            **{k: float(m[k]) for k in ['accuracy','precision','recall','f1','auc']},
            'duration_sec': m.get('duration_sec', 0),
            'category_f1': {k: float(v) for k, v in m.get('category_f1', {}).items()},
        } for i, m in enumerate(all_fold_metrics)],
        'table_8_row': row,
    }

    out = os.path.join(args.output_dir, f'plm_crf_results_{args.model}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    csv_out = os.path.join(args.output_dir, f'plm_crf_table8_{args.model}.csv')
    pd.DataFrame([row]).to_csv(csv_out, index=False)
    print(f"\nResults: {out}")
    print(f"Table 8: {csv_out}")

    print(f"\n{'='*70}")
    print("TABLE 8 ROW (copy to manuscript):")
    print(f"{'='*70}")
    print(f"  Target | {row['Method']} | {row['Accuracy']} | {row['Precision']} | "
          f"{row['Recall']} | {row['F1-score']} | {row['AUC']} | {row['Duration']}")
    print(f"{'='*70}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PLM + CRF Baselines for PII Detection')

    parser.add_argument('--model', type=str, default='roberta',
                        choices=['roberta', 'deberta'],
                        help='Model: roberta (RoBERTa-base) or deberta (DeBERTa-v3-base)')
    parser.add_argument('--data_path', type=str, default='../../data/sample/sample_pii_tweets.csv',
                        help='Path to PII dataset CSV. Default uses anonymized sample data.')
    parser.add_argument('--output_dir', type=str, default='./plm_crf_output')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip folds that already have saved results')

    args = parser.parse_args()
    run_experiment(args)