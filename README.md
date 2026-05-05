# MCA-PIIE Reference Implementation

Anonymous code repository accompanying the manuscript:
**"A Decision Support Framework for Privacy Risk Assessment on Social Media
through Automated Personally Identifiable Information Detection"**
(submitted to a journal in the Information Systems area).

This repository provides the **reference implementation** of MCA-PIIE
(Multi-Context Attention for PII Entity Extraction) and all baseline models
reported in the paper.

---

## Purpose of This Repository

This codebase is shared to support **methodological transparency and
implementation-level reproducibility**. Specifically, reviewers and other
researchers can use it to:

- Verify the **architecture** of MCA-PIIE matches the description in the
  manuscript (Section 5).
- Inspect the **hyperparameters, training procedure, and Deep Transfer
  Learning (DTL) pipeline** in code form.
- Examine the **prompting strategies and evaluation protocols** used for
  every baseline reported in Table 8 (RoBERTa+CRF, GPT-NER with GPT-4o,
  UniNER-7B, GLiNER, LLaMA-3-8B, Gemma-2-9B).
- Run the end-to-end pipeline on a small de-identified sample to confirm
  that all components (Char Bi-LSTM, Transformer Self-Attention, PII-GAT,
  Global Attention, CRF) execute as described.

> **Note on numerical reproducibility.** Due to data-access constraints
> (described in the *Data Availability* section below), the exact
> numerical results reported in the paper cannot be reproduced from this
> repository alone. The code is provided as a *reference implementation*,
> not as a bit-exact replication package.

---

## Data Availability Statement

### Target Domain (PII_tweet)

The primary target dataset (PII_tweet, 7,768 manually annotated tweets
across 7 PII categories) **is not redistributed** for the following
reasons:

1. **Privacy considerations.** The dataset contains tweets with real
   personally identifiable information (names, contact numbers, account
   identifiers, locations) extracted from public social media posts.
   Redistributing a curated collection of PII-bearing posts would amplify
   privacy risks for the original users.
2. **Institutional Review Board (IRB) restrictions.** The data collection
   and annotation were conducted under an institutional IRB protocol that
   restricts secondary redistribution of the corpus.
3. **Platform Terms of Service.** Twitter / X's developer policies allow
   sharing of tweet IDs but restrict redistribution of full tweet text.

A **de-identified sample of 20 tweets** is provided in
`data/sample/sample_pii_tweets.csv` so that the entire pipeline (training
and inference) can be executed end-to-end. All real names, phone numbers,
account identifiers, and URLs in this sample have been replaced with
placeholders while preserving the original BIOES annotation structure.

The full corpus is **available from the corresponding author upon
reasonable request**, subject to a Data Use Agreement and confirmation of
the requestor's institutional IRB approval.

### Source Domain (Six Public NER Datasets)

The six source-domain datasets used in the DTL framework
(CoNLL-2003, GMB, WNUT-17, Broad Twitter, Resume-NER, and i2b2/n2c2 2014)
**are not redistributed** in this repository because each is governed by
its own license, redistribution policy, or DUA. Please see
[`data/source/README.md`](data/source/README.md) for download links,
citations, and access procedures for every source dataset, including the
access-restricted i2b2 component.

---

## Repository Structure

```
MCAPIIE/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── evaluate.py                        # Shared entity-level F1 evaluation
├── .gitignore
│
├── MCA-PIIE/
│   └── script_MCA_PIIE.py             # Main model + DTL training pipeline
│
├── baselines/
│   ├── RoBERTa-CRF/                   # Fine-tuned PLM baseline
│   │   └── script_RoBERTa_CRF.py
│   ├── GPT-4o/                        # GPT-NER prompting (Wang et al., 2025)
│   │   └── script_GPT_NER.py
│   ├── UniNER-7B/                     # Universal NER (Zhou et al., 2024)
│   │   └── script_UniNER.py
│   ├── GLiNER/                        # GLiNER (Zaratiana et al., 2024)
│   │   └── script_GLiNER.py
│   ├── LLaMA-3-8B/                    # LLaMA-3-8B-Instruct, 3-shot
│   │   └── script_LLaMA3.py
│   └── Gemma-2-9B/                    # Gemma-2-9B-IT, zero-shot
│       └── script_Gemma2.py
│
└── data/
    ├── sample/
    │   └── sample_pii_tweets.csv      # 20 de-identified tweets
    └── source/
        └── README.md                  # Source-dataset access guide
```

---

## Setup

### 1. Clone

```bash
git clone <anonymous-repo-url>
cd MCAPIIE
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# Install spaCy English model (used by PII-GAT for dependency parsing)
python -m spacy download en_core_web_sm
```

### 3. (Optional) Download GloVe Twitter Embeddings

Required only for the main MCA-PIIE model:

```bash
mkdir -p embeddings
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -P embeddings/
unzip embeddings/glove.twitter.27B.zip -d embeddings/
# Use glove.twitter.27B.200d.txt
```

### 4. (Optional) API Key for GPT-4o Baseline

```bash
export OPENAI_API_KEY='your-key-here'
```

### 5. (Optional) Source Datasets for Full DTL Pipeline

To run the full DTL pre-training stage, download the source datasets
listed in [`data/source/README.md`](data/source/README.md) and place
them under `data/source/` in the schema described there.

---

## Running the Models

### MCA-PIIE (Main Model)

Modes correspond to experiments in the paper:

```bash
cd MCA-PIIE/

# Full pipeline with Deep Transfer Learning (Table 8)
python script_MCA_PIIE.py --mode full_dtl

# Target-only training (no DTL) — ablation in Table 9
python script_MCA_PIIE.py --mode target_only

# Source-only training/eval — Table 8 source-domain results
python script_MCA_PIIE.py --mode source_only

# All ablation variants — Table 9
python script_MCA_PIIE.py --mode ablation
```

Hyperparameter overrides:

```bash
python script_MCA_PIIE.py --mode full_dtl --epochs 30 --batch_size 16 --lr 1e-3
```

### Baselines

Every baseline defaults to the de-identified sample data in
`data/sample/sample_pii_tweets.csv`. Override `--data_path` (or the
`DATA_PATH` constant near the top of the file) to point at your own
corpus.

> **Note.** The 20-row sample data is intended only for verifying that
> the pipeline executes end-to-end; it is too small for meaningful F1
> evaluation. To obtain interpretable metrics, point the baselines at a
> full PII-annotated corpus following the BIOES schema documented above.

```bash
# Fine-tuned PLM baseline
cd baselines/RoBERTa-CRF/
python script_RoBERTa_CRF.py

# GPT-NER with GPT-4o (requires OPENAI_API_KEY)
cd ../GPT-4o/
python script_GPT_NER.py --model gpt-4o

# UniNER-7B-all
cd ../UniNER-7B/
python script_UniNER.py

# GLiNER
cd ../GLiNER/
python script_GLiNER.py

# LLaMA-3-8B-Instruct (3-shot ICL)
cd ../LLaMA-3-8B/
python script_LLaMA3.py

# Gemma-2-9B-IT (zero-shot)
cd ../Gemma-2-9B/
python script_Gemma2.py
```

---

## Hyperparameters (As Used in the Paper)

The default values in `MCA-PIIE/script_MCA_PIIE.py` (class `Config`) match
the configuration reported in the manuscript:

| Component | Setting |
|---|---|
| Word embeddings | GloVe-Twitter-200d |
| Char embedding dim | 30 |
| Char Bi-LSTM hidden | 50 (per direction) |
| Hidden dim | 300 |
| Transformer heads / layers | 6 / 2 |
| Transformer FF dim | 512 |
| GAT heads / hidden | 4 / 64 |
| Global attention dim | 300 |
| Max sequence length | 128 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Epochs | 30 |
| Patience (early stop) | 5 |
| Dropout | 0.3 |
| Gradient clip | 5.0 |
| Cross-validation | 5-fold |

---

## Hardware

- **MCA-PIIE training:** NVIDIA RTX 3090 (24 GB VRAM) is sufficient.
- **LLaMA-3-8B / Gemma-2-9B / UniNER-7B inference:** ≥ 24 GB VRAM, or use
  `bitsandbytes` 4-bit quantization on smaller GPUs.
- **GLiNER:** runs comfortably on CPU; GPU recommended.
- **GPT-4o:** API-only; no local GPU needed.

---

## Anonymity

This repository is intended for **double-blind peer review**. All author,
institution, email, and acknowledgement information has been removed from
the code and documentation. Please do not attempt to identify the authors
through the commit history of this repository.
