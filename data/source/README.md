# Source Domain Datasets

The Deep Transfer Learning (DTL) framework in MCA-PIIE pre-trains the model
on **six publicly available NER datasets** before fine-tuning on the target
PII Twitter corpus. To respect each dataset's license, terms, and Data Use
Agreement (DUA), **none of these datasets are redistributed in this
repository**. Instead, we provide the original sources, citations, and
access procedures below.

After obtaining the raw data, place each dataset (preprocessed into the
shared three-column BIOES format described in the main `README.md`) under
this directory:

```
data/source/
├── conll2003_all.csv
├── gmb.csv
├── wnut17.csv
├── broad_twitter.csv
├── resume_ner.csv
└── i2b2_2014.csv          # access-restricted; see below
```

A reference preprocessing schema (`sentence_id`, `Tokens`,
`Word_Level_BIOES`) is illustrated by `data/sample/sample_pii_tweets.csv`.

---

## 1. CoNLL-2003

General-purpose newswire NER (PER, LOC, ORG, MISC).

- **Source:** https://www.clips.uantwerpen.be/conll2003/ner/
- **HuggingFace mirror:** https://huggingface.co/datasets/eriktks/conll2003
- **Citation:** Sang, E.T.K. and De Meulder, F. *Introduction to the
  CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.*
  In Proceedings of CoNLL-2003, 142–147.
- **License note:** The Reuters newswire text underlying CoNLL-2003 is
  subject to a Reuters research-only redistribution restriction. We
  therefore do not redistribute the corpus.

## 2. GMB (Groningen Meaning Bank)

Multi-genre English NER with extended entity types.

- **Source:** https://gmb.let.rug.nl/data.php
- **Citation:** Bos, J., Basile, V., Evang, K., Venhuizen, N.J., and Bjerva,
  J. *The Groningen Meaning Bank.* In Handbook of Linguistic Annotation
  (2017), Springer, 463–496.
- **License:** Public domain / CC.

## 3. WNUT-17 (Emerging and Rare Entity Recognition in User-Generated Text)

Twitter / social-media NER focused on emerging and rare entities.

- **Source:** https://noisy-text.github.io/2017/emerging-rare-entities.html
- **GitHub:** https://github.com/leondz/emerging_entities_17
- **HuggingFace mirror:** https://huggingface.co/datasets/leondz/wnut_17
- **Citation:** Derczynski, L., Nichols, E., van Erp, M., and Limsopatham, N.
  *Results of the WNUT2017 Shared Task on Novel and Emerging Entity
  Recognition.* In Proceedings of the 3rd Workshop on Noisy User-generated
  Text (W-NUT 2017), 140–147.

## 4. Broad Twitter Corpus

Twitter NER with broader coverage and noisier text than WNUT-17.

- **Source:** https://github.com/GateNLP/broad_twitter_corpus
- **Citation:** Derczynski, L., Bontcheva, K., and Roberts, I. *Broad
  Twitter Corpus: A Diverse Named Entity Recognition Resource.* In
  Proceedings of COLING 2016, 1169–1179.

## 5. Resume-NER

Named entity recognition on résumé text, used as a profession-rich source
domain in the DTL framework.

- **Source:** https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
- **Citation:** DATATURKS. *Resume Entities for NER, Version 1.* 2018.
  https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner
- **License:** Refer to the Kaggle dataset page for current usage terms.

## 6. i2b2 / n2c2 2014 De-identification Corpus (Access-Restricted)

Clinical narratives annotated for protected health information (PHI),
used in our DTL framework as a medical-domain PII source.

- **Source / current portal:** https://n2c2.dbmi.hms.harvard.edu/
- **HuggingFace metadata-only listing:**
  https://huggingface.co/datasets/bigbio/n2c2_2014_deid
- **Citation:** Stubbs, A., Kotfila, C., and Uzuner, Ö. *Automated systems
  for the de-identification of longitudinal clinical narratives: Overview
  of 2014 i2b2/UTHealth shared task Track 1.* Journal of Biomedical
  Informatics 58S (2015), S11–S19.

> **Important access note.** The i2b2 / n2c2 datasets are governed by a
> strict Data Use Agreement that *prohibits redistribution to any third
> party, including via GitHub or any other public website.* At the time
> of this writing, n2c2 has additionally indicated that the datasets are
> **temporarily unavailable** through their portal pending administrative
> review. Researchers seeking to reproduce experiments involving this
> dataset must apply directly through the n2c2 / DBMI Data Portal once
> access is restored, and complete the required DUA.

---

## Total Corpus Size in the Paper

The manuscript reports **24,762 documents** as the total source-domain
corpus size used during DTL pre-training. This figure aggregates
PII-relevant subsets of the six datasets above after applying the
preprocessing and entity-mapping pipeline described in Section 5 of the
manuscript. Because the i2b2 component is access-restricted and the
remaining five public datasets are subject to their respective licenses,
this exact corpus cannot be redistributed; it can be reconstructed by
following the source links above and applying the preprocessing schema
documented in the main `README.md`.
