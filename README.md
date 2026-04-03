# Symptom-Based Disease Identification via Hybrid FP-Growth + RAG

A hybrid diagnostic system that fuses **FP-Growth association rule mining** with **dense retrieval (RAG)** to rank likely diagnoses from a patient's symptom set — backed by biomedical literature.

> **Course:** CS 226 Data Mining — San José State University, Spring 2026  
> **Team:** Sakshat Nandkumar Patil · Vineet Kumar · Aishwarya Madhave

---

## Results

Ablation study across 200 synthetic test cases, 41 diseases:

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR |
|------|----------|----------|-----------|-----|
| Retrieval-only | 7.5% | 23.5% | 38.5% | 0.144 |
| **Mining-only** | **56.0%** | **88.5%** | **92.5%** | **0.587** |
| Fused (α=0.6) | 54.0% | 93.5% | **99.0%** | 0.592 |

**Key finding:** Retrieval collapsed to 7.5% Recall@1 due to a vocabulary mismatch — the MedQuAD corpus uses clinical terminology (e.g. *myalgia*) while the Kaggle test cases use snake_case identifiers (e.g. `muscle_pain`). The FP-Growth rules, derived from the same vocabulary as the test set, were unaffected and drove strong performance. Fused scoring recovers at Recall@10 (99%) because retrieval still adds marginal signal at wider cutoffs, but lowering α (retrieval weight) would help Recall@1.

This finding motivates using domain-specific embeddings such as PubMedBERT that are trained on clinical vocabulary and can bridge the terminology gap.

---

## Architecture

```
Patient Symptoms (query)
        │
        ├──► FP-Growth Rule Lookup
        │         └── match symptom subsets → (disease, confidence, lift)
        │
        ├──► Sentence-Transformer Encoder
        │         └── FAISS flat-IP index over MedQuAD passages
        │                   └── top-K passages → retrieval similarity score
        │
        └──► Hybrid Fusion Reranker
                  FusedScore = α × RetrievalSim + (1−α) × MiningConf
                  (default α = 0.6)
                        │
                        ▼
             Ranked disease list + supporting citations
```

---

## Project Structure

```
Code/
├── README.md
├── requirements.txt
├── src/
│   ├── etl.py                  # Ingest Synthea FHIR, Kaggle CSV, or synthetic data → transactions.csv
│   ├── mining.py               # FP-Growth frequent itemset mining → association_rules.csv
│   ├── retrieval.py            # Sentence-transformer encoder + FAISS dense retriever
│   ├── fusion_reranker.py      # Hybrid fusion: α·retrieval + (1−α)·mining confidence
│   ├── evaluation.py           # Recall@K, Precision@K, F1@K, MRR, ablation runner
│   ├── run_experiment.py       # End-to-end experiment pipeline
│   ├── medquad_preprocessor.py # MedQuAD XML → chunked JSONL passages
│   └── config.py               # Shared paths and hyperparameters
├── data/
│   ├── raw/                    # Source data (not committed — see Data Setup below)
│   ├── processed/
│   │   ├── transactions.csv    # 4,920 records × 41 diseases (Kaggle)
│   │   └── association_rules.csv  # 508 mined rules (support≥0.01, conf≥0.5)
│   └── results/
│       ├── ablation_summary.csv   # Per-mode metrics
│       └── experiment_results.csv # Per-case results
├── notebooks/
└── docs/
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

**Option A — Kaggle dataset (recommended, free)**

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p data/raw
unzip data/raw/disease-symptom-description-dataset.zip -d data/raw
```

**Option B — Synthea synthetic EHR**

Download [Synthea](https://github.com/synthetichealth/synthea), generate patients, place FHIR bundles in `data/raw/`.

**MedQuAD corpus (for retrieval)**

```bash
git clone https://github.com/abachaa/MedQuAD data/raw/MedQuAD
```

### 3. Run the pipeline

```bash
# Step 1 — ETL: raw data → transactions
python src/etl.py --source kaggle --csv data/raw/dataset.csv

# Step 2 — Mine association rules (raise --min_support if OOM)
python src/mining.py --min_support 0.03 --min_confidence 0.5

# Step 3 — Preprocess MedQuAD for retrieval
python src/medquad_preprocessor.py

# Step 4 — Run full experiment + ablation
python src/run_experiment.py --rules data/processed/association_rules.csv \
    --corpus data/raw/passages.jsonl --n_cases 200 --alpha 0.6
```

Results are written to `data/results/`.

---

## Module Overview

### `etl.py`
Ingests three source types in order of preference: Synthea FHIR JSON → Kaggle wide-format CSV → built-in synthetic fallback. Outputs `transactions.csv` with columns `patient_id | condition | symptoms` (pipe-separated symptom list per row).

### `mining.py`
Runs [mlxtend](https://rasbt.github.io/mlxtend/) FP-Growth on the one-hot transaction matrix. Each row is a (symptom₁, …, symptomₙ, DX:disease) basket. Filters rules to those with a single disease consequent and no disease in the antecedent. Outputs `association_rules.csv` with `symptom_set | disease | support | confidence | lift`.

> **Memory note:** The one-hot matrix can be large. If you hit OOM, use `--min_support 0.03` instead of the default `0.01`.

### `retrieval.py`
Encodes a corpus of MedQuAD passages using a sentence-transformer model (default: `all-MiniLM-L6-v2`) and stores them in a FAISS flat inner-product index (cosine similarity after L2 normalisation). Falls back to a 20-passage demo corpus when no JSONL file is provided.

### `fusion_reranker.py`
Combines per-disease signals:
- **Mining score** — maximum confidence across all rules whose antecedent is a subset of the query symptoms
- **Retrieval score** — cosine similarity of the closest passage mentioning each disease
- **Fused** = α × retrieval + (1−α) × mining (default α = 0.6)

### `evaluation.py`
Generates synthetic test cases from the transaction table, runs the full pipeline per case, and computes Recall@K, Precision@K, F1@K, and MRR for K ∈ {1, 3, 5, 10}. Also runs the three-way ablation (retrieval-only / mining-only / fused).

---

## Discussion

### Why did retrieval underperform?
The MedQuAD corpus describes symptoms in clinical prose ("myalgia", "pyrexia") while the Kaggle dataset uses normalised snake_case tokens (`muscle_pain`, `high_fever`). A general-purpose sentence encoder cannot bridge this gap because the surface forms are too different. Solutions include:
- Using **PubMedBERT** or **BioSentBERT** embeddings trained on biomedical text
- Adding a **synonym expansion** step (UMLS / MeSH) to the query before encoding
- Replacing MedQuAD with a corpus that uses the same vocabulary as the training data

### Why does fused beat mining at Recall@10?
At Recall@10 the fused model reaches 99% vs 92.5% for mining-only. Even noisy retrieval signal is enough to pull the true disease into the top-10 when it was just outside mining's coverage. Lowering α (e.g. α=0.3) would likely improve Recall@1 as well.

---

## References

1. Walonoski et al. (2018). *Synthea: An approach, method, and software for generating synthetic patients.* JAMIA.
2. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
3. Ben Abacha & Demner-Fushman (2019). *A Question-Entailment Approach to Question Answering.* BMC Bioinformatics.
4. Han et al. (2000). *Mining Frequent Patterns without Candidate Generation.* ACM SIGMOD.
5. Gu et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical NLP.* ACM TOCH.
6. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.

---

*Academic project only — not intended for clinical use.*
