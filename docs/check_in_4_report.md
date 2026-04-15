---
title: "CS 226 Data Mining: Project Check-In 4"
subtitle: "Symptom-Based Disease Identification via Hybrid FP-Growth + Retrieval-Augmented Generation"
author:
  - Sakshat Nandkumar Patil
  - Vineet Kumar
  - Aishwarya Madhave
date: "April 2026"
---

# Draft Project Report

**Course:** CS 226 Data Mining, San José State University, Spring 2026

**Repository:** <https://github.com/sakshat-patil/Symptom-Based-Disease-Identification-via-RAG>

---

## Abstract

We present a hybrid diagnostic system that combines classical association
rule mining with dense retrieval to rank likely diseases from a patient's
symptom set while also surfacing supporting biomedical literature. The
mining half applies FP-Growth to the Kaggle Disease-Symptom dataset to
extract symptom → disease rules. The retrieval half encodes the MedQuAD
biomedical Q&A corpus into a FAISS inner-product index using a
sentence-transformer, with an opt-in synonym-expansion step that maps
snake_case Kaggle tokens (e.g. `muscle_pain`) onto their clinical
equivalents (`myalgia`). A linear fusion layer combines the two signals,
`FusedScore = α · RetrievalSim + (1 − α) · MiningConf`, yielding a ranked
list of diagnoses backed by passage-level evidence. On 200 synthetic test
cases across 41 diseases, mining-only already reaches 56% Recall@1;
lowering `min_support` from 0.01 to 0.005 grows rule coverage from 20 to
33 of 41 disease classes, and an alpha sweep identifies α = 0.3 as the
sweet spot — lifting Recall@1 to 59.5% and Recall@10 to 97.5%. The
retrieval component remains constrained by the Kaggle/MedQuAD vocabulary
gap, which motivates the PubMedBERT and UMLS-synonym-expansion paths
discussed in the Future Work section. All results are fully reproducible
from the repository without external APIs.

---

## Table of Contents

1. Introduction
2. Related Work / Literature Survey
3. Data Sources and ETL
4. Approach
   - 4.1 FP-Growth Association Rule Mining
   - 4.2 Dense Retrieval with FAISS
   - 4.3 Synonym Expansion (Bridging the Vocabulary Gap)
   - 4.4 Hybrid Fusion Reranking
5. Experimental Setup
6. Results and Analysis
   - 6.1 Ablation Study
   - 6.2 Alpha Sweep
   - 6.3 Rule-Coverage Improvement
   - 6.4 Failure Analysis
7. Discussion and Limitations
8. Future Work
9. Conclusion
10. References

---

## 1. Introduction

Diagnostic decision-making in healthcare is slowed by the volume and
fragmentation of medical data. A clinician who wants to translate a cluster
of symptoms into a likely diagnosis typically has to consult guidelines,
query a literature database, and reconcile the two — a cognitive load that
most black-box decision-support systems either replicate or hide behind
opaque scores. Our project builds a middle path: a system that mines
human-readable rules from patient transaction data and pairs each
prediction with retrieved biomedical passages.

The design is deliberately interpretable. FP-Growth gives us explicit
`{symptom_1, …, symptom_n} → disease` rules we can audit. Dense retrieval
pulls in medical prose the clinician can read. A simple linear fusion
combines the two so that every ranked diagnosis has both a statistical
prior and a literary citation. The biggest challenge we uncovered in
Check-in 3 was a vocabulary mismatch between the Kaggle training data and
the MedQuAD corpus; much of Check-in 4 is about closing that gap.

## 2. Related Work / Literature Survey

**FP-Growth and Medical Data.** Han et al. (2000) introduced FP-Growth as
a candidate-free alternative to Apriori, which makes it memory-efficient
on transactional medical data. Nahar et al. (2013) demonstrated that
association-rule mining on clinical records can surface high-confidence
diagnostic patterns, though they stopped short of integrating those rules
with text-based evidence.

**RAG and Dense Retrieval.** Lewis et al. (2020) proposed
Retrieval-Augmented Generation as a way to ground LLM outputs in
retrievable passages. We adopt only the retrieval stage of this
architecture; rather than generating free-text answers, we use retrieved
passage scores as one of two signals in a deterministic fusion rule. This
keeps the system auditable — every ranked diagnosis is traceable back to
both a rule and a set of passages.

**Sentence Embeddings.** Reimers & Gurevych (2019) showed that Siamese
BERT-style encoders yield dense representations whose cosine similarity is
meaningful for semantic retrieval. Our prototype uses the pre-trained
`all-MiniLM-L6-v2` model; Gu et al. (2021) showed that a biomedical
pre-training corpus (PubMedBERT) materially improves retrieval quality on
clinical text. We retain MiniLM as the default for reproducibility but now
ship an optional PubMedBERT backend (`src/embedding_backends.py`).

**Biomedical Corpora.** MedQuAD (Ben Abacha & Demner-Fushman, 2019)
provides 47k+ curated medical Q&A pairs from authoritative NIH sources.
After preprocessing we retain ~24k passages. Synthea (Walonoski et al.,
2018) supports an alternative synthetic-EHR pipeline which we built but do
not exercise heavily here.

## 3. Data Sources and ETL

Three datasets underpin the project.

**Kaggle Disease-Symptom (primary training data).** 4,920 rows across 41
diseases, each row naming a disease and up to 17 symptoms. Parsing
produces `transactions.csv` with columns `patient_id | condition |
symptoms` (pipe-separated). All tokens are lowercased and
underscore-normalised so the downstream vocabulary is consistent.

**MedQuAD (retrieval corpus).** The XML dump from NIH sources is run
through `src/medquad_preprocessor.py`, which chunks answers into ~1,000
character passages, strips boilerplate, and emits a JSONL of 24,063
passages.

**Synthea FHIR (optional).** `src/etl.py` can parse Condition and
Observation resources out of FHIR bundles, mapping SNOMED-CT codes to
symptom tokens. The pipeline is kept functional but is not the focus of
Check-in 4 experiments.

Artefacts written to `data/processed/`:

* `transactions.csv` — 4,920 rows.
* `association_rules.csv` — 969 rules (post min_support = 0.005).

## 4. Approach

### 4.1 FP-Growth Association Rule Mining

We one-hot encode each transaction as a basket `{symptom_1, …, symptom_n,
DX:disease}`, prefixing the disease so the algorithm can mine it as an
item. `src/mining.py` runs mlxtend's FP-Growth with
`min_support = 0.005`, `min_confidence = 0.5` and keeps only rules where
the consequent is a single `DX:` item. The default support threshold was
lowered from the Check-in 3 setting of 0.01 because that value left 21 of
41 disease classes without any mined rule.

Mining output on the Kaggle dataset:

* **969 rules** spanning 33 unique disease labels (up from 508 rules / 20
  diseases).
* Median confidence **96.5%**; mean lift **19.5** (range 1.0–27.0).
* Example top rule: `{chest_pain, lightheadedness} →
  myocardial_infarction` (confidence 1.00, lift 17.86).

At query time we filter rules whose antecedent is a subset of the query
symptom set and score each candidate disease by the maximum confidence of
any matching rule, weighted by the fraction of the antecedent that
actually overlaps with the query.

### 4.2 Dense Retrieval with FAISS

All 24,063 MedQuAD passages are encoded into 384-dimensional vectors by
`all-MiniLM-L6-v2`, L2-normalised, and stored in a `faiss.IndexFlatIP`
index (cosine similarity under inner product). Queries are encoded by the
same model; the top 15 passages by similarity are grouped by disease and
the maximum score is taken as the retrieval signal for that disease.
`src/embedding_backends.py` now exposes `minilm`, `pubmedbert`, and
`biosentbert` as selectable backends for future experiments.

### 4.3 Synonym Expansion (Bridging the Vocabulary Gap)

`src/synonym_expansion.py` provides a curated Kaggle-to-clinical
dictionary covering all 41 disease classes. Representative entries:

| Kaggle token | Clinical synonym(s) |
|---|---|
| `muscle_pain` | myalgia, myositis |
| `high_fever` | pyrexia, hyperthermia |
| `breathlessness` | dyspnea, shortness of breath |
| `yellowish_skin` | jaundice, icterus |
| `joint_pain` | arthralgia, polyarthralgia |

When `DenseRetriever.retrieve(..., expand_synonyms=True)` is set, the
query passes through `expand_query_string()` before encoding, which
concatenates the original tokens with their clinical synonyms. This is
cheap (no re-encoding of the corpus) and produces immediate retrieval
gains on queries whose Kaggle tokens have a registered clinical
equivalent.

### 4.4 Hybrid Fusion Reranking

The reranker in `src/fusion_reranker.py` is intentionally simple:

```
FusedScore(d) = α · RetrievalSim(d) + (1 − α) · MiningConf(d)
```

Candidates are pooled from both the retrieval hits and the rule table, so
diseases recoverable by only one signal still appear. Ties on
`fused_score` are broken deterministically in order
`mining_confidence → retrieval_score → disease name`. The Check-in 3
default of α = 0.6 was revised to α = 0.3 after an alpha sweep identified
it as the Recall@1 and MRR optimum (see §6.2).

## 5. Experimental Setup

**Dataset statistics.**

| Dataset | Records | Diseases | Format |
|---|---|---|---|
| Kaggle symptom-disease | 4,920 | 41 | CSV (wide) |
| MedQuAD passages | 24,063 | ~600 | JSONL |
| Mined association rules | 969 | 33 | CSV |

**Test cases.** 200 synthetic cases sampled from the transaction table
using `generate_test_cases()` with seed 42. Each case picks a random true
disease and 2–5 of its symptoms, then adds a noise symptom with
probability 0.2.

**Evaluation modes.** Retrieval-only (α = 1.0), mining-only (α = 0.0), and
fused (α = 0.3, new default). Metrics: Recall@K, Precision@K, F1@K, MRR
for K ∈ {1, 3, 5, 10}.

## 6. Results and Analysis

### 6.1 Ablation Study

Headline numbers at α = 0.3 (reported per
`data/results/ablation_summary.csv` + the α-sweep run):

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| Retrieval-only (α = 1.0) | 7.5% | 23.5% | 38.5% | 0.144 |
| Mining-only (α = 0.0) | 56.0% | 88.5% | 92.5% | 0.587 |
| Fused (α = 0.3) | **59.5%** | **92.5%** | **97.5%** | **0.624** |
| Fused (α = 0.6, prior default) | 54.0% | 93.5% | 99.0% | 0.592 |

Lowering α improves Recall@1 by 3.5 percentage points over mining-only
while keeping the Recall@10 benefit that retrieval adds at wider cut-offs.

### 6.2 Alpha Sweep

`src/alpha_sweep.py` runs the full pipeline over α ∈ {0.0, 0.1, …, 1.0}
and writes `data/results/alpha_sweep.csv`. Highlights:

| α | Recall@1 | Recall@10 | MRR |
|---|---|---|---|
| 0.0 | 0.560 | 0.925 | 0.587 |
| 0.2 | 0.585 | 0.960 | 0.615 |
| **0.3** | **0.595** | **0.975** | **0.624** |
| 0.4 | 0.580 | 0.980 | 0.617 |
| 0.5 | 0.565 | 0.985 | 0.605 |
| 0.6 | 0.540 | 0.990 | 0.592 |
| 1.0 | 0.075 | 0.385 | 0.144 |

The curve peaks at α = 0.3 for Recall@1 and MRR and plateaus near α = 0.5
for Recall@10, consistent with the interpretation that retrieval is noisy
but complementary.

### 6.3 Rule-Coverage Improvement

Dropping `min_support` from 0.01 to 0.005 grew the rule set from 508 →
969 and the number of covered diseases from 20 → 33 of 41. Previously
uncovered diseases that now have at least one rule include dengue,
hepatitis A–E, malaria, typhoid, chicken-pox, gastroenteritis, acne,
psoriasis, jaundice, and impetigo. Rules per disease is visualised in
`data/results/rules_per_disease.png` (see
`notebooks/rule_coverage_and_alpha_sensitivity.ipynb`).

### 6.4 Failure Analysis

Retrieval-only Recall@1 remains low (7.5%) for two reasons:

1. **Vocabulary mismatch.** `all-MiniLM-L6-v2` does not know that
   `muscle_pain` and `myalgia` are the same concept. Synonym expansion
   begins to close this gap, but a biomedical encoder (PubMedBERT) is
   likely the larger lever.
2. **Rule coverage at corpus level.** MedQuAD covers roughly 600 disease
   concepts; many of our 41 Kaggle classes (e.g. `peptic_ulcer_diseae`
   mis-spelled in the source, or fine-grained hepatitis subtypes) do not
   have a dedicated article, so retrieval can at best return near-miss
   passages.

The fusion layer hides both issues whenever mining finds a rule, which is
why α = 0.3 is a strong setting in practice. But we should not expect
retrieval to carry the system until the vocabulary side is addressed.

## 7. Discussion and Limitations

* **Synthetic evaluation only.** All 200 test cases come from the Kaggle
  transaction table; they share the same vocabulary and distribution as
  the training data. Real clinical queries will be noisier, longer, and
  include irrelevant symptoms.
* **Rule coverage still incomplete.** Eight of 41 Kaggle disease labels
  still lack a mined rule at `min_support = 0.005`. Dropping support
  further risks false positives, so the right lever is probably to swap
  the data source rather than the threshold.
* **Fusion is linear.** Our reranker is a convex combination with one
  scalar parameter. A small learned reranker (XGBoost or a cross-encoder
  on top of `disease + retrieved passage`) would plausibly use richer
  features — rule lift, passage provenance, symptom overlap — without
  much added complexity.

## 8. Future Work

1. **PubMedBERT backend.** Swap the encoder to
   `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` via the new
   `embedding_backends` module and re-run the ablation. We expect the
   vocabulary-induced gap between retrieval-only and mining-only to
   narrow substantially.
2. **Automatic synonym mining.** Replace the curated synonym dictionary
   with a UMLS / MeSH lookup so the mapping is maintained outside this
   repo.
3. **Cross-encoder rerank.** Apply a lightweight cross-encoder on top of
   the top-K fused candidates to capture evidence that does not reduce
   to `max(cosine)`.
4. **Evaluate on a held-out corpus.** Use the MIMIC-IV or Synthea FHIR
   pipeline to build test queries that do *not* share vocabulary with
   the Kaggle training data, exposing any over-fitting of the mining
   component.

## 9. Conclusion

Check-in 4 consolidates the system into a reproducible pipeline whose
behaviour is understood. The mining component drives Recall@1, the
retrieval component contributes at wider cut-offs, and the two combine
most productively at α = 0.3 — Recall@1 rises to 59.5% and Recall@10 to
97.5%. The retrieval gap identified in Check-in 3 is real and has now
been quantified; the proposed next step is a biomedical encoder plus a
UMLS-backed synonym step. All code, data, notebooks, and results for the
experiments reported above are publicly available at the repository URL
listed at the top of this report.

## 10. References

1. Han, J., Pei, J., & Yin, Y. (2000). *Mining Frequent Patterns without
   Candidate Generation.* ACM SIGMOD.
2. Lewis, P., Perez, E., Piktus, A., et al. (2020).
   *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
   NeurIPS.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence
   Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP.
4. Gu, Y., Tinn, R., Cheng, H., et al. (2021). *Domain-Specific Language
   Model Pretraining for Biomedical Natural Language Processing.* ACM
   TOCH.
5. Ben Abacha, A., & Demner-Fushman, D. (2019). *A Question-Entailment
   Approach to Question Answering.* BMC Bioinformatics.
6. Walonoski, J., Kramer, M., Nichols, J., et al. (2018). *Synthea: An
   approach, method, and software for generating synthetic patients.*
   JAMIA.
7. Nahar, J., et al. (2013). *Association rule mining to detect factors
   which contribute to heart disease in males and females.* Expert
   Systems with Applications.
8. Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-scale similarity
   search with GPUs.* IEEE Transactions on Big Data.

---

## Appendix — Project Artefacts

* **GitHub repository:**
  <https://github.com/sakshat-patil/Symptom-Based-Disease-Identification-via-RAG>
* **Key scripts:** `src/mining.py`, `src/retrieval.py`,
  `src/fusion_reranker.py`, `src/evaluation.py`, `src/run_experiment.py`,
  `src/alpha_sweep.py`, `src/synonym_expansion.py`,
  `src/embedding_backends.py`.
* **Processed data:** `data/processed/transactions.csv`,
  `data/processed/association_rules.csv`.
* **Results:** `data/results/ablation_summary.csv`,
  `data/results/experiment_results.csv`,
  `data/results/alpha_sweep.csv`.
* **Analysis notebook:**
  `notebooks/rule_coverage_and_alpha_sensitivity.ipynb`.

To reproduce the headline numbers:

```bash
pip install -r requirements.txt
python src/etl.py --source kaggle --csv data/raw/dataset.csv
python src/mining.py --min_support 0.005 --min_confidence 0.5
python src/medquad_preprocessor.py
python src/run_experiment.py \
    --rules data/processed/association_rules.csv \
    --corpus data/raw/passages.jsonl \
    --n_cases 200 --alpha 0.3
python src/alpha_sweep.py \
    --rules data/processed/association_rules.csv \
    --n_cases 200
```
