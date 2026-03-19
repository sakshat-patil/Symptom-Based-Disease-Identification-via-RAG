# Record-Based Medical Diagnostic Assistant

A hybrid diagnostic assistant that integrates **Frequent Pattern Mining** with **Retrieval-Augmented Generation (RAG)** to suggest accurate diagnoses based on patient symptom clusters, backed by peer-reviewed medical evidence.

> **Course Project** — San José State University  
> **Team:** Sakshat Nandkumar Patil, Vineet Kumar, Aishwarya Madhave

---

## Overview

Clinicians face significant cognitive load when synthesizing patient symptoms with the ever-expanding body of medical literature. This system addresses that gap by combining two complementary approaches:

1. **FP-Growth Mining** on synthetic EHR data (Synthea) to extract high-confidence symptom→disease association rules as statistical priors.
2. **Dense Retrieval (RAG)** over a biomedical corpus (MedQuAD + PubMed Central) to fetch relevant medical literature for evidence grounding.
3. **Hybrid Fusion Reranking** that merges retrieval similarity scores with mined association confidence to produce a final ranked diagnostic output.

## Architecture

```
User Query (symptoms)
        │
        ├──► FP-Growth Rule Lookup ──► Mining Confidence Score
        │
        ├──► PubMedBERT Encoder ──► FAISS Retrieval ──► Retrieval Similarity Score
        │
        └──► Hybrid Fusion Reranker
                    │
                    ▼
        Ranked Diagnoses + Supporting Citations
```

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── synthea_etl.py            # ETL pipeline for Synthea FHIR data
│   ├── fp_growth_mining.py       # Association rule mining module
│   ├── medquad_preprocessor.py   # MedQuAD corpus cleaning and chunking
│   ├── embedding_index.py        # PubMedBERT embedding + FAISS index (WIP)
│   ├── retrieval.py              # Dense retrieval pipeline (WIP)
│   └── fusion_reranker.py        # Hybrid reranking fusion (WIP)
├── data/
│   ├── raw/                      # Raw Synthea JSON + MedQuAD XML (not committed)
│   ├── processed/                # Cleaned CSVs, JSONL chunks
│   └── rules/                    # Mined association rules
├── notebooks/
│   └── exploratory_analysis.ipynb
├── docs/
│   └── Project_CheckIn_Report.pdf
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.9+
- [Synthea](https://github.com/synthetichealth/synthea) (for generating EHR data)
- [MedQuAD](https://github.com/abachaa/MedQuAD) (clone into `data/raw/MedQuAD`)

### Installation

```bash
git clone https://github.com/<your-username>/medical-diagnostic-assistant.git
cd medical-diagnostic-assistant
pip install -r requirements.txt
```

### Requirements

```
pandas
mlxtend
lxml
scikit-learn
transformers
faiss-cpu
torch
```

## Usage

### 1. Generate and process Synthea data

```bash
# Generate synthetic patients (run Synthea separately first)
# Then parse FHIR JSON into transaction tables:
python src/synthea_etl.py --input_dir ./data/raw/synthea_fhir --output_dir ./data/processed
```

This produces `transactions.csv` and `symptom_baskets.csv` in the output directory.

### 2. Run FP-Growth mining

```bash
python src/fp_growth_mining.py --input ./data/processed/symptom_baskets.csv \
                               --output ./data/rules/association_rules.csv \
                               --min_support 0.01 --min_confidence 0.5
```

Outputs a CSV of symptom→disease association rules ranked by confidence and lift.

### 3. Preprocess MedQuAD corpus

```bash
python src/medquad_preprocessor.py --input_dir ./data/raw/MedQuAD \
                                   --output ./data/processed/medquad_chunks.jsonl \
                                   --chunk_size 256 --chunk_overlap 64
```

Produces a JSONL file of overlapping text passages ready for vector embedding.

## Evaluation Plan

| Metric | Component | Target |
|--------|-----------|--------|
| Recall@K (K=5,10,20) | Retrieval | ≥ 0.75 |
| MRR | Retrieval | ≥ 0.60 |
| Precision@K | Diagnostic accuracy | ≥ 0.70 |
| F1-Score | Top-1 diagnosis | ≥ 0.65 |
| Latency | End-to-end | < 3 sec |

An ablation study comparing retrieval-only, mining-only, and fused approaches is planned.

## References

1. Walonoski et al. (2018). *Synthea: An approach, method, and software for generating synthetic patients and the synthetic electronic health care record.* JAMIA.
2. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
3. Ben Abacha & Demner-Fushman (2019). *A Question-Entailment Approach to Question Answering.* BMC Bioinformatics.
4. Han et al. (2000). *Mining Frequent Patterns without Candidate Generation.* ACM SIGMOD.
5. Lee et al. (2020). *BioBERT: A pre-trained biomedical language representation model.* Bioinformatics.
6. Gu et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical NLP.* ACM TOCH.

## License

This project is for academic purposes only. Not intended for clinical use.
