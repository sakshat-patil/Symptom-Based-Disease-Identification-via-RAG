# Symptom-Based-Disease-Identification-via-RAG

A hybrid system combining frequent pattern mining (FP-Growth) with Retrieval-Augmented Generation (RAG) for symptom-based disease identification using synthetic patient data (Synthea) and medical literature (MedQuAD).

## Project Overview

This project builds a pipeline that:
1. Generates synthetic patient records using Synthea
2. Extracts symptom-condition associations via FP-Growth mining
3. Retrieves relevant medical passages using dense retrieval (FAISS + PubMedBERT)
4. Combines mining confidence with retrieval similarity for hybrid disease identification

## Repository Structure

```
├── data/               # Raw and processed datasets
├── src/                # Source code for pipeline modules
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── docs/               # Project documentation and reports
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Tech Stack

- **Data Generation**: Synthea (synthetic FHIR patient records)
- **Mining**: FP-Growth (mlxtend)
- **Embeddings**: PubMedBERT
- **Retrieval**: FAISS
- **Mapping**: SNOMED-CT code mapping
- **Corpus**: MedQuAD medical Q&A dataset
