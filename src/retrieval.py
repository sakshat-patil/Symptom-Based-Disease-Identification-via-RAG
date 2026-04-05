"""Dense retrieval over MedQuAD with FAISS IndexFlatIP.

Now with optional clinical-synonym expansion at query time. See
synonym_expansion.SYMPTOM_SYNONYMS — the expansion appends formal
clinical terms to the natural-language probe before encoding, which
closes part of the Kaggle/MedQuAD vocabulary gap without re-embedding
the corpus.

Owner: Vineet.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from .disease_keywords import diseases_matching
from .embedding_backends import Backend, get_backend
from .synonym_expansion import expand_query_string

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PASSAGES = ROOT / "data" / "processed" / "passages.jsonl"
PROCESSED = ROOT / "data" / "processed"


def _emb_path(name: str) -> Path:
    return PROCESSED / f"embeddings_{name}.npy"


def _index_path(name: str) -> Path:
    return PROCESSED / f"faiss_{name}.index"


def load_passages(path: Path = DEFAULT_PASSAGES) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_index(backend: Backend, passages: list[dict],
                 batch_size: int = 64) -> tuple[np.ndarray, faiss.Index]:
    texts = [p["text"] for p in passages]
    emb = backend.encode(texts, batch_size=batch_size, show_progress=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return emb, index


def save_index(name: str, emb: np.ndarray, index: faiss.Index) -> None:
    np.save(_emb_path(name), emb)
    faiss.write_index(index, str(_index_path(name)))


def load_or_build(backend: Backend, passages: list[dict],
                   rebuild: bool = False) -> faiss.Index:
    p = _index_path(backend.name)
    if p.exists() and not rebuild:
        return faiss.read_index(str(p))
    emb, idx = build_index(backend, passages)
    save_index(backend.name, emb, idx)
    return idx


def _candidate_diseases_from_passage(p: dict, universe: set[str]) -> list[str]:
    """Match a passage to the disease universe via curated keywords. Focus
    first (highest precision), then question. Don't fall back to text — the
    "What is Chest Pain?"-style articles inflate scores otherwise."""
    cands = diseases_matching(p.get("focus", ""), universe)
    if cands:
        return cands
    return diseases_matching(p.get("question", ""), universe)


@dataclass
class DenseRetriever:
    backend: Backend
    passages: list[dict]
    index: faiss.Index
    disease_universe: set[str]

    @classmethod
    def from_files(cls, name: str, diseases: Iterable[str],
                   passages_path: Path = DEFAULT_PASSAGES) -> "DenseRetriever":
        backend = get_backend(name)
        passages = load_passages(passages_path)
        index = load_or_build(backend, passages)
        return cls(backend=backend, passages=passages, index=index,
                   disease_universe=set(diseases))

    def retrieve(self, symptoms: list[str], top_k: int = 30,
                 expand_synonyms: bool = False
                  ) -> tuple[dict[str, float], dict[str, list[dict]]]:
        query = "symptoms: " + ", ".join(s.replace("_", " ") for s in symptoms)
        if expand_synonyms:
            query = expand_query_string(query, symptoms)
        q = self.backend.encode([query])
        scores, ids = self.index.search(q, top_k)
        per_disease_score: dict[str, float] = {}
        per_disease_passages: dict[str, list[dict]] = defaultdict(list)
        for s, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i < 0:
                continue
            p = self.passages[i]
            cands = _candidate_diseases_from_passage(p, self.disease_universe)
            if not cands:
                continue
            for d in cands:
                if s > per_disease_score.get(d, -1.0):
                    per_disease_score[d] = float(s)
                per_disease_passages[d].append({**p, "score": float(s)})
        return per_disease_score, dict(per_disease_passages)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="minilm",
                   choices=["minilm", "pubmedbert"])
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    import pandas as pd
    diseases = set(pd.read_csv(PROCESSED / "transactions.csv")["condition"].unique())
    backend = get_backend(args.backend)
    passages = load_passages()
    print(f"[retrieval] passages: {len(passages)}; backend={backend.name}")
    emb, index = build_index(backend, passages)
    save_index(backend.name, emb, index)


if __name__ == "__main__":
    main()
