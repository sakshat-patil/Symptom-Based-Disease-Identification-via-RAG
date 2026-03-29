"""Dense retrieval over MedQuAD with FAISS IndexFlatIP.

Inner-product on L2-normalised vectors == cosine sim. The heavy work is
encoding the 24k passages once and caching the matrix and index to disk;
everything after that is millisecond-scale.

Query path:
    1. Build a natural-language version of the symptom list.
    2. Encode and ask FAISS for the top-K passages.
    3. Map each passage to one of the 41 Kaggle diseases via
       disease_keywords.diseases_matching.
    4. Aggregate to per-disease scores by max() of passage similarities.

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

    def retrieve(self, symptoms: list[str], top_k: int = 30
                  ) -> tuple[dict[str, float], dict[str, list[dict]]]:
        query = "symptoms: " + ", ".join(s.replace("_", " ") for s in symptoms)
        q = self.backend.encode([query])
        scores, ids = self.index.search(q, top_k)
        per_disease_score: dict[str, float] = {}
        per_disease_passages: dict[str, list[dict]] = defaultdict(list)
        for s, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i < 0:
                continue
            p = self.passages[i]
            cands = diseases_matching(p.get("focus", ""), self.disease_universe)
            if not cands:
                cands = diseases_matching(p.get("question", ""),
                                            self.disease_universe)
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
