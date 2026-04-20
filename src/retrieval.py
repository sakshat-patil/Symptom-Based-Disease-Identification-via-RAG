"""Dense retrieval over MedQuAD with FAISS IndexFlatIP.

We do an inner-product index on L2-normalised vectors (= cosine sim). The
heavy work is encoding the 24k passages once and caching the matrix and
the FAISS index to disk; everything after that is millisecond-scale.

Query path:
    1. Build a natural-language version of the symptom list.
    2. Optionally append clinical synonyms (synonym_expansion.expand_query_string).
    3. Encode and ask FAISS for the top-K passages.
    4. Map each passage to one of our 41 Kaggle diseases via the curated
       keyword index in disease_keywords.py. We match on Focus first
       (highest precision), fall back to Question. We *don't* fall back
       to the full answer text -- learned the hard way that "What is
       Chest Pain?" articles get attributed to every related disease.
    5. Aggregate to per-disease scores by max() of passage similarities.

The first run takes about 80 seconds for MiniLM and ~5 minutes for
PubMedBERT on Vineet's M3 Pro; reruns load the index from disk in <1s.

Owner: Vineet. The focus-only attribution policy was Sakshat's catch
during a debugging session -- before that, retrieval scores were noisy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from .disease_keywords import diseases_matching
from .embedding_backends import Backend, get_backend
from .synonym_expansion import expand_query_string
from .vector_store import VectorMatch, VectorStore, get_vector_store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PASSAGES = ROOT / "data" / "processed" / "passages.jsonl"
PROCESSED = ROOT / "data" / "processed"


def _emb_path(backend_name: str) -> Path:
    return PROCESSED / f"embeddings_{backend_name}.npy"


def _index_path(backend_name: str) -> Path:
    return PROCESSED / f"faiss_{backend_name}.index"


def load_passages(path: Path = DEFAULT_PASSAGES) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_index(backend: Backend, passages: list[dict], batch_size: int = 64) -> tuple[np.ndarray, faiss.Index]:
    texts = [p["text"] for p in passages]
    emb = backend.encode(texts, batch_size=batch_size, show_progress=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return emb, index


def save_index(backend_name: str, emb: np.ndarray, index: faiss.Index) -> None:
    np.save(_emb_path(backend_name), emb)
    faiss.write_index(index, str(_index_path(backend_name)))


def load_or_build(backend: Backend, passages: list[dict], rebuild: bool = False) -> faiss.Index:
    idx_p = _index_path(backend.name)
    if idx_p.exists() and not rebuild:
        return faiss.read_index(str(idx_p))
    emb, index = build_index(backend, passages)
    save_index(backend.name, emb, index)
    return index


def _normalise_focus(focus: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", focus.lower()).strip("_")
    return s


def _candidate_diseases_from_passage(passage: dict, all_disease_set: set[str]) -> list[str]:
    """Match a MedQuAD passage to our disease universe via curated keywords.

    Returns the strongest match: focus matches are preferred (highest
    precision), then question matches; text matches are NOT used as a
    fallback because generic articles (e.g. "What is Chest Pain?") mention
    every related disease and would inflate scores for unrelated candidates.
    """
    cands = diseases_matching(passage.get("focus", ""), all_disease_set)
    if cands:
        return cands
    return diseases_matching(passage.get("question", ""), all_disease_set)


@dataclass
class DenseRetriever:
    backend: Backend
    passages: list[dict]
    store: VectorStore
    disease_universe: set[str]

    @classmethod
    def from_files(cls, backend_name: str, disease_universe: Iterable[str],
                   passages_path: Path = DEFAULT_PASSAGES, rebuild: bool = False,
                   build_local_faiss_if_missing: bool = True,
                   index_name: str | None = None) -> "DenseRetriever":
        """Construct a retriever using whichever VectorStore is selected.

        `index_name` overrides PINECONE_INDEX_NAME when running Pinecone;
        used by the API service for per-request index switching (e.g.
        switch between 255-data-mining for Azure-3072d and
        medical-rag-medquad for PubMedBERT-768d).

        For FAISS we still build the local index on first use; for Pinecone
        we assume `scripts/seed_pinecone.py` was already run.
        """
        backend = get_backend(backend_name)
        passages = load_passages(passages_path)
        store_kind = os.environ.get("VECTOR_STORE", "faiss").lower()
        is_remote_only_backend = backend_name in ("azure-openai", "openai", "azure")
        if store_kind == "pinecone" or is_remote_only_backend:
            store = (get_vector_store(backend_name, dim=backend.dim,
                                        index_name=index_name)
                     if store_kind == "pinecone" else None)
            if store is None:
                raise RuntimeError(
                    "Azure OpenAI embedding backend requires "
                    "VECTOR_STORE=pinecone (no local FAISS index for "
                    "OpenAI vectors)")
        else:
            if build_local_faiss_if_missing:
                load_or_build(backend, passages, rebuild=rebuild)
            store = get_vector_store(backend_name, passages=passages)
        return cls(backend=backend, passages=passages, store=store,
                   disease_universe=set(disease_universe))

    def _passage_by_id(self, pid: str) -> dict | None:
        # Pinecone returns metadata directly; FAISS returns indices into our
        # passage list. Provide a generic lookup by id.
        for p in self.passages:
            if p["id"] == pid:
                return p
        return None

    def retrieve(self, symptoms: list[str], top_k: int = 30,
                 expand_synonyms: bool = False,
                 metadata_filter: dict | None = None
                 ) -> tuple[dict[str, float], dict[str, list[dict]]]:
        """Retrieve and bucket results by Kaggle disease.

        metadata_filter is a Pinecone-style filter (e.g., {"source": "9_CDC_QA"})
        that gets pushed down to the vector store. FAISS honours it via a
        Python post-filter; Pinecone does it server-side.
        """
        query = "symptoms: " + ", ".join(s.replace("_", " ") for s in symptoms)
        if expand_synonyms:
            query = expand_query_string(query, symptoms)
        q = self.backend.encode([query])
        matches = self.store.query(q, top_k=top_k, filter=metadata_filter)

        per_disease_score: dict[str, float] = {}
        per_disease_passages: dict[str, list[dict]] = defaultdict(list)
        for m in matches:
            md = m.metadata or {}
            # Reconstruct a passage-shaped dict from whatever the backend gave us.
            p = {
                "id": m.passage_id,
                "source": md.get("source", ""),
                "focus": md.get("focus", ""),
                "question": md.get("question", ""),
                "text": md.get("text", ""),
            }
            cands = _candidate_diseases_from_passage(p, self.disease_universe)
            if not cands:
                continue
            for d in cands:
                if m.score > per_disease_score.get(d, -1.0):
                    per_disease_score[d] = float(m.score)
                per_disease_passages[d].append({
                    **p, "score": float(m.score),
                })
        return per_disease_score, dict(per_disease_passages)

    def retrieve_passages(self, symptoms: list[str], top_k: int = 10,
                           expand_synonyms: bool = False,
                           metadata_filter: dict | None = None) -> list[dict]:
        """Return the top-K passages by semantic similarity, no disease filter."""
        query = "symptoms: " + ", ".join(s.replace("_", " ") for s in symptoms)
        if expand_synonyms:
            query = expand_query_string(query, symptoms)
        q = self.backend.encode([query])
        matches = self.store.query(q, top_k=top_k, filter=metadata_filter)
        out: list[dict] = []
        for m in matches:
            md = m.metadata or {}
            out.append({
                "id": m.passage_id,
                "source": md.get("source", ""),
                "focus": md.get("focus", ""),
                "question": md.get("question", ""),
                "text": md.get("text", ""),
                "score": float(m.score),
            })
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="minilm", choices=["minilm", "pubmedbert"])
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    import pandas as pd
    diseases = set(pd.read_csv(PROCESSED / "transactions.csv")["condition"].unique())
    backend = get_backend(args.backend)
    passages = load_passages()
    print(f"[retrieval] passages: {len(passages)}; backend: {backend.name} (dim={backend.dim})")
    emb, index = build_index(backend, passages)
    save_index(backend.name, emb, index)
    print(f"[retrieval] saved index to {_index_path(backend.name)}")


if __name__ == "__main__":
    main()
