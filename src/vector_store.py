"""Pluggable vector store: FAISS (default, offline) or Pinecone (managed).

Both backends implement the same `query(vec, top_k, filter)` interface so
the retrieval module doesn't need to know which one is in use. The
selection is made via the `VECTOR_STORE` environment variable:

    VECTOR_STORE=faiss        (default; uses local FAISS index)
    VECTOR_STORE=pinecone     (uses Pinecone with metadata filtering)

Why both? FAISS is faster at our scale and runs offline. Pinecone gives
us first-class metadata filtering -- a clinician can ask "only show me
NIH NHLBI passages of type=symptoms" and we push that filter down to
the index instead of post-filtering in Python. The Next.js UI surfaces
this as a 'filter by source / passage type' control that's only enabled
when the Pinecone backend is active.

Owner: Vineet (FAISS infra), Sakshat (Pinecone integration), Aishwarya
(metadata schema design).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


@dataclass
class VectorMatch:
    passage_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Interface every backend implements."""
    name: str = "abstract"

    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int = 30,
              filter: dict[str, Any] | None = None) -> list[VectorMatch]:
        ...

    @abstractmethod
    def supports_filter(self) -> bool:
        ...


# --- FAISS backend ---------------------------------------------------------

class FAISSStore(VectorStore):
    """Local FAISS IndexFlatIP. Free, offline, the fast default."""
    name = "faiss"

    def __init__(self, backend_name: str, passages: list[dict]):
        import faiss
        self.passages = passages
        self.index = faiss.read_index(str(PROCESSED / f"faiss_{backend_name}.index"))

    def query(self, vector: np.ndarray, top_k: int = 30,
              filter: dict[str, Any] | None = None) -> list[VectorMatch]:
        scores, ids = self.index.search(vector, top_k)
        out: list[VectorMatch] = []
        for s, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i < 0:
                continue
            p = self.passages[i]
            md = {"source": p.get("source", ""),
                   "focus": p.get("focus", ""),
                   "question": p.get("question", ""),
                   "text": p.get("text", "")}
            # Post-filter in Python (FAISS has no native filter).
            if filter and not _matches_filter(md, filter):
                continue
            out.append(VectorMatch(passage_id=p["id"], score=float(s),
                                     metadata=md))
        return out

    def supports_filter(self) -> bool:
        # We expose true here because we DO honour filters (just slower).
        return True


def _matches_filter(md: dict, flt: dict) -> bool:
    for k, v in (flt or {}).items():
        cur = md.get(k)
        if isinstance(v, dict):
            # Mongo-ish ops: {"$in": [...]}
            if "$in" in v and cur not in v["$in"]:
                return False
            if "$eq" in v and cur != v["$eq"]:
                return False
            if "$ne" in v and cur == v["$ne"]:
                return False
        else:
            if cur != v:
                return False
    return True


# --- Pinecone backend ------------------------------------------------------

class PineconeStore(VectorStore):
    """Pinecone Serverless backend. Pushes metadata filters to the index.

    Env vars required:
        PINECONE_API_KEY      - account key
        PINECONE_INDEX_NAME   - default 'medical-rag-medquad'
        PINECONE_CLOUD        - default 'aws'
        PINECONE_REGION       - default 'us-east-1'
    """
    name = "pinecone"

    def __init__(self, *, dim: int, index_name: str | None = None):
        """Initialize against a specific index. Defaults to the env var
        PINECONE_INDEX_NAME so the existing API stays backward compatible;
        callers can pass `index_name` explicitly to switch at runtime
        (used by /diagnose's per-request `pinecone_index` override).
        """
        from pinecone import Pinecone, ServerlessSpec
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name or os.environ.get(
            "PINECONE_INDEX_NAME", "medical-rag-medquad")
        self.cloud = os.environ.get("PINECONE_CLOUD", "aws")
        self.region = os.environ.get("PINECONE_REGION", "us-east-1")
        # Create index if missing (idempotent).
        existing = {idx["name"] for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self.index = self.pc.Index(self.index_name)

    def upsert(self, passages: list[dict], embeddings: np.ndarray,
                batch_size: int = 100) -> None:
        """One-shot population. Run scripts/seed_pinecone.py to invoke."""
        from tqdm import tqdm
        items = []
        for p, vec in zip(passages, embeddings):
            items.append({
                "id": p["id"],
                "values": vec.astype(np.float32).tolist(),
                "metadata": {
                    "source": p.get("source", ""),
                    "focus": p.get("focus", "")[:200],
                    "question": p.get("question", "")[:300],
                    "text": p.get("text", "")[:1500],   # Pinecone metadata cap is 40 KB
                },
            })
        for i in tqdm(range(0, len(items), batch_size), desc="upsert"):
            self.index.upsert(vectors=items[i:i+batch_size])

    def query(self, vector: np.ndarray, top_k: int = 30,
              filter: dict[str, Any] | None = None) -> list[VectorMatch]:
        # Pinecone wants a flat python list of floats.
        vec = vector[0].astype(np.float32).tolist() if vector.ndim == 2 else vector.astype(np.float32).tolist()
        resp = self.index.query(vector=vec, top_k=top_k,
                                  filter=filter or None,
                                  include_metadata=True)
        out: list[VectorMatch] = []
        for match in resp["matches"]:
            md = dict(match.get("metadata") or {})
            out.append(VectorMatch(passage_id=str(match["id"]),
                                     score=float(match["score"]),
                                     metadata=md))
        return out

    def supports_filter(self) -> bool:
        return True


# --- factory ---------------------------------------------------------------

def get_vector_store(backend_name: str, passages: list[dict] | None = None,
                       *, dim: int | None = None,
                       index_name: str | None = None) -> VectorStore:
    """Pick a vector store based on the VECTOR_STORE env var.

    `index_name` only applies when kind=='pinecone' and lets callers
    override the env-default index per call (used by per-request runtime
    switching in /diagnose).
    """
    kind = os.environ.get("VECTOR_STORE", "faiss").lower()
    if kind == "faiss":
        if passages is None:
            raise ValueError("FAISSStore needs the passages list")
        return FAISSStore(backend_name, passages)
    if kind == "pinecone":
        if dim is None:
            raise ValueError("PineconeStore needs `dim` (vector dimension)")
        return PineconeStore(dim=dim, index_name=index_name)
    raise ValueError(f"unknown VECTOR_STORE: {kind}")
