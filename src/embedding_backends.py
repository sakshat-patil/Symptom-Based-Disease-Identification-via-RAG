"""Two interchangeable text encoders behind a tiny common interface.

We ship two so the comparison is part of the report:
    - MiniLM (sentence-transformers/all-MiniLM-L6-v2, 384d): general
      pre-trained encoder, fast, small.
    - PubMedBERT (neuml/pubmedbert-base-embeddings, 768d): biomedical
      pre-training.

Each backend gives back a [n, dim] L2-normalised float32 matrix so the
downstream FAISS IndexFlatIP just does cosine similarity.

Owner: Vineet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_PUBMEDBERT = "neuml/pubmedbert-base-embeddings"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Backend:
    name: str
    dim: int
    _model: object

    def encode(self, texts: list[str], batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
        v = self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_progress,
        )
        return v.astype(np.float32)


def get_backend(name: str = "minilm") -> Backend:
    from sentence_transformers import SentenceTransformer
    name = name.lower()
    if name == "minilm":
        m = SentenceTransformer(_MINILM, device=_device())
        return Backend("minilm", m.get_sentence_embedding_dimension(), m)
    if name == "pubmedbert":
        m = SentenceTransformer(_PUBMEDBERT, device=_device())
        return Backend("pubmedbert", m.get_sentence_embedding_dimension(), m)
    raise ValueError(f"unknown backend: {name}")


def list_backends() -> Iterable[str]:
    return ("minilm", "pubmedbert")
