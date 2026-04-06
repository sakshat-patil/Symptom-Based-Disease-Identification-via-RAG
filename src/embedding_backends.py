"""
Embedding-backend selector for the dense retriever.

Lets us swap between a general-purpose sentence-transformer and a
biomedical model (PubMedBERT / BioSentBERT) without touching retrieval.py.
This was motivated by the Check-in 3 finding that all-MiniLM-L6-v2 cannot
bridge the vocabulary gap between snake_case Kaggle symptoms (muscle_pain)
and MedQuAD clinical prose (myalgia).

Supported backends:
    - minilm       : sentence-transformers/all-MiniLM-L6-v2          (default)
    - pubmedbert   : microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
    - biosentbert  : pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb

Usage:
    from src.embedding_backends import get_encoder
    encoder = get_encoder("pubmedbert")
    vec = encoder.encode(["fever cough"])
"""

from __future__ import annotations

from typing import Any

_BACKENDS: dict[str, str] = {
    "minilm":       "sentence-transformers/all-MiniLM-L6-v2",
    "pubmedbert":   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "biosentbert":  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
}

_CACHE: dict[str, Any] = {}


def list_backends() -> list[str]:
    """Return the list of supported backend keys."""
    return list(_BACKENDS.keys())


def resolve_model_name(backend: str) -> str:
    """Return the HuggingFace model id for a backend key."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[backend]


def get_encoder(backend: str = "minilm"):
    """Return a SentenceTransformer-compatible encoder for the given backend.

    Caches instances so repeated calls with the same backend do not reload
    model weights.

    Parameters
    ----------
    backend:
        One of ``minilm``, ``pubmedbert``, or ``biosentbert``.
    """
    model_name = resolve_model_name(backend)
    if model_name in _CACHE:
        return _CACHE[model_name]

    from sentence_transformers import SentenceTransformer

    print(f"[embedding_backends] Loading {backend} ({model_name}) …")
    model = SentenceTransformer(model_name)
    _CACHE[model_name] = model
    return model


def embed_passages(passages: list[str], backend: str = "minilm") -> "np.ndarray":
    """Convenience wrapper: embed a list of passages with the chosen backend."""
    import numpy as np

    encoder = get_encoder(backend)
    vecs = encoder.encode(passages, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")
