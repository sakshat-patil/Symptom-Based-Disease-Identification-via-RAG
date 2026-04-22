"""Three interchangeable text encoders behind a tiny common interface.

We ship three because the comparison is part of the report and the demo:
    - MiniLM (sentence-transformers/all-MiniLM-L6-v2, 384d): general
      pre-trained encoder, fast, small. Local.
    - PubMedBERT (neuml/pubmedbert-base-embeddings, 768d): biomedical
      pre-training -- closes part of the vocabulary gap before synonym
      expansion kicks in. Local.
    - Azure OpenAI text-embedding-3-large (3072d): hits an
      OpenAI-compatible v1 endpoint (Azure deployment via
      OPENAI_BASE_URL). Strongest semantic quality of the three but
      requires a key and adds a network round-trip. Used in the demo
      against our 255-data-mining Pinecone index.

Each backend gives back a [n, dim] L2-normalised float32 matrix so the
downstream FAISS IndexFlatIP just does cosine similarity. The local
backends auto-pick MPS on Apple silicon and CUDA on NVIDIA; Azure runs
remotely so device selection doesn't matter.

Owner: Vineet. The three-backend design lets us demonstrate that domain
pre-training matters (PubMedBERT > MiniLM), AND that scale matters (OpenAI
3072d outperforms both on our retrieval ablation), AND that fusion + rules
beat all three retrieval-only baselines.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_PUBMEDBERT = "neuml/pubmedbert-base-embeddings"
_AZURE_DEFAULT_MODEL = "truestar-text-embedding-3-large"
_AZURE_DEFAULT_DIM = 3072


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
    _model: object  # SentenceTransformer

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        from sentence_transformers import SentenceTransformer  # noqa: F401  (type hint only)
        v = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return v.astype(np.float32)


@dataclass
class AzureOpenAIBackend:
    """Embedding backend that talks to an OpenAI-compatible v1 endpoint.

    Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment; falls
    back to the well-known Azure deployment name if OPENAI_EMBEDDING_MODEL
    isn't set. Output is L2-normalised float32 to match the local
    backends so the rest of the pipeline doesn't care about provenance.
    """
    name: str = "azure-openai"
    dim: int = _AZURE_DEFAULT_DIM
    model: str = _AZURE_DEFAULT_MODEL
    _client: object = None

    @classmethod
    def load(cls) -> "AzureOpenAIBackend":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", _AZURE_DEFAULT_MODEL)
        dim = int(os.environ.get("OPENAI_EMBEDDING_DIMENSION", _AZURE_DEFAULT_DIM))
        client = OpenAI()  # picks up OPENAI_API_KEY + OPENAI_BASE_URL from env
        return cls(model=model, dim=dim, _client=client)

    def encode(self, texts: list[str], batch_size: int = 64,
                show_progress: bool = False) -> np.ndarray:
        # Azure has a 16-input-per-call cap on the embeddings endpoint by
        # default; we batch defensively at 16 even when the caller asks
        # for more.
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        cap = min(batch_size, 16)
        out: list[list[float]] = []
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), cap), desc=f"{self.name}")
        else:
            iterator = range(0, len(texts), cap)
        for i in iterator:
            chunk = texts[i:i + cap]
            resp = self._client.embeddings.create(
                model=self.model, input=chunk
            )
            for d in resp.data:
                out.append(d.embedding)
        v = np.asarray(out, dtype=np.float32)
        # L2-normalise so inner-product == cosine similarity.
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (v / norms).astype(np.float32)


def get_backend(name: str = "minilm"):
    from sentence_transformers import SentenceTransformer
    name = name.lower()
    if name == "minilm":
        model = SentenceTransformer(_MINILM, device=_device())
        return Backend(name="minilm", dim=model.get_sentence_embedding_dimension(), _model=model)
    if name == "pubmedbert":
        model = SentenceTransformer(_PUBMEDBERT, device=_device())
        return Backend(name="pubmedbert", dim=model.get_sentence_embedding_dimension(), _model=model)
    if name in ("azure-openai", "openai", "azure"):
        return AzureOpenAIBackend.load()
    raise ValueError(f"unknown backend: {name}")


def list_backends() -> Iterable[str]:
    out = ["minilm", "pubmedbert"]
    if os.environ.get("OPENAI_API_KEY"):
        out.append("azure-openai")
    return tuple(out)
