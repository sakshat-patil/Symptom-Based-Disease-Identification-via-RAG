"""Cross-encoder re-ranking on top of the FAISS bi-encoder retrieval.

Bi-encoders (MiniLM, PubMedBERT) score query and passage independently --
fast but lose joint context. A cross-encoder reads the (query, passage) pair
together and produces a single relevance score, which usually fixes a
fraction of the ranking errors at the top of the bi-encoder list.

We use the lightweight `cross-encoder/ms-marco-MiniLM-L-6-v2` checkpoint by
default; it scores a (query, passage) pair in a few milliseconds on CPU and
about a millisecond on Apple MPS. The cost per query is bounded because we
only re-score the top-K=15 passages we already retrieved.

Aishwarya wired this in after we noticed in the Check-in 3 ablation that
retrieval recall stalled at moderate K -- the cross-encoder gives us a
second pass to pull genuinely relevant passages forward without changing the
bi-encoder index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class CrossEncoderReranker:
    model_name: str
    _model: object  # sentence_transformers.CrossEncoder

    @classmethod
    def load(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        m = CrossEncoder(model_name, device=device)
        return cls(model_name=model_name, _model=m)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=16, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, items: list[dict], text_key: str = "text",
               top_k: int | None = None) -> list[dict]:
        """Return items sorted by cross-encoder score, with the score added.

        Items unchanged in shape; mutates by adding a `ce_score` field.
        """
        texts = [it[text_key] for it in items]
        scores = self.score(query, texts)
        for it, s in zip(items, scores):
            it["ce_score"] = s
        out = sorted(items, key=lambda it: it["ce_score"], reverse=True)
        return out[:top_k] if top_k else out
