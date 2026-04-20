"""Pinecone Inference reranker (Cohere Rerank 3.5 by default).

When the project runs in Pinecone mode, we can use Pinecone's hosted
reranker as an alternative to the local sentence-transformers
cross-encoder. Same interface as `cross_encoder_rerank.CrossEncoderReranker`
so the FastAPI service can swap between them at runtime based on which
vector store is active.

Why this is interesting for the demo:
    - Cohere Rerank 3.5 is a stronger reranker than the local MS-MARCO
      cross-encoder we ship as the default.
    - It runs server-side, so per-query latency is dominated by the
      network round-trip to Pinecone (~100 ms) rather than local CPU.
    - It's a managed service: no model to download, no version drift.

Owner: Vineet (added as a paired option to the local cross-encoder so
we have apples-to-apples comparison numbers between FAISS+local-rerank
and Pinecone+Cohere-rerank in the report).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass
class PineconeReranker:
    """Calls the Pinecone Inference rerank API.

    Reads PINECONE_API_KEY from env (same key as the vector store) and
    PINECONE_RERANK_MODEL. Default is 'bge-reranker-v2-m3' which is free
    on every Pinecone project. 'cohere-rerank-3.5' is stronger but requires
    a Cohere-enabled Pinecone project; if you set PINECONE_RERANK_MODEL
    explicitly we'll use it. Falls back automatically on 403 to the free
    Pinecone-hosted reranker.

    Compatible with the same `rerank()` signature as the local
    CrossEncoderReranker so the FastAPI service can swap them at runtime.
    """
    name: str = "pinecone-rerank"
    model: str = "bge-reranker-v2-m3"
    _client: object = None

    @classmethod
    def load(cls, model: str | None = None) -> "PineconeReranker":
        from pinecone import Pinecone
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        m = model or os.environ.get("PINECONE_RERANK_MODEL",
                                       "bge-reranker-v2-m3")
        return cls(model=m, _client=Pinecone(api_key=api_key))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        try:
            resp = self._client.inference.rerank(
                model=self.model,
                query=query,
                documents=list(passages),
                top_n=len(passages),
                return_documents=False,
            )
        except Exception as exc:
            # Auto-fallback: if the configured model isn't authorised on
            # this Pinecone project, retry with the free default.
            from pinecone.exceptions.exceptions import ForbiddenException
            if (isinstance(exc, ForbiddenException)
                    and self.model != "bge-reranker-v2-m3"):
                self.model = "bge-reranker-v2-m3"
                resp = self._client.inference.rerank(
                    model=self.model,
                    query=query,
                    documents=list(passages),
                    top_n=len(passages),
                    return_documents=False,
                )
            else:
                raise

        # Response carries (index, score) per document; we want scores
        # in the original order so the caller can attach them.
        out = [0.0] * len(passages)
        for r in resp.data:
            out[r.index] = float(r.score)
        return out

    def rerank(self, query: str, items: list[dict], text_key: str = "text",
                top_k: int | None = None) -> list[dict]:
        """Match the local cross-encoder API: mutates items with `ce_score`
        and returns them sorted high-to-low.
        """
        texts = [it[text_key] for it in items]
        scores = self.score(query, texts)
        for it, s in zip(items, scores):
            it["ce_score"] = s
        out = sorted(items, key=lambda it: it["ce_score"], reverse=True)
        return out[:top_k] if top_k else out
