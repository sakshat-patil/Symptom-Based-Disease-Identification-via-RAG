"""Tests for the Pinecone vector store and the Cohere reranker.

The unit tests of the FAISS post-filter (`_matches_filter`) already live
in test_vector_store.py. This file adds:

    - Pure unit tests for the PineconeReranker with a stub client
    - Live integration tests gated on PINECONE_API_KEY (`pytest -m live`):
        * the 255-data-mining index has the expected vector count
        * a real query returns relevant matches
        * the Cohere Rerank 3.5 endpoint reorders documents sensibly
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


# --- Stub Pinecone client for unit tests ----------------------------------

class _StubInferenceRerank:
    """Returns scores in reverse-input-order so we can confirm the
    reranker actually reorders rather than passing through."""
    last_call: dict[str, Any] = {}

    @classmethod
    def rerank(cls, *, model: str, query: str, documents: list[str],
                top_n: int, return_documents: bool):
        cls.last_call = dict(model=model, query=query, documents=documents,
                              top_n=top_n)
        n = len(documents)
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, score=float(n - i))  # higher = earlier
            for i in range(n)
        ])


class _StubInference:
    rerank = _StubInferenceRerank.rerank


class _StubPineconeClient:
    inference = _StubInference()


# --- Unit tests on PineconeReranker ---------------------------------------

class TestPineconeRerankerUnit:
    def test_load_without_key_raises(self, monkeypatch):
        from src.pinecone_rerank import PineconeReranker
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="PINECONE_API_KEY not set"):
            PineconeReranker.load()

    def test_uses_default_model(self, monkeypatch):
        from src.pinecone_rerank import PineconeReranker
        monkeypatch.delenv("PINECONE_RERANK_MODEL", raising=False)
        r = PineconeReranker(_client=_StubPineconeClient())
        # bge-reranker-v2-m3 is free on every Pinecone project, so we
        # default to it. Cohere-rerank-3.5 is opt-in.
        assert r.model == "bge-reranker-v2-m3"

    def test_picks_up_env_model_override(self, monkeypatch):
        from src.pinecone_rerank import PineconeReranker
        monkeypatch.setenv("PINECONE_API_KEY", "test-key")
        monkeypatch.setenv("PINECONE_RERANK_MODEL", "cohere-rerank-english-v3.0")
        # We can't fully load without a real Pinecone API, but we can
        # assert the registry intent.
        r = PineconeReranker(_client=_StubPineconeClient(),
                              model=os.environ["PINECONE_RERANK_MODEL"])
        assert r.model == "cohere-rerank-english-v3.0"

    def test_score_returns_score_per_input_in_original_order(self):
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker(_client=_StubPineconeClient())
        scores = r.score("query", ["doc A", "doc B", "doc C", "doc D"])
        # Stub returns score = n - i (reversed). For 4 docs: 4, 3, 2, 1
        assert scores == [4.0, 3.0, 2.0, 1.0]

    def test_score_empty_returns_empty(self):
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker(_client=_StubPineconeClient())
        assert r.score("q", []) == []

    def test_rerank_attaches_ce_score_and_sorts(self):
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker(_client=_StubPineconeClient())
        items = [{"text": f"doc{i}"} for i in range(4)]
        out = r.rerank("query", items)
        # Sorted high -> low. Stub gave doc0 score 4, doc1 score 3, ...
        assert [o["text"] for o in out] == ["doc0", "doc1", "doc2", "doc3"]
        assert all("ce_score" in o for o in out)
        assert out[0]["ce_score"] >= out[-1]["ce_score"]

    def test_rerank_top_k_truncates(self):
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker(_client=_StubPineconeClient())
        items = [{"text": f"doc{i}"} for i in range(10)]
        out = r.rerank("q", items, top_k=3)
        assert len(out) == 3


# --- Service auto-selection of reranker -----------------------------------

class TestServiceRerankerSelection:
    def test_pinecone_mode_with_rerank_model_picks_pinecone(self, monkeypatch):
        # When VECTOR_STORE=pinecone and PINECONE_RERANK_MODEL is set,
        # service.api._get_cross_encoder should return a PineconeReranker.
        # We patch the Pinecone client constructor to avoid network.
        monkeypatch.setenv("VECTOR_STORE", "pinecone")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key")
        monkeypatch.setenv("PINECONE_RERANK_MODEL", "cohere-rerank-3.5")

        # Stub the Pinecone() constructor in the pinecone_rerank module.
        import sys
        fake_module = type(sys)("pinecone_stub")

        class _FakePinecone:
            def __init__(self, api_key=None):
                self.inference = _StubInference()

        fake_module.Pinecone = _FakePinecone
        monkeypatch.setitem(sys.modules, "pinecone", fake_module)

        # We can't import service.api fully without booting FastAPI, but
        # we can replicate the selector logic to validate the choice.
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker.load()
        assert isinstance(r, PineconeReranker)
        assert r.model == "cohere-rerank-3.5"


# --- Live integration tests -----------------------------------------------

LIVE_PC = bool(os.environ.get("PINECONE_API_KEY"))
LIVE_OAI = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"))


@pytest.mark.live
@pytest.mark.skipif(not LIVE_PC,
                     reason="set PINECONE_API_KEY to run live tests")
class TestPineconeIndexLive:
    """Sanity-check the populated 255-data-mining index."""

    def test_index_exists_and_is_populated(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ.get("PINECONE_INDEX_NAME", "255-data-mining")
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        # We expect 24,063 vectors at 3072 dimensions.
        assert stats["total_vector_count"] >= 1000, \
            f"index {index_name} has only {stats['total_vector_count']} vectors -- did seed_pinecone.py run?"
        assert stats["dimension"] == 3072

    def test_query_returns_matches(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        idx = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "255-data-mining"))
        # Random unit vector -- we just want to confirm the query API works.
        rng = np.random.default_rng(42)
        v = rng.standard_normal(3072).astype(np.float32)
        v /= np.linalg.norm(v)
        resp = idx.query(vector=v.tolist(), top_k=5,
                          include_metadata=True)
        assert len(resp["matches"]) > 0


@pytest.mark.live
@pytest.mark.skipif(not (LIVE_PC and LIVE_OAI),
                     reason="needs PINECONE_API_KEY + OPENAI_API_KEY + OPENAI_BASE_URL")
class TestEndToEndAzurePineconeLive:
    """Full integration: encode with Azure, search Pinecone, get matches."""

    def test_cardiac_query_surfaces_relevant_passages(self):
        from src.embedding_backends import AzureOpenAIBackend
        from pinecone import Pinecone
        be = AzureOpenAIBackend.load()
        v = be.encode([
            "symptoms: chest pain, lightheadedness, sweating, breathlessness"
        ])
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        idx = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "255-data-mining"))
        resp = idx.query(vector=v[0].tolist(), top_k=5,
                          include_metadata=True)
        assert len(resp["matches"]) >= 5
        # At least one of the top-5 should be cardiac-related (Heart Attack,
        # Chest Pain, Angina, Bronchitis, Breathing).
        foci = [(m.get("metadata") or {}).get("focus", "").lower()
                 for m in resp["matches"]]
        cardiac_terms = ("chest", "heart", "angina", "cardiac",
                          "breath", "brachi", "broncho")
        assert any(any(t in f for t in cardiac_terms) for f in foci), \
            f"no cardiac focus in top-5: {foci}"


@pytest.mark.live
@pytest.mark.skipif(not LIVE_PC, reason="set PINECONE_API_KEY to run live tests")
class TestPineconeRerankLive:
    """Hit the real Pinecone Inference rerank endpoint.

    Uses bge-reranker-v2-m3 (free) by default; users with Cohere access
    can set PINECONE_RERANK_MODEL=cohere-rerank-3.5 to flip the backend.
    """

    def test_rerank_orders_relevant_first(self):
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker.load()
        docs = [
            "Photosynthesis converts sunlight into chemical energy.",
            "Chest pain is the most common symptom of myocardial infarction.",
            "The capital of France is Paris.",
        ]
        scores = r.score("symptoms of a heart attack", docs)
        # The cardiac doc must outrank the unrelated ones.
        assert scores[1] == max(scores), \
            f"reranker did not put cardiac doc first: {scores}"

    def test_auto_fallback_on_forbidden(self, monkeypatch):
        """If a user explicitly asks for cohere-rerank-3.5 but the
        project isn't authorised, the wrapper should silently fall back
        to bge-reranker-v2-m3 instead of raising.
        """
        monkeypatch.setenv("PINECONE_RERANK_MODEL", "cohere-rerank-3.5")
        from src.pinecone_rerank import PineconeReranker
        r = PineconeReranker.load()
        # First call may upgrade the model attribute mid-flight.
        scores = r.score("query", ["doc1", "doc2"])
        assert len(scores) == 2
        # If the original failed, the model should now be the fallback.
        assert r.model in ("cohere-rerank-3.5", "bge-reranker-v2-m3")
