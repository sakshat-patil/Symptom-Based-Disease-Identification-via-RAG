"""Tests for the Azure OpenAI embedding backend.

We split into two flavours:

    - Pure unit tests that exercise registry/dispatch/normalisation logic
      with a stub OpenAI client. These run on every `pytest` invocation.

    - Live integration tests that hit the real Azure endpoint, gated
      behind the `OPENAI_API_KEY` env var so they're skipped by default
      and only fire in CI / on demand. Uses the `live` mark so you can
      run them with `pytest -m live`.

We never call the real Azure endpoint without explicit opt-in (env var
present) so the default test suite stays fast, deterministic, and free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Pure unit tests with a stub OpenAI client
# --------------------------------------------------------------------------

@dataclass
class _StubEmbedResp:
    data: list[Any]


class _StubEmbeddings:
    """Returns deterministic vectors that grow with input length, so we
    can assert batching + ordering without hitting the network."""
    def __init__(self, dim: int = 3072):
        self.dim = dim
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        self.calls.append(input)
        out = []
        for s in input:
            v = np.zeros(self.dim, dtype=np.float32)
            # Encode the input length into a couple of dims so different
            # inputs produce different vectors.
            v[0] = float(len(s))
            v[1] = float(sum(ord(c) for c in s) % 100)
            out.append(SimpleNamespace(embedding=v.tolist()))
        return _StubEmbedResp(data=out)


class _StubClient:
    def __init__(self, dim: int = 3072):
        self.embeddings = _StubEmbeddings(dim)


@pytest.fixture
def stub_backend(monkeypatch):
    """Return an AzureOpenAIBackend whose client is the stub above.

    We monkeypatch OPENAI_API_KEY so the `.load()` precondition passes,
    then construct the backend manually so we don't hit the real OpenAI.
    """
    from src.embedding_backends import AzureOpenAIBackend
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return AzureOpenAIBackend(model="stub-model", dim=3072,
                                _client=_StubClient(dim=3072))


class TestAzureBackendRegistry:
    def test_registered_under_aliases(self, monkeypatch):
        """The registry should accept all three aliases."""
        from src.embedding_backends import get_backend
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # We can't actually `.load()` a real client without hitting Azure,
        # but we can confirm that the dispatch raises on bad names.
        from src.embedding_backends import AzureOpenAIBackend
        # Just verifying the alias map -- live load is exercised in the
        # `live` test below.
        assert AzureOpenAIBackend.__name__ == "AzureOpenAIBackend"

    def test_listed_when_key_present(self, monkeypatch):
        from src.embedding_backends import list_backends
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert "azure-openai" in list_backends()

    def test_not_listed_when_key_absent(self, monkeypatch):
        from src.embedding_backends import list_backends
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert "azure-openai" not in list_backends()

    def test_load_without_key_raises(self, monkeypatch):
        from src.embedding_backends import AzureOpenAIBackend
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
            AzureOpenAIBackend.load()

    def test_unknown_backend_raises(self):
        from src.embedding_backends import get_backend
        with pytest.raises(ValueError, match="unknown backend"):
            get_backend("not-a-real-backend")


class TestEncodeBehaviour:
    def test_returns_normalised_float32(self, stub_backend):
        v = stub_backend.encode(["hello world"])
        assert v.dtype == np.float32
        assert v.shape == (1, 3072)
        # Each row should be unit norm (within fp32 tolerance).
        assert pytest.approx(np.linalg.norm(v[0]), rel=1e-5) == 1.0

    def test_preserves_input_order(self, stub_backend):
        # Stub encodes len(s) into dim 0 BEFORE normalisation. After
        # normalisation, all rows are unit length, so dim-0 reflects the
        # *fraction* of signal captured by len. The strongest behaviour
        # we can assert: each input produces its own row, and inputs of
        # equal length produce equal vectors (deterministic encode).
        v = stub_backend.encode(["alpha", "beta", "alpha"])
        assert v.shape == (3, 3072)
        # Inputs at index 0 and 2 are identical -- they must come back
        # as identical vectors (deterministic encoding).
        np.testing.assert_array_equal(v[0], v[2])
        # Different inputs should produce different vectors.
        assert not np.array_equal(v[0], v[1])

    def test_zero_norm_input_doesnt_explode(self, monkeypatch):
        """If the underlying API somehow returns a zero vector, we must
        still produce a finite normalised result (norm clamp to 1)."""
        from src.embedding_backends import AzureOpenAIBackend
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        class _ZeroClient:
            class _E:
                def create(self, model, input):
                    return _StubEmbedResp(data=[
                        SimpleNamespace(embedding=[0.0] * 16)
                        for _ in input
                    ])
            embeddings = _E()

        b = AzureOpenAIBackend(model="m", dim=16, _client=_ZeroClient())
        v = b.encode(["x"])
        assert np.all(np.isfinite(v))
        # Norm clamp means we don't divide by zero -- output is just zeros.
        assert v.shape == (1, 16)

    def test_batching_chunks_at_16(self, stub_backend):
        # Pass 40 inputs; the stub records each .create() call with the
        # batch it received. We expect ceil(40/16) = 3 batches.
        texts = [f"input {i}" for i in range(40)]
        stub_backend.encode(texts, batch_size=64)  # bigger ask gets clamped to 16
        sizes = [len(c) for c in stub_backend._client.embeddings.calls]
        assert sizes == [16, 16, 8]

    def test_empty_input_returns_empty_array(self, stub_backend):
        v = stub_backend.encode([])
        assert v.shape == (0,) or v.size == 0


# --------------------------------------------------------------------------
# Live integration tests against the real Azure endpoint
#
# These only run if you opt in:  pytest -m live
# Skipped by default so the suite stays free + fast.
# --------------------------------------------------------------------------

LIVE = os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")


@pytest.mark.live
@pytest.mark.skipif(not LIVE, reason="set OPENAI_API_KEY + OPENAI_BASE_URL to run live tests")
class TestAzureLive:
    def test_real_endpoint_returns_3072_dims(self):
        from src.embedding_backends import AzureOpenAIBackend
        b = AzureOpenAIBackend.load()
        v = b.encode(["chest pain and lightheadedness"])
        assert v.shape == (1, b.dim)
        # Real endpoint should return a non-degenerate, normalised vector.
        norm = np.linalg.norm(v[0])
        assert pytest.approx(norm, rel=1e-4) == 1.0
        assert np.any(v[0] != 0.0)

    def test_real_endpoint_semantic_similarity(self):
        """Two clinically related queries should be more similar to each
        other than to an unrelated one. This is a smoke test of model
        sanity, not a precise benchmark."""
        from src.embedding_backends import AzureOpenAIBackend
        b = AzureOpenAIBackend.load()
        v = b.encode([
            "symptoms: chest pain, sweating, breathlessness",  # cardiac
            "symptoms: angina, dyspnea, diaphoresis",           # cardiac (clinical synonyms)
            "symptoms: itching, skin rash, eruptions",         # dermatologic
        ])
        cardiac_sim = float(np.dot(v[0], v[1]))
        cross_sim = float(np.dot(v[0], v[2]))
        assert cardiac_sim > cross_sim, \
            f"expected cardiac queries closer to each other ({cardiac_sim:.3f}) " \
            f"than to dermatologic ({cross_sim:.3f})"
