"""Tests for the OpenAI clinical explainer.

Like the Azure backend tests, we split into pure unit tests with a mocked
OpenAI client (always run) and live integration tests gated on
OPENAI_API_KEY (`pytest -m live`).

The unit tests exercise:
    - JSON schema parsing of a well-formed response
    - Citation footer is appended deterministically (not by the LLM)
    - Graceful fallback to TemplateClinicalExplainer when the API errors
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.clinical_explanation import (ClinicalExplanation,
                                          OpenAIClinicalExplainer,
                                          TemplateClinicalExplainer,
                                          get_clinical_explainer)
from src.evidence import EvidenceCard, EvidenceClaim


def _card(**kw) -> EvidenceCard:
    base = dict(
        passage_id="abc123",
        source_id="8_NHLBI_QA_XML",
        source_label="NIH NHLBI",
        source_tier=1,
        passage_type="symptoms",
        focus="Heart Attack",
        question="What are the symptoms?",
        full_text="Chest pain or discomfort. Shortness of breath. Sweating.",
        claims=[EvidenceClaim("Chest pain or discomfort.", 0, 26, ["chest pain"])],
        specificity=0.75, retrieval_score=0.78, ce_score=None,
    )
    base.update(kw)
    return EvidenceCard(**base)


# --- Stub OpenAI client ----------------------------------------------------

class _StubChatCompletions:
    """Reads a canned JSON response from a class attribute so tests can
    swap the response per scenario."""
    canned: str = ""
    raise_exc: Exception | None = None
    last_kwargs: dict[str, Any] = {}

    @classmethod
    def create(cls, **kwargs):
        cls.last_kwargs = kwargs
        if cls.raise_exc:
            raise cls.raise_exc
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=cls.canned)
            )]
        )


class _StubChat:
    completions = _StubChatCompletions


class _StubClient:
    chat = _StubChat


@pytest.fixture(autouse=True)
def reset_stub():
    _StubChatCompletions.canned = json.dumps({
        "symptom_disease_link": "Chest pain + diaphoresis are classic MI presenters.",
        "statistical_prior": "Confidence 1.00, lift 41 in our 4920-row table.",
        "evidence_quality": "NIH NHLBI tier-1 source; specificity 0.75.",
        "whats_missing": "ECG and troponin required for definitive diagnosis.",
    })
    _StubChatCompletions.raise_exc = None
    _StubChatCompletions.last_kwargs = {}


@pytest.fixture
def stub_explainer(monkeypatch):
    """Construct an OpenAIClinicalExplainer with the stub client.

    We don't go through .load() because that would talk to the real
    OpenAI library; instead we set the env and instantiate directly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAIClinicalExplainer(model="gpt-4o-mini", _client=_StubClient())


# --- Unit tests ------------------------------------------------------------

class TestOpenAIExplainerWellFormed:
    def test_parses_four_fields(self, stub_explainer):
        out = stub_explainer.explain(
            disease="heart_attack",
            query_symptoms=["chest_pain", "sweating"],
            matching_rules=[{"antecedent": ["chest_pain"], "confidence": 1.0,
                              "lift": 41.0, "size": 1}],
            evidence_cards=[_card()],
        )
        assert isinstance(out, ClinicalExplanation)
        assert "MI" in out.symptom_disease_link
        assert "lift 41" in out.statistical_prior
        assert "tier-1" in out.evidence_quality
        assert "ECG" in out.whats_missing
        assert out.backend == "openai:gpt-4o-mini"

    def test_citations_are_added_by_us_not_llm(self, stub_explainer):
        # Even if the LLM doesn't include a citation field, our code
        # appends one from the evidence card metadata.
        _StubChatCompletions.canned = json.dumps({
            "symptom_disease_link": "x", "statistical_prior": "x",
            "evidence_quality": "x", "whats_missing": "x",
        })
        out = stub_explainer.explain(
            disease="heart_attack", query_symptoms=["chest_pain"],
            matching_rules=[],
            evidence_cards=[_card(passage_id="P-X", source_label="NIH NHLBI")])
        assert any("NIH NHLBI" in c and "P-X" in c for c in out.citations)

    def test_request_uses_strict_json_format(self, stub_explainer):
        # We MUST send response_format=json_object so the model can't
        # return prose; verify our payload says so.
        stub_explainer.explain(
            disease="x", query_symptoms=["q"],
            matching_rules=[], evidence_cards=[])
        kw = _StubChatCompletions.last_kwargs
        assert kw["response_format"] == {"type": "json_object"}
        assert kw["model"] == "gpt-4o-mini"
        # Temperature should be low for clinical content.
        assert kw["temperature"] <= 0.3


class TestOpenAIExplainerFallback:
    def test_api_error_falls_back_to_template(self, stub_explainer):
        _StubChatCompletions.raise_exc = RuntimeError("Azure 503")
        out = stub_explainer.explain(
            disease="heart_attack", query_symptoms=["chest_pain"],
            matching_rules=[{"antecedent": ["chest_pain"], "confidence": 1.0,
                              "lift": 41.0, "size": 1}],
            evidence_cards=[_card()])
        # Fallback marker should be visible.
        assert "openai-failed" in out.backend
        assert "template" in out.backend
        # Content should be filled in by the template explainer.
        assert out.symptom_disease_link
        assert out.whats_missing

    def test_malformed_json_falls_back(self, stub_explainer):
        _StubChatCompletions.canned = "not actually json"
        out = stub_explainer.explain(
            disease="x", query_symptoms=["q"],
            matching_rules=[],
            evidence_cards=[_card()])
        assert "openai-failed" in out.backend


class TestPromptConstruction:
    def test_includes_disease_symptom_and_evidence(self, stub_explainer):
        stub_explainer.explain(
            disease="heart_attack",
            query_symptoms=["chest_pain", "sweating"],
            matching_rules=[{"antecedent": ["chest_pain"], "confidence": 1.0,
                              "lift": 41.0, "size": 1}],
            evidence_cards=[_card()])
        kw = _StubChatCompletions.last_kwargs
        user_msg = kw["messages"][1]["content"]
        assert "heart attack" in user_msg.lower()
        assert "chest pain" in user_msg.lower()
        assert "NIH NHLBI" in user_msg or "tier 1" in user_msg.lower()

    def test_no_rules_emits_placeholder(self, stub_explainer):
        stub_explainer.explain(disease="x", query_symptoms=["q"],
                                  matching_rules=[], evidence_cards=[])
        msg = _StubChatCompletions.last_kwargs["messages"][1]["content"]
        assert "no matching rule" in msg.lower()


class TestFactoryAutoSelect:
    def test_auto_uses_openai_when_key_set(self, monkeypatch):
        # Bust the lru_cache on the factory and set the env.
        get_clinical_explainer.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Patching the OpenAI() constructor so .load() doesn't actually
        # instantiate the real client.
        from src import clinical_explanation as ce
        class _FakeOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key
            chat = _StubChat
        # The real load() does `from openai import OpenAI`, so we stub at
        # import time via sys.modules.
        import sys
        fake_module = type(sys)("openai_stub")
        fake_module.OpenAI = _FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_module)

        ex = get_clinical_explainer("auto")
        assert ex.name == "openai"

    def test_auto_falls_back_to_template_without_key(self, monkeypatch):
        get_clinical_explainer.cache_clear()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ex = get_clinical_explainer("auto")
        assert ex.name == "template"


# --- Live integration ------------------------------------------------------

LIVE = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"))


@pytest.mark.live
@pytest.mark.skipif(
    not (LIVE and os.environ.get("OPENAI_CHAT_MODEL")),
    reason=("set OPENAI_API_KEY + OPENAI_BASE_URL + OPENAI_CHAT_MODEL "
             "(must be a deployed Azure chat model name)"))
class TestOpenAIExplainerLive:
    def test_real_endpoint_returns_four_fields(self):
        # Real Azure OpenAI must produce a parseable JSON object.
        # Note: this test is skipped unless OPENAI_CHAT_MODEL is set
        # because the embedding-only Azure deployment doesn't support
        # chat completions. The Vineet/Aishwarya demo path uses the
        # template explainer when a chat deployment isn't available.
        from src.clinical_explanation import OpenAIClinicalExplainer
        ex = OpenAIClinicalExplainer.load(
            model=os.environ["OPENAI_CHAT_MODEL"])
        out = ex.explain(
            disease="heart_attack",
            query_symptoms=["chest_pain", "sweating", "lightheadedness"],
            matching_rules=[{"antecedent": ["chest_pain", "lightheadedness"],
                              "confidence": 1.0, "lift": 41.0, "size": 2}],
            evidence_cards=[_card()],
        )
        # All four fields populated.
        assert out.symptom_disease_link
        assert out.statistical_prior
        assert out.evidence_quality
        assert out.whats_missing
        assert out.backend.startswith("openai:")
