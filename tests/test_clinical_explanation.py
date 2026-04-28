"""Tests for the clinical-structured explanation layer.

The OpenAI backend isn't unit-testable without a real API key (or a heavy
mock). We exercise the deterministic template explainer thoroughly and
check the factory dispatch. Citation format is also verified.
"""
from __future__ import annotations

import pytest

from src.clinical_explanation import (ClinicalExplanation,
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
        specificity=0.75,
        retrieval_score=0.78,
        ce_score=None,
    )
    base.update(kw)
    return EvidenceCard(**base)


class TestTemplateExplainerStructure:
    def test_returns_four_field_dataclass(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(
            disease="heart_attack",
            query_symptoms=["chest_pain", "sweating"],
            matching_rules=[{"antecedent": ["chest_pain"], "confidence": 1.0,
                              "lift": 41.0, "size": 1}],
            evidence_cards=[_card()],
        )
        assert isinstance(out, ClinicalExplanation)
        assert out.symptom_disease_link
        assert out.statistical_prior
        assert out.evidence_quality
        assert out.whats_missing
        assert out.backend == "template"

    def test_link_section_includes_disease_name(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="heart_attack",
                          query_symptoms=["chest_pain"],
                          matching_rules=[{"antecedent": ["chest_pain"],
                                            "confidence": 1.0, "lift": 41.0, "size": 1}],
                          evidence_cards=[_card()])
        assert "heart attack" in out.symptom_disease_link.lower()

    def test_prior_section_uses_rule_numbers(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="heart_attack",
                          query_symptoms=["chest_pain"],
                          matching_rules=[{"antecedent": ["chest_pain"],
                                            "confidence": 0.85, "lift": 12.5, "size": 1}],
                          evidence_cards=[])
        assert "0.85" in out.statistical_prior
        assert "12.5" in out.statistical_prior

    def test_no_rules_says_retrieval_alone(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="diabetes",
                          query_symptoms=["fatigue"],
                          matching_rules=[],
                          evidence_cards=[_card(focus="Diabetes")])
        assert "retrieval" in out.statistical_prior.lower()

    def test_no_evidence_says_no_passage(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="heart_attack",
                          query_symptoms=["chest_pain"],
                          matching_rules=[{"antecedent": ["chest_pain"],
                                            "confidence": 1.0, "lift": 41.0, "size": 1}],
                          evidence_cards=[])
        # When no evidence cards, the function should still produce something
        # in evidence_quality and acknowledge the lack.
        assert out.evidence_quality
        assert "no" in out.evidence_quality.lower() or "fell back" in out.evidence_quality.lower()

    def test_whats_missing_always_warns_about_clinical_caveats(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="heart_attack",
                          query_symptoms=["chest_pain"],
                          matching_rules=[{"antecedent": ["chest_pain"],
                                            "confidence": 1.0, "lift": 41.0, "size": 1}],
                          evidence_cards=[_card()])
        # Should mention exam / labs / imaging or general clinical limitation.
        text = out.whats_missing.lower()
        assert ("examination" in text or "lab" in text or "imaging" in text
                 or "replace" in text)

    def test_modest_lift_triggers_extra_caveat(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="x",
                          query_symptoms=["q"],
                          matching_rules=[{"antecedent": ["q"], "confidence": 0.6,
                                            "lift": 1.2, "size": 1}],
                          evidence_cards=[_card()])
        # Lift < 5 should add a "modest lift" sentence.
        assert "lift" in out.whats_missing.lower()


class TestCitationsList:
    def test_citations_use_source_label_and_passage_id(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="heart_attack",
                          query_symptoms=["chest_pain"],
                          matching_rules=[],
                          evidence_cards=[_card(passage_id="P-X",
                                                 source_label="NIH NHLBI")])
        assert any("NIH NHLBI" in c and "P-X" in c for c in out.citations)

    def test_citation_count_capped_at_three(self):
        ex = TemplateClinicalExplainer()
        cards = [_card(passage_id=f"p{i}") for i in range(7)]
        out = ex.explain(disease="x", query_symptoms=["q"],
                          matching_rules=[], evidence_cards=cards)
        assert len(out.citations) == 3


class TestFactoryDispatch:
    def test_template_kind_returns_template(self):
        ex = get_clinical_explainer("template")
        assert ex.name == "template"

    def test_auto_falls_back_to_template_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Bust the lru_cache so the factory recomputes.
        get_clinical_explainer.cache_clear()
        ex = get_clinical_explainer("auto")
        assert ex.name == "template"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown clinical explainer"):
            get_clinical_explainer("vapourware")


class TestSerialisation:
    def test_to_dict_is_pure_json(self):
        ex = TemplateClinicalExplainer()
        out = ex.explain(disease="x", query_symptoms=["q"],
                          matching_rules=[], evidence_cards=[])
        d = out.to_dict()
        assert set(d) == {"symptom_disease_link", "statistical_prior",
                            "evidence_quality", "whats_missing",
                            "citations", "backend"}
        # All values are simple types -- JSON-serialisable.
        import json
        json.dumps(d)
