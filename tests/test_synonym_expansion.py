"""Tests for clinical synonym expansion."""
from __future__ import annotations

from src.synonym_expansion import (SYMPTOM_SYNONYMS, expand_query_string,
                                      expand_tokens)


class TestExpandTokens:
    def test_known_token_pulls_in_synonyms(self):
        out = expand_tokens(["muscle_pain"])
        assert "muscle_pain" in out
        assert "myalgia" in out
        assert "muscle ache" in out

    def test_unknown_token_passes_through(self):
        out = expand_tokens(["fictional_symptom"])
        assert out == ["fictional_symptom"]

    def test_multiple_tokens_concatenate_synonyms(self):
        out = expand_tokens(["high_fever", "joint_pain"])
        assert "pyrexia" in out
        assert "arthralgia" in out


class TestExpandQueryString:
    def test_appends_clinical_terms_section(self):
        q = expand_query_string("symptoms: muscle pain", ["muscle_pain"])
        assert "muscle pain" in q
        assert "myalgia" in q.lower()
        assert "clinical terms:" in q.lower()

    def test_no_synonyms_returns_query_unchanged(self):
        q = expand_query_string("symptoms: foo", ["fictional_symptom"])
        assert q == "symptoms: foo"


class TestDictionaryCoverage:
    def test_dictionary_is_nonempty(self):
        assert len(SYMPTOM_SYNONYMS) > 100

    def test_critical_clinical_synonyms_present(self):
        # Sanity: the headline synonyms we cite in the report must be there.
        for k in ["muscle_pain", "high_fever", "breathlessness",
                   "yellowish_skin", "chest_pain"]:
            assert k in SYMPTOM_SYNONYMS

    def test_no_empty_synonym_lists(self):
        for k, v in SYMPTOM_SYNONYMS.items():
            assert v, f"empty synonyms for {k}"
            assert all(isinstance(s, str) and s for s in v)
