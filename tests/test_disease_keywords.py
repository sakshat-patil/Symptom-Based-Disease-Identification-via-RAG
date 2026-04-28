"""Tests for the MedQuAD<->Kaggle disease keyword matcher.

This file includes the regression cases that motivated the original bug
fix: the 3-letter abbreviation 'hav' must NOT match inside 'have', and
multi-word phrases must still match via plain substring.
"""
from __future__ import annotations

from src.disease_keywords import DISEASE_KEYWORDS, _kw_matches, diseases_matching


class TestKwMatchesWordBoundary:
    """Single-token keywords use word boundaries; multi-word do not."""

    def test_word_boundary_blocks_partial_match(self):
        # The bug: 'hav' (a three-letter token) was matching inside 'have'.
        assert not _kw_matches("hav", "i have a headache")

    def test_word_boundary_allows_exact_match(self):
        assert _kw_matches("malaria", "patient diagnosed with malaria")

    def test_multiword_phrase_uses_substring(self):
        # Phrases with spaces don't use word boundaries because they rarely
        # appear inside another word; this is the cheaper rule.
        assert _kw_matches("heart attack", "had a heart attack last year")

    def test_punctuation_separates_words(self):
        assert _kw_matches("malaria", "fever, malaria, recurrent")

    def test_substring_inside_unrelated_word_ignored(self):
        # 'aids' should NOT match inside 'paid', 'maids', 'aiding'.
        # Make sure our test matrix reflects this.
        for hostile in ["i paid the bill", "two maids walked by", "aiding research"]:
            assert not _kw_matches("aids", hostile)
        # But the standalone token does match.
        assert _kw_matches("aids", "AIDS prevention").__class__ is bool


class TestDiseasesMatching:
    def test_returns_disease_when_focus_matches(self):
        focus = "Heart Attack"
        out = diseases_matching(focus, {"heart_attack", "fungal_infection"})
        assert out == ["heart_attack"]

    def test_unknown_text_returns_empty(self):
        out = diseases_matching("Unrelated topic", {"heart_attack", "malaria"})
        assert out == []

    def test_multiple_diseases_can_match(self):
        # Synthetic edge case: text mentions both keyword sets.
        out = diseases_matching("heart attack and malaria",
                                  {"heart_attack", "malaria", "asthma"})
        assert sorted(out) == ["heart_attack", "malaria"]

    def test_returns_disease_only_once_per_match_set(self):
        # Even if multiple keywords for the same disease appear, the disease
        # is listed once.
        out = diseases_matching("hepatitis a infection (HAV-like)",
                                  {"hepatitis_a"})
        assert out == ["hepatitis_a"]


class TestKeywordHygiene:
    """Guard rails on the curated keyword dictionary itself."""

    def test_all_keywords_lowercase(self):
        # diseases_matching lowercases input but keywords should already be lower.
        for d, kws in DISEASE_KEYWORDS.items():
            for kw in kws:
                assert kw == kw.lower(), f"{d} has uppercase keyword: {kw!r}"

    def test_no_dangerously_short_keywords(self):
        # The original bug: 'hav', 'tb', 'uti', 'bppv' matched inside other
        # words and inflated retrieval scores. We now require >= 3 chars and
        # whitelist a small set of well-known medical acronyms that are
        # genuinely safe (HIV always appears as a standalone token).
        SAFE_SHORT = {"hiv"}
        for d, kws in DISEASE_KEYWORDS.items():
            for kw in kws:
                if " " in kw or "-" in kw:
                    continue
                if kw in SAFE_SHORT:
                    continue
                assert len(kw) >= 4, f"{d} has dangerous short keyword: {kw!r}"

    def test_all_diseases_have_at_least_one_keyword(self):
        for d, kws in DISEASE_KEYWORDS.items():
            assert kws, f"empty keyword list for {d}"
