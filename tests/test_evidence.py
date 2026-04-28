"""Tests for the claim-level evidence extractor.

Covers: sentence splitting + offsets, claim ranking by match count,
source-tier mapping, passage-type classification, specificity scoring,
and the (tier asc, specificity desc) sort that surfaces the
highest-authority + most-specific evidence first.
"""
from __future__ import annotations

from src import evidence
from src.evidence import (DEFAULT_SOURCE, SOURCE_AUTHORITY, EvidenceCard,
                            build_evidence_card, cards_for_disease)


class TestSentenceSplit:
    def test_splits_on_period(self):
        s = "First sentence. Second sentence. Third sentence."
        sentences = evidence._split_sentences_with_offsets(s)
        assert len(sentences) == 3

    def test_offsets_match_substrings(self):
        s = "Alpha is one. Beta is two."
        sents = evidence._split_sentences_with_offsets(s)
        for sent, start, end in sents:
            assert s[start:end] == sent

    def test_empty_input_returns_empty(self):
        assert evidence._split_sentences_with_offsets("") == []


class TestPassageTypeClassifier:
    def test_symptoms_matches_first(self):
        assert evidence._classify_passage_type(
            "What are the symptoms of Asthma?") == "symptoms"

    def test_diagnosis_pattern(self):
        assert evidence._classify_passage_type(
            "How is diabetes diagnosed?") == "diagnosis"

    def test_treatment_pattern(self):
        assert evidence._classify_passage_type(
            "What are the treatments for hepatitis?") == "treatment"

    def test_what_is_falls_to_overview(self):
        assert evidence._classify_passage_type(
            "What is (are) Asthma ?") == "overview"

    def test_unknown_falls_to_general(self):
        assert evidence._classify_passage_type(
            "Random unrelated string") == "general"

    def test_empty_question(self):
        assert evidence._classify_passage_type("") == "general"
        assert evidence._classify_passage_type(None) == "general"


class TestSourceMeta:
    def test_known_source_returns_tier1(self):
        meta = evidence._source_meta("8_NHLBI_QA_XML")
        assert meta["tier"] == 1
        assert "NHLBI" in meta["label"]

    def test_unknown_source_falls_back_to_default(self):
        meta = evidence._source_meta("999_unknown_source")
        assert meta == DEFAULT_SOURCE

    def test_authority_table_has_no_blank_labels(self):
        for src, meta in SOURCE_AUTHORITY.items():
            assert meta["label"], f"{src} has blank label"
            assert meta["tier"] in (1, 2, 3)


class TestFindClaims:
    def test_picks_sentence_with_most_matches(self):
        # The middle sentence has two matched terms (thirst, urination); the
        # first has only one (diabetes); the third has zero. The two-match
        # sentence must rank first.
        text = ("Diabetes is a chronic disease. "
                 "Patients report increased thirst and frequent urination. "
                 "Diet and exercise help.")
        claims = evidence._find_claims(
            text,
            query_terms=["increased_thirst", "frequent_urination"],
            disease_keywords=["diabetes"],
            max_claims=2)
        assert claims, "expected at least one claim"
        assert "thirst" in claims[0].sentence.lower()
        assert len(claims[0].matched_terms) >= 2

    def test_returns_empty_when_no_match(self):
        claims = evidence._find_claims(
            "Random sentence about nothing.",
            query_terms=["chest_pain"], disease_keywords=["heart attack"])
        assert claims == []

    def test_offsets_recover_substring(self):
        text = "Alpha cause. Beta cause. Gamma cause of asthma."
        claims = evidence._find_claims(text, query_terms=[],
                                          disease_keywords=["asthma"], max_claims=1)
        assert claims
        c = claims[0]
        assert text[c.char_start:c.char_end] == c.sentence

    def test_multiple_claims_sorted_by_match_count(self):
        text = ("This sentence has only fever. "
                 "This other sentence has both fever and chest pain. "
                 "This last one has chest pain only.")
        claims = evidence._find_claims(
            text, query_terms=["fever", "chest_pain"], disease_keywords=[],
            max_claims=3)
        # The middle sentence has the most matches (2) so should rank first.
        assert "both" in claims[0].sentence


class TestSpecificity:
    def test_full_specificity_when_all_symptoms_appear(self):
        text = "Chest pain and lightheadedness can indicate a cardiac event."
        s = evidence._specificity(text, ["chest_pain", "lightheadedness"])
        assert s == 1.0

    def test_partial_specificity(self):
        text = "Chest pain alone is not diagnostic."
        s = evidence._specificity(text, ["chest_pain", "lightheadedness"])
        assert s == 0.5

    def test_zero_when_no_symptoms_in_text(self):
        s = evidence._specificity("Unrelated content.",
                                     ["chest_pain", "fever"])
        assert s == 0.0

    def test_empty_query_returns_zero(self):
        assert evidence._specificity("any text", []) == 0.0


class TestBuildEvidenceCard:
    def test_basic_card_round_trip(self, sample_passage_record):
        card = build_evidence_card(
            sample_passage_record,
            query_symptoms=["chest_pain", "lightheadedness", "sweating"],
            disease_keywords=["heart attack"],
        )
        assert isinstance(card, EvidenceCard)
        assert card.passage_id == "p001"
        assert card.source_tier == 1
        assert card.passage_type == "symptoms"
        assert card.specificity > 0.5  # 3 symptoms appear in passage
        assert card.claims  # claim list non-empty


class TestCardsForDisease:
    def _make_passage(self, **kw):
        base = {
            "id": "x", "source": "8_NHLBI_QA_XML", "focus": "Heart Attack",
            "question": "What are the symptoms of Heart Attack?",
            "text": "Chest pain is the most common symptom.",
            "score": 0.5,
        }
        base.update(kw)
        return base

    def test_drops_cards_with_no_evidence(self):
        # Passage has no symptom mentions and no claims.
        p = self._make_passage(text="Unrelated content.", id="empty")
        cards = cards_for_disease([p],
                                     query_symptoms=["chest_pain"],
                                     disease_keywords=["xyz"])
        assert cards == []

    def test_sorts_tier1_before_tier3(self):
        p1 = self._make_passage(source="8_NHLBI_QA_XML", id="t1",
                                 text="Chest pain warning sign.")
        p2 = self._make_passage(source="12_MPlusHerbsSupplements_QA", id="t3",
                                 text="Chest pain may be relieved by herbs.")
        cards = cards_for_disease([p2, p1],   # input deliberately reversed
                                     query_symptoms=["chest_pain"],
                                     disease_keywords=["heart attack"])
        assert [c.passage_id for c in cards] == ["t1", "t3"]

    def test_specificity_breaks_tier_tie(self):
        p_lo = self._make_passage(id="lo",
                                    text="Chest pain alone is concerning.")
        p_hi = self._make_passage(id="hi",
                                    text="Chest pain and lightheadedness together suggest cardiac event.")
        cards = cards_for_disease([p_lo, p_hi],
                                     query_symptoms=["chest_pain", "lightheadedness"],
                                     disease_keywords=["heart attack"])
        # Both tier 1; higher specificity wins.
        assert cards[0].passage_id == "hi"

    def test_max_cards_cap_respected(self):
        passages = [self._make_passage(id=f"p{i}") for i in range(10)]
        cards = cards_for_disease(passages,
                                     query_symptoms=["chest_pain"],
                                     disease_keywords=["heart attack"],
                                     max_cards=3)
        assert len(cards) == 3
