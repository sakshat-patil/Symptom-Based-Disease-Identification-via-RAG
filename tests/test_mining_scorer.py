"""Tests for query-time mining scoring.

Score formula recap:
    MiningConf(Q,d) = max over rules (A -> d) with A ⊆ Q of
                          confidence(A->d) * (|A ∩ Q| / |A|)
"""
from __future__ import annotations

from src.mining_scorer import MiningScorer


class TestMiningScorerLoading:
    def test_loads_rules_grouped_by_disease(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        # We loaded 2 rules for heart_attack and 1 for fungal_infection.
        assert set(m.rules_by_disease) == {"heart_attack", "fungal_infection"}
        assert len(m.rules_by_disease["heart_attack"]) == 2
        assert len(m.rules_by_disease["fungal_infection"]) == 1

    def test_diseases_property(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        assert m.diseases == {"heart_attack", "fungal_infection"}


class TestMiningScorerScore:
    def test_full_match_returns_rule_confidence(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        # Q = exactly the antecedent of the highest-confidence rule.
        scores = m.score(["chest_pain", "lightheadedness"])
        # heart_attack rule (chest_pain|lightheadedness, conf=1.0) matches fully.
        # Overlap = 2/2 = 1.0, so score = 1.0 * 1.0 = 1.0
        assert scores["heart_attack"] == 1.0

    def test_subset_query_uses_max_rule_with_overlap(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        # Q = just "chest_pain" -- the 2-rule won't fire (subset check fails),
        # but the single-token rule does.
        scores = m.score(["chest_pain"])
        # Single-token rule: conf=0.75, overlap=1/1 -> 0.75
        assert scores["heart_attack"] == 0.75

    def test_no_match_disease_absent(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        scores = m.score(["bladder_discomfort"])
        # Neither disease has a rule with this antecedent.
        assert scores == {}

    def test_extra_query_symptoms_dont_hurt(self, sample_rules_csv):
        # Q has the rule's antecedent plus extra junk -- should still fire,
        # because the rule only requires its antecedent ⊆ Q.
        m = MiningScorer.from_file(sample_rules_csv)
        scores = m.score(["chest_pain", "lightheadedness", "kitchen_sink"])
        assert scores["heart_attack"] == 1.0

    def test_takes_max_across_rules(self, sample_rules_csv):
        # Both rules fire for ["chest_pain", "lightheadedness"]; max() should pick
        # the conf=1.0 rule, not the conf=0.75 one.
        m = MiningScorer.from_file(sample_rules_csv)
        s = m.score(["chest_pain", "lightheadedness"])
        assert s["heart_attack"] == 1.0  # conf 1.0, not 0.75


class TestMatchingRules:
    def test_returns_top_n_sorted(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        rules = m.matching_rules(["chest_pain", "lightheadedness"], "heart_attack",
                                  top_n=5)
        assert len(rules) == 2
        # Sort: confidence desc, size desc, lift desc
        assert rules[0]["confidence"] >= rules[1]["confidence"]
        assert rules[0]["antecedent"] == ["chest_pain", "lightheadedness"]

    def test_returns_empty_for_unmatched_disease(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        assert m.matching_rules(["foo"], "heart_attack") == []

    def test_returns_empty_for_unknown_disease(self, sample_rules_csv):
        m = MiningScorer.from_file(sample_rules_csv)
        assert m.matching_rules(["chest_pain"], "no_such_disease") == []
