"""Tests for evaluation metrics and synthetic test-case generation.

The metrics are simple but the test-case generator must be deterministic
under a fixed seed (we cite per-run numbers in the report; non-deterministic
generation would invalidate them).
"""
from __future__ import annotations

import pytest

from src.evaluation import (TestCase, generate_test_cases, recall_at_k,
                              reciprocal_rank)


class TestRecallAtK:
    def test_hit_at_top1(self):
        assert recall_at_k("x", ["x", "y", "z"], 1) == 1.0

    def test_hit_within_topk(self):
        assert recall_at_k("y", ["a", "y", "b"], 3) == 1.0

    def test_miss_returns_zero(self):
        assert recall_at_k("not_present", ["a", "b", "c"], 3) == 0.0

    def test_k_zero_is_miss(self):
        assert recall_at_k("a", ["a"], 0) == 0.0

    def test_empty_ranked_list(self):
        assert recall_at_k("a", [], 5) == 0.0


class TestReciprocalRank:
    def test_top1(self):
        assert reciprocal_rank("a", ["a", "b", "c"]) == 1.0

    def test_rank_three(self):
        assert reciprocal_rank("c", ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_not_found_zero(self):
        assert reciprocal_rank("z", ["a", "b", "c"]) == 0.0


class TestGenerateTestCases:
    def test_returns_requested_count(self, tmp_path):
        # Synthesise a tiny transactions.csv to feed the generator.
        p = tmp_path / "tx.csv"
        p.write_text(
            "patient_id,condition,symptoms\n"
            "P0001,heart_attack,chest_pain|sweating|lightheadedness\n"
            "P0002,heart_attack,chest_pain|breathlessness\n"
            "P0003,fungal_infection,itching|skin_rash|nodal_skin_eruptions\n"
            "P0004,fungal_infection,itching|skin_rash\n"
        )
        cases = generate_test_cases(p, n=10, seed=42)
        assert len(cases) == 10
        for c in cases:
            assert isinstance(c, TestCase)
            assert c.true_disease in {"heart_attack", "fungal_infection"}
            assert 2 <= len(c.symptoms) <= 6  # 2-5 + maybe 1 noise

    def test_deterministic_under_same_seed(self, tmp_path):
        p = tmp_path / "tx.csv"
        p.write_text(
            "patient_id,condition,symptoms\n"
            "P1,heart_attack,chest_pain|sweating|lightheadedness|breathlessness\n"
            "P2,fungal_infection,itching|skin_rash|nodal_skin_eruptions\n"
        )
        a = generate_test_cases(p, n=20, seed=42)
        b = generate_test_cases(p, n=20, seed=42)
        assert [(c.true_disease, c.symptoms) for c in a] == \
               [(c.true_disease, c.symptoms) for c in b]

    def test_different_seeds_produce_different_outputs(self, tmp_path):
        p = tmp_path / "tx.csv"
        p.write_text(
            "patient_id,condition,symptoms\n"
            "P1,heart_attack,chest_pain|sweating|lightheadedness|breathlessness\n"
            "P2,fungal_infection,itching|skin_rash|nodal_skin_eruptions\n"
        )
        a = generate_test_cases(p, n=20, seed=1)
        b = generate_test_cases(p, n=20, seed=99)
        # Extremely high probability they differ for non-trivial n.
        assert [(c.true_disease, c.symptoms) for c in a] != \
               [(c.true_disease, c.symptoms) for c in b]

    def test_noise_can_be_disabled(self, tmp_path):
        p = tmp_path / "tx.csv"
        p.write_text(
            "patient_id,condition,symptoms\n"
            "P1,heart_attack,chest_pain|sweating|lightheadedness\n"
            "P2,fungal_infection,itching|skin_rash\n"
        )
        cases = generate_test_cases(p, n=30, seed=42, noise_prob=0.0)
        for c in cases:
            cond_symptoms = {
                "heart_attack": {"chest_pain", "sweating", "lightheadedness"},
                "fungal_infection": {"itching", "skin_rash"},
            }[c.true_disease]
            assert set(c.symptoms).issubset(cond_symptoms), \
                f"Noise-free test produced foreign symptom: {c.symptoms}"
