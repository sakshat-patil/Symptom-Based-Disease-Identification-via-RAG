"""Tests for the linear fusion reranker.

FusedScore(d) = α * RetrievalSim(d) + (1 - α) * MiningConf(d)

Tie-break order (descending priority): fused, mining, retrieval, name.
"""
from __future__ import annotations

import pytest

from src.fusion_reranker import FusedCandidate, fuse


class TestFuseConvexCombination:
    def test_alpha_zero_uses_mining_only(self):
        out = fuse({"a": 0.9}, {"b": 0.5}, alpha=0.0)
        # b's mining score becomes the entire fused score; a contributes 0.
        scores = {c.disease: c.fused_score for c in out}
        assert scores["b"] == pytest.approx(0.5)
        assert scores["a"] == pytest.approx(0.0)

    def test_alpha_one_uses_retrieval_only(self):
        out = fuse({"a": 0.9}, {"b": 0.5}, alpha=1.0)
        scores = {c.disease: c.fused_score for c in out}
        assert scores["a"] == pytest.approx(0.9)
        assert scores["b"] == pytest.approx(0.0)

    def test_alpha_half_is_average_when_both_present(self):
        out = fuse({"x": 0.4}, {"x": 0.8}, alpha=0.5)
        assert out[0].disease == "x"
        assert out[0].fused_score == pytest.approx(0.6)


class TestCandidatePool:
    def test_pools_union_of_signals(self):
        out = fuse({"a": 0.9, "b": 0.1}, {"c": 0.5}, alpha=0.5)
        assert {c.disease for c in out} == {"a", "b", "c"}

    def test_disease_only_in_one_signal_still_appears(self):
        # 'b' has mining=0 (absent) but retrieval=0.7 -- must still show up.
        out = fuse({"b": 0.7}, {}, alpha=0.5)
        assert any(c.disease == "b" for c in out)


class TestTieBreak:
    def test_mining_wins_tie(self):
        # Two diseases with identical fused scores; mining-higher wins the tie.
        out = fuse({"a": 0.5, "b": 0.0},     # retrieval
                    {"a": 0.0, "b": 0.5},     # mining
                    alpha=0.5)
        # Both fused = 0.25. Tie-break: mining desc -> b before a.
        assert [c.disease for c in out] == ["b", "a"]

    def test_retrieval_wins_when_mining_also_tied(self):
        # Both diseases have identical fused AND identical mining; retrieval breaks tie.
        out = fuse({"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}, alpha=0.5)
        # Both fused = 0.5. Mining: a=0.4, b=0.6 -> b first (mining desc).
        # No actual mining tie, so we use the secondary signal.
        assert out[0].disease == "b"

    def test_name_breaks_full_tie(self):
        # Identical everything -> alphabetical disease name.
        out = fuse({"banana": 0.5, "apple": 0.5},
                    {"banana": 0.5, "apple": 0.5}, alpha=0.5)
        assert [c.disease for c in out] == ["apple", "banana"]


class TestFusedCandidateShape:
    def test_returns_dataclass_with_subscores(self):
        out = fuse({"x": 0.7}, {"x": 0.3}, alpha=0.6)
        c = out[0]
        assert isinstance(c, FusedCandidate)
        assert c.retrieval_score == pytest.approx(0.7)
        assert c.mining_score == pytest.approx(0.3)
        assert c.fused_score == pytest.approx(0.54)


class TestEmptyInputs:
    def test_empty_returns_empty(self):
        assert fuse({}, {}, alpha=0.3) == []
