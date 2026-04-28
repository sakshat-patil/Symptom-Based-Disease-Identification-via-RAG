"""Tests for the vector-store abstraction.

We don't talk to Pinecone in unit tests (no network calls). Instead we
exercise the FAISS post-filter (`_matches_filter`) which mimics Pinecone's
filter syntax so the same UI control works for both backends.
"""
from __future__ import annotations

from src.vector_store import _matches_filter


class TestMatchesFilterEqualityShortForm:
    def test_simple_equality(self):
        md = {"source": "9_CDC_QA", "tier": 1}
        assert _matches_filter(md, {"source": "9_CDC_QA"})

    def test_equality_negative(self):
        md = {"source": "9_CDC_QA"}
        assert not _matches_filter(md, {"source": "8_NHLBI_QA_XML"})

    def test_missing_key_filters_out(self):
        md = {"source": "9_CDC_QA"}
        assert not _matches_filter(md, {"foo": "bar"})

    def test_empty_filter_passes(self):
        md = {"source": "x"}
        assert _matches_filter(md, {})
        assert _matches_filter(md, None)


class TestMatchesFilterMongoOps:
    def test_dollar_in(self):
        md = {"source": "9_CDC_QA"}
        assert _matches_filter(md, {"source": {"$in": ["9_CDC_QA", "8_NHLBI_QA_XML"]}})

    def test_dollar_in_negative(self):
        md = {"source": "12_MPlusHerbsSupplements_QA"}
        assert not _matches_filter(md,
            {"source": {"$in": ["9_CDC_QA", "8_NHLBI_QA_XML"]}})

    def test_dollar_eq(self):
        assert _matches_filter({"x": 1}, {"x": {"$eq": 1}})
        assert not _matches_filter({"x": 1}, {"x": {"$eq": 2}})

    def test_dollar_ne(self):
        assert _matches_filter({"x": 1}, {"x": {"$ne": 2}})
        assert not _matches_filter({"x": 2}, {"x": {"$ne": 2}})


class TestMultipleConditions:
    def test_all_must_match(self):
        md = {"source": "9_CDC_QA", "passage_type": "symptoms"}
        assert _matches_filter(md, {"source": "9_CDC_QA",
                                      "passage_type": "symptoms"})

    def test_any_failure_rejects(self):
        md = {"source": "9_CDC_QA", "passage_type": "symptoms"}
        assert not _matches_filter(md, {"source": "9_CDC_QA",
                                          "passage_type": "treatment"})
