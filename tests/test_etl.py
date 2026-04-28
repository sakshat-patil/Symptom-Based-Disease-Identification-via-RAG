"""Tests for the Kaggle-style transaction ETL.

We focus on the bits that have actual logic: token normalisation, symptom
deduplication, output schema. The pandas IO is exercised end-to-end via a
small synthetic CSV fixture so a bug in the parser would surface here.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import etl


class TestNormalise:
    """Token normalisation: lowercase, snake_case, ASCII-only."""

    @pytest.mark.parametrize("raw, expected", [
        ("Chest Pain",         "chest_pain"),
        ("  HIGH  Fever  ",    "high_fever"),
        ("yellowish-skin",     "yellowishskin"),
        ("Pain (jaw / arm)",   "pain_jaw__arm"),
        ("",                   ""),
    ])
    def test_normalise_known_inputs(self, raw, expected):
        assert etl.normalise(raw) == expected

    def test_normalise_idempotent(self):
        # Already-normalised tokens should pass through unchanged.
        for tok in ("chest_pain", "high_fever", "yellowish_skin"):
            assert etl.normalise(tok) == tok


class TestParseKaggle:
    """End-to-end CSV parse roundtrip on the tiny fixture."""

    def test_columns(self, tiny_transactions_csv):
        df = etl.parse_kaggle(tiny_transactions_csv)
        assert set(df.columns) == {"patient_id", "condition", "symptoms"}

    def test_row_count_matches_input(self, tiny_transactions_csv):
        df = etl.parse_kaggle(tiny_transactions_csv)
        assert len(df) == 5

    def test_patient_ids_unique_and_padded(self, tiny_transactions_csv):
        df = etl.parse_kaggle(tiny_transactions_csv)
        assert df["patient_id"].is_unique
        assert all(p.startswith("P") and len(p) == 6 for p in df["patient_id"])

    def test_conditions_normalised(self, tiny_transactions_csv):
        df = etl.parse_kaggle(tiny_transactions_csv)
        assert set(df["condition"].unique()) == {"heart_attack", "fungal_infection"}

    def test_symptoms_pipe_separated_and_sorted(self, tiny_transactions_csv):
        df = etl.parse_kaggle(tiny_transactions_csv)
        # First row symptoms: "chest pain|sweating|lightheadedness" -> sorted alpha
        first = df.iloc[0]["symptoms"].split("|")
        assert first == sorted(first)

    def test_symptoms_dedup_within_row(self, tmp_path):
        # Duplicates in a row should be deduped by parse_kaggle.
        p = tmp_path / "dup.csv"
        p.write_text(
            "Disease,Symptom_1,Symptom_2,Symptom_3,Symptom_4\n"
            "x,Itching,itching,Itching,\n"
        )
        df = etl.parse_kaggle(p)
        assert df.iloc[0]["symptoms"] == "itching"

    def test_missing_disease_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="expected 'Disease' column"):
            etl.parse_kaggle(p)
