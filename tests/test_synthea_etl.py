"""Tests for the Synthea FHIR ETL.

The headline behaviour: a Bundle's Conditions and Observations get tied
together by encounter reference, then emitted one row per (encounter,
condition) pair. We also check normalisation and the 'no condition,
no row' rule.
"""
from __future__ import annotations

from src import synthea_etl


class TestParseBundle:
    def test_basic_bundle_emits_one_row(self, fhir_bundle):
        rows = synthea_etl.parse_bundle(fhir_bundle)
        assert len(rows) == 1
        row = rows[0]
        assert row["patient_id"] == "P-001"
        assert row["encounter_id"] == "E-1"
        assert row["condition"] == "heart_attack"
        # Symptoms are sorted alphabetically and pipe-joined.
        assert row["symptoms"] == "chest_pain|lightheadedness|sweating"

    def test_observation_without_condition_drops(self):
        # Encounter has observations but no Condition -> no row.
        bundle = {
            "resourceType": "Bundle", "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "P-2"}},
                {"resource": {
                    "resourceType": "Encounter", "id": "E-9",
                }},
                {"resource": {
                    "resourceType": "Observation",
                    "encounter": {"reference": "Encounter/E-9"},
                    "code": {"text": "Cough"},
                }},
            ],
        }
        assert synthea_etl.parse_bundle(bundle) == []

    def test_multiple_conditions_per_encounter(self, fhir_bundle):
        # Add a second Condition tied to the same encounter; expect 2 rows.
        b = dict(fhir_bundle)
        b["entry"] = list(fhir_bundle["entry"]) + [
            {"resource": {
                "resourceType": "Condition",
                "encounter": {"reference": "Encounter/E-1"},
                "code": {"text": "Coronary artery disease"},
            }},
        ]
        rows = synthea_etl.parse_bundle(b)
        conds = sorted(r["condition"] for r in rows)
        assert conds == ["coronary_artery_disease", "heart_attack"]

    def test_handles_missing_encounter_ref(self):
        # Resource without encounter ref still parses as 'unknown' bucket.
        bundle = {
            "resourceType": "Bundle", "type": "collection",
            "entry": [
                {"resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Asthma"},
                }},
                {"resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Wheezing"},
                }},
            ],
        }
        rows = synthea_etl.parse_bundle(bundle)
        assert len(rows) == 1
        assert rows[0]["condition"] == "asthma"
        assert "wheezing" in rows[0]["symptoms"]


class TestSampleBundleRoundtrip:
    """write_sample_bundle -> parse_directory should recover one row."""

    def test_sample_then_parse(self, tmp_path):
        sample = tmp_path / "sample.json"
        synthea_etl.write_sample_bundle(sample)
        rows = synthea_etl.parse_directory(tmp_path)
        assert len(rows) == 1
        assert rows[0]["condition"] == "heart_attack"


class TestNormalise:
    def test_strips_punctuation(self):
        assert synthea_etl._normalise("Chest pain (acute)!") == "chest_pain_acute"

    def test_handles_empty(self):
        assert synthea_etl._normalise("") == ""
        assert synthea_etl._normalise("   ") == ""
