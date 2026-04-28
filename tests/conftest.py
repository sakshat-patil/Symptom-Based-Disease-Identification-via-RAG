"""Shared pytest fixtures for the test suite.

Most fixtures are tiny synthetic inputs -- no model loading, no FAISS index,
no network calls. The integration-flavoured fixtures (e.g. a built FAISS
index) live behind a `@pytest.mark.slow` marker so the default `pytest` run
stays under a second on a laptop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_transactions_csv(tmp_path: Path) -> Path:
    """Five-row Kaggle-shape CSV. Big enough to mine, small enough to read."""
    p = tmp_path / "dataset.csv"
    p.write_text(
        "Disease,Symptom_1,Symptom_2,Symptom_3,Symptom_4\n"
        "heart_attack,chest pain,sweating,lightheadedness,\n"
        "heart_attack,chest pain,sweating,,\n"
        "heart_attack,chest pain,lightheadedness,breathlessness,\n"
        "fungal_infection,itching,skin rash,,\n"
        "fungal_infection,itching,skin rash,nodal skin eruptions,\n"
    )
    return p


@pytest.fixture
def sample_passage_record() -> dict:
    """One MedQuAD-shaped passage, used by retrieval and evidence tests."""
    return {
        "id": "p001",
        "source": "8_NHLBI_QA_XML",
        "focus": "Heart Attack",
        "question": "What are the symptoms of Heart Attack?",
        "text": (
            "Q: What are the symptoms of Heart Attack? "
            "A: Chest pain or discomfort is the most common symptom. "
            "Other warning signs include shortness of breath, "
            "sweating, lightheadedness, and pain in the jaw or arm. "
            "Seek emergency care immediately if these are present."
        ),
    }


@pytest.fixture
def sample_rules_csv(tmp_path: Path) -> Path:
    """A miniature association_rules.csv, three rules across two diseases."""
    p = tmp_path / "rules.csv"
    p.write_text(
        "antecedent,consequent,support,confidence,lift,antecedent_len\n"
        "chest_pain|lightheadedness,heart_attack,0.20,1.00,5.00,2\n"
        "chest_pain,heart_attack,0.40,0.75,3.75,1\n"
        "itching|skin_rash,fungal_infection,0.40,1.00,2.50,2\n"
    )
    return p


@pytest.fixture
def fhir_bundle() -> dict:
    """A minimal Synthea-style FHIR bundle (Patient + Encounter + Condition + 3 Observations)."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "P-001"}},
            {"resource": {
                "resourceType": "Encounter", "id": "E-1",
                "subject": {"reference": "Patient/P-001"},
            }},
            {"resource": {
                "resourceType": "Condition",
                "encounter": {"reference": "Encounter/E-1"},
                "code": {"text": "Heart attack"},
            }},
            {"resource": {
                "resourceType": "Observation",
                "encounter": {"reference": "Encounter/E-1"},
                "code": {"text": "Chest pain"},
            }},
            {"resource": {
                "resourceType": "Observation",
                "encounter": {"reference": "Encounter/E-1"},
                "code": {"text": "Lightheadedness"},
            }},
            {"resource": {
                "resourceType": "Observation",
                "encounter": {"reference": "Encounter/E-1"},
                "code": {"text": "Sweating"},
            }},
        ],
    }
