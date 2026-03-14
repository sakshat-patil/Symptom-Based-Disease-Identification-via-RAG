"""Metric helpers + synthetic test-case generator.

We generate test cases by sampling 2-5 symptoms from a random disease's
canonical symptom set, then optionally adding a noise symptom from a
different disease. Seed 42 makes everything reproducible. The 200-case
default matches the harness used in our Check-in 3 and 4 numbers.

Limitation: cases share vocabulary with the training table by
construction. We flag this in the Discussion section of the report and
recommend a Synthea-based held-out eval as future work.

Owner: Sakshat (case generator) + Aishwarya (metrics).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TestCase:
    true_disease: str
    symptoms: list[str]


# Tell pytest not to try collecting this dataclass as a test class; the
# `Test` prefix is just unfortunate naming.
TestCase.__test__ = False


def generate_test_cases(transactions_csv: Path, n: int = 200, seed: int = 42,
                         min_k: int = 2, max_k: int = 5,
                         noise_prob: float = 0.2) -> list[TestCase]:
    df = pd.read_csv(transactions_csv).fillna("")
    by_disease: dict[str, list[set[str]]] = {}
    all_symptoms: set[str] = set()
    for _, r in df.iterrows():
        cond = r["condition"]
        syms = {s for s in r["symptoms"].split("|") if s}
        all_symptoms.update(syms)
        by_disease.setdefault(cond, []).append(syms)

    rng = random.Random(seed)
    diseases = sorted(by_disease.keys())
    all_symptoms_list = sorted(all_symptoms)
    cases: list[TestCase] = []
    for _ in range(n):
        d = rng.choice(diseases)
        candidate_syms = list({s for basket in by_disease[d] for s in basket})
        k = rng.randint(min_k, min(max_k, len(candidate_syms)))
        chosen = rng.sample(candidate_syms, k)
        if rng.random() < noise_prob:
            noise = rng.choice([s for s in all_symptoms_list if s not in candidate_syms]
                                or all_symptoms_list)
            chosen.append(noise)
        cases.append(TestCase(true_disease=d, symptoms=sorted(set(chosen))))
    return cases


def recall_at_k(true: str, ranked: list[str], k: int) -> float:
    return 1.0 if true in ranked[:k] else 0.0


def precision_at_k(true: str, ranked: list[str], k: int) -> float:
    if not ranked[:k]:
        return 0.0
    return (1.0 if true in ranked[:k] else 0.0) / 1.0  # single ground truth -> at most 1 correct


def reciprocal_rank(true: str, ranked: list[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d == true:
            return 1.0 / i
    return 0.0


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k not in ("true_disease", "ranked")]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
