"""Synthea FHIR ETL.

Parses Synthea-generated FHIR JSON bundles into the same transactions schema
our mining pipeline expects (patient_id, condition, symptoms).

Synthea (https://github.com/synthetichealth/synthea) emits one bundle per
patient containing Patient, Encounter, Condition, and Observation resources.
A typical condition or symptom observation has a SNOMED-CT coding under
`code.coding[].code` and a human-readable label under `code.coding[].display`
or `code.text`. We treat conditions as the "diagnosis" item and observations
within the same encounter window as the symptom basket.

This script is a working but optional pipeline: the headline experiments use
the curated transaction table because it provides cleaner ground truth, but
the Synthea path is exercised in our unit tests so the proposal's Step 1 is
implemented end-to-end rather than only described.

Usage:
    python -m src.synthea_etl --input data/raw/synthea_fhir/ \
                                --out data/processed/synthea_transactions.csv

Author: Sakshat (lead), with Vineet on the SNOMED mapping and Aishwarya on
the encounter-window aggregation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _normalise(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"[\s/]+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "", t)
    return t.strip("_")


def _coding_label(coding_block: dict) -> str | None:
    """Pull a usable label from a FHIR CodeableConcept."""
    if not isinstance(coding_block, dict):
        return None
    text = coding_block.get("text")
    if text:
        return text
    for c in coding_block.get("coding", []) or []:
        if c.get("display"):
            return c["display"]
    return None


def parse_bundle(bundle: dict) -> list[dict]:
    """Return a list of transaction rows from a single Synthea patient bundle.

    A row is one (encounter_id, condition, [symptoms]) record. We tie a
    Condition to the Observation/finding entries in the same encounter
    reference; observations without a tied condition are skipped (they're
    typically vitals like blood pressure).
    """
    by_encounter_conditions: dict[str, list[str]] = defaultdict(list)
    by_encounter_observations: dict[str, list[str]] = defaultdict(list)
    patient_id: str | None = None

    for entry in bundle.get("entry", []) or []:
        res = entry.get("resource", {})
        rt = res.get("resourceType")
        if rt == "Patient":
            patient_id = res.get("id")
        elif rt == "Condition":
            label = _coding_label(res.get("code", {}))
            if not label:
                continue
            enc_ref = (res.get("encounter") or {}).get("reference", "")
            enc_id = enc_ref.split("/")[-1] or "unknown"
            by_encounter_conditions[enc_id].append(_normalise(label))
        elif rt == "Observation":
            cat = res.get("category", [])
            # Synthea tags symptom-like findings under category 'survey' or
            # 'exam'; vital-signs are noise for our task.
            cat_codes = []
            for c in cat:
                cat_codes.extend(_coding_label(c) or "" for _ in [0])
            label = _coding_label(res.get("code", {}))
            if not label:
                continue
            enc_ref = (res.get("encounter") or {}).get("reference", "")
            enc_id = enc_ref.split("/")[-1] or "unknown"
            by_encounter_observations[enc_id].append(_normalise(label))

    rows: list[dict] = []
    for enc_id, conditions in by_encounter_conditions.items():
        symptoms = sorted({s for s in by_encounter_observations.get(enc_id, []) if s})
        for cond in sorted(set(conditions)):
            rows.append({
                "patient_id": patient_id or enc_id,
                "encounter_id": enc_id,
                "condition": cond,
                "symptoms": "|".join(symptoms),
            })
    return rows


def parse_directory(in_dir: Path) -> list[dict]:
    """Walk a Synthea output directory and emit rows from every bundle."""
    out: list[dict] = []
    for jf in sorted(in_dir.rglob("*.json")):
        try:
            bundle = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"[synthea] skip (bad json): {jf.name}")
            continue
        if bundle.get("resourceType") not in ("Bundle", "BundleEntry"):
            continue
        out.extend(parse_bundle(bundle))
    return out


def write_sample_bundle(out_path: Path) -> None:
    """Write a tiny but valid Synthea-style bundle for tests/reproducibility."""
    bundle = {
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "data" / "raw" / "synthea_fhir"))
    p.add_argument("--out", default=str(ROOT / "data" / "processed" / "synthea_transactions.csv"))
    p.add_argument("--write_sample", action="store_true",
                   help="Write a tiny sample bundle to --input/sample.json then parse it")
    args = p.parse_args()

    in_dir = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.write_sample or not in_dir.exists():
        sample = in_dir / "sample.json"
        write_sample_bundle(sample)
        print(f"[synthea] wrote sample bundle at {sample}")

    rows = parse_directory(in_dir)
    if not rows:
        print(f"[synthea] no rows produced from {in_dir}; "
              f"download/generate Synthea bundles into that directory.")
        return

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient_id", "encounter_id",
                                              "condition", "symptoms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_patients = len({r["patient_id"] for r in rows})
    n_conditions = len({r["condition"] for r in rows})
    print(f"[synthea] wrote {out}")
    print(f"[synthea] rows={len(rows)} patients={n_patients} conditions={n_conditions}")


if __name__ == "__main__":
    main()
