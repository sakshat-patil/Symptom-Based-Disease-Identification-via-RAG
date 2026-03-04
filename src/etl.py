"""Kaggle disease-symptom CSV -> our internal transactions table.

We tried two flavours of source data: the curated Kaggle wide CSV used here,
and the Synthea FHIR pipeline (see synthea_etl.py). The curated table has
cleaner per-row labels which is what mining wants, so we run our headline
experiments off this one. Synthea remains a working alternative.

Schema we emit (data/processed/transactions.csv):
    patient_id : str   -- "P00001" .. "PNNNNN"
    condition  : str   -- snake_case disease label
    symptoms   : str   -- pipe-separated symptom tokens, sorted, deduped

Owner: Sakshat. Reviewed by Vineet (token normalisation rules).
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "raw" / "dataset.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "transactions.csv"


def normalise(token: str) -> str:
    t = token.strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "", t)
    return t


def parse_kaggle(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "Disease" not in df.columns:
        raise ValueError("expected 'Disease' column in raw csv")
    sym_cols = [c for c in df.columns if c.lower().startswith("symptom_")]
    rows = []
    for i, r in df.iterrows():
        cond = normalise(r["Disease"])
        symptoms = [normalise(r[c]) for c in sym_cols if r[c]]
        symptoms = sorted({s for s in symptoms if s})
        rows.append({
            "patient_id": f"P{i:05d}",
            "condition": cond,
            "symptoms": "|".join(symptoms),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="kaggle", choices=["kaggle"])
    p.add_argument("--csv", default=str(DEFAULT_RAW))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = parse_kaggle(Path(args.csv))
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[etl] wrote {out}")
    print(f"[etl] rows: {len(df)}  unique diseases: {df['condition'].nunique()}")
    sym_set = {s for cs in df["symptoms"] for s in cs.split("|") if s}
    print(f"[etl] unique symptoms: {len(sym_set)}")


if __name__ == "__main__":
    main()
