"""
ETL pipeline for ingesting medical symptom-disease data into a transaction table.

Supports three input sources (tried in order):
  1. Synthea FHIR Bundle JSON files  (data/raw/*.json)
  2. Kaggle symptom-disease CSV       (data/raw/disease_symptom.csv  or similar)
  3. Built-in synthetic fallback      (generated on-the-fly so every script runs)

Output: data/processed/transactions.csv
  Columns: patient_id, symptoms (pipe-separated), condition

Usage:
    python src/etl.py
    python src/etl.py --source synthea --raw_dir data/raw --out data/processed/transactions.csv
    python src/etl.py --source kaggle  --csv data/raw/disease_symptom.csv
    python src/etl.py --source synthetic --n_patients 1000
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FHIR / Synthea parsing
# ---------------------------------------------------------------------------

# SNOMED / display-name patterns that indicate a *symptom* resource vs a
# primary diagnosis.  Synthea encodes both as Condition resources; we
# distinguish by checking the category code.
_SYMPTOM_CATEGORY_CODES = {"55607006", "418799008", "symptom"}
_CONDITION_CATEGORY_CODES = {"encounter-diagnosis", "problem-list-item"}


def parse_synthea_bundle(filepath: str | Path) -> dict:
    """Parse one Synthea FHIR Bundle JSON and return structured patient data.

    Parameters
    ----------
    filepath:
        Path to a ``<patient-uuid>.json`` Synthea output file.

    Returns
    -------
    dict with keys:
        ``patient_id`` (str),
        ``conditions`` (list[str])  — display names of diagnoses,
        ``symptoms``   (list[str])  — display names of symptom observations.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    patient_id: str = filepath.stem
    conditions: list[str] = []
    symptoms: list[str] = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        # ----------------------------------------------------------------
        # Condition resources → diagnoses or symptoms
        # ----------------------------------------------------------------
        if rtype == "Condition":
            display = _extract_display(resource.get("code", {}))
            if not display:
                continue

            cats = [
                c.get("coding", [{}])[0].get("code", "").lower()
                for c in resource.get("category", [])
            ]
            cat_str = " ".join(cats)

            if any(s in cat_str for s in _SYMPTOM_CATEGORY_CODES):
                symptoms.append(_normalise_term(display))
            else:
                conditions.append(_normalise_term(display))

        # ----------------------------------------------------------------
        # Observation resources with category "survey" or "exam" → symptoms
        # ----------------------------------------------------------------
        elif rtype == "Observation":
            cats = [
                c.get("coding", [{}])[0].get("code", "").lower()
                for c in resource.get("category", [])
            ]
            if any(c in ("survey", "exam", "symptoms") for c in cats):
                display = _extract_display(resource.get("code", {}))
                if display:
                    symptoms.append(_normalise_term(display))

    return {
        "patient_id": patient_id,
        "conditions": list(set(conditions)),
        "symptoms":   list(set(symptoms)),
    }


def _extract_display(code_obj: dict) -> str:
    """Pull the human-readable display from a FHIR CodeableConcept."""
    # Try top-level text first
    if code_obj.get("text"):
        return code_obj["text"]
    # Fall back to first coding display
    for coding in code_obj.get("coding", []):
        if coding.get("display"):
            return coding["display"]
    return ""


def _normalise_term(term: str) -> str:
    """Lower-case, strip, replace spaces with underscores."""
    import re
    return re.sub(r"[\s\-/]+", "_", term.strip().lower())


def build_transaction_table(records: list[dict]) -> pd.DataFrame:
    """Convert a list of parsed patient records into a transaction table.

    Each row represents one (patient, primary_condition) pair.
    The ``symptoms`` column is a pipe-separated string of symptom tokens.

    Parameters
    ----------
    records:
        Output of repeated calls to ``parse_synthea_bundle()``.

    Returns
    -------
    pd.DataFrame  columns: patient_id, condition, symptoms
    """
    rows = []
    for rec in records:
        if not rec["conditions"]:
            continue
        # Use the first (primary) condition as the label
        primary = rec["conditions"][0]
        syms = "|".join(rec["symptoms"]) if rec["symptoms"] else ""
        rows.append({
            "patient_id": rec["patient_id"],
            "condition":  primary,
            "symptoms":   syms,
        })
    return pd.DataFrame(rows, columns=["patient_id", "condition", "symptoms"])


# ---------------------------------------------------------------------------
# Kaggle / CSV ingest
# The most common public symptom-disease CSV on Kaggle has columns like:
#   Disease, Symptom_1, Symptom_2, … Symptom_N
# We handle that format plus a "tidy" format with columns Disease, Symptom.
# ---------------------------------------------------------------------------

def load_kaggle_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a Kaggle-style symptom-disease CSV into a transaction table.

    Handles two common layouts:
      - **Wide**: ``Disease, Symptom_1, Symptom_2, …``
      - **Tidy**: ``Disease, Symptom``

    Returns
    -------
    pd.DataFrame  columns: patient_id, condition, symptoms
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect layout
    symptom_cols = [c for c in df.columns if c.startswith("symptom") and c != "symptom"]

    rows = []
    if symptom_cols:
        # Wide layout
        disease_col = "disease" if "disease" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            disease = _normalise_term(str(row[disease_col]))
            syms = [
                _normalise_term(str(row[c]))
                for c in symptom_cols
                if pd.notna(row[c]) and str(row[c]).strip()
            ]
            if syms:
                rows.append({
                    "patient_id": str(uuid.uuid4())[:8],
                    "condition":  disease,
                    "symptoms":   "|".join(syms),
                })
    elif "symptom" in df.columns and "disease" in df.columns:
        # Tidy layout — group by disease
        for disease, grp in df.groupby("disease"):
            syms = [_normalise_term(s) for s in grp["symptom"].dropna()]
            if syms:
                rows.append({
                    "patient_id": str(uuid.uuid4())[:8],
                    "condition":  _normalise_term(str(disease)),
                    "symptoms":   "|".join(syms),
                })
    else:
        raise ValueError(
            f"Cannot parse '{csv_path}': expected columns 'disease' + "
            f"'symptom_N' (wide) or 'disease'+'symptom' (tidy). "
            f"Found: {list(df.columns)}"
        )

    result = pd.DataFrame(rows, columns=["patient_id", "condition", "symptoms"])
    print(f"[etl] Loaded {len(result)} records from '{csv_path}'.")
    return result


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

SYNTHETIC_DISEASE_SYMPTOM_MAP: dict[str, list[str]] = {
    "influenza":             ["fever", "headache", "chills", "muscle_pain", "cough", "fatigue", "sore_throat"],
    "type_2_diabetes":       ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision", "weight_loss", "slow_healing"],
    "hypertension":          ["headache", "shortness_of_breath", "nosebleed", "dizziness", "chest_pain"],
    "pneumonia":             ["cough", "fever", "chills", "shortness_of_breath", "chest_pain", "fatigue"],
    "asthma":                ["wheezing", "shortness_of_breath", "cough", "chest_tightness", "fatigue"],
    "myocardial_infarction": ["chest_pain", "shortness_of_breath", "nausea", "sweating", "lightheadedness", "arm_pain"],
    "urinary_tract_infection":["burning_urination", "frequent_urination", "pelvic_pain", "cloudy_urine", "fever"],
    "migraine":              ["headache", "nausea", "light_sensitivity", "vomiting", "aura"],
    "copd":                  ["cough", "wheezing", "shortness_of_breath", "fatigue", "mucus_production"],
    "rheumatoid_arthritis":  ["joint_pain", "stiffness", "fatigue", "swelling", "fever"],
    "appendicitis":          ["abdominal_pain", "nausea", "fever", "vomiting", "loss_of_appetite"],
    "hypothyroidism":        ["fatigue", "weight_gain", "cold_intolerance", "constipation", "dry_skin"],
    "gerd":                  ["heartburn", "acid_regurgitation", "chest_pain", "difficulty_swallowing", "nausea"],
    "depression":            ["sad_mood", "fatigue", "loss_of_interest", "sleep_disturbance", "concentration_difficulty"],
    "anxiety_disorder":      ["worry", "restlessness", "fatigue", "muscle_tension", "sleep_disturbance"],
    "tuberculosis":          ["cough", "night_sweats", "weight_loss", "fever", "fatigue", "haemoptysis"],
    "lupus":                 ["fever", "joint_pain", "rash", "fatigue", "hair_loss"],
    "sepsis":                ["fever", "confusion", "rapid_heartbeat", "shortness_of_breath", "low_blood_pressure"],
    "stroke":                ["sudden_numbness", "confusion", "speech_difficulty", "headache", "vision_loss"],
    "anemia":                ["fatigue", "pallor", "shortness_of_breath", "dizziness", "cold_hands"],
}


def generate_synthetic_transactions(
    n_patients: int = 1000,
    disease_symptom_map: dict[str, list[str]] | None = None,
    noise_prob: float = 0.15,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a synthetic transaction table for testing.

    Parameters
    ----------
    n_patients:
        Number of synthetic patient records to create.
    disease_symptom_map:
        Override the built-in map if desired.
    noise_prob:
        Probability of adding one random unrelated symptom per patient.
    rng:
        NumPy random generator.

    Returns
    -------
    pd.DataFrame  columns: patient_id, condition, symptoms
    """
    if disease_symptom_map is None:
        disease_symptom_map = SYNTHETIC_DISEASE_SYMPTOM_MAP
    if rng is None:
        rng = np.random.default_rng(42)

    all_symptoms = list({s for syms in disease_symptom_map.values() for s in syms})
    diseases = list(disease_symptom_map.keys())
    rows = []

    for i in range(n_patients):
        disease = rng.choice(diseases)
        pool = disease_symptom_map[disease]
        k = int(rng.integers(2, min(len(pool) + 1, 6)))
        chosen = list(rng.choice(pool, size=k, replace=False))
        if rng.random() < noise_prob:
            noise = rng.choice(all_symptoms)
            if noise not in chosen:
                chosen.append(noise)
        rows.append({
            "patient_id": f"syn_{i:05d}",
            "condition":  disease,
            "symptoms":   "|".join(chosen),
        })

    df = pd.DataFrame(rows, columns=["patient_id", "condition", "symptoms"])
    print(f"[etl] Generated {len(df)} synthetic patient records.")
    return df


# ---------------------------------------------------------------------------
# Top-level run function
# ---------------------------------------------------------------------------

def run_etl(
    source: str = "auto",
    raw_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
    n_patients: int = 1000,
    out_path: str | Path = "data/processed/transactions.csv",
) -> pd.DataFrame:
    """Run the full ETL and save the transaction table.

    Parameters
    ----------
    source:
        ``"synthea"``, ``"kaggle"``, ``"synthetic"``, or ``"auto"``
        (auto tries synthea → kaggle → synthetic).
    raw_dir:
        Directory containing Synthea JSON files.
    csv_path:
        Path to a Kaggle-style CSV.
    n_patients:
        Number of synthetic patients (only used for ``"synthetic"`` source).
    out_path:
        Where to write the output CSV.

    Returns
    -------
    pd.DataFrame
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame | None = None

    # ---- Synthea ----
    if source in ("synthea", "auto"):
        raw = Path(raw_dir) if raw_dir else Path("data/raw")
        json_files = list(raw.glob("*.json"))
        if json_files:
            print(f"[etl] Found {len(json_files)} Synthea JSON files in '{raw}'.")
            records = [parse_synthea_bundle(f) for f in json_files]
            df = build_transaction_table(records)
            if len(df) == 0:
                print("[etl] Synthea parse produced 0 records — falling through.")
                df = None
        elif source == "synthea":
            raise FileNotFoundError(f"No JSON files found in '{raw}'.")

    # ---- Kaggle CSV ----
    if df is None and source in ("kaggle", "auto"):
        candidates = []
        if csv_path:
            candidates.append(Path(csv_path))
        else:
            raw = Path(raw_dir) if raw_dir else Path("data/raw")
            candidates = list(raw.glob("*.csv"))

        for p in candidates:
            try:
                df = load_kaggle_csv(p)
                break
            except Exception as e:
                print(f"[etl] Could not load '{p}': {e}")

    # ---- Synthetic fallback ----
    if df is None:
        if source not in ("synthetic", "auto"):
            raise RuntimeError(f"ETL source '{source}' produced no data.")
        print("[etl] No real data found — generating synthetic transactions.")
        df = generate_synthetic_transactions(n_patients=n_patients)

    df.to_csv(out_path, index=False)
    print(f"[etl] Saved {len(df)} records to '{out_path}'.")
    print(f"[etl] Unique conditions: {df['condition'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="ETL pipeline for symptom-disease transactions.")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "synthea", "kaggle", "synthetic"],
                        help="Data source. 'auto' tries each in order.")
    parser.add_argument("--raw_dir", default="data/raw",
                        help="Directory containing Synthea JSON or CSV files.")
    parser.add_argument("--csv", default=None,
                        help="Explicit path to a Kaggle-style CSV file.")
    parser.add_argument("--n_patients", type=int, default=1000,
                        help="Number of synthetic patients (synthetic mode only).")
    parser.add_argument("--out", default="data/processed/transactions.csv",
                        help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    args = _parse_args()

    print("=" * 60)
    print("ETL Pipeline")
    print("=" * 60)

    df = run_etl(
        source=args.source,
        raw_dir=args.raw_dir,
        csv_path=args.csv,
        n_patients=args.n_patients,
        out_path=args.out,
    )

    print("\nSample records:")
    print(df.head(5).to_string(index=False))
    print(f"\nCondition distribution (top 10):")
    print(df["condition"].value_counts().head(10).to_string())
