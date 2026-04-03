"""
Evaluation module for the Symptom-Based Disease Identification system.

Metrics implemented:
    - Recall@K   (K = 1, 3, 5, 10)
    - Precision@K
    - F1@K
    - MRR  (Mean Reciprocal Rank)

Ablation modes compared:
    - retrieval-only  (alpha = 1.0)
    - mining-only     (alpha = 0.0)
    - fused           (alpha = 0.6)

Usage (standalone demo):
    python src/evaluation.py
    python src/evaluation.py --rules data/processed/association_rules.csv \
        --out_dir data/results --n_cases 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(ground_truth: list[str], predictions: list[str], k: int) -> float:
    """Fraction of relevant items found in the top-K predictions.

    For single-label evaluation (one true disease), this is 1 if the true
    label appears in predictions[:k], else 0.

    Parameters
    ----------
    ground_truth:
        List of true disease labels (usually length 1 for single-label).
    predictions:
        Ranked list of predicted disease names, best first.
    k:
        Cut-off rank.

    Returns
    -------
    float in [0, 1]
    """
    gt_set = set(d.lower() for d in ground_truth)
    top_k = [p.lower() for p in predictions[:k]]
    hits = sum(1 for p in top_k if p in gt_set)
    return hits / max(len(gt_set), 1)


def precision_at_k(ground_truth: list[str], predictions: list[str], k: int) -> float:
    """Fraction of top-K predictions that are relevant.

    Parameters
    ----------
    ground_truth:
        List of true disease labels.
    predictions:
        Ranked list of predicted disease names.
    k:
        Cut-off rank.

    Returns
    -------
    float in [0, 1]
    """
    if k == 0:
        return 0.0
    gt_set = set(d.lower() for d in ground_truth)
    top_k = [p.lower() for p in predictions[:k]]
    hits = sum(1 for p in top_k if p in gt_set)
    return hits / k


def f1_at_k(ground_truth: list[str], predictions: list[str], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K.

    Returns
    -------
    float in [0, 1]
    """
    p = precision_at_k(ground_truth, predictions, k)
    r = recall_at_k(ground_truth, predictions, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(ground_truth: list[str], predictions: list[str]) -> float:
    """Reciprocal rank of the first relevant item in *predictions*.

    Returns 0 if no relevant item is found.
    """
    gt_set = set(d.lower() for d in ground_truth)
    for rank, pred in enumerate(predictions, start=1):
        if pred.lower() in gt_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(ground_truths: list[list[str]], all_predictions: list[list[str]]) -> float:
    """Average reciprocal rank over a list of queries."""
    rrs = [
        reciprocal_rank(gt, preds)
        for gt, preds in zip(ground_truths, all_predictions)
    ]
    return float(np.mean(rrs))


def evaluate_batch(
    ground_truths: list[list[str]],
    all_predictions: list[list[str]],
    ks: list[int] | None = None,
) -> dict:
    """Compute all metrics over a batch of (ground_truth, predictions) pairs.

    Parameters
    ----------
    ground_truths:
        One list of true labels per query.
    all_predictions:
        Ranked prediction list per query.
    ks:
        Cut-off ranks to evaluate.  Defaults to [1, 3, 5, 10].

    Returns
    -------
    dict  — keys: ``recall@K``, ``precision@K``, ``f1@K``, ``mrr``
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    n = len(ground_truths)
    metrics: dict[str, float] = {}

    for k in ks:
        metrics[f"recall@{k}"]    = float(np.mean([recall_at_k(gt, p, k)    for gt, p in zip(ground_truths, all_predictions)]))
        metrics[f"precision@{k}"] = float(np.mean([precision_at_k(gt, p, k) for gt, p in zip(ground_truths, all_predictions)]))
        metrics[f"f1@{k}"]        = float(np.mean([f1_at_k(gt, p, k)        for gt, p in zip(ground_truths, all_predictions)]))

    metrics["mrr"] = mean_reciprocal_rank(ground_truths, all_predictions)
    metrics["n_queries"] = n
    return metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    ground_truths: list[list[str]],
    all_predictions_per_alpha: dict[str, list[list[str]]],
    ks: list[int] | None = None,
) -> pd.DataFrame:
    """Run evaluate_batch for each alpha mode and return a summary DataFrame.

    Parameters
    ----------
    ground_truths:
        One list of true labels per query.
    all_predictions_per_alpha:
        Dict mapping mode name → ranked prediction lists.
        e.g. ``{"retrieval-only": [...], "mining-only": [...], "fused": [...]}``.
    ks:
        Cut-off ranks.

    Returns
    -------
    pd.DataFrame  with one row per mode.
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    rows = []
    for mode, preds in all_predictions_per_alpha.items():
        m = evaluate_batch(ground_truths, preds, ks=ks)
        m["mode"] = mode
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["mode"] + [c for c in df.columns if c != "mode"]
    return df[cols]


def print_ablation_table(df: pd.DataFrame) -> None:
    """Pretty-print the ablation results table."""
    ks = sorted({int(c.split("@")[1]) for c in df.columns if c.startswith("recall@")})
    metric_cols = (
        [f"recall@{k}" for k in ks]
        + [f"precision@{k}" for k in ks]
        + [f"f1@{k}" for k in ks]
        + ["mrr"]
    )
    metric_cols = [c for c in metric_cols if c in df.columns]

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    header = f"{'Mode':<20}" + "".join(f"{c:>12}" for c in metric_cols)
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        line = f"{row['mode']:<20}" + "".join(f"{row[c]:>12.4f}" for c in metric_cols)
        print(line)

    print("=" * 80)


# ---------------------------------------------------------------------------
# Synthetic test-case generation
# ---------------------------------------------------------------------------

# Disease → canonical symptom list mapping (used in demo / __main__)
DISEASE_SYMPTOM_MAP: dict[str, list[str]] = {
    "Influenza":            ["fever", "headache", "chills", "muscle_pain", "cough", "fatigue"],
    "Type 2 Diabetes":      ["increased_thirst", "frequent_urination", "fatigue", "blurred_vision", "weight_loss"],
    "Hypertension":         ["headache", "shortness_of_breath", "nosebleed", "dizziness"],
    "Pneumonia":            ["cough", "fever", "chills", "shortness_of_breath", "chest_pain"],
    "Asthma":               ["wheezing", "shortness_of_breath", "cough", "chest_tightness"],
    "Myocardial Infarction":["chest_pain", "shortness_of_breath", "nausea", "sweating", "lightheadedness"],
    "Urinary Tract Infection":["burning_urination", "frequent_urination", "pelvic_pain", "cloudy_urine"],
    "Migraine":             ["headache", "nausea", "light_sensitivity", "vomiting"],
    "COPD":                 ["cough", "wheezing", "shortness_of_breath", "fatigue"],
    "Rheumatoid Arthritis": ["joint_pain", "stiffness", "fatigue", "swelling"],
    "Appendicitis":         ["abdominal_pain", "nausea", "fever", "vomiting"],
    "Hypothyroidism":       ["fatigue", "weight_gain", "cold_intolerance", "constipation"],
    "GERD":                 ["heartburn", "acid_regurgitation", "chest_pain", "difficulty_swallowing"],
    "Depression":           ["sad_mood", "fatigue", "loss_of_interest", "sleep_disturbance"],
    "Anxiety Disorder":     ["worry", "restlessness", "fatigue", "muscle_tension"],
    "Tuberculosis":         ["cough", "night_sweats", "weight_loss", "fever", "fatigue"],
    "Lupus":                ["fever", "joint_pain", "rash", "fatigue"],
    "Sepsis":               ["fever", "confusion", "rapid_heartbeat", "shortness_of_breath"],
    "Stroke":               ["sudden_numbness", "confusion", "speech_difficulty", "headache"],
    "Anemia":               ["fatigue", "pallor", "shortness_of_breath", "dizziness"],
}


def generate_test_cases(
    n: int = 200,
    disease_symptom_map: dict[str, list[str]] | None = None,
    noise_prob: float = 0.2,
    min_symptoms: int = 2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate synthetic patient encounters for evaluation.

    Each encounter randomly selects a disease, samples 2–5 of its symptoms,
    and with probability *noise_prob* adds one random unrelated symptom.

    Parameters
    ----------
    n:
        Number of test cases to generate.
    disease_symptom_map:
        Mapping of disease → symptom list.  Uses ``DISEASE_SYMPTOM_MAP`` if None.
    noise_prob:
        Probability of injecting a random noise symptom.
    min_symptoms:
        Minimum number of symptoms per case.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    list[dict]  with keys ``"disease"``, ``"symptoms"`` (list[str]).
    """
    if disease_symptom_map is None:
        disease_symptom_map = DISEASE_SYMPTOM_MAP
    if rng is None:
        rng = np.random.default_rng(42)

    all_symptoms = list({s for symptoms in disease_symptom_map.values() for s in symptoms})
    diseases = list(disease_symptom_map.keys())
    cases = []

    for _ in range(n):
        disease = rng.choice(diseases)
        pool = disease_symptom_map[disease]
        k = int(rng.integers(min_symptoms, min(len(pool) + 1, 6)))
        chosen = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        if rng.random() < noise_prob:
            noise = rng.choice(all_symptoms)
            if noise not in chosen:
                chosen.append(noise)
        cases.append({"disease": disease, "symptoms": chosen})

    return cases


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluation module demo and ablation study.")
    parser.add_argument("--rules", default=None, help="Path to association rules CSV.")
    parser.add_argument("--out_dir", default="data/results", help="Directory to save result CSVs.")
    parser.add_argument("--n_cases", type=int, default=200, help="Number of synthetic test cases.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.retrieval import initialise as init_retriever
    from src.fusion_reranker import FusionReranker

    args = _parse_args()

    print("=" * 70)
    print("Evaluation Module Demo — Ablation Study")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("\n[eval] Building retriever …")
    retriever = init_retriever()

    print("[eval] Loading fusion reranker …")
    reranker_base = FusionReranker(rules_path=args.rules, alpha=0.6)

    # ------------------------------------------------------------------
    # Generate test cases
    # ------------------------------------------------------------------
    print(f"\n[eval] Generating {args.n_cases} synthetic test cases …")
    test_cases = generate_test_cases(n=args.n_cases)

    # ------------------------------------------------------------------
    # Run predictions for three alpha modes
    # ------------------------------------------------------------------
    MODES = {
        "retrieval-only": 1.0,
        "mining-only":    0.0,
        "fused":          0.6,
    }

    ground_truths: list[list[str]] = []
    all_preds: dict[str, list[list[str]]] = {m: [] for m in MODES}

    print("[eval] Running predictions …")
    for i, case in enumerate(test_cases):
        if (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{args.n_cases}")

        query = " ".join(case["symptoms"])
        ret_results = retriever.retrieve(query, top_k=15)
        ground_truths.append([case["disease"]])

        for mode, alpha in MODES.items():
            ranked = reranker_base.rank(
                query_symptoms=case["symptoms"],
                retrieval_results=ret_results,
                top_k=10,
                alpha=alpha,
            )
            preds = [r["disease"] for r in ranked]
            all_preds[mode].append(preds)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    ablation_df = run_ablation(ground_truths, all_preds)
    print_ablation_table(ablation_df)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_summary.csv"
    ablation_df.to_csv(out_path, index=False)
    print(f"\n[eval] Ablation table saved to '{out_path}'.")
