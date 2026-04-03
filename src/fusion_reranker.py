"""
Hybrid fusion reranker combining dense retrieval scores with FP-Growth
association-rule mining confidence scores.

Fused score formula:
    FusedScore = alpha * RetrievalSim + (1 - alpha) * MiningConf

where:
    RetrievalSim  — cosine similarity from the FAISS dense retriever (0–1)
    MiningConf    — association-rule confidence for the disease given the
                    observed symptom set (0–1)
    alpha         — interpolation weight, default 0.6

Usage (standalone demo):
    python src/fusion_reranker.py
    python src/fusion_reranker.py --rules data/processed/association_rules.csv \
        --symptoms "fever,headache,chills" --alpha 0.6 --top_k 5
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic rules used when no CSV is available
# ---------------------------------------------------------------------------
_DEMO_RULES = [
    {"symptom_set": "fever,headache,chills", "disease": "Influenza",            "support": 0.12, "confidence": 0.82, "lift": 3.1},
    {"symptom_set": "fever,cough,fatigue",   "disease": "Influenza",            "support": 0.10, "confidence": 0.75, "lift": 2.8},
    {"symptom_set": "fever,muscle_pain",     "disease": "Influenza",            "support": 0.09, "confidence": 0.70, "lift": 2.6},
    {"symptom_set": "fever,rash",            "disease": "Measles",              "support": 0.04, "confidence": 0.78, "lift": 4.2},
    {"symptom_set": "fever,rash,cough",      "disease": "Measles",              "support": 0.03, "confidence": 0.85, "lift": 5.0},
    {"symptom_set": "chest_pain,shortness_of_breath,sweating", "disease": "Myocardial Infarction", "support": 0.05, "confidence": 0.88, "lift": 6.1},
    {"symptom_set": "chest_pain,shortness_of_breath",          "disease": "Myocardial Infarction", "support": 0.06, "confidence": 0.72, "lift": 5.0},
    {"symptom_set": "shortness_of_breath,wheezing",  "disease": "Asthma",       "support": 0.08, "confidence": 0.80, "lift": 4.4},
    {"symptom_set": "shortness_of_breath,cough,wheezing", "disease": "Asthma",  "support": 0.07, "confidence": 0.84, "lift": 4.7},
    {"symptom_set": "increased_thirst,frequent_urination,fatigue", "disease": "Type 2 Diabetes", "support": 0.06, "confidence": 0.79, "lift": 3.8},
    {"symptom_set": "fatigue,weight_gain,cold_intolerance", "disease": "Hypothyroidism", "support": 0.05, "confidence": 0.74, "lift": 3.5},
    {"symptom_set": "joint_pain,stiffness,fatigue", "disease": "Rheumatoid Arthritis", "support": 0.04, "confidence": 0.71, "lift": 3.3},
    {"symptom_set": "abdominal_pain,nausea,fever", "disease": "Appendicitis",   "support": 0.03, "confidence": 0.83, "lift": 5.5},
    {"symptom_set": "heartburn,acid_regurgitation", "disease": "GERD",          "support": 0.09, "confidence": 0.76, "lift": 2.9},
    {"symptom_set": "sad_mood,fatigue,loss_of_interest", "disease": "Depression","support": 0.07, "confidence": 0.73, "lift": 3.0},
    {"symptom_set": "worry,restlessness,fatigue", "disease": "Anxiety Disorder","support": 0.06, "confidence": 0.68, "lift": 2.7},
    {"symptom_set": "pelvic_pain,dysmenorrhoea",  "disease": "Endometriosis",   "support": 0.03, "confidence": 0.77, "lift": 4.1},
    {"symptom_set": "bloody_diarrhoea,abdominal_pain", "disease": "Ulcerative Colitis", "support": 0.03, "confidence": 0.80, "lift": 4.6},
    {"symptom_set": "diarrhoea,abdominal_pain,weight_loss", "disease": "Crohn's Disease", "support": 0.03, "confidence": 0.78, "lift": 4.3},
    {"symptom_set": "headache,nausea,light_sensitivity", "disease": "Migraine", "support": 0.08, "confidence": 0.86, "lift": 4.9},
    {"symptom_set": "cough,night_sweats,weight_loss", "disease": "Tuberculosis","support": 0.03, "confidence": 0.84, "lift": 5.2},
    {"symptom_set": "fever,joint_pain,rash",     "disease": "Lupus",            "support": 0.02, "confidence": 0.72, "lift": 4.0},
    {"symptom_set": "confusion,fever,rapid_heartbeat", "disease": "Sepsis",     "support": 0.02, "confidence": 0.89, "lift": 7.0},
    {"symptom_set": "sudden_numbness,confusion,speech_difficulty", "disease": "Stroke", "support": 0.02, "confidence": 0.91, "lift": 7.5},
    {"symptom_set": "fatigue,pallor,shortness_of_breath", "disease": "Anemia",  "support": 0.06, "confidence": 0.74, "lift": 3.4},
    {"symptom_set": "upper_abdominal_pain,nausea,vomiting", "disease": "Pancreatitis", "support": 0.03, "confidence": 0.81, "lift": 4.8},
    {"symptom_set": "right_abdominal_pain,nausea,fatty_food", "disease": "Gallstones", "support": 0.04, "confidence": 0.77, "lift": 3.7},
    {"symptom_set": "burning_urination,frequent_urination,pelvic_pain", "disease": "Urinary Tract Infection", "support": 0.07, "confidence": 0.85, "lift": 4.5},
    {"symptom_set": "skin_rash,itching,dry_skin", "disease": "Eczema",          "support": 0.06, "confidence": 0.70, "lift": 2.9},
    {"symptom_set": "joint_pain,redness,swelling", "disease": "Gout",           "support": 0.04, "confidence": 0.82, "lift": 4.4},
]


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

def load_rules(csv_path: str | Path | None) -> pd.DataFrame:
    """Load association rules from a CSV file.

    Expected columns: ``symptom_set``, ``disease``, ``support``,
    ``confidence``, ``lift``.

    Falls back to built-in demo rules if *csv_path* is ``None`` or missing.

    Parameters
    ----------
    csv_path:
        Path to the rules CSV produced by ``src/mining.py``.

    Returns
    -------
    pd.DataFrame
    """
    if csv_path is not None:
        path = Path(csv_path)
        if path.exists():
            df = pd.read_csv(path)
            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]
            required = {"symptom_set", "disease", "support", "confidence", "lift"}
            missing = required - set(df.columns)
            if not missing:
                print(f"[fusion] Loaded {len(df)} rules from '{path}'.")
                return df
            else:
                print(f"[fusion] CSV missing columns {missing} — using demo rules.")
        else:
            print(f"[fusion] '{path}' not found — using demo rules.")

    df = pd.DataFrame(_DEMO_RULES)
    print(f"[fusion] Using {len(df)} built-in demo rules.")
    return df


# ---------------------------------------------------------------------------
# Symptom normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip, replace spaces/hyphens with underscores."""
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _symptom_set_from_string(s: str) -> set[str]:
    """Parse a comma-separated symptom string into a normalised set."""
    return {_normalise(tok) for tok in s.split(",") if tok.strip()}


# ---------------------------------------------------------------------------
# Mining-based score
# ---------------------------------------------------------------------------

def _mining_confidence(disease: str, query_symptoms: set[str], rules_df: pd.DataFrame) -> float:
    """Return the maximum confidence among all rules that:
      1. Have ``disease`` as consequent.
      2. Have at least one symptom in common with *query_symptoms*.

    Returns 0.0 if no matching rule exists.
    """
    relevant = rules_df[
        rules_df["disease"].str.lower() == disease.lower()
    ]
    if relevant.empty:
        return 0.0

    best = 0.0
    for _, row in relevant.iterrows():
        rule_symptoms = _symptom_set_from_string(str(row["symptom_set"]))
        if rule_symptoms & query_symptoms:  # non-empty intersection
            # Weight by overlap ratio for a smoother score
            overlap = len(rule_symptoms & query_symptoms) / max(len(rule_symptoms), 1)
            weighted = float(row["confidence"]) * (0.5 + 0.5 * overlap)
            best = max(best, weighted)
    return best


# ---------------------------------------------------------------------------
# Disease extraction from passages
# ---------------------------------------------------------------------------

def _diseases_from_results(retrieval_results: list[dict], rules_df: pd.DataFrame) -> set[str]:
    """Collect candidate disease names from retrieval results and rule table."""
    diseases: set[str] = set()

    # From retrieval passages that carry a "disease" tag
    for r in retrieval_results:
        if r.get("disease"):
            diseases.add(r["disease"])

    # All diseases in the rule table (ensures mining-only candidates appear)
    diseases.update(rules_df["disease"].unique().tolist())

    return diseases


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse_and_rank(
    query_symptoms: str | list[str],
    retrieval_results: list[dict],
    rules_df: pd.DataFrame,
    alpha: float = 0.6,
    top_k: int = 10,
) -> list[dict]:
    """Compute fused scores for each candidate disease and return ranked list.

    Parameters
    ----------
    query_symptoms:
        Either a comma-separated symptom string or a list of symptom strings.
    retrieval_results:
        Output of ``DenseRetriever.retrieve()``.  Each dict should have at
        minimum ``{"score": float, "text": str}``.  An optional
        ``"disease"`` key is used for passage attribution.
    rules_df:
        DataFrame of association rules (from ``load_rules()``).
    alpha:
        Weight for the retrieval score.  ``(1 - alpha)`` weights mining.
    top_k:
        Maximum number of diseases to return.

    Returns
    -------
    list[dict]
        Sorted descending by ``fused_score``.  Each entry::

            {
                "rank": int,
                "disease": str,
                "fused_score": float,
                "retrieval_score": float,
                "mining_confidence": float,
                "supporting_passages": list[str],
            }
    """
    if isinstance(query_symptoms, list):
        symptom_set = {_normalise(s) for s in query_symptoms}
    else:
        symptom_set = _symptom_set_from_string(query_symptoms)

    candidates = _diseases_from_results(retrieval_results, rules_df)

    # Build a per-disease retrieval score: max cosine sim among passages that
    # mention that disease, then a fallback using overall top passage score.
    global_max_sim = max((r["score"] for r in retrieval_results), default=0.0)

    disease_ret: dict[str, float] = {}
    disease_passages: dict[str, list[str]] = {d: [] for d in candidates}

    for r in retrieval_results:
        d = r.get("disease")
        if d and d in candidates:
            disease_ret[d] = max(disease_ret.get(d, 0.0), float(r["score"]))
            disease_passages[d].append(r["text"])

    rows = []
    for disease in candidates:
        ret_sim = disease_ret.get(disease, global_max_sim * 0.3)
        mine_conf = _mining_confidence(disease, symptom_set, rules_df)
        fused = alpha * ret_sim + (1.0 - alpha) * mine_conf
        rows.append({
            "disease": disease,
            "fused_score": round(fused, 6),
            "retrieval_score": round(ret_sim, 6),
            "mining_confidence": round(mine_conf, 6),
            "supporting_passages": disease_passages.get(disease, []),
        })

    rows.sort(key=lambda x: x["fused_score"], reverse=True)
    for i, row in enumerate(rows[:top_k], start=1):
        row["rank"] = i

    return rows[:top_k]


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

class FusionReranker:
    """Stateful wrapper around ``fuse_and_rank`` that holds the rules table.

    Parameters
    ----------
    rules_path:
        Path to association rules CSV, or ``None`` to use demo rules.
    alpha:
        Default interpolation weight.
    """

    def __init__(self, rules_path: str | Path | None = None, alpha: float = 0.6):
        self.rules_df = load_rules(rules_path)
        self.alpha = alpha

    def rank(
        self,
        query_symptoms: str | list[str],
        retrieval_results: list[dict],
        top_k: int = 10,
        alpha: float | None = None,
    ) -> list[dict]:
        """Return fused disease ranking (see ``fuse_and_rank`` for details)."""
        a = alpha if alpha is not None else self.alpha
        return fuse_and_rank(
            query_symptoms=query_symptoms,
            retrieval_results=retrieval_results,
            rules_df=self.rules_df,
            alpha=a,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid fusion reranker demo."
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Path to association rules CSV. Uses built-in demo rules if omitted.",
    )
    parser.add_argument(
        "--symptoms",
        default="fever,headache,chills,muscle_pain",
        help="Comma-separated symptom query.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Retrieval weight (0–1). Default 0.6.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of ranked diseases to show.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import retrieval here (not at module top) to keep imports optional.
    from src.retrieval import initialise as init_retriever

    args = _parse_args()

    print("=" * 70)
    print("Hybrid Fusion Reranker Demo")
    print("=" * 70)

    # Build retriever with demo corpus
    retriever = init_retriever()

    # Run retrieval
    query = args.symptoms.replace(",", " ")
    print(f"\nSymptom query: {args.symptoms}")
    retrieval_results = retriever.retrieve(query, top_k=15)
    print(f"Retrieved {len(retrieval_results)} passages.\n")

    # Load rules and rank
    reranker = FusionReranker(rules_path=args.rules, alpha=args.alpha)
    ranked = reranker.rank(
        query_symptoms=args.symptoms,
        retrieval_results=retrieval_results,
        top_k=args.top_k,
    )

    # Display results
    print(f"{'Rank':<5} {'Disease':<30} {'Fused':>8} {'Retrieval':>10} {'Mining':>8}")
    print("-" * 65)
    for r in ranked:
        print(
            f"{r['rank']:<5} {r['disease']:<30} "
            f"{r['fused_score']:>8.4f} {r['retrieval_score']:>10.4f} "
            f"{r['mining_confidence']:>8.4f}"
        )

    print("\nDone.")
