"""
End-to-end experiment runner for the Symptom-Based Disease Identification system.

Pipeline:
    1. Build / load the dense FAISS retriever (demo corpus or JSONL file).
    2. Load association rules (CSV or built-in demo rules).
    3. Generate 200 synthetic patient encounters.
    4. For each encounter: retrieve passages → fuse-and-rank diseases.
    5. Evaluate with Recall@K, Precision@K, F1@K, and MRR.
    6. Run ablation (retrieval-only, mining-only, fused).
    7. Save per-case results and ablation summary to data/results/.

Usage:
    python src/run_experiment.py
    python src/run_experiment.py --rules data/processed/association_rules.csv \
        --corpus data/raw/passages.jsonl --n_cases 200 --alpha 0.6 --top_k 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end experiment runner for symptom-based disease identification."
    )
    parser.add_argument("--corpus",  default=None,             help="JSONL passages file (falls back to demo corpus).")
    parser.add_argument("--rules",   default=None,             help="Association rules CSV (falls back to demo rules).")
    parser.add_argument("--index",   default=None,             help="Pre-built FAISS index path to load/save.")
    parser.add_argument("--n_cases", type=int,   default=200,  help="Number of synthetic test cases.")
    parser.add_argument("--alpha",   type=float, default=0.6,  help="Fusion weight for retrieval score.")
    parser.add_argument("--top_k",   type=int,   default=10,   help="Retrieval and ranking cut-off.")
    parser.add_argument("--out_dir", default="data/results",   help="Output directory for CSV files.")
    parser.add_argument("--seed",    type=int,   default=42,   help="Random seed for synthetic data.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # Ensure project root is on the path
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))

    from src.retrieval       import initialise as init_retriever
    from src.fusion_reranker import FusionReranker
    from src.evaluation      import (
        generate_test_cases,
        evaluate_batch,
        run_ablation,
        print_ablation_table,
        DISEASE_SYMPTOM_MAP,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Symptom-Based Disease Identification — Experiment Runner")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build retriever
    # ------------------------------------------------------------------
    print("\n[run] Initialising dense retriever …")
    t0 = time.time()
    retriever = init_retriever(
        corpus_path=args.corpus,
        index_path=args.index,
        save_index_to=args.index,   # save if a path was given and index didn't exist yet
    )
    print(f"[run] Retriever ready in {time.time() - t0:.1f}s.")

    # ------------------------------------------------------------------
    # 2. Load fusion reranker
    # ------------------------------------------------------------------
    print("\n[run] Loading fusion reranker …")
    reranker = FusionReranker(rules_path=args.rules, alpha=args.alpha)

    # ------------------------------------------------------------------
    # 3. Generate test set
    # ------------------------------------------------------------------
    print(f"\n[run] Generating {args.n_cases} synthetic patient encounters (seed={args.seed}) …")
    rng = np.random.default_rng(args.seed)
    test_cases = generate_test_cases(
        n=args.n_cases,
        disease_symptom_map=DISEASE_SYMPTOM_MAP,
        rng=rng,
    )
    disease_counts = pd.Series([c["disease"] for c in test_cases]).value_counts()
    print(f"[run] Disease distribution (top 5):\n{disease_counts.head(5).to_string()}\n")

    # ------------------------------------------------------------------
    # 4. Run pipeline for all three alpha modes
    # ------------------------------------------------------------------
    MODES = {
        "retrieval-only": 1.0,
        "mining-only":    0.0,
        "fused":          args.alpha,
    }

    ground_truths: list[list[str]] = []
    all_preds: dict[str, list[list[str]]] = {m: [] for m in MODES}
    per_case_rows: list[dict] = []

    print(f"[run] Running {args.n_cases} test cases (top_k={args.top_k}) …")
    t_start = time.time()

    for i, case in enumerate(test_cases):
        query = " ".join(case["symptoms"])
        ret_results = retriever.retrieve(query, top_k=max(args.top_k + 5, 15))
        ground_truths.append([case["disease"]])

        case_row: dict = {
            "case_id":       i,
            "true_disease":  case["disease"],
            "symptoms":      ",".join(case["symptoms"]),
        }

        for mode, alpha in MODES.items():
            ranked = reranker.rank(
                query_symptoms=case["symptoms"],
                retrieval_results=ret_results,
                top_k=args.top_k,
                alpha=alpha,
            )
            preds = [r["disease"] for r in ranked]
            all_preds[mode].append(preds)

            # Store top-1 prediction and its fused score
            top1 = ranked[0] if ranked else {}
            case_row[f"{mode}_pred1"]        = top1.get("disease", "")
            case_row[f"{mode}_fused_score"]  = top1.get("fused_score", 0.0)
            case_row[f"{mode}_correct@1"]    = int(
                top1.get("disease", "").lower() == case["disease"].lower()
            )
            # Correct@5
            case_row[f"{mode}_correct@5"] = int(
                any(p.lower() == case["disease"].lower() for p in preds[:5])
            )

        per_case_rows.append(case_row)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  … {i + 1}/{args.n_cases} cases  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t_start
    print(f"[run] Pipeline complete in {total_time:.1f}s ({total_time/args.n_cases:.2f}s/case).\n")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    KS = [1, 3, 5, 10]
    ablation_df = run_ablation(ground_truths, all_preds, ks=KS)
    print_ablation_table(ablation_df)

    # Quick summary of the fused mode
    fused_metrics = ablation_df[ablation_df["mode"] == "fused"].iloc[0]
    print("\n[run] Fused mode headline metrics:")
    for k in KS:
        print(f"  Recall@{k:<3} = {fused_metrics[f'recall@{k}']:.4f}  |  "
              f"Precision@{k:<3} = {fused_metrics[f'precision@{k}']:.4f}  |  "
              f"F1@{k:<3} = {fused_metrics[f'f1@{k}']:.4f}")
    print(f"  MRR       = {fused_metrics['mrr']:.4f}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    per_case_df = pd.DataFrame(per_case_rows)
    exp_path = out_dir / "experiment_results.csv"
    per_case_df.to_csv(exp_path, index=False)
    print(f"\n[run] Per-case results saved to '{exp_path}'.")

    abl_path = out_dir / "ablation_summary.csv"
    ablation_df.to_csv(abl_path, index=False)
    print(f"[run] Ablation summary saved to '{abl_path}'.")

    # Also print per-mode accuracy@1 shortcut
    print("\n[run] Quick accuracy@1 per mode:")
    for mode in MODES:
        col = f"{mode}_correct@1"
        if col in per_case_df.columns:
            acc = per_case_df[col].mean()
            print(f"  {mode:<20} acc@1 = {acc:.4f}")

    print("\n[run] Experiment complete.")
    return ablation_df, per_case_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
