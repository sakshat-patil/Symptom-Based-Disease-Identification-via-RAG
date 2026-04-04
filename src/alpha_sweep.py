"""
Alpha grid search for the hybrid fusion reranker.

Runs the full evaluation pipeline over a range of alpha values and reports
the Recall@K, Precision@K, F1@K and MRR for each setting. Useful for picking
the best retrieval vs. mining weight for our dataset.

Usage:
    python src/alpha_sweep.py --rules data/processed/association_rules.csv \
        --n_cases 200 --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

Outputs:
    data/results/alpha_sweep.csv  — one row per alpha value
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Alpha grid search for hybrid fusion reranker."
    )
    parser.add_argument("--rules",   default=None,            help="Association rules CSV.")
    parser.add_argument("--corpus",  default=None,            help="JSONL passages for retrieval.")
    parser.add_argument("--n_cases", type=int, default=200,   help="Number of synthetic test cases.")
    parser.add_argument("--top_k",   type=int, default=10,    help="Ranking cut-off.")
    parser.add_argument("--alphas",  default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values to try.")
    parser.add_argument("--out_dir", default="data/results",  help="Output directory.")
    parser.add_argument("--seed",    type=int, default=42,    help="Random seed.")
    return parser.parse_args()


def main():
    args = _parse_args()
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))

    from src.retrieval       import initialise as init_retriever
    from src.fusion_reranker import FusionReranker
    from src.evaluation      import (
        generate_test_cases,
        evaluate_batch,
        DISEASE_SYMPTOM_MAP,
    )

    alphas = [float(a) for a in args.alphas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Alpha sweep over {len(alphas)} values: {alphas}")
    print("=" * 70)

    # Build retriever once
    retriever = init_retriever(corpus_path=args.corpus)

    # Build reranker once (alpha will be overridden per run)
    reranker = FusionReranker(rules_path=args.rules, alpha=0.5)

    # Generate test cases once so all alpha values see the same queries
    test_cases = generate_test_cases(
        n=args.n_cases,
        disease_symptom_map=DISEASE_SYMPTOM_MAP,
        seed=args.seed,
    )

    # Pre-compute retrieval results per case so we only encode once
    cached_retrieval = []
    for case in test_cases:
        query_str = " ".join(case["symptoms"])
        cached_retrieval.append(retriever.retrieve(query_str, top_k=15))

    # Sweep
    rows = []
    for alpha in alphas:
        ground_truths = []
        all_preds = []
        for case, retrieval_results in zip(test_cases, cached_retrieval):
            ranked = reranker.rank(
                query_symptoms=case["symptoms"],
                retrieval_results=retrieval_results,
                top_k=args.top_k,
                alpha=alpha,
            )
            preds = [r["disease"] for r in ranked]
            ground_truths.append([case["true_disease"]])
            all_preds.append(preds)

        metrics = evaluate_batch(ground_truths, all_preds)
        metrics["alpha"] = alpha
        rows.append(metrics)
        print(
            f"  alpha={alpha:.2f} | R@1={metrics['recall@1']:.3f} "
            f"R@5={metrics['recall@5']:.3f} R@10={metrics['recall@10']:.3f} "
            f"MRR={metrics['mrr']:.3f}"
        )

    df = pd.DataFrame(rows)
    cols = ["alpha"] + [c for c in df.columns if c != "alpha"]
    df = df[cols]
    out_path = out_dir / "alpha_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote sweep summary to {out_path}")

    # Highlight best alpha per metric
    best_r1   = df.loc[df["recall@1"].idxmax()]
    best_r10  = df.loc[df["recall@10"].idxmax()]
    best_mrr  = df.loc[df["mrr"].idxmax()]
    print("\nBest settings:")
    print(f"  Recall@1   : alpha={best_r1['alpha']:.2f}  (R@1={best_r1['recall@1']:.3f})")
    print(f"  Recall@10  : alpha={best_r10['alpha']:.2f}  (R@10={best_r10['recall@10']:.3f})")
    print(f"  MRR        : alpha={best_mrr['alpha']:.2f}  (MRR={best_mrr['mrr']:.3f})")


if __name__ == "__main__":
    main()
