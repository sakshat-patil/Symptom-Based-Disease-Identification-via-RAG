"""End-to-end experiment runner.

Modes:
    retrieval-only  (alpha = 1.0)
    mining-only     (alpha = 0.0)
    fused           (alpha = configurable, default 0.3)

Variants of the retrieval backbone:
    minilm                          -- baseline, no synonym expansion
    minilm + synonyms               -- adds clinical synonym expansion
    pubmedbert                      -- biomedical encoder
    pubmedbert + synonyms           -- biomedical encoder + synonym expansion

Saves data/results/ablation_summary.csv with a row per (variant, mode).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .evaluation import (TestCase, generate_test_cases, recall_at_k,
                          reciprocal_rank)
from .fusion_reranker import fuse
from .mining_scorer import MiningScorer
from .retrieval import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def evaluate_mode(cases: list[TestCase], retriever: DenseRetriever | None,
                   miner: MiningScorer, alpha: float, expand_synonyms: bool) -> dict[str, float]:
    n = len(cases)
    sums = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    for case in cases:
        if alpha == 0.0:
            retr_scores: dict[str, float] = {}
        else:
            assert retriever is not None
            retr_scores, _ = retriever.retrieve(case.symptoms, top_k=15,
                                                 expand_synonyms=expand_synonyms)
        if alpha == 1.0:
            mine_scores: dict[str, float] = {}
        else:
            mine_scores = miner.score(case.symptoms)
        ranked = [c.disease for c in fuse(retr_scores, mine_scores, alpha=alpha)]
        for k in (1, 3, 5, 10):
            sums[f"recall@{k}"] += recall_at_k(case.true_disease, ranked, k)
        sums["mrr"] += reciprocal_rank(case.true_disease, ranked)
    return {k: v / n for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", default=str(PROCESSED / "association_rules.csv"))
    p.add_argument("--transactions", default=str(PROCESSED / "transactions.csv"))
    p.add_argument("--n_cases", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--variants", nargs="+",
                   default=["minilm", "minilm+syn", "pubmedbert", "pubmedbert+syn"])
    p.add_argument("--out", default=str(RESULTS / "ablation_summary.csv"))
    args = p.parse_args()

    miner = MiningScorer.from_file(Path(args.rules))
    cases = generate_test_cases(Path(args.transactions), n=args.n_cases, seed=42)
    diseases = miner.diseases | set(pd.read_csv(args.transactions)["condition"].unique())

    rows = []
    cache: dict[str, DenseRetriever] = {}

    def get_retr(name: str) -> DenseRetriever:
        if name not in cache:
            cache[name] = DenseRetriever.from_files(name, diseases)
        return cache[name]

    print(f"[run] {len(cases)} test cases, {len(diseases)} disease universe, alpha={args.alpha}")

    # mining-only is the same regardless of retrieval variant; compute once
    mining_only = evaluate_mode(cases, None, miner, alpha=0.0, expand_synonyms=False)
    rows.append({"variant": "(mining-only)", "mode": "mining-only", **mining_only})
    print(f"  mining-only: {mining_only}")

    for variant in args.variants:
        if "+syn" in variant:
            backend_name, expand = variant.replace("+syn", ""), True
        else:
            backend_name, expand = variant, False
        retr = get_retr(backend_name)
        # retrieval-only
        ret_only = evaluate_mode(cases, retr, miner, alpha=1.0, expand_synonyms=expand)
        rows.append({"variant": variant, "mode": "retrieval-only", **ret_only})
        print(f"  {variant} retrieval-only: {ret_only}")
        # fused
        fused = evaluate_mode(cases, retr, miner, alpha=args.alpha, expand_synonyms=expand)
        rows.append({"variant": variant, "mode": f"fused(a={args.alpha})", **fused})
        print(f"  {variant} fused(a={args.alpha}): {fused}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[run] wrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
