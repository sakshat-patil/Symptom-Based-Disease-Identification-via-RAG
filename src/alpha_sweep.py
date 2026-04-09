"""Sweep alpha in {0.0, 0.1, ..., 1.0} for the best retrieval variant."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluation import generate_test_cases, recall_at_k, reciprocal_rank
from .fusion_reranker import fuse
from .mining_scorer import MiningScorer
from .retrieval import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", default=str(PROCESSED / "association_rules.csv"))
    p.add_argument("--transactions", default=str(PROCESSED / "transactions.csv"))
    p.add_argument("--backend", default="pubmedbert")
    p.add_argument("--expand_synonyms", action="store_true", default=True)
    p.add_argument("--n_cases", type=int, default=200)
    p.add_argument("--out", default=str(RESULTS / "alpha_sweep.csv"))
    args = p.parse_args()

    miner = MiningScorer.from_file(Path(args.rules))
    cases = generate_test_cases(Path(args.transactions), n=args.n_cases, seed=42)
    diseases = miner.diseases | set(pd.read_csv(args.transactions)["condition"].unique())
    retr = DenseRetriever.from_files(args.backend, diseases)

    rows = []
    print(f"[sweep] backend={args.backend} synonyms={args.expand_synonyms} "
          f"n_cases={args.n_cases}")
    for alpha in [round(x * 0.1, 1) for x in range(11)]:
        sums = {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
        for case in cases:
            retr_scores = ({} if alpha == 0.0 else
                           retr.retrieve(case.symptoms, top_k=15,
                                          expand_synonyms=args.expand_synonyms)[0])
            mine_scores = ({} if alpha == 1.0 else miner.score(case.symptoms))
            ranked = [c.disease for c in fuse(retr_scores, mine_scores, alpha=alpha)]
            for k in (1, 3, 5, 10):
                sums[f"recall@{k}"] += recall_at_k(case.true_disease, ranked, k)
            sums["mrr"] += reciprocal_rank(case.true_disease, ranked)
        avg = {k: v / len(cases) for k, v in sums.items()}
        rows.append({"alpha": alpha, **avg})
        print(f"  alpha={alpha}: {avg}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
