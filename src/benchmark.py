"""End-to-end latency benchmark.

Reports per-stage and total wall-clock latency over a sample of queries:
    1. mining_score   -- FP-Growth rule lookup
    2. retrieval      -- bi-encoder + FAISS top-K
    3. cross_encode   -- optional re-rank of top-K
    4. fuse           -- linear combination
    5. explain        -- template or LLM explanation for the top-1 disease

Outputs data/results/latency_summary.csv with mean/p50/p95 in milliseconds.

Sakshat ran these on the M3 Pro to confirm we hit the < 1s/query target on
the default configuration (mining + bi-encoder + fusion + template
explainer); the cross-encoder and LLM paths are slower but still
sub-second.
"""
from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .clinical_explanation import (TemplateClinicalExplainer,
                                       get_clinical_explainer)
from .cross_encoder_rerank import CrossEncoderReranker
from .disease_keywords import DISEASE_KEYWORDS
from .evaluation import generate_test_cases
from .evidence import cards_for_disease
from .fusion_reranker import fuse
from .mining_scorer import MiningScorer
from .retrieval import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"


@contextmanager
def timer(stage: str, sink: dict[str, list[float]]):
    t = time.perf_counter()
    yield
    sink.setdefault(stage, []).append((time.perf_counter() - t) * 1000.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_cases", type=int, default=50)
    p.add_argument("--backend", default="minilm")
    p.add_argument("--with_cross_encoder", action="store_true")
    p.add_argument("--with_llm", action="store_true",
                   help="Use the OpenAI clinical explainer instead of the template "
                        "stitch (requires OPENAI_API_KEY)")
    p.add_argument("--alpha", type=float, default=0.3)
    args = p.parse_args()

    miner = MiningScorer.from_file(PROCESSED / "association_rules.csv")
    diseases = set(pd.read_csv(PROCESSED / "transactions.csv")["condition"].unique())
    retr = DenseRetriever.from_files(args.backend, diseases)
    rerank = CrossEncoderReranker.load() if args.with_cross_encoder else None
    # Use the same clinical explainer the live API uses, so the benchmark
    # numbers are comparable to production latency.
    explainer = (get_clinical_explainer("openai") if args.with_llm
                 else TemplateClinicalExplainer())

    cases = generate_test_cases(PROCESSED / "transactions.csv",
                                  n=args.n_cases, seed=42)

    timings: dict[str, list[float]] = {}
    print(f"[bench] backend={args.backend} cases={len(cases)} "
          f"cross_encoder={bool(rerank)} explainer={explainer.name}")

    for case in cases:
        with timer("mining_score", timings):
            mscore = miner.score(case.symptoms)
        with timer("retrieval", timings):
            rscore, rpassages = retr.retrieve(case.symptoms, top_k=15,
                                                expand_synonyms=True)
        if rerank is not None:
            top_d = max(rscore.items(), key=lambda kv: kv[1], default=("", 0.0))[0]
            ps = rpassages.get(top_d, [])
            if ps:
                with timer("cross_encode", timings):
                    rerank.rerank("symptoms: " + ", ".join(case.symptoms),
                                   ps[:15], top_k=10)
        with timer("fuse", timings):
            ranked = fuse(rscore, mscore, alpha=args.alpha)
        with timer("explain", timings):
            top = ranked[0] if ranked else None
            if top is not None:
                rules = miner.matching_rules(case.symptoms, top.disease, top_n=3)
                # Build evidence cards from the top passages so the
                # explainer call has the same input shape it does at
                # inference time in the live API.
                cards = cards_for_disease(
                    rpassages.get(top.disease, []),
                    query_symptoms=case.symptoms,
                    disease_keywords=DISEASE_KEYWORDS.get(top.disease, []),
                    max_cards=5)
                explainer.explain(
                    disease=top.disease,
                    query_symptoms=case.symptoms,
                    matching_rules=rules,
                    evidence_cards=cards,
                )

    rows = []
    for stage, vals in timings.items():
        if not vals:
            continue
        rows.append({
            "stage": stage,
            "n": len(vals),
            "mean_ms": round(statistics.mean(vals), 2),
            "p50_ms": round(statistics.median(vals), 2),
            "p95_ms": round(statistics.quantiles(vals, n=20)[-1] if len(vals) >= 20
                              else max(vals), 2),
            "max_ms": round(max(vals), 2),
        })
    n_cases = len(cases)
    totals = []
    for i in range(n_cases):
        s = 0.0
        for vals in timings.values():
            if i < len(vals):
                s += vals[i]
        totals.append(s)
    rows.append({
        "stage": "TOTAL",
        "n": len(totals),
        "mean_ms": round(statistics.mean(totals), 2),
        "p50_ms": round(statistics.median(totals), 2),
        "p95_ms": round(statistics.quantiles(totals, n=20)[-1] if len(totals) >= 20
                          else max(totals), 2),
        "max_ms": round(max(totals), 2),
    })

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "latency_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[bench] wrote {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
