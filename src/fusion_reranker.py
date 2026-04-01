"""The actual fusion step: combine the two scores into one ranking.

We pool candidates from BOTH signals -- a disease that only the rule
side knows about still appears in the output (with retrieval_score=0)
and vice versa. Score is a one-parameter convex combination:

    FusedScore(d) = alpha * RetrievalSim(d) + (1 - alpha) * MiningConf(d)

Tie-break is deterministic: fused desc, then mining desc, then retrieval
desc, then disease name asc. Mining wins ties because we trust its
top-1 precision more than retrieval's.

The Check-in 4 default of alpha=0.6 was wrong (we picked it before we'd
done the sweep). The sweep peaks at alpha ∈ [0.1, 0.4]; we ship 0.3.

Owner: Aishwarya. Tie-break order was the part we debated longest.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedCandidate:
    disease: str
    fused_score: float
    retrieval_score: float
    mining_score: float


def fuse(retrieval_scores: dict[str, float], mining_scores: dict[str, float],
         alpha: float = 0.3) -> list[FusedCandidate]:
    universe = set(retrieval_scores) | set(mining_scores)
    out: list[FusedCandidate] = []
    for d in universe:
        r = float(retrieval_scores.get(d, 0.0))
        m = float(mining_scores.get(d, 0.0))
        f = alpha * r + (1 - alpha) * m
        out.append(FusedCandidate(disease=d, fused_score=f, retrieval_score=r, mining_score=m))
    out.sort(key=lambda c: (-c.fused_score, -c.mining_score, -c.retrieval_score, c.disease))
    return out
