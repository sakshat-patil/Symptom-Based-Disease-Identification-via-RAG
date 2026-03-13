"""Query-time scorer over the mined rules.

For an input symptom set Q, we score each disease as:

    MiningConf(Q, d) = max over rules (A -> d) with A ⊆ Q of
                          conf(A -> d) * (|A ∩ Q| / |A|)

The overlap factor is the bit we kept fiddling with. Without it, a 5-symptom
rule that matched 5/5 of Q got the same score as a 1-symptom rule that
matched. With it, longer rules that match more of Q win, which is what we
actually want clinically.

Indexed by disease so the per-query lookup is cheap (~1 ms over our 23,839
rules). matching_rules() is what the UI uses to surface the supporting rule.

Owner: Sakshat (algorithm) + Aishwarya (UI integration).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "data" / "processed" / "association_rules.csv"


@dataclass
class MiningScorer:
    rules_by_disease: dict[str, list[tuple[frozenset, float, float]]]

    @classmethod
    def from_file(cls, path: Path = DEFAULT_RULES) -> "MiningScorer":
        df = pd.read_csv(path)
        out: dict[str, list[tuple[frozenset, float, float]]] = defaultdict(list)
        for _, r in df.iterrows():
            ante = frozenset(s for s in str(r["antecedent"]).split("|") if s)
            out[r["consequent"]].append((ante, float(r["confidence"]), float(r["lift"])))
        return cls(rules_by_disease=dict(out))

    def score(self, symptoms: list[str]) -> dict[str, float]:
        q = set(symptoms)
        out: dict[str, float] = {}
        for disease, ruleset in self.rules_by_disease.items():
            best = 0.0
            for ante, conf, _lift in ruleset:
                if not ante.issubset(q):
                    continue
                overlap = len(ante & q) / max(1, len(ante))
                s = conf * overlap
                if s > best:
                    best = s
            if best > 0:
                out[disease] = best
        return out

    def matching_rules(self, symptoms: list[str], disease: str, top_n: int = 3) -> list[dict]:
        q = set(symptoms)
        matches = []
        for ante, conf, lift in self.rules_by_disease.get(disease, []):
            if not ante.issubset(q):
                continue
            matches.append({
                "antecedent": sorted(ante),
                "confidence": conf,
                "lift": lift,
                "size": len(ante),
            })
        matches.sort(key=lambda m: (m["confidence"], m["size"], m["lift"]), reverse=True)
        return matches[:top_n]

    @property
    def diseases(self) -> set[str]:
        return set(self.rules_by_disease.keys())
