"""FP-Growth pattern mining for {symptom set} -> disease rules.

Each transaction goes in as a basket of symptom tokens plus a `DX:<disease>`
item; FP-Growth from mlxtend handles candidate generation, then we filter
for rules whose consequent is exactly one DX: item and whose antecedent is
all symptom tokens. The DX: prefix is what lets FP-Growth treat the
diagnosis as just another item without polluting symptom co-occurrence.

We dropped min_support from 0.01 (Check-in 3) to 0.005 here -- the higher
threshold left 21 of 41 diseases without any rule, which made fusion
useless for those classes. 0.005 covers all 41 at the cost of a noisier
rule list; we mitigate that downstream with the overlap weighting in
mining_scorer.MiningScorer.score().

Output: data/processed/association_rules.csv with one rule per row.

Owner: Sakshat. Threshold debate happened in our Check-in 3 retro.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TX = ROOT / "data" / "processed" / "transactions.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "association_rules.csv"

DX_PREFIX = "DX:"


def build_baskets(tx_csv: Path) -> list[list[str]]:
    df = pd.read_csv(tx_csv).fillna("")
    baskets: list[list[str]] = []
    for _, r in df.iterrows():
        items = [s for s in r["symptoms"].split("|") if s]
        items.append(f"{DX_PREFIX}{r['condition']}")
        baskets.append(items)
    return baskets


def mine(baskets: list[list[str]], min_support: float, min_confidence: float) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    df = pd.DataFrame(arr, columns=te.columns_)

    freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)

    keep_rows = []
    for _, r in rules.iterrows():
        ante = set(r["antecedents"])
        cons = set(r["consequents"])
        if len(cons) != 1:
            continue
        cons_item = next(iter(cons))
        if not cons_item.startswith(DX_PREFIX):
            continue
        if any(it.startswith(DX_PREFIX) for it in ante):
            continue
        keep_rows.append({
            "antecedent": "|".join(sorted(ante)),
            "consequent": cons_item[len(DX_PREFIX):],
            "support": float(r["support"]),
            "confidence": float(r["confidence"]),
            "lift": float(r["lift"]),
            "antecedent_len": len(ante),
        })
    return pd.DataFrame(keep_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", default=str(DEFAULT_TX))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min_support", type=float, default=0.005)
    p.add_argument("--min_confidence", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    baskets = build_baskets(Path(args.transactions))
    rules = mine(baskets, args.min_support, args.min_confidence)
    rules = rules.sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)
    rules.to_csv(out, index=False)

    print(f"[mining] wrote {out}")
    print(f"[mining] rules: {len(rules)}  diseases covered: {rules['consequent'].nunique()}")
    if len(rules):
        print(f"[mining] median confidence: {rules['confidence'].median():.3f}")
        print(f"[mining] mean lift: {rules['lift'].mean():.3f}")
        top = rules.iloc[0]
        print(f"[mining] top rule: {{{top['antecedent']}}} -> {top['consequent']} "
              f"(conf={top['confidence']:.3f}, lift={top['lift']:.2f})")


if __name__ == "__main__":
    main()
