"""Generate the figures embedded in the IEEE report.

Outputs:
    data/results/alpha_sweep.png          (Recall@1, Recall@10, MRR vs alpha)
    data/results/rules_per_disease.png    (bar chart, top-N diseases by rule count)
    data/results/ablation_bar.png         (grouped bar: Recall@1/5/10 per variant)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PROCESSED = ROOT / "data" / "processed"

sns.set_theme(style="whitegrid", context="paper")


def alpha_sweep_plot():
    src = RESULTS / "alpha_sweep.csv"
    if not src.exists():
        print(f"[plots] missing {src}; skipping")
        return
    df = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for col, marker in [("recall@1", "o"), ("recall@10", "s"), ("mrr", "^")]:
        ax.plot(df["alpha"], df[col], marker=marker, label=col)
    ax.set_xlabel("α (retrieval weight)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fusion sensitivity to α (PubMedBERT + synonyms)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = RESULTS / "alpha_sweep.png"
    fig.savefig(out, dpi=180)
    print(f"[plots] wrote {out}")
    plt.close(fig)


def rules_per_disease_plot(top_n: int = 25):
    src = PROCESSED / "association_rules.csv"
    if not src.exists():
        print(f"[plots] missing {src}; skipping")
        return
    df = pd.read_csv(src)
    counts = df.groupby("consequent").size().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.values, y=[d.replace("_", " ") for d in counts.index],
                ax=ax, color="#3a86ff")
    ax.set_xlabel("# mined association rules")
    ax.set_ylabel("")
    ax.set_title(f"Top-{top_n} diseases by FP-Growth rule count")
    fig.tight_layout()
    out = RESULTS / "rules_per_disease.png"
    fig.savefig(out, dpi=180)
    print(f"[plots] wrote {out}")
    plt.close(fig)


def ablation_bar_plot():
    src = RESULTS / "ablation_summary.csv"
    if not src.exists():
        print(f"[plots] missing {src}; skipping")
        return
    df = pd.read_csv(src)
    df = df.copy()
    df["label"] = df["variant"].astype(str) + " / " + df["mode"].astype(str)
    metrics = ["recall@1", "recall@5", "recall@10"]
    n = len(df)
    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[m], width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: retrieval / mining / fused across variants")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    out = RESULTS / "ablation_bar.png"
    fig.savefig(out, dpi=180)
    print(f"[plots] wrote {out}")
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    alpha_sweep_plot()
    rules_per_disease_plot()
    ablation_bar_plot()


if __name__ == "__main__":
    main()
