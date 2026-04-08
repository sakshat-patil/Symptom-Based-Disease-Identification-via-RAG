"""
FP-Growth frequent pattern mining on symptom-condition transactions.

Reads the transaction table produced by etl.py, runs FP-Growth to find
frequent symptom itemsets, generates association rules, and saves them to
data/processed/association_rules.csv.

Rule format:
    symptom_set  — pipe-separated antecedent symptoms
    disease      — consequent (condition label)
    support      — fraction of transactions containing the full itemset
    confidence   — P(disease | symptoms)
    lift         — confidence / P(disease)

Usage:
    python src/mining.py
    python src/mining.py --transactions data/processed/transactions.csv \
        --min_support 0.01 --min_confidence 0.5 --out data/processed/association_rules.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    """Load the transaction table written by etl.py.

    Parameters
    ----------
    csv_path:
        Path to transactions CSV with columns: patient_id, condition, symptoms.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"patient_id", "condition", "symptoms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transactions CSV missing columns: {missing}")
    print(f"[mining] Loaded {len(df)} transactions ({df['condition'].nunique()} unique conditions).")
    return df


def build_item_matrix(transactions_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert symptom strings into a one-hot encoded boolean DataFrame.

    Each row is a transaction; each column is a unique symptom token.
    The *condition* label is included as a column so we can mine
    symptom → disease rules in one pass.

    Parameters
    ----------
    transactions_df:
        DataFrame with at least ``symptoms`` (pipe-separated) and
        ``condition`` columns.

    Returns
    -------
    (one_hot_df, item_list)
        one_hot_df — bool DataFrame suitable for mlxtend
        item_list  — list of all item names (symptoms + condition labels)
    """
    records = []
    for _, row in transactions_df.iterrows():
        syms = [s.strip() for s in str(row["symptoms"]).split("|") if s.strip()]
        # Prefix condition with "DX:" so we can identify it later
        condition_item = f"DX:{str(row['condition']).strip()}"
        records.append(syms + [condition_item])

    te = TransactionEncoder()
    te_array = te.fit_transform(records)
    one_hot = pd.DataFrame(te_array, columns=te.columns_)
    return one_hot, list(te.columns_)


def run_fpgrowth(
    transactions_df: pd.DataFrame,
    min_support: float = 0.005,
) -> pd.DataFrame:
    """Run FP-Growth on the transaction table.

    Parameters
    ----------
    transactions_df:
        Output of ``load_transactions()``.
    min_support:
        Minimum support threshold (fraction of transactions).

    Returns
    -------
    pd.DataFrame  — mlxtend frequent_itemsets with ``support`` and
                    ``itemsets`` columns.
    """
    one_hot, _ = build_item_matrix(transactions_df)
    print(f"[mining] Running FP-Growth (min_support={min_support}, "
          f"{one_hot.shape[1]} unique items, {len(one_hot)} transactions) …")
    freq_itemsets = fpgrowth(one_hot, min_support=min_support, use_colnames=True)
    print(f"[mining] Found {len(freq_itemsets)} frequent itemsets.")
    return freq_itemsets


def generate_association_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets.

    Filters to rules where the *consequent* is exactly one disease item
    (prefixed ``"DX:"``), so every rule has the form:
        {symptom_1, …, symptom_N} → {DX:disease}

    Parameters
    ----------
    frequent_itemsets:
        Output of ``run_fpgrowth()``.
    min_confidence:
        Minimum confidence threshold.
    min_lift:
        Minimum lift threshold (default 1.0 = any positive association).

    Returns
    -------
    pd.DataFrame  columns: symptom_set, disease, support, confidence, lift
    """
    if frequent_itemsets.empty:
        print("[mining] No frequent itemsets — returning empty rules table.")
        return pd.DataFrame(columns=["symptom_set", "disease", "support", "confidence", "lift"])

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    rules = rules[rules["lift"] >= min_lift]

    # Keep only rules where consequent is a single disease item
    disease_rules = rules[
        rules["consequents"].apply(
            lambda c: len(c) == 1 and next(iter(c)).startswith("DX:")
        )
    ].copy()

    if disease_rules.empty:
        print("[mining] No disease-consequent rules found after filtering.")
        return pd.DataFrame(columns=["symptom_set", "disease", "support", "confidence", "lift"])

    # Filter out antecedents that contain disease items
    disease_rules = disease_rules[
        disease_rules["antecedents"].apply(
            lambda a: not any(item.startswith("DX:") for item in a)
        )
    ].copy()

    # Build clean output table
    out = pd.DataFrame({
        "symptom_set": disease_rules["antecedents"].apply(
            lambda a: "|".join(sorted(a))
        ),
        "disease": disease_rules["consequents"].apply(
            lambda c: next(iter(c))[3:]  # strip "DX:" prefix
        ),
        "support":    disease_rules["support"].round(6).values,
        "confidence": disease_rules["confidence"].round(6).values,
        "lift":       disease_rules["lift"].round(6).values,
    })

    out = out.sort_values("confidence", ascending=False).reset_index(drop=True)
    print(f"[mining] Generated {len(out)} association rules "
          f"({out['disease'].nunique()} unique diseases).")
    return out


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_mining(
    transactions_path: str | Path = "data/processed/transactions.csv",
    min_support: float = 0.005,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    out_path: str | Path = "data/processed/association_rules.csv",
    fallback_synthetic: bool = True,
) -> pd.DataFrame:
    """Full mining pipeline: load → FP-Growth → rules → save.

    Parameters
    ----------
    transactions_path:
        CSV produced by etl.py.
    min_support, min_confidence, min_lift:
        FP-Growth / rule-generation thresholds.
    out_path:
        Where to write the rules CSV.
    fallback_synthetic:
        If the transactions file doesn't exist, generate synthetic data
        so the script still runs standalone.

    Returns
    -------
    pd.DataFrame  — association rules table.
    """
    transactions_path = Path(transactions_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or generate transactions
    if transactions_path.exists():
        df = load_transactions(transactions_path)
    elif fallback_synthetic:
        print(f"[mining] '{transactions_path}' not found — generating synthetic transactions.")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.etl import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_patients=2000)
    else:
        raise FileNotFoundError(f"Transactions file not found: '{transactions_path}'")

    # FP-Growth
    freq_itemsets = run_fpgrowth(df, min_support=min_support)

    # Rules
    rules = generate_association_rules(freq_itemsets, min_confidence=min_confidence,
                                       min_lift=min_lift)

    # Save
    rules.to_csv(out_path, index=False)
    print(f"[mining] Rules saved to '{out_path}'.")
    return rules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="FP-Growth association rule mining.")
    parser.add_argument("--transactions", default="data/processed/transactions.csv",
                        help="Input transaction table CSV.")
    parser.add_argument("--min_support",    type=float, default=0.005,
                        help="Minimum support threshold (lowered from 0.01 to improve "
                             "rule coverage across the 21 previously uncovered diseases).")
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--min_lift",       type=float, default=1.0)
    parser.add_argument("--out", default="data/processed/association_rules.csv",
                        help="Output rules CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    args = _parse_args()

    print("=" * 60)
    print("FP-Growth Association Rule Mining")
    print("=" * 60)

    rules = run_mining(
        transactions_path=args.transactions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        out_path=args.out,
    )

    if not rules.empty:
        print("\nTop 10 rules by confidence:")
        print(rules.head(10).to_string(index=False))

        print(f"\nRule statistics:")
        print(f"  Total rules:       {len(rules)}")
        print(f"  Unique diseases:   {rules['disease'].nunique()}")
        print(f"  Avg confidence:    {rules['confidence'].mean():.4f}")
        print(f"  Avg lift:          {rules['lift'].mean():.4f}")
        print(f"  Max lift:          {rules['lift'].max():.4f}")
