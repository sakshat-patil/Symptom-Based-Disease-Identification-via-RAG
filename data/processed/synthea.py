"""
Exploratory Data Analysis: Synthea EHR & Association Rule Mining
================================================================
This notebook-style script performs end-to-end exploratory analysis:
1. Generates a sample synthetic dataset (simulating Synthea EHR output)
2. Builds symptom-disease transaction baskets
3. Runs FP-Growth to mine association rules
4. Produces summary statistics and visualizations

Usage:
    python notebooks/eda_synthea_mining.py

Output:
    - Console summary statistics
    - Plots saved to notebooks/figures/
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEED = 42
NUM_PATIENTS = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Synthetic EHR Generation (simulating Synthea)
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Generating Synthetic EHR Data")
print("=" * 60)

# Disease-symptom mapping based on common medical knowledge
# Each disease has a set of likely symptoms with probabilities
DISEASE_SYMPTOM_MAP = {
    "Upper Respiratory Infection": {
        "primary": ["Cough", "Fever", "Fatigue", "Sore Throat", "Nasal Congestion"],
        "secondary": ["Headache", "Body Aches", "Chills", "Sneezing"],
    },
    "Pneumonia": {
        "primary": ["Cough", "Fever", "Shortness of Breath", "Chest Pain", "Fatigue"],
        "secondary": ["Chills", "Nausea", "Body Aches", "Rapid Breathing"],
    },
    "Acute Bronchitis": {
        "primary": ["Cough", "Chest Discomfort", "Fatigue", "Sore Throat"],
        "secondary": ["Fever", "Body Aches", "Shortness of Breath", "Wheezing"],
    },
    "Influenza": {
        "primary": ["Fever", "Body Aches", "Cough", "Fatigue", "Headache"],
        "secondary": ["Chills", "Sore Throat", "Nasal Congestion", "Nausea"],
    },
    "Gastroenteritis": {
        "primary": ["Nausea", "Vomiting", "Diarrhea", "Abdominal Pain"],
        "secondary": ["Fever", "Fatigue", "Dehydration", "Loss of Appetite"],
    },
    "Urinary Tract Infection": {
        "primary": ["Painful Urination", "Frequent Urination", "Pelvic Pain"],
        "secondary": ["Fever", "Cloudy Urine", "Fatigue", "Back Pain"],
    },
    "Migraine": {
        "primary": ["Headache", "Nausea", "Light Sensitivity", "Visual Disturbance"],
        "secondary": ["Dizziness", "Fatigue", "Neck Stiffness", "Vomiting"],
    },
    "Hypertension": {
        "primary": ["Headache", "Dizziness", "Shortness of Breath"],
        "secondary": ["Chest Pain", "Fatigue", "Blurred Vision", "Nosebleed"],
    },
    "Type 2 Diabetes": {
        "primary": ["Frequent Urination", "Increased Thirst", "Fatigue", "Blurred Vision"],
        "secondary": ["Slow Healing Wounds", "Tingling Hands", "Weight Loss", "Hunger"],
    },
    "Allergic Rhinitis": {
        "primary": ["Sneezing", "Nasal Congestion", "Itchy Eyes", "Runny Nose"],
        "secondary": ["Headache", "Fatigue", "Sore Throat", "Cough"],
    },
    "Asthma": {
        "primary": ["Wheezing", "Shortness of Breath", "Chest Tightness", "Cough"],
        "secondary": ["Fatigue", "Rapid Breathing", "Anxiety", "Sleep Disturbance"],
    },
    "Sinusitis": {
        "primary": ["Nasal Congestion", "Facial Pain", "Headache", "Thick Nasal Discharge"],
        "secondary": ["Cough", "Fever", "Fatigue", "Sore Throat", "Bad Breath"],
    },
}

# Disease prevalence weights (approximate relative frequency)
DISEASE_WEIGHTS = {
    "Upper Respiratory Infection": 0.18,
    "Influenza": 0.12,
    "Allergic Rhinitis": 0.11,
    "Gastroenteritis": 0.10,
    "Acute Bronchitis": 0.09,
    "Hypertension": 0.09,
    "Urinary Tract Infection": 0.07,
    "Sinusitis": 0.07,
    "Migraine": 0.06,
    "Type 2 Diabetes": 0.05,
    "Asthma": 0.04,
    "Pneumonia": 0.02,
}


def generate_patient_record(patient_id):
    """Generate a single synthetic patient encounter."""
    # Select disease based on prevalence weights
    diseases = list(DISEASE_WEIGHTS.keys())
    weights = list(DISEASE_WEIGHTS.values())
    disease = random.choices(diseases, weights=weights, k=1)[0]

    symptom_info = DISEASE_SYMPTOM_MAP[disease]

    # Select symptoms: most primary symptoms + some secondary
    num_primary = random.randint(2, len(symptom_info["primary"]))
    num_secondary = random.randint(0, min(2, len(symptom_info["secondary"])))

    symptoms = set(random.sample(symptom_info["primary"], num_primary))
    symptoms.update(random.sample(symptom_info["secondary"], num_secondary))

    # Occasionally add a noise symptom (unrelated)
    all_symptoms = set()
    for d in DISEASE_SYMPTOM_MAP.values():
        all_symptoms.update(d["primary"])
        all_symptoms.update(d["secondary"])
    if random.random() < 0.15:
        noise = random.choice(list(all_symptoms - symptoms))
        symptoms.add(noise)

    # Patient demographics
    age = random.randint(18, 85)
    gender = random.choice(["M", "F"])

    return {
        "patient_id": f"P{patient_id:05d}",
        "age": age,
        "gender": gender,
        "disease": disease,
        "symptoms": sorted(symptoms),
        "num_symptoms": len(symptoms),
    }


# Generate dataset
print(f"Generating {NUM_PATIENTS} synthetic patient records...")
records = [generate_patient_record(i) for i in range(NUM_PATIENTS)]
df = pd.DataFrame(records)

print(f"  Total records: {len(df)}")
print(f"  Unique diseases: {df['disease'].nunique()}")
print(f"  Avg symptoms per encounter: {df['num_symptoms'].mean():.2f}")

# Save raw transactions
df.to_csv(os.path.join(DATA_DIR, "synthetic_ehr_records.csv"), index=False)
print(f"  Saved to {DATA_DIR}/synthetic_ehr_records.csv")

# ──────────────────────────────────────────────
# 2. Dataset Statistics & Visualizations
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("STEP 2: Dataset Statistics")
print("=" * 60)

# Disease distribution
print("\nDisease Distribution:")
disease_counts = df["disease"].value_counts()
for disease, count in disease_counts.items():
    pct = count / len(df) * 100
    print(f"  {disease:<35s} {count:>5d}  ({pct:.1f}%)")

# Plot disease distribution
fig, ax = plt.subplots(figsize=(10, 6))
disease_counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(disease_counts)))
ax.set_xlabel("Number of Encounters")
ax.set_title("Disease Frequency Distribution (n={:,})".format(NUM_PATIENTS))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "disease_distribution.png"), dpi=150)
plt.close()
print(f"\n  Saved: {OUTPUT_DIR}/disease_distribution.png")

# Symptom frequency
all_symptom_list = []
for syms in df["symptoms"]:
    all_symptom_list.extend(eval(syms) if isinstance(syms, str) else syms)
symptom_freq = Counter(all_symptom_list)
symptom_df = pd.DataFrame(symptom_freq.most_common(), columns=["Symptom", "Count"])

print(f"\nTop 15 Symptoms:")
for _, row in symptom_df.head(15).iterrows():
    pct = row["Count"] / NUM_PATIENTS * 100
    print(f"  {row['Symptom']:<30s} {row['Count']:>5d}  ({pct:.1f}%)")

# Plot symptom frequency
fig, ax = plt.subplots(figsize=(10, 8))
top_symptoms = symptom_df.head(20)
ax.barh(top_symptoms["Symptom"], top_symptoms["Count"], color=sns.color_palette("mako", 20))
ax.set_xlabel("Frequency")
ax.set_title("Top 20 Symptom Frequencies")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "symptom_frequency.png"), dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/symptom_frequency.png")

# Symptoms per encounter distribution
fig, ax = plt.subplots(figsize=(8, 5))
df["num_symptoms"].hist(bins=range(1, df["num_symptoms"].max() + 2), ax=ax,
                         color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Number of Symptoms")
ax.set_ylabel("Number of Encounters")
ax.set_title("Distribution of Symptoms per Encounter")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "symptoms_per_encounter.png"), dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/symptoms_per_encounter.png")

# Age distribution by disease (top 6)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
top_diseases = disease_counts.head(6).index.tolist()
for i, disease in enumerate(top_diseases):
    ax = axes[i // 3][i % 3]
    subset = df[df["disease"] == disease]["age"]
    ax.hist(subset, bins=20, color=sns.color_palette("Set2")[i], edgecolor="white", alpha=0.85)
    ax.set_title(disease, fontsize=10)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
plt.suptitle("Age Distribution by Disease (Top 6)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_by_disease.png"), dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/age_by_disease.png")


# ──────────────────────────────────────────────
# 3. Build Transaction Baskets & Run FP-Growth
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("STEP 3: FP-Growth Association Rule Mining")
print("=" * 60)

# Build transactions: each transaction is [symptom1, symptom2, ..., DX:disease]
transactions = []
for _, row in df.iterrows():
    syms = eval(row["symptoms"]) if isinstance(row["symptoms"], str) else row["symptoms"]
    items = [f"SX:{s}" for s in syms] + [f"DX:{row['disease']}"]
    transactions.append(items)

print(f"  Built {len(transactions)} transaction baskets")

# One-hot encode
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"  Basket matrix shape: {basket_df.shape}")
print(f"  Symptom items: {sum(1 for c in basket_df.columns if c.startswith('SX:'))}")
print(f"  Diagnosis items: {sum(1 for c in basket_df.columns if c.startswith('DX:'))}")

# Run FP-Growth
print(f"\n  Running FP-Growth (min_support={MIN_SUPPORT})...")
frequent_itemsets = fpgrowth(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
print(f"  Found {len(frequent_itemsets)} frequent itemsets")

# Generate association rules
print(f"  Generating rules (min_confidence={MIN_CONFIDENCE})...")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

# Filter to symptom -> disease rules only
def consequent_is_disease(row):
    return all(item.startswith("DX:") for item in row["consequents"])

def antecedent_is_symptoms(row):
    return all(item.startswith("SX:") for item in row["antecedents"])

mask = rules.apply(consequent_is_disease, axis=1) & rules.apply(antecedent_is_symptoms, axis=1)
sx_dx_rules = rules[mask].copy()

# Add readable columns
sx_dx_rules["symptom_set"] = sx_dx_rules["antecedents"].apply(
    lambda x: ", ".join(sorted(i.replace("SX:", "") for i in x))
)
sx_dx_rules["disease"] = sx_dx_rules["consequents"].apply(
    lambda x: ", ".join(i.replace("DX:", "") for i in x)
)
sx_dx_rules["num_symptoms"] = sx_dx_rules["antecedents"].apply(len)

sx_dx_rules = sx_dx_rules.sort_values("confidence", ascending=False).reset_index(drop=True)

print(f"  Found {len(sx_dx_rules)} symptom -> disease rules")

# Save rules
rules_path = os.path.join(DATA_DIR, "association_rules.csv")
sx_dx_rules[["symptom_set", "disease", "support", "confidence", "lift", "num_symptoms"]].to_csv(
    rules_path, index=False
)
print(f"  Saved rules to {rules_path}")

# ──────────────────────────────────────────────
# 4. Rule Analysis
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("STEP 4: Association Rule Analysis")
print("=" * 60)

# Top 25 rules by confidence
print("\nTop 25 Rules by Confidence:")
print("-" * 90)
print(f"{'#':<4} {'Symptoms':<45} {'Disease':<30} {'Conf':>6} {'Lift':>6}")
print("-" * 90)
for i, row in sx_dx_rules.head(25).iterrows():
    sym_str = row["symptom_set"][:43]
    print(f"{i+1:<4} {sym_str:<45} {row['disease']:<30} {row['confidence']:.3f} {row['lift']:.2f}")

# Rule statistics
print(f"\nRule Statistics:")
print(f"  Total symptom->disease rules: {len(sx_dx_rules)}")
print(f"  Avg confidence: {sx_dx_rules['confidence'].mean():.3f}")
print(f"  Avg lift: {sx_dx_rules['lift'].mean():.2f}")
print(f"  Max confidence: {sx_dx_rules['confidence'].max():.3f}")
print(f"  Rules with lift > 5: {(sx_dx_rules['lift'] > 5).sum()}")
print(f"  Rules with confidence > 0.8: {(sx_dx_rules['confidence'] > 0.8).sum()}")

# Confidence distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(sx_dx_rules["confidence"], bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Confidence")
axes[0].set_ylabel("Number of Rules")
axes[0].set_title("Rule Confidence Distribution")
axes[0].axvline(x=0.8, color="red", linestyle="--", label="0.8 threshold")
axes[0].legend()

axes[1].hist(sx_dx_rules["lift"], bins=30, color="#FF9800", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Lift")
axes[1].set_ylabel("Number of Rules")
axes[1].set_title("Rule Lift Distribution")
axes[1].axvline(x=1.0, color="red", linestyle="--", label="lift = 1")
axes[1].legend()

axes[2].scatter(sx_dx_rules["confidence"], sx_dx_rules["lift"],
                alpha=0.5, c="#2196F3", edgecolors="white", linewidth=0.3, s=30)
axes[2].set_xlabel("Confidence")
axes[2].set_ylabel("Lift")
axes[2].set_title("Confidence vs Lift")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rule_analysis.png"), dpi=150)
plt.close()
print(f"\n  Saved: {OUTPUT_DIR}/rule_analysis.png")

# Rules per disease
rules_per_disease = sx_dx_rules.groupby("disease").agg(
    num_rules=("confidence", "count"),
    avg_confidence=("confidence", "mean"),
    max_confidence=("confidence", "max"),
    avg_lift=("lift", "mean"),
).sort_values("num_rules", ascending=False)

print("\nRules per Disease:")
print("-" * 75)
print(f"{'Disease':<35} {'Rules':>6} {'Avg Conf':>9} {'Max Conf':>9} {'Avg Lift':>9}")
print("-" * 75)
for disease, row in rules_per_disease.iterrows():
    print(f"{disease:<35} {row['num_rules']:>6.0f} {row['avg_confidence']:>9.3f} "
          f"{row['max_confidence']:>9.3f} {row['avg_lift']:>9.2f}")

# Heatmap: symptom co-occurrence with diseases
print("\nGenerating symptom-disease co-occurrence heatmap...")
top_syms = symptom_df.head(15)["Symptom"].tolist()
top_dis = disease_counts.head(8).index.tolist()

cooccurrence = pd.DataFrame(0, index=top_syms, columns=top_dis)
for _, row in df.iterrows():
    syms = eval(row["symptoms"]) if isinstance(row["symptoms"], str) else row["symptoms"]
    disease = row["disease"]
    if disease in top_dis:
        for s in syms:
            if s in top_syms:
                cooccurrence.loc[s, disease] += 1

# Normalize by disease count
for col in cooccurrence.columns:
    cooccurrence[col] = cooccurrence[col] / disease_counts[col]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cooccurrence, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Symptom-Disease Co-occurrence Matrix (Normalized by Disease Frequency)")
ax.set_ylabel("Symptom")
ax.set_xlabel("Disease")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "symptom_disease_heatmap.png"), dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/symptom_disease_heatmap.png")

# ──────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)
print(f"  Patients generated:       {NUM_PATIENTS:,}")
print(f"  Unique diseases:          {df['disease'].nunique()}")
print(f"  Unique symptoms:          {len(symptom_freq)}")
print(f"  Frequent itemsets found:  {len(frequent_itemsets)}")
print(f"  Association rules mined:  {len(sx_dx_rules)}")
print(f"  High-confidence (>0.8):   {(sx_dx_rules['confidence'] > 0.8).sum()}")
print(f"  High-lift (>5):           {(sx_dx_rules['lift'] > 5).sum()}")
print(f"\n  Figures saved to:  {OUTPUT_DIR}/")
print(f"  Data saved to:     {DATA_DIR}/")
print(f"\n{'=' * 60}")
print("Exploratory analysis complete.")
