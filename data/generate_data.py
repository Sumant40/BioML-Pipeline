"""
generate_data.py — REAL DATA version
Replaces the synthetic data generator.

Reads the UCI PANCAN RNA-Seq dataset:
  data/data.csv   — 801 patients × 20531 gene expression values
  data/labels.csv — 801 cancer type labels

Cancer types in the dataset:
  BRCA — Breast carcinoma
  KIRC — Kidney renal clear-cell carcinoma
  COAD — Colon adenocarcinoma
  LUAD — Lung adenocarcinoma
  PRAD — Prostate adenocarcinoma

What this script does:
  1. Loads both files
  2. Merges them into one DataFrame
  3. Encodes cancer type strings → integers (0-4)
  4. Saves a single gene_expression.csv identical in shape
     to what the synthetic version produced — so ALL
     downstream scripts (preprocess.py, train_tf.py, etc.)
     work with zero changes
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Load ────────────────────────────────────────────────────
print("📂 Loading real PANCAN RNA-Seq data...")

data   = pd.read_csv("data/data.csv",   index_col=0)   # shape: (801, 20531)
labels = pd.read_csv("data/labels.csv", index_col=0)   # shape: (801, 1)

print(f"   data.csv   : {data.shape[0]} patients × {data.shape[1]} genes")
print(f"   labels.csv : {labels.shape[0]} labels")

# ── Rename columns to match our GENE_XXXX convention ────────
# Original columns are named gene_0, gene_1, ... gene_20530
# We rename to GENE_00000 ... GENE_20530 so preprocess.py
# picks them up with its startswith("GENE_") filter unchanged
data.columns = [f"GENE_{i:05d}" for i in range(data.shape[1])]

# ── Encode labels ────────────────────────────────────────────
# Map cancer type strings to integers
CANCER_MAP = {
    "BRCA": 0,   # Breast
    "KIRC": 1,   # Kidney
    "COAD": 2,   # Colon
    "LUAD": 3,   # Lung
    "PRAD": 4,   # Prostate
}

# labels.csv column is called "Class"
labels.columns = ["cancer_type_name"]
labels["cancer_type"] = labels["cancer_type_name"].map(CANCER_MAP)

# ── Merge & shuffle ──────────────────────────────────────────
df = data.copy()
df["cancer_type"] = labels["cancer_type"].values

# Shuffle rows so classes aren't grouped together
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sanity checks
assert df["cancer_type"].isna().sum() == 0, "Unmapped cancer labels found!"
assert df.shape == (801, 20532), f"Unexpected shape: {df.shape}"

# ── Save ────────────────────────────────────────────────────
out_path = "data/gene_expression.csv"
df.to_csv(out_path, index=False)

print(f"\n✅ Saved {df.shape[0]} patients × {df.shape[1]-1} genes → {out_path}")
print(f"\n   Cancer type distribution:")

name_map = {v: k for k, v in CANCER_MAP.items()}
for code, count in df["cancer_type"].value_counts().sort_index().items():
    print(f"     {code} = {name_map[code]:<6}  {count} patients")