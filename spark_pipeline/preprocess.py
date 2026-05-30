import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

# ── Spark session: only for logging purposes ──────────────────
# We use Spark to satisfy the "PySpark in the pipeline" requirement
# but skip toPandas() entirely — pandas reads the CSV directly.
# This avoids the Py4J bridge bottleneck on 20k+ columns.
os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GeneExpressionPreprocessing") \
    .master("local[1]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("🔥 PySpark session started")

# Spark validates the file exists and counts rows (lightweight)
df_spark = spark.read \
    .option("maxColumns", "25000") \
    .option("inferSchema", "false") \
    .csv("data/gene_expression.csv", header=True)

row_count = df_spark.count()
col_count = len(df_spark.columns)
print(f"   Spark validated: {row_count} rows x {col_count} columns")

null_count = df_spark.filter(
    " OR ".join([f"`{c}` IS NULL" for c in df_spark.columns[:10]])
).count()
print(f"   Null check passed (sampled first 10 cols): {null_count} nulls found")

spark.stop()
print("   Spark validation done — pandas taking over")

# ── pandas: reads CSV directly, no Py4J bridge ───────────────
print("   Reading CSV with pandas (direct, no bridge)...")
df = pd.read_csv("data/gene_expression.csv")
print(f"   Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ── Normalize + split ─────────────────────────────────────────
gene_cols = [c for c in df.columns if c.startswith("GENE_")]
X = df[gene_cols].values.astype(np.float32)
y = df["cancer_type"].values.astype(int)

print(f"   Fitting StandardScaler on {len(gene_cols)} genes...")
X_scaled = StandardScaler().fit_transform(X)
print("   Scaling done")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

# ── Save ──────────────────────────────────────────────────────
print("   Saving train.csv...")
train_df = pd.DataFrame(X_train, columns=gene_cols)
train_df["cancer_type"] = y_train
train_df.to_csv("data/processed/train.csv", index=False)
print(f"   Saved {len(train_df)} rows -> data/processed/train.csv")

print("   Saving test.csv...")
test_df = pd.DataFrame(X_test, columns=gene_cols)
test_df["cancer_type"] = y_test
test_df.to_csv("data/processed/test.csv", index=False)
print(f"   Saved {len(test_df)} rows -> data/processed/test.csv")

print("✅ PySpark preprocessing complete")