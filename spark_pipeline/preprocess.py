import os

os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

spark = SparkSession.builder \
    .appName("GeneExpressionPreprocessing") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("🔥 PySpark session started")

df = spark.read.csv("data/gene_expression.csv", header=True, inferSchema=True)
print(f"   Loaded: {df.count()} rows x {len(df.columns)} columns")

df = df.dropna()

gene_cols = [c for c in df.columns if c.startswith("GENE_")]

assembler = VectorAssembler(inputCols=gene_cols, outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                           withMean=True, withStd=True)

pipeline       = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(df)
df_scaled      = pipeline_model.transform(df)

train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)

def save_partition(spark_df, path):
    rows     = spark_df.select("features", "cancer_type").collect()
    features = [r["features"].toArray() for r in rows]
    labels   = [r["cancer_type"] for r in rows]
    out      = pd.DataFrame(features, columns=gene_cols)
    out["cancer_type"] = labels
    out.to_csv(path, index=False)
    print(f"   Saved {len(out)} rows → {path}")

save_partition(train_df, "data/processed/train.csv")
save_partition(test_df,  "data/processed/test.csv")

spark.stop()
print("✅ PySpark preprocessing complete")
