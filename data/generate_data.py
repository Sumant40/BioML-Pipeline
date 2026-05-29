import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

np.random.seed(42)

N_SAMPLES  = 900
N_GENES    = 500
N_CLASSES  = 3

samples, labels = [], []

for cancer_type in range(N_CLASSES):
    mean_expression = np.random.uniform(0.5, 2.0, N_GENES)
    for _ in range(N_SAMPLES // N_CLASSES):
        expression = np.random.normal(mean_expression, 0.3)
        samples.append(expression)
        labels.append(cancer_type)

gene_columns = [f"GENE_{i:04d}" for i in range(N_GENES)]
df = pd.DataFrame(samples, columns=gene_columns)
df["cancer_type"] = labels
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/gene_expression.csv", index=False)

print(f"✅ Generated {len(df)} samples x {N_GENES} genes → data/gene_expression.csv")
print(f"   Class distribution:\n{df['cancer_type'].value_counts().to_string()}")
