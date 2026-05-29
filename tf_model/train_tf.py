import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.makedirs("models", exist_ok=True)
print("🧠 TensorFlow model training")

train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")

gene_cols = [c for c in train.columns if c.startswith("GENE_")]

X_train = train[gene_cols].values.astype(np.float32)
y_train = train["cancer_type"].values
X_test  = test[gene_cols].values.astype(np.float32)
y_test  = test["cancer_type"].values

print(f"   Train: {X_train.shape}  Test: {X_test.shape}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=30, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ TensorFlow Test Accuracy: {accuracy*100:.1f}%")

model.save("models/tf_model.keras")
np.save("models/tf_probs.npy",    model.predict(X_test, verbose=0))
np.save("models/test_labels.npy", y_test)
print("   Saved → models/tf_model.keras + tf_probs.npy")
