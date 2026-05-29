import numpy as np
from sklearn.metrics import classification_report

CANCER_NAMES = {0: "BRCA (Breast)", 1: "LUAD (Lung)", 2: "PRAD (Prostate)"}

print("🔬 Ensemble prediction")

tf_probs      = np.load("models/tf_probs.npy")
pytorch_probs = np.load("models/pytorch_probs.npy")
true_labels   = np.load("models/test_labels.npy")

ensemble_probs = (tf_probs + pytorch_probs) / 2.0
ensemble_preds = np.argmax(ensemble_probs, axis=1)
tf_preds       = np.argmax(tf_probs,       axis=1)
pytorch_preds  = np.argmax(pytorch_probs,  axis=1)

def acc(p, t): return (p == t).mean() * 100

print(f"\n   TensorFlow  accuracy : {acc(tf_preds,       true_labels):.1f}%")
print(f"   PyTorch     accuracy : {acc(pytorch_preds,  true_labels):.1f}%")
print(f"   Ensemble    accuracy : {acc(ensemble_preds, true_labels):.1f}%")

print("\n📊 Classification report (Ensemble):")
print(classification_report(
    true_labels, ensemble_preds,
    target_names=[CANCER_NAMES[i] for i in range(3)]
))

print("🧬 Sample predictions (first 10 test patients):")
print(f"{'#':<4} {'True':<22} {'Predicted':<22} {'OK?'}")
print("─" * 55)
for i in range(10):
    t = CANCER_NAMES[true_labels[i]]
    p = CANCER_NAMES[ensemble_preds[i]]
    ok = "✅" if true_labels[i] == ensemble_preds[i] else "❌"
    print(f"{i:<4} {t:<22} {p:<22} {ok}")
