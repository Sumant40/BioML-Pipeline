import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.makedirs("models", exist_ok=True)
print("🔥 PyTorch model training")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {DEVICE}")

class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):            return len(self.y)
    def __getitem__(self, idx):   return self.X[idx], self.y[idx]

class CancerMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.network(x)

train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")

gene_cols = [c for c in train.columns if c.startswith("GENE_")]

X_train = train[gene_cols].values.astype(np.float32)
y_train = train["cancer_type"].values.astype(np.int64)
X_test  = test[gene_cols].values.astype(np.float32)
y_test  = test["cancer_type"].values.astype(np.int64)

train_loader = DataLoader(GeneDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(GeneDataset(X_test,  y_test),  batch_size=64, shuffle=False)

model     = CancerMLP(X_train.shape[1], 5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(30):
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1:02d}/30 — Loss: {total_loss/len(train_loader):.4f}")

model.eval()
all_probs, correct, total = [], 0, 0
with torch.no_grad():
    for X_b, y_b in test_loader:
        probs = torch.softmax(model(X_b.to(DEVICE)), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        correct += (preds == y_b.numpy()).sum()
        total   += len(y_b)

print(f"\n✅ PyTorch Test Accuracy: {correct/total*100:.1f}%")

torch.save(model.state_dict(), "models/pytorch_model.pth")
np.save("models/pytorch_probs.npy", np.vstack(all_probs))
print("   Saved → models/pytorch_model.pth + pytorch_probs.npy")
