"""
Train a HoeffdingTree on embeddings from the trained MLP (best_acc_model.pt).

Steps:
- Load the trained MLP checkpoint.
- Define a "feature extractor" that returns activations from the 2nd hidden layer (64D).
- Stream through the same training dataset as before.
- Convert regression target to binary: 1 if >0, else 0.
- Train a River HoeffdingTreeClassifier on the embeddings.
- Save the trained tree with pickle.
- Print all rules with their class distributions.
"""

import os
import glob
import pickle
import torch
from torch import nn
import pyarrow.parquet as pq
import yaml
from river import tree

# -----------------------------
# Load config + features
# -----------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = f"{params['DATA_PREP']['PROCESSED_PATH']}_shuffled"
target = params['MODELING']['TARGET']
feature_path = params['DATA_PREP'].get('FEATURE_PATH')
model_path = params['MODELING']['MODEL_PATH']

# Read features
features = []
with open(feature_path, "r") as f:
    for line in f:
        features.append(line.strip())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define the same MLP and load weights
# -----------------------------
hidden1 = params['MODELING'].get('HIDDEN1', 128)
hidden2 = params['MODELING'].get('HIDDEN2', 64)
dropout_p = params['MODELING'].get('DROPOUT', 0.1)

class MLP(nn.Module):
    """MLP architecture matching the training script (Sequential under `net`).

    net.0: Linear(in_dim -> h1)
    net.1: ReLU
    net.2: Dropout
    net.3: Linear(h1 -> h2)
    net.4: Dropout
    net.5: Linear(h2 -> 1)

    The embed() method returns the representation after net.4 (i.e., output of second hidden layer), shape (N, h2).
    """
    def __init__(self, in_dim, h1, h2, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_p),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)

    def embed(self, x):
        # Forward through all but the last linear layer (net[5])
        # Equivalent to: for i in range(5): x = self.net[i](x)
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        return x

checkpoint = torch.load(os.path.join(model_path, "best_acc_model_temp.pt"), map_location=device)

# If checkpoint provides the exact feature order/selection, prefer it to avoid shape mismatches
ckpt_features = checkpoint.get("feature_cols")
if ckpt_features and isinstance(ckpt_features, (list, tuple)):
    if list(ckpt_features) != list(features):
        print("[WARN] Using feature list from checkpoint to match trained model.")
    features = list(ckpt_features)

# If checkpoint provides the hidden config, prefer it to guarantee architectural match
ckpt_cfg = checkpoint.get("config", {})
hidden1 = ckpt_cfg.get('hidden1', hidden1)
hidden2 = ckpt_cfg.get('hidden2', hidden2)
dropout_p = ckpt_cfg.get('dropout', dropout_p)

in_dim = len(features)
model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# Initialize HoeffdingTree
# -----------------------------
ht = tree.HoeffdingTreeClassifier(grace_period=10000, max_depth=8)

# -----------------------------
# Train the HoeffdingTree on embeddings
# -----------------------------
train_files = sorted(glob.glob(os.path.join(processed_path, "train_shard_*.parquet")))

# Progress counters
num_files = len(train_files)
overall_samples = 0

for file_idx, file in enumerate(train_files, start=1):
    print(f"[File {file_idx}/{num_files}] Processing: {os.path.basename(file)}")
    parquet_file = pq.ParquetFile(file)

    processed_in_file = 0
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=2048, columns=features + [target]), start=1):
        # Convert RecordBatch directly to pandas DataFrame
        df = batch.to_pandas()

        X_tensor = torch.tensor(df[features].values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(df[target].values, dtype=torch.float32)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.embed(X_tensor).cpu().numpy()

        # Convert target to binary
        y_binary = (y_tensor >= 0).long().numpy()

        # Feed sample by sample to River HoeffdingTree
        for e, y in zip(embeddings, y_binary):
            sample = {f"emb_{i}": float(val) for i, val in enumerate(e)}
            ht.learn_one(sample, int(y))

        # Progress tracking per batch
        batch_rows = len(df)
        processed_in_file += batch_rows
        overall_samples += batch_rows

    print(f"[Done] {os.path.basename(file)} -> {processed_in_file} rows (total so far: {overall_samples})")

# -----------------------------
# Save the trained tree
# -----------------------------
tree_path = os.path.join(model_path, "hoeffding_tree_temp.pkl")
with open(tree_path, "wb") as f:
    pickle.dump(ht, f)
print(f"Saved HoeffdingTree to {tree_path}")

# Visualization of tree structure
g = ht.draw()
g.render("graph", format="png", view=True)

