"""
Evaluate HoeffdingTreeClassifier on validation and test sets,
including per-leaf performance measures.

Steps:
- Load trained MLP checkpoint (best_acc_model.pt).
- Load trained HoeffdingTreeClassifier (hoeffding_tree.pkl).
- Stream validation and test datasets, compute 64D embeddings.
- Predict with the tree and compare to binary target (y>=0).
- Print overall accuracy and confusion matrix.
- Print per-leaf performance (n_samples, class distribution, accuracy).
"""

import os
import glob
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.metrics import accuracy_score, confusion_matrix
import hashlib

# -----------------------------
# Config and paths
# -----------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = f"{params['DATA_PREP']['PROCESSED_PATH']}_shuffled"
target = params['MODELING']['TARGET']
feature_path = params['DATA_PREP'].get('FEATURE_PATH')
model_path = params['MODELING']['MODEL_PATH']

# Load features list
features = []
with open(feature_path, "r") as f:
    for line in f:
        features.append(line.strip())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Define MLP with embed() function
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

# Load checkpoint
checkpoint = torch.load(os.path.join(model_path, "best_acc_model_temp.pt"), map_location=device)
in_dim = len(features)
model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------------
# Load trained HoeffdingTree
# -----------------------------
tree_path = os.path.join(model_path, "hoeffding_tree_temp.pkl")
with open(tree_path, "rb") as f:
    ht = pickle.load(f)

# Also get node structure for leaf mapping (for reporting totals)
df = ht.to_dataframe().reset_index()
df = df.rename(columns={df.columns[0]: "node_id"})
leaf_total = int(df.loc[df["is_leaf"], "node_id"].shape[0])

# Helper to get a stable-ish leaf key from a traversed node
def _leaf_key(leaf_node):
    # Try common attribute names first
    for attr in ("node_id", "id", "nid", "nodeid"):
        if hasattr(leaf_node, attr):
            try:
                val = getattr(leaf_node, attr)
                # Convert to str to ensure hashability/serialization
                return str(val)
            except Exception:
                pass
    # Fallback: repr of the node object (may include class probabilities and is unstable across runs)
    return repr(leaf_node)

# Prefer stable IDs derived from the debug path

def _normalize_debug_path(s: str) -> str:
    if s is None:
        return ""
    # collapse whitespace to make the string canonical
    return " ".join(str(s).strip().split())


def _leaf_id_from_debug(s: str) -> str:
    norm = _normalize_debug_path(s)
    if not norm:
        return None
    # Short, stable identifier derived from the path
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
    return f"leaf_{h}"

# -----------------------------
# Helper to stream data
# -----------------------------
def stream_split(split, batch_size=2048):
    files = sorted(glob.glob(os.path.join(processed_path, f"{split}_shard_*.parquet")))
    print(f"[stream] Split '{split}': {len(files)} files found.")
    for file_idx, file in enumerate(files, start=1):
        print(f"[stream] {split}: reading file {file_idx}/{len(files)} -> {os.path.basename(file)}")
        parquet_file = pq.ParquetFile(file)
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=features + [target]), start=1):
            table = pa.Table.from_batches(batches=[batch])
            df_batch = table.to_pandas()
            X_tensor = torch.tensor(df_batch[features].values, dtype=torch.float32).to(device)
            with torch.no_grad():
                emb = model.embed(X_tensor).cpu().numpy()
            y = (df_batch[target].values >= 0).astype(int)
            #print(f"[stream] {split}: file {file_idx}/{len(files)} batch {batch_idx}: rows={len(df_batch)}")
            yield emb, y

# -----------------------------
# Evaluation with per-leaf stats
# -----------------------------
def evaluate_split(split):
    print(f"[eval] Start evaluating split='{split}' ...")
    y_true, y_pred = [], []
    # accumulate per-leaf stats dynamically
    per_leaf = {}

    total = 0
    for X_emb, y in stream_split(split):
        for xi, yi in zip(X_emb, y):
            sample = {f"emb_{i}": float(xi[i]) for i in range(len(xi))}
            pred = ht.predict_one(sample)
            # Try to derive a stable leaf id from the debug path; fall back to node attributes if needed
            leaf_id_key = None
            path_trace = None
            # Prefer path-based stable id
            try:
                path_trace = ht.debug_one(sample)
                if path_trace:
                    leaf_id_key = _leaf_id_from_debug(path_trace)
            except Exception:
                path_trace = None
            # Fallback: traverse to leaf node and try known id attributes
            if leaf_id_key is None:
                try:
                    leaf_node = ht._root.traverse(sample)
                    leaf_id_key = _leaf_key(leaf_node)
                except Exception:
                    pass

            y_true.append(int(yi))
            y_pred.append(int(pred))

            if leaf_id_key is not None:
                if leaf_id_key not in per_leaf:
                    per_leaf[leaf_id_key] = {"n": 0, "true": [], "pred": [], "path": None}
                per_leaf[leaf_id_key]["n"] += 1
                per_leaf[leaf_id_key]["true"].append(int(yi))
                per_leaf[leaf_id_key]["pred"].append(int(pred))
                # Store a representative path for this leaf once (normalized for readability)
                if per_leaf[leaf_id_key]["path"] is None and path_trace is not None:
                    per_leaf[leaf_id_key]["path"] = _normalize_debug_path(path_trace)
            total += 1
        if total > 0 and total % 50000 == 0:
            non_empty = sum(1 for d in per_leaf.values() if d["n"] > 0)
            print(f"[eval] {split}: processed {total} samples, non-empty leaves={non_empty}")

    # Overall
    if len(y_true) == 0:
        print(f"[eval] {split}: no samples found. Returning empty metrics.")
        acc = 0.0
        cm = np.zeros((2, 2), dtype=int)
    else:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
    print(f"[eval] {split}: done. total={len(y_true)}, acc={acc:.4f}")

    # Per-leaf summary
    leaf_summary = []
    for nid_key, d in per_leaf.items():
        if d["n"] > 0:
            acc_leaf = accuracy_score(d["true"], d["pred"]) if len(d["true"]) > 0 else 0.0
            dist = {
                0: int(np.sum(np.array(d["true"]) == 0)),
                1: int(np.sum(np.array(d["true"]) == 1))
            }
            leaf_summary.append({
                "node_id": nid_key,
                "n_samples": d["n"],
                "accuracy": acc_leaf,
                "distribution": dist,
                "path": d.get("path")
            })

    if len(leaf_summary) == 0:
        leaf_df = pd.DataFrame(columns=["node_id", "n_samples", "accuracy", "distribution", "path"])
    else:
        leaf_df = pd.DataFrame(leaf_summary).sort_values("n_samples", ascending=False)

    print(f"[eval] {split}: leaves with samples={len(leaf_df)} / {leaf_total}")
    return acc, cm, leaf_df

# -----------------------------
# Run evaluation
# -----------------------------
val_acc, val_cm, val_leaf = evaluate_split("validation")
test_acc, test_cm, test_leaf = evaluate_split("test")

print("Validation accuracy:", val_acc)
print("Validation confusion matrix:\n", val_cm)
print("\nPer-leaf validation performance:")
print(val_leaf.head(10))  # top 10 by sample count

print("\nTest accuracy:", test_acc)
print("Test confusion matrix:\n", test_cm)
print("\nPer-leaf test performance:")
print(test_leaf.head(10))  # top 10 by sample count

# -----------------------------
# Flatten and save per-leaf DataFrames
# -----------------------------

def _flatten_leaf_df(df_in: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if df_in is None or len(df_in) == 0:
        return pd.DataFrame(columns=[
            "split", "node_id", "n_samples", "accuracy",
            "n_class_0", "n_class_1", "ratio_class_0", "ratio_class_1", "path"
        ])
    out_rows = []
    for _, row in df_in.iterrows():
        dist = row.get("distribution", {}) or {}
        n0 = int(dist.get(0, 0))
        n1 = int(dist.get(1, 0))
        n = int(row.get("n_samples", n0 + n1) or 0)
        r0 = (n0 / n) if n > 0 else 0.0
        r1 = (n1 / n) if n > 0 else 0.0
        out_rows.append({
            "split": split_name,
            "node_id": row.get("node_id"),
            "n_samples": n,
            "accuracy": float(row.get("accuracy", 0.0) or 0.0),
            "n_class_0": n0,
            "n_class_1": n1,
            "ratio_class_0": r0,
            "ratio_class_1": r1,
            "path": row.get("path")
        })
    return pd.DataFrame(out_rows).sort_values(["n_samples", "accuracy"], ascending=[False, False])

val_leaf_flat = _flatten_leaf_df(val_leaf, "validation")
test_leaf_flat = _flatten_leaf_df(test_leaf, "test")

# Save CSVs under the experiment's data folder
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(_base_dir, exist_ok=True)
val_csv = os.path.join(_base_dir, "per_leaf_validation.csv")
test_csv = os.path.join(_base_dir, "per_leaf_test.csv")
val_leaf_flat.to_csv(val_csv, index=False)
test_leaf_flat.to_csv(test_csv, index=False)

print(f"Saved per-leaf validation CSV to: {val_csv}")
print(f"Saved per-leaf test CSV to: {test_csv}")
