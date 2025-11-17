"""
Evaluate a trained scikit-learn DecisionTree on validation and test splits.
- Loads the MLP checkpoint and uses it to compute embeddings from raw features.
- Loads as much data as possible (memory-wise) from *_shard_*.parquet (like 02_decision_tree_scikit.py).
- Computes overall metrics (accuracy, confusion matrix) for validation and test.
- Prints tree stats (like evaluation.py): exports per-node CSV and prints per-node subset stats using decision rules.
"""

import os
import sys
import glob
import pickle
import yaml
import psutil
import numpy as np
import pandas as pd
import torch
from torch import nn
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, confusion_matrix

# Make tree_utilities importable from the training scripts folder (cannot use package import due to numeric folder name)
THIS_DIR = os.path.dirname(__file__)
UTILS_DIR = os.path.abspath(os.path.join(THIS_DIR, "../07_model_training"))
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)
from tree_utilities import get_tree_stats, get_decision_path, apply_decision_rules  # noqa: E402

# -----------------------------
# Config and paths
# -----------------------------
params = yaml.safe_load(open(os.path.join(THIS_DIR, "../../conf/params.yaml")))
processed_path = f"{params['DATA_PREP']['PROCESSED_PATH']}_shuffled"
target = params['MODELING']['TARGET']
feature_path = params['DATA_PREP'].get('FEATURE_PATH')
model_path = params['MODELING']['MODEL_PATH']
os.makedirs(model_path, exist_ok=True)

# Read features list
features = []
with open(feature_path, "r") as f:
    for line in f:
        feat = line.strip()
        if feat:
            features.append(feat)

# -----------------------------
# Define MLP and load checkpoint for embeddings
# -----------------------------
hidden1 = params['MODELING'].get('HIDDEN1', 128)
hidden2 = params['MODELING'].get('HIDDEN2', 64)
dropout_p = params['MODELING'].get('DROPOUT', 0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
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
        x = self.net[0](x); x = self.net[1](x); x = self.net[2](x)
        x = self.net[3](x); x = self.net[4](x)
        return x

checkpoint = torch.load(os.path.join(model_path, "best_acc_model_temp.pt"), map_location=device)
ckpt_features = checkpoint.get("feature_cols")
if ckpt_features and isinstance(ckpt_features, (list, tuple)):
    features = list(ckpt_features)
ckpt_cfg = checkpoint.get("config", {})
hidden1 = ckpt_cfg.get('hidden1', hidden1)
hidden2 = ckpt_cfg.get('hidden2', hidden2)
dropout_p = ckpt_cfg.get('dropout', dropout_p)

in_dim = len(features)
mlp = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)
mlp.load_state_dict(checkpoint["model_state_dict"])
mlp.eval()

# -----------------------------
# Load trained scikit Decision Tree
# -----------------------------
model_file = os.path.join(model_path, "decision_tree_scikit.pkl")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}. Train it first with 02_decision_tree_scikit.py.")
with open(model_file, "rb") as f:
    bundle = pickle.load(f)
clf = bundle["model"]
embed_dim = int(bundle.get("embed_dim", hidden2))
emb_cols = [f"emb_{i}" for i in range(embed_dim)]

# -----------------------------
# Memory budget and loaders
# -----------------------------
init_avail = psutil.virtual_memory().available
limit_bytes = int(min(8 * (1024**3), init_avail * 0.60))  # a bit more conservative for eval
print(f"[MEM] Available: {init_avail/1e9:.2f} GB, Budget (per split): {limit_bytes/1e9:.2f} GB")

def load_split_to_df(split: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(processed_path, f"{split}_shard_*.parquet")))
    if len(files) == 0:
        print(f"[WARN] No files found for split '{split}' in {processed_path}")
        return pd.DataFrame(columns=emb_cols + ["target", "target_binary"])

    X_chunks, y_cont_chunks, y_bin_chunks = [], [], []
    bytes_used = 0
    rows = 0
    for file_idx, file in enumerate(files, start=1):
        pq_file = pq.ParquetFile(file)
        for batch_idx, batch in enumerate(pq_file.iter_batches(batch_size=2048, columns=features + [target]), start=1):
            df = batch.to_pandas()
            if df.empty:
                continue
            X_t = torch.tensor(df[features].values, dtype=torch.float32, device=device)
            with torch.no_grad():
                emb = mlp.embed(X_t).detach().cpu().numpy().astype(np.float32, copy=False)
            y_cont = df[target].values.astype(np.float32, copy=False)
            y_bin = (df[target].values >= 0).astype(np.int8, copy=False)

            batch_bytes = emb.nbytes + y_cont.nbytes + y_bin.nbytes
            if bytes_used + batch_bytes > limit_bytes:
                remaining = max(0, limit_bytes - bytes_used)
                per_row_bytes = emb.shape[1] * emb.itemsize + y_cont.itemsize + y_bin.itemsize
                n_can_add = int(remaining // per_row_bytes)
                if n_can_add > 0:
                    X_chunks.append(emb[:n_can_add])
                    y_cont_chunks.append(y_cont[:n_can_add])
                    y_bin_chunks.append(y_bin[:n_can_add])
                    bytes_used += n_can_add * per_row_bytes
                    rows += n_can_add
                    print(f"[MEM] {split}: Added partial rows={n_can_add} (budget filled).")
                print(f"[MEM] {split}: Budget reached at file {file_idx}, batch {batch_idx}.")
                break
            else:
                X_chunks.append(emb)
                y_cont_chunks.append(y_cont)
                y_bin_chunks.append(y_bin)
                bytes_used += batch_bytes
                rows += len(df)
        else:
            continue
        break
    if rows == 0:
        print(f"[WARN] {split}: No rows loaded within memory budget.")
        return pd.DataFrame(columns=emb_cols + ["target", "target_binary"])

    X = np.concatenate(X_chunks, axis=0)
    y_cont = np.concatenate(y_cont_chunks, axis=0)
    y_bin = np.concatenate(y_bin_chunks, axis=0)
    df_out = pd.DataFrame(X, columns=emb_cols)
    df_out["target"] = y_cont
    df_out["target_binary"] = y_bin
    print(f"[LOAD] {split}: rows={len(df_out)}, used≈{bytes_used/1e9:.2f} GB")
    return df_out

# -----------------------------
# Load validation and test
# -----------------------------
val_df = load_split_to_df("validation")
test_df = load_split_to_df("test")

# -----------------------------
# Tree stats and per-node rule evaluation (like evaluation.py)
# -----------------------------
# Compute and save per-node stats
stats_df = get_tree_stats(clf, feature_cols=emb_cols)
# Filter to class==1 nodes and sort by impurity desc (similar to provided evaluation.py)
stats_pos = stats_df[stats_df['class'] == 1].sort_values(by='impurity', ascending=False)

# Save to experiment's data folder
data_out_dir = os.path.abspath(os.path.join(THIS_DIR, "../../data"))
os.makedirs(data_out_dir, exist_ok=True)
stats_csv = os.path.join(data_out_dir, "tree_stats.csv")
stats_pos.to_csv(stats_csv, index=False)
print(f"[STATS] Saved node stats (class==1) to: {stats_csv}")

node_ids = stats_pos['node_id'].tolist()

# Helper to print subset stats for a split

def print_subset_stats(df: pd.DataFrame, split_name: str):
    if df.empty:
        print(f"[RULES] {split_name}: no data loaded.")
        return
    for node_id in node_ids[:200]:  # cap to avoid too verbose output
        rules = get_decision_path(clf, emb_cols, node_id)
        df_f = apply_decision_rules(df, rules)
        n = len(df_f)
        mean_bin = float(df_f['target_binary'].mean()) if n > 0 else 0.0
        mean_cont = float(df_f['target'].mean()) if n > 0 else 0.0
        print(f"Node {node_id}: rules={rules} | {split_name} samples={n} | mean_bin={mean_bin:.4f} | mean_target={mean_cont:.6f} | n_rules={len(rules)}")

# New: compute and store subset stats per node as DataFrame(s)

def compute_subset_stats_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cols = [
        "split",
        "node_id",
        "n_samples",
        "mean_target_binary",
        "mean_target",
        "n_rules",
        "rules",
    ]
    if df.empty:
        print(f"[RULES] {split_name}: no data loaded (returning empty stats df).")
        return pd.DataFrame(columns=cols)

    rows = []
    for node_id in node_ids:  # compute for all selected nodes
        rules = get_decision_path(clf, emb_cols, node_id)
        df_f = apply_decision_rules(df, rules)
        n = int(len(df_f))
        mean_bin = float(df_f['target_binary'].mean()) if n > 0 else 0.0
        mean_cont = float(df_f['target'].mean()) if n > 0 else 0.0
        rows.append({
            "split": split_name,
            "node_id": node_id,
            "n_samples": n,
            "mean_target_binary": mean_bin,
            "mean_target": mean_cont,
            "n_rules": len(rules),
            "rules": rules,
        })
    df_stats = pd.DataFrame(rows, columns=cols)
    return df_stats

# Print a concise console view (capped) and save full DataFrames
print_subset_stats(val_df, "validation")
print_subset_stats(test_df, "test")

val_node_stats = compute_subset_stats_df(val_df, "validation")
val_stats_csv = os.path.join(data_out_dir, "node_subset_stats_validation.csv")
val_node_stats.to_csv(val_stats_csv, index=False)
print(f"[STATS] Saved validation per-node subset stats to: {val_stats_csv}")

test_node_stats = compute_subset_stats_df(test_df, "test")
test_stats_csv = os.path.join(data_out_dir, "node_subset_stats_test.csv")
test_node_stats.to_csv(test_stats_csv, index=False)
print(f"[STATS] Saved test per-node subset stats to: {test_stats_csv}")

# Optional: also save a combined view for convenience
both_node_stats = pd.concat([val_node_stats, test_node_stats], ignore_index=True)
both_stats_csv = os.path.join(data_out_dir, "node_subset_stats_combined.csv")
both_node_stats.to_csv(both_stats_csv, index=False)
print(f"[STATS] Saved combined per-node subset stats to: {both_stats_csv}")

# -----------------------------
# Overall metrics (accuracy, confusion matrix) for validation and test
# -----------------------------
if not val_df.empty:
    y_true_v = val_df["target_binary"].astype(int).values
    y_pred_v = clf.predict(val_df[emb_cols].values)
    acc_v = accuracy_score(y_true_v, y_pred_v)
    cm_v = confusion_matrix(y_true_v, y_pred_v)
    print(f"[VAL] accuracy={acc_v:.4f}\n[VAL] confusion_matrix=\n{cm_v}")
else:
    print("[VAL] No data for validation metrics.")

if not test_df.empty:
    y_true_t = test_df["target_binary"].astype(int).values
    y_pred_t = clf.predict(test_df[emb_cols].values)
    acc_t = accuracy_score(y_true_t, y_pred_t)
    cm_t = confusion_matrix(y_true_t, y_pred_t)
    print(f"[TEST] accuracy={acc_t:.4f}\n[TEST] confusion_matrix=\n{cm_t}")
else:
    print("[TEST] No data for test metrics.")

print("[DONE] Evaluation completed.")
