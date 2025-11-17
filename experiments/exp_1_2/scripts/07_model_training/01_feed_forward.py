"""
Feed-forward neural network training for exp_1_2.

This script:
- Loads experiment config and the engineered feature list
- Streams shuffled Parquet shards for train/validation/test via an IterableDataset
- Builds a simple MLP regressor (configurable hidden sizes/dropout)
- Trains with MSE loss and tracks both validation loss and a sign-accuracy metric
- Saves best checkpoints for lowest validation loss and best validation accuracy
- Logs per-epoch metrics to a log file and saves a PNG plot of Val acc, Val acc rand, and Val loss

Assumptions:
- Shuffled shard files exist under `<PROCESSED_PATH>_shuffled` and are named like
  `{split}_shard_*.parquet` for split in {train, validation, test}.
- The target column name is provided via params['MODELING']['TARGET'] and exists in the shards.
- features.txt contains one feature name per line; these columns exist in the shards.
"""

import glob
import os
import pandas as pd
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import matplotlib
matplotlib.use('Agg')  # headless-safe backend for saving figures
import matplotlib.pyplot as plt

# -----------------------------
# Config + data loading
# -----------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
ticker_list_df = pd.read_csv('../../data/sp500_companies_as_of_jan_2025.csv')

# Load paths and parameters
processed_path = f"{params['DATA_PREP']['PROCESSED_PATH']}_shuffled"
target = params['MODELING']['TARGET']
feature_path = params['DATA_PREP'].get('FEATURE_PATH')
model_path = params['MODELING']['MODEL_PATH']

# Ensure model path exists and set up logging to file (keep prints for console)
os.makedirs(model_path, exist_ok=True)
log_file = os.path.join(model_path, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filename=log_file,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Read features from txt file (one per line)
features = []
with open(feature_path, "r") as f:
    for line in f:
        features.append(line.strip())


class ParquetShardDataset(IterableDataset):
    """Streams batches from a set of Parquet shard files for a given split.

    Args:
        data_dir: Base directory containing *_shuffled parquet files.
        split: One of ['train', 'validation', 'test'].
        features: List of feature column names to load.
        target_col: Column to treat as target.
        batch_size: Number of samples per batch (for Parquet batch iteration).
    """

    def __init__(self, data_dir, split, features, target_col="target", batch_size=32):
        self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_shard_*.parquet")))
        if not self.files:
            raise ValueError(f"No parquet files found for split '{split}' in {data_dir}")
        self.features = features
        self.target_col = target_col
        self.batch_size = batch_size

    def __iter__(self):
        for file in self.files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.features + [self.target_col]):
                table = pa.Table.from_batches([batch])
                df = table.to_pandas()

                # Extract features + target explicitly
                X = torch.tensor(df[self.features].values, dtype=torch.float32)
                y = torch.tensor(df[self.target_col].values, dtype=torch.float32)

                yield X, y


def create_dataloaders(base_path, features, target_col="target", batch_size=32):
    """Create train/val/test DataLoaders from shuffled parquet shards."""
    data_dir = f"{base_path}"

    train_ds = ParquetShardDataset(data_dir, "train", features, target_col, batch_size)
    val_ds   = ParquetShardDataset(data_dir, "validation", features, target_col, batch_size)
    test_ds  = ParquetShardDataset(data_dir, "test", features, target_col, batch_size)

    # batch_size=None because the dataset already yields batches
    train_loader = DataLoader(train_ds, batch_size=None)
    val_loader   = DataLoader(val_ds, batch_size=None)
    test_loader  = DataLoader(test_ds, batch_size=None)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = create_dataloaders(
    processed_path,
    features=features,
    target_col=target,
    batch_size=2048
)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model
# -----------------------------
in_dim = len(features)
hidden1 = params['MODELING'].get('HIDDEN1', 128)
hidden2 = params['MODELING'].get('HIDDEN2', 64)
dropout_p = params['MODELING'].get('DROPOUT', 0.1)


class MLP(nn.Module):
    """Simple 2-hidden-layer MLP regressor."""

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


model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
lr = params['MODELING'].get('LR', 1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=params['MODELING'].get('WEIGHT_DECAY', 1e-4))

# -----------------------------
# Training Loop with Early Stopping
# -----------------------------

epochs = params['MODELING'].get('EPOCHS', 25)
patience = params['MODELING'].get('PATIENCE', 5)
best_val_loss = np.inf
best_val_acc = 0.0
best_val_epoch = 0
best_state = None
best_acc_state = None
no_improve = 0

# Metric histories for plotting at the end
epoch_hist = []
val_acc_hist = []
val_acc_rand_hist = []
val_loss_hist = []

for epoch in range(1, epochs + 1):
    # Train
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        yb = yb.unsqueeze(1)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    yb_binary_sum = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = yb.unsqueeze(1)
            logits = model(xb)

            # transform yb to binary for accuracy calculation
            yb_binary = (yb >= 0).float()
            yb_binary_sum += yb_binary.sum().item()

            val_correct += (logits >= 0).eq(yb_binary).sum().item()
            val_total += yb.numel()

            # MSE Loss
            loss = criterion(logits, yb)
            val_loss += loss.item() / yb.size(0)  # sum the batch loss

    val_acc = val_correct / val_total
    val_acc_rand = yb_binary_sum / val_total
    # Console output remains for real-time feedback
    print(f"Epoch {epoch:03d} | Val acc={val_acc:.12f} | Best val acc={best_val_acc:.12f} | Best val epoch={best_val_epoch:.12f} | Val acc rand={val_acc_rand:.12f} | Val loss={val_loss} | Best val loss={best_val_loss}")
    # Log the same metrics to file
    logger.info(
        f"Epoch {epoch:03d} | Val acc={val_acc:.12f} | Best val acc={best_val_acc:.12f} | Best val epoch={best_val_epoch:.12f} | Val acc rand={val_acc_rand:.12f} | Val loss={val_loss} | Best val loss={best_val_loss}"
    )

    # Save metrics for plotting
    epoch_hist.append(epoch)
    val_acc_hist.append(val_acc)
    val_acc_rand_hist.append(val_acc_rand)
    val_loss_hist.append(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch
        best_acc_state = model.state_dict()
        torch.save({
            "model_state_dict": best_acc_state,
            "in_dim": in_dim,
            "feature_cols": features,
            "config": {"hidden1": hidden1, "hidden2": hidden2, "dropout": dropout_p}
        }, f"{model_path}/best_acc_model.pt")
        logger.info(f"Saved best-accuracy checkpoint at epoch {epoch} -> {model_path}/best_acc_model.pt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        no_improve = 0
        torch.save({
            "model_state_dict": best_state,
            "in_dim": in_dim,
            "feature_cols": features,
            "config": {"hidden1": hidden1, "hidden2": hidden2, "dropout": dropout_p}
        }, f"{model_path}/best_loss_model.pt")
        logger.info(f"Saved best-loss checkpoint at epoch {epoch} -> {model_path}/best_loss_model.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            logger.info(f"Early stopping at epoch {epoch}")
            break

# -----------------------------
# Plot metrics at the end
# -----------------------------
try:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Accuracies
    ax1.plot(epoch_hist, val_acc_hist, label='Val acc', color='tab:blue')
    ax1.plot(epoch_hist, val_acc_rand_hist, label='Val acc rand', color='tab:orange', linestyle='--')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':')

    # Loss
    ax2.plot(epoch_hist, val_loss_hist, label='Val loss', color='tab:red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':')

    fig.suptitle('Validation Metrics')
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))

    metrics_png = os.path.join(model_path, 'training_metrics.png')
    fig.savefig(metrics_png)
    plt.close(fig)
    logger.info(f"Saved training metrics plot to: {metrics_png}")
except Exception as e:
    logger.exception(f"Failed to plot/save training metrics: {e}")
