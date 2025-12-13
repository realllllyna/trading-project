"""
LSTM-Training für kurzfristige Volatilitätsvorhersage auf Sequenz-Shards (.npz).

- Liest Sequenzen aus SEQUENCE_PATH (train/validation/test_shard_*.npz)
- Verwendet nur eine begrenzte Anzahl Shards pro Split (laptop-freundlich)
- Nutzt Features aus features.txt und Hyperparameter aus params.yaml
"""

from __future__ import annotations

import glob
import os
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from metrics import classification_metrics


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_feature_list() -> list[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(this_dir, "../03_pre_split_prep/features.txt")
    with open(feat_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Dataset mit Lazy Loading und begrenzter Shard-Anzahl
# ---------------------------------------------------------
class LSTMSequenceDataset(Dataset):
    """
    Dataset, das eine Liste von npz-Shards lazy lädt.
    Es hält immer nur einen Shard im RAM (Cache).
    """

    def __init__(self, shard_files: List[str], split_name: str = ""):
        """
        shard_files: Liste von Pfaden zu *.npz-Dateien für diesen Split
        split_name: nur für Logging (z.B. 'train', 'validation', 'test')
        """
        if not shard_files:
            raise ValueError(f"Keine Shards übergeben für Split '{split_name}'")

        self.files = sorted(shard_files)

        # Shardgrößen + Klassenverteilung bestimmen (nur y laden)
        self.shard_sizes = []
        self.cum_sizes = []
        total = 0
        total_pos = 0.0
        total_neg = 0.0

        print(f"[{split_name}] Analysiere {len(self.files)} Shards ...")
        for fpath in self.files:
            data = np.load(fpath, allow_pickle=True)
            y = data["y"].astype(np.float32)
            n = y.shape[0]
            self.shard_sizes.append(n)
            total += n
            self.cum_sizes.append(total)

            pos = float(y.sum())
            neg = float(n - pos)
            total_pos += pos
            total_neg += neg

        self.class_weights = None
        if total_pos > 0 and total_neg > 0:
            w_pos = total_neg / (total_pos + total_neg)
            w_neg = total_pos / (total_pos + total_neg)
            self.class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)

        print(
            f"[{split_name}] {total} Sequenzen, "
            f"Positives: {int(total_pos)}, Negatives: {int(total_neg)}, "
            f"Shards: {len(self.files)}"
        )

        # Cache
        self._cache_shard_idx = None
        self._cache_X = None
        self._cache_y = None

    def __len__(self) -> int:
        return self.cum_sizes[-1]

    def _load_shard(self, shard_idx: int):
        if self._cache_shard_idx == shard_idx:
            return
        fpath = self.files[shard_idx]
        data = np.load(fpath, allow_pickle=True)
        self._cache_X = data["X"].astype(np.float32)
        self._cache_y = data["y"].astype(np.float32)
        self._cache_shard_idx = shard_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # globalen Index auf (shard_idx, local_idx) abbilden
        shard_idx = 0
        while idx >= self.cum_sizes[shard_idx]:
            shard_idx += 1

        prev_cum = 0 if shard_idx == 0 else self.cum_sizes[shard_idx - 1]
        local_idx = idx - prev_cum

        self._load_shard(shard_idx)

        x_np = self._cache_X[local_idx]  # [T, F]
        y_np = self._cache_y[local_idx]  # Skalar

        x = torch.from_numpy(x_np)
        y = torch.tensor(y_np, dtype=torch.float32)
        return x, y


# ---------------------------------------------------------
# Modell
# ---------------------------------------------------------
class LSTMVolModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]          # [B, H]
        h = self.dropout(h_last)
        logit = self.fc(h).squeeze(-1)  # [B]
        return logit


# ---------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------
def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion,
        optimizer=None,
):
    model.train(mode=optimizer is not None)
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y_batch.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = classification_metrics(y_true, y_prob)

    return avg_loss, metrics


def main():
    print("Starte LSTM-Training ...")

    params = load_params()
    feature_cols = load_feature_list()
    seq_path = params["SEQUENCE_BUILD"]["SEQUENCE_PATH"]
    model_cfg = params["MODEL"]

    input_size = len(feature_cols)
    hidden_size = model_cfg.get("HIDDEN_SIZE", 64)
    num_layers = model_cfg.get("NUM_LAYERS", 2)
    dropout = model_cfg.get("DROPOUT", 0.1)
    lr = model_cfg.get("LEARNING_RATE", 1e-3)
    batch_size = model_cfg.get("BATCH_SIZE", 1024)
    epochs = model_cfg.get("EPOCHS", 20)
    patience = model_cfg.get("PATIENCE", 10)
    save_dir = model_cfg.get("SAVE_PATH", "../../models")
    os.makedirs(save_dir, exist_ok=True)

    device_str = model_cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Features: {input_size}")

    set_seed(42)

    # ----- Shards finden -----
    train_files_all = sorted(
        glob.glob(os.path.join(seq_path, "train", "train_shard_*.npz"))
    )
    val_files_all = sorted(
        glob.glob(os.path.join(seq_path, "validation", "validation_shard_*.npz"))
    )
    test_files_all = sorted(
        glob.glob(os.path.join(seq_path, "test", "test_shard_*.npz"))
    )

    print(f"Gefundene Shards: train={len(train_files_all)}, val={len(val_files_all)}, test={len(test_files_all)}")

    if not train_files_all or not val_files_all or not test_files_all:
        raise FileNotFoundError("Mindestens ein Split hat keine Shards. Bitte Schritt 4 prüfen.")

    # Begrenzung: wir nehmen erstmal nur Teilmenge
    MAX_TRAIN_SHARDS = 10
    MAX_VAL_SHARDS = 3
    MAX_TEST_SHARDS = 3

    train_files = train_files_all[:MAX_TRAIN_SHARDS]
    val_files = val_files_all[:MAX_VAL_SHARDS]
    test_files = test_files_all[:MAX_TEST_SHARDS]

    print("Verwende für dieses Experiment:")
    print("  Train-Shards:", [os.path.basename(f) for f in train_files])
    print("  Val-Shards:  ", [os.path.basename(f) for f in val_files])
    print("  Test-Shards: ", [os.path.basename(f) for f in test_files])

    # ----- Datasets / Loader -----
    train_ds = LSTMSequenceDataset(train_files, split_name="train")
    val_ds = LSTMSequenceDataset(val_files, split_name="validation")
    test_ds = LSTMSequenceDataset(test_files, split_name="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ----- Modell & Optimizer -----
    model = LSTMVolModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    if train_ds.class_weights is not None:
        class_weights = train_ds.class_weights.to(device)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Verwende pos_weight={pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----- Training mit Early Stopping -----
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_metrics = run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, device, criterion, optimizer=None
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            print("  → Bestes Modell aktualisiert.")
        else:
            epochs_no_improve += 1
            print(f"  Keine Verbesserung ({epochs_no_improve}/{patience}).")
            if epochs_no_improve >= patience:
                print("  Early Stopping ausgelöst.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_name = f"lstm_{model_cfg.get('TARGET', 'vol_label_30m')}.pt"
    save_path = os.path.join(save_dir, model_name)
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg}, save_path)
    print(f"\nBestes Modell gespeichert unter: {save_path}")

    # ----- Test -----
    test_loss, test_metrics = run_epoch(
        model, test_loader, device, criterion, optimizer=None
    )
    print("\nTest-Ergebnisse:")
    print(f"Test Loss: {test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
