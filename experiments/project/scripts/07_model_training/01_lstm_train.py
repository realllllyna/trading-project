"""
LSTM model training for intraday volatility prediction (shard-wise & sampled).

- Trainiert das Modell shard-weise, um RAM zu sparen.
- Pro Epoche wird nur eine zufällige Teilmenge der Shards und Batches verwendet,
  damit das Training auf einem Laptop in vernünftiger Zeit läuft.
"""

import os
import random

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset_utils import build_sequences, VolatilityDataset, list_shard_files
from metrics import compute_metrics


# ---------------- LSTM MODEL ----------------

class LSTMVolatility(nn.Module):
    def __init__(self, n_features, hidden1, hidden2, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden1, hidden2)
        self.fc2 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        out, _ = self.lstm(x)
        # letztes Hidden-State der Sequenz
        out = out[:, -1, :]  # shape: (batch_size, hidden1)
        out = self.dropout(self.act(self.fc1(out)))
        out = self.fc2(out)  # shape: (batch_size, 1)
        return out  # logits (noch ohne Sigmoid)


# ---------------- MAIN TRAINING ----------------

def main():
    # 1) Konfiguration laden
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    SEQ_LEN = params["FEATURE_ENGINEERING"]["SEQUENCE_LENGTH"]
    PROCESSED_SHUFFLED = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_shuffled"
    TARGET_COL = params["MODEL"]["TARGET"]
    SAVE_PATH = params["MODEL"]["SAVE_PATH"]

    os.makedirs(SAVE_PATH, exist_ok=True)

    # 2) Features laden
    feature_file = "../../scripts/03_pre_split_prep/features.txt"
    feature_cols = open(feature_file, "r", encoding="utf-8").read().splitlines()
    feature_cols = [c for c in feature_cols if c]  # evtl. leere Zeilen entfernen

    # 3) Shard-Dateien (nur Namen) auflisten
    train_shards = list_shard_files(PROCESSED_SHUFFLED, "train")
    val_shards = list_shard_files(PROCESSED_SHUFFLED, "validation")

    print(f"Found {len(train_shards)} train shards, {len(val_shards)} validation shards.")

    # Sampling-Parameter
    MAX_TRAIN_SHARDS_PER_EPOCH = 6  # 6 Train-Shards pro Epoche
    MAX_VAL_SHARDS_PER_EPOCH = 2  # 2 Val-Shard pro Epoche
    MAX_BATCHES_PER_TRAIN_SHARD = 100  # max. Batches pro Train-Shard
    MAX_BATCHES_PER_VAL_SHARD = 100  # max. Batches pro Val-Shard

    # 4) Modell + Optimizer + Loss definieren
    model = LSTMVolatility(
        n_features=len(feature_cols),
        hidden1=params["MODEL"]["HIDDEN_SIZE_1"],
        hidden2=params["MODEL"]["HIDDEN_SIZE_2"],
        dropout=params["MODEL"]["DROPOUT"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["MODEL"]["LEARNING_RATE"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # 5) Epochen-Loop
    for epoch in range(1, params["MODEL"]["EPOCHS"] + 1):
        model.train()
        train_losses = []

        print(f"\n===== Epoch {epoch} / {params['MODEL']['EPOCHS']} =====")

        # ---- TRAINING: zufällige Teilmenge der Train-Shards ----
        epoch_train_shards = random.sample(
            train_shards,
            k=min(MAX_TRAIN_SHARDS_PER_EPOCH, len(train_shards))
        )
        print("Training shard-wise on:", [os.path.basename(s) for s in epoch_train_shards])

        for shard_file in epoch_train_shards:
            print(f"  Training on shard: {os.path.basename(shard_file)}")

            cols = feature_cols + [TARGET_COL]
            df = pd.read_parquet(shard_file, columns=cols)

            X_train, y_train = build_sequences(df, feature_cols, TARGET_COL, SEQ_LEN)
            if len(X_train) == 0:
                print("    No sequences in this shard (too short?), skipping.")
                continue

            train_ds = VolatilityDataset(X_train, y_train)
            train_loader = DataLoader(
                train_ds,
                batch_size=params["MODEL"]["BATCH_SIZE"],
                shuffle=True,
            )

            for batch_idx, (X, y) in enumerate(train_loader):
                if batch_idx >= MAX_BATCHES_PER_TRAIN_SHARD:
                    break

                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            del df, X_train, y_train, train_ds, train_loader

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float("nan")

        # ---- VALIDIERUNG: zufällige Teilmenge der Val-Shards ----
        model.eval()
        val_probs = []
        val_true = []

        epoch_val_shards = random.sample(
            val_shards,
            k=min(MAX_VAL_SHARDS_PER_EPOCH, len(val_shards))
        )
        print("Validating shard-wise on:", [os.path.basename(s) for s in epoch_val_shards])

        with torch.no_grad():
            for shard_file in epoch_val_shards:
                print(f"  Validating on shard: {os.path.basename(shard_file)}")

                cols = feature_cols + [TARGET_COL]
                df_val = pd.read_parquet(shard_file, columns=cols)

                X_val, y_val = build_sequences(df_val, feature_cols, TARGET_COL, SEQ_LEN)
                if len(X_val) == 0:
                    print("    No sequences in this shard (too short?), skipping.")
                    continue

                val_ds = VolatilityDataset(X_val, y_val)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=params["MODEL"]["BATCH_SIZE"],
                    shuffle=False,
                )

                for batch_idx, (X, y) in enumerate(val_loader):
                    if batch_idx >= MAX_BATCHES_PER_VAL_SHARD:
                        break

                    X = X.to(device)
                    y = y.to(device)

                    logits = model(X)
                    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                    val_probs.extend(probs)
                    val_true.extend(y.detach().cpu().numpy().flatten())

                del df_val, X_val, y_val, val_ds, val_loader

        if len(val_true) > 0:
            metrics = compute_metrics(val_true, val_probs)
            val_auc = metrics["auc"]
            val_f1 = metrics["f1"]
        else:
            val_auc = float("nan")
            val_f1 = float("nan")

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_file = os.path.join(SAVE_PATH, "lstm_best.pt")
            torch.save(model.state_dict(), save_file)
            print(f"  → New best model saved with AUC {best_val_auc:.4f} to {save_file}")

    print("\nTraining complete.")
    print(f"Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
