"""
LSTM training for intraday volatility prediction (using prebuilt sequences).

- Lädt vorgefertigte Sequenzen (X, y) aus .npz-Dateien,
  die von 06_build_lstm_sequences.py erzeugt wurden.
- Trainiert ein LSTM auf diesen Sequenzen.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import VolatilityDataset
from metrics import compute_metrics


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
        out, _ = self.lstm(x)         # (batch, seq_len, hidden1)
        out = out[:, -1, :]           # letztes Hidden-State
        out = self.dropout(self.act(self.fc1(out)))
        out = self.fc2(out)           # (batch, 1)
        return out                   # logits


def main():
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    seq_path = processed_path + "_lstm_sequences"

    save_path = params["MODEL"]["SAVE_PATH"]
    os.makedirs(save_path, exist_ok=True)

    # Featureliste nur für n_features
    feature_file = "../../scripts/03_pre_split_prep/features.txt"
    feature_cols = [
        line.strip()
        for line in open(feature_file, "r", encoding="utf-8")
        if line.strip()
    ]

    # Sequenz-Dateien
    train_file = os.path.join(seq_path, "lstm_sequences_train.npz")
    val_file = os.path.join(seq_path, "lstm_sequences_validation.npz")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train sequence file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation sequence file not found: {val_file}")

    print(f"Loading train sequences from: {train_file}")
    train_data = np.load(train_file)
    X_train = train_data["X"]
    y_train = train_data["y"]

    print(f"Loading validation sequences from: {val_file}")
    val_data = np.load(val_file)
    X_val = val_data["X"]
    y_val = val_data["y"]

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    # Datasets / DataLoader
    train_ds = VolatilityDataset(X_train, y_train)
    val_ds = VolatilityDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=params["MODEL"]["BATCH_SIZE"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params["MODEL"]["BATCH_SIZE"],
        shuffle=False,
    )

    # Modell
    model = LSTMVolatility(
        n_features=X_train.shape[-1],
        hidden1=params["MODEL"]["HIDDEN_SIZE_1"],
        hidden2=params["MODEL"]["HIDDEN_SIZE_2"],
        dropout=params["MODEL"]["DROPOUT"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["MODEL"]["LEARNING_RATE"])
    loss_fn = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    best_val_auc = 0.0
    n_epochs = params["MODEL"]["EPOCHS"]

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        print(f"\n===== Epoch {epoch} / {n_epochs} =====")

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                val_probs.extend(probs)
                val_true.extend(y.detach().cpu().numpy().flatten())

        metrics = compute_metrics(val_true, val_probs)
        val_auc = metrics["auc"]
        val_f1 = metrics["f1"]

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_path = os.path.join(save_path, "lstm_best_from_sequences.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  → New best model saved with AUC {best_val_auc:.4f} to {best_path}")

    print("\nTraining complete.")
    print(f"Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
