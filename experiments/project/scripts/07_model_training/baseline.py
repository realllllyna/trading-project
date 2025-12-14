"""
Baseline model for short-term volatility classification.

Baseline definition:
- Constant-probability classifier
- Predicts the same probability for all samples, equal to the
  positive class frequency in the training data.

Purpose:
- Provides a trivial reference model (random / majority-class baseline)
- Used to demonstrate that the GBT model significantly outperforms chance

Metrics:
- ROC-AUC
- LogLoss
- Balanced Accuracy
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import yaml

from metrics import evaluate


# -----------------------------
# Helper functions
# -----------------------------

def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(
        shard_pattern: str,
        target_col: str,
        max_shards: int | None = None,
) -> pd.Series:
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No shards found for pattern: {shard_pattern}")

    if max_shards is not None:
        files = files[:max_shards]

    print(f"Lade {len(files)} Shards für Pattern: {shard_pattern}")

    ys = []
    for fp in files:
        print("  ->", os.path.basename(fp))
        df = pd.read_parquet(fp, columns=[target_col])
        df = df.dropna()
        ys.append(df[target_col].astype(int))

    return pd.concat(ys, ignore_index=True)


# -----------------------------
# Main
# -----------------------------

def main():
    params = load_params()
    target_col = params["MODEL"]["TARGET"]

    # Pfad zu den geshardeten Daten (Schritt 4)
    SHARDED_ROOT = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_sharded"

    train_pattern = os.path.join(SHARDED_ROOT, "train", "train_shard_*.parquet")
    test_pattern = os.path.join(SHARDED_ROOT, "test", "test_shard_*.parquet")

    # Begrenzen, um schneller zu laufen
    MAX_TRAIN_SHARDS = 20
    MAX_TEST_SHARDS = 5

    print("=== Lade Trainingslabels (für Baseline-Schätzung) ===")
    y_train = load_labels(train_pattern, target_col, MAX_TRAIN_SHARDS)

    pos_rate = float(y_train.mean())
    print(f"\nPositive Rate (Train): {pos_rate:.4f}")

    print("\n=== Lade Testlabels (für Evaluation) ===")
    y_test = load_labels(test_pattern, target_col, MAX_TEST_SHARDS)

    # Baseline-Vorhersage: konstante Wahrscheinlichkeit
    y_prob = np.full_like(y_test, fill_value=pos_rate, dtype=float)

    print("\n=== Baseline Evaluation (Test) ===")
    metrics = evaluate(y_test.to_numpy(), y_prob, threshold=0.5)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
