"""
Step 7 – Model Training (Gradient Boosted Trees / LightGBM)

Ziel:
- Trainiere ein GBT-Modell zur Vorhersage kurzfristiger Volatilität
- Nutzt Train- und Validation-Daten
- Early Stopping auf Validation

Input:
- Gesplittete Shards aus Schritt 4 (Processed_sharded)

Output:
- Trainiertes Modell als .txt-Datei
"""

from __future__ import annotations

import glob
import os

import lightgbm as lgb
import pandas as pd
import yaml


# ---------------- Hilfsfunktionen ----------------

def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, "../../conf/params.yaml"), "r") as f:
        return yaml.safe_load(f)


def load_feature_list() -> list[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, "../03_pre_split_prep/features.txt"), "r") as f:
        return [l.strip() for l in f if l.strip()]


def load_shards(pattern: str, feature_cols: list[str], target_col: str, max_shards: int):
    files = sorted(glob.glob(pattern))[:max_shards]
    if not files:
        raise FileNotFoundError(f"Keine Dateien für {pattern}")

    dfs = []
    cols = ["symbol"] + feature_cols + [target_col]

    for fp in files:
        df = pd.read_parquet(fp, columns=cols).dropna()
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["symbol"] = df["symbol"].astype("category")

    y = df[target_col].astype("int8")
    X = df.drop(columns=[target_col])

    return X, y


# ---------------- Hauptprogramm ----------------

def main():
    params = load_params()
    feature_cols = load_feature_list()
    target_col = params["MODEL"]["TARGET"]

    SHARDED_ROOT = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_sharded"
    model_dir = params["MODEL"]["SAVE_PATH"]
    os.makedirs(model_dir, exist_ok=True)

    print("=== Lade Trainingsdaten ===")
    X_train, y_train = load_shards(
        os.path.join(SHARDED_ROOT, "train", "train_shard_*.parquet"),
        feature_cols,
        target_col,
        max_shards=20
    )

    print("=== Lade Validierungsdaten ===")
    X_val, y_val = load_shards(
        os.path.join(SHARDED_ROOT, "validation", "validation_shard_*.parquet"),
        feature_cols,
        target_col,
        max_shards=5
    )

    # Klassenungleichgewicht
    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=params["MODEL"]["N_ESTIMATORS"],
        learning_rate=params["MODEL"]["LEARNING_RATE"],
        num_leaves=params["MODEL"]["NUM_LEAVES"],
        subsample=params["MODEL"]["SUBSAMPLE"],
        colsample_bytree=params["MODEL"]["COLSAMPLE_BYTREE"],
        min_child_samples=params["MODEL"]["MIN_CHILD_SAMPLES"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )

    print("=== Training mit Early Stopping ===")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=["symbol"],
        callbacks=[lgb.early_stopping(30)],
    )

    model_path = os.path.join(model_dir, f"gbt_{target_col}.txt")
    model.booster_.save_model(model_path)
    print(f"Modell gespeichert unter:\n{model_path}")


if __name__ == "__main__":
    main()
