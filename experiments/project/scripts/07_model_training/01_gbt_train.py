"""
Gradient Boosted Trees (LightGBM SKLEARN API) Training für kurzfristige Volatilitätsvorhersage.

- Nutzt Shards aus Schritt 4 (Processed_sharded).
- Lädt nur eine begrenzte Anzahl Shards (RAM-schonend).
- Nutzt 'symbol' als kategoriales Feature (Panel Learning).
- Early Stopping auf Validation.
- Evaluierung auf Test mit 3 Metriken: AUC, LogLoss, Balanced Accuracy.
"""

from __future__ import annotations

import glob
import os

import lightgbm as lgb
import pandas as pd
import yaml

from metrics import evaluate  # <- deine minimal-metrics.py


# ---------------- Hilfsfunktionen ----------------

def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_list() -> list[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(this_dir, "../03_pre_split_prep/features.txt")
    with open(feat_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_shards(
        shard_pattern: str,
        feature_cols: list[str],
        target_col: str,
        max_shards: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"Keine Shards gefunden für Pattern: {shard_pattern}")

    if max_shards is not None:
        files = files[:max_shards]

    print(f"Lade {len(files)} Shards für Pattern: {shard_pattern}")

    dfs = []
    needed = ["symbol"] + feature_cols + [target_col]

    for fp in files:
        print("  ->", os.path.basename(fp))
        df = pd.read_parquet(fp, columns=needed)
        df = df.dropna()
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # symbol als category (LightGBM benötigt category dtype)
    df_all["symbol"] = df_all["symbol"].astype("category")

    y = df_all[target_col].astype("int8")
    X = df_all.drop(columns=[target_col])

    print(f"  Gesamt: {len(df_all)} Zeilen, {X.shape[1]} Features")
    return X, y


# ---------------- Haupttraining ----------------

def main():
    params = load_params()
    feature_cols = load_feature_list()
    target_col = params["MODEL"]["TARGET"]
    model_save_dir = params["MODEL"]["SAVE_PATH"]
    os.makedirs(model_save_dir, exist_ok=True)

    # >>> Deine Shards liegen hier (aus Schritt 4)
    SHARDED_ROOT = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_sharded"

    train_pattern = os.path.join(SHARDED_ROOT, "train", "train_shard_*.parquet")
    val_pattern = os.path.join(SHARDED_ROOT, "validation", "validation_shard_*.parquet")
    test_pattern = os.path.join(SHARDED_ROOT, "test", "test_shard_*.parquet")

    # Shard-Limits (RAM-schonend)
    MAX_TRAIN_SHARDS = 20
    MAX_VAL_SHARDS = 5
    MAX_TEST_SHARDS = 5

    print("Target:", target_col)
    print("Num features:", len(feature_cols))

    # ---- Daten laden ----
    print("\n=== Lade Trainingsdaten ===")
    X_train, y_train = load_shards(train_pattern, feature_cols, target_col, MAX_TRAIN_SHARDS)

    print("\n=== Lade Validierungsdaten ===")
    X_val, y_val = load_shards(val_pattern, feature_cols, target_col, MAX_VAL_SHARDS)

    print("\n=== Lade Testdaten ===")
    X_test, y_test = load_shards(test_pattern, feature_cols, target_col, MAX_TEST_SHARDS)

    # Kategorien konsistent machen (wichtig, falls Symbole in Val/Test fehlen/extra sind)
    all_cats = pd.Index(X_train["symbol"].astype(str).unique()).union(
        pd.Index(X_val["symbol"].astype(str).unique())
    ).union(
        pd.Index(X_test["symbol"].astype(str).unique())
    )

    X_train["symbol"] = pd.Categorical(X_train["symbol"].astype(str), categories=all_cats)
    X_val["symbol"] = pd.Categorical(X_val["symbol"].astype(str), categories=all_cats)
    X_test["symbol"] = pd.Categorical(X_test["symbol"].astype(str), categories=all_cats)

    pos_rate = float(y_train.mean())
    scale_pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0
    print(f"\nPositivrate (Train): {pos_rate:.3f}")
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    # ---------------- LightGBM SKLEARN CLASSIFIER ----------------
    model = lgb.LGBMClassifier(
        n_estimators=int(params["MODEL"]["N_ESTIMATORS"]),
        learning_rate=float(params["MODEL"]["LEARNING_RATE"]),
        num_leaves=int(params["MODEL"]["NUM_LEAVES"]),
        max_depth=-1,
        subsample=float(params["MODEL"]["SUBSAMPLE"]),
        colsample_bytree=float(params["MODEL"]["COLSAMPLE_BYTREE"]),
        reg_lambda=1.0,
        min_child_samples=int(params["MODEL"]["MIN_CHILD_SAMPLES"]),
        scale_pos_weight=scale_pos_weight,   # <- statt class_weight
        n_jobs=-1,
    )

    print("\n=== Training LightGBM (mit Early Stopping) ===")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=["symbol"],
        callbacks=[lgb.early_stopping(30)],
    )

    # ---- Modell speichern ----
    model_path = os.path.join(model_save_dir, f"gbt_{target_col}.txt")
    model.booster_.save_model(model_path)
    print(f"Modell gespeichert unter: {model_path}")

    # ---- Testevaluierung ----
    print("\n=== Test-Evaluierung ===")
    y_prob = model.predict_proba(X_test)[:, 1]
    m = evaluate(y_test.to_numpy(), y_prob, threshold=0.5)

    for k, v in m.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
