"""
Step 9a – Deployment / Inference

Dieses Skript zeigt, dass das trainierte LightGBM-Modell
außerhalb des Trainings lauffähig ist und Vorhersagen erzeugt.

Output:
- CSV mit p(t) = Wahrscheinlichkeit für High Volatility pro Minute
"""

from __future__ import annotations

import os
import glob
import pandas as pd
import lightgbm as lgb
import yaml


def load_params() -> dict:
    with open("../../conf/params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features() -> list[str]:
    with open("../03_pre_split_prep/features.txt", "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def find_latest_model(model_dir: str) -> str:
    models = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not models:
        raise FileNotFoundError("Kein Modell gefunden.")
    return models[-1]


def main():
    params = load_params()
    feature_cols = load_features()

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    model_dir = params["MODEL"]["SAVE_PATH"]

    model_path = find_latest_model(model_dir)
    model = lgb.Booster(model_file=model_path)

    # Beispiel: erste Test-Datei
    test_files = sorted(glob.glob(os.path.join(processed_path, "*_test.parquet")))
    df = pd.read_parquet(test_files[0])

    X = df[["symbol"] + feature_cols].copy()
    X["symbol"] = X["symbol"].astype("category")

    df["p_high_vol"] = model.predict(X)

    out_dir = "../../results/deployment"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions_example.csv")
    df[["timestamp", "symbol", "p_high_vol"]].to_csv(out_path, index=False)

    print("Deployment Prediction gespeichert unter:")
    print(out_path)


if __name__ == "__main__":
    main()
