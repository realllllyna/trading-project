"""
Gradient Boosted Trees (GBT) Training für das Volatilitäts-Experiment (Experiment 2.1).

Ablauf:
1) Projekt-Konfiguration aus ../../conf/params.yaml laden
2) Geshuffelte Shards (train / validation) von D: laden (nur Teilmenge, um RAM zu sparen)
3) Featureliste aus features.txt einlesen
4) Trainings- und Validierungs-Matrizen X, y bauen (jede Zeile = 1 Minute)
5) HistGradientBoostingClassifier trainieren
6) Performance auf dem Validation-Set berechnen (Accuracy, F1, ROC-AUC)
7) Modell als gbt_volatility.pkl speichern
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from dataset_utils import load_shards
from metrics import compute_metrics


def main():
    # ---------------------------------------------------------
    # 1. Konfiguration laden
    # ---------------------------------------------------------
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    # Shuffled-Daten
    SHUFFLED_PATH = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_shuffled"

    # Target-Spalte
    target_col = "vol_label_30m"

    # Modell-Speicherort aus params.yaml
    model_dir = params["MODEL"]["SAVE_PATH"]
    os.makedirs(model_dir, exist_ok=True)

    # Hyperparameter für GBT (aus params.yaml oder Defaults)
    lr = params["MODEL"].get("LEARNING_RATE", 0.05)
    max_depth = params["MODEL"].get("MAX_DEPTH", 6)
    n_estimators = params["MODEL"].get("N_ESTIMATORS", 300)

    # ---------------------------------------------------------
    # 2. Featureliste laden
    # ---------------------------------------------------------
    feature_list_path = "../../scripts/03_pre_split_prep/features.txt"
    with open(feature_list_path, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"Lade {len(feature_cols)} Features aus {feature_list_path}")

    # ---------------------------------------------------------
    # 3. Shards laden (Train & Validation) – mit Begrenzung
    # ---------------------------------------------------------
    # Anzahl Dateien, die wir pro Split nutzen
    max_files_train = 10
    max_files_val = 2

    print("\nLade Training-Shards ...")
    train_df = load_shards(SHUFFLED_PATH, "train", max_files=max_files_train)
    print(f"Train-Shape (vor Dropna): {train_df.shape}")

    print("Lade Validation-Shards ...")
    val_df = load_shards(SHUFFLED_PATH, "validation", max_files=max_files_val)
    print(f"Validation-Shape (vor Dropna): {val_df.shape}")

    # Nur Features + Target behalten und NaNs entfernen
    cols_needed = feature_cols + [target_col]
    train_df = train_df[cols_needed].dropna().reset_index(drop=True)
    val_df = val_df[cols_needed].dropna().reset_index(drop=True)

    print(f"Train-Shape (nach Dropna): {train_df.shape}")
    print(f"Validation-Shape (nach Dropna): {val_df.shape}")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values.astype(int)

    # Nochmal innerhalb des DataFrames subsamplen
    max_train_samples = 200_000
    if len(X_train) > max_train_samples:
        idx = np.random.choice(len(X_train), max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Train-Subset für schnellen Lauf: {X_train.shape}")

    # ---------------------------------------------------------
    # 4. Modell definieren
    # ---------------------------------------------------------
    print("\nInitialisiere HistGradientBoostingClassifier ...")
    clf = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_depth=max_depth,
        max_iter=n_estimators,
        class_weight="balanced",
        random_state=42,
    )

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------
    print("Starte Training des GBT-Modells ...")
    clf.fit(X_train, y_train)
    print("Training abgeschlossen.")

    # ---------------------------------------------------------
    # 6. Evaluation auf Validation-Set
    # ---------------------------------------------------------
    val_proba = clf.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_proba)

    print("\nValidation-Performance (GBT):")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['auc']:.4f}")

    # ---------------------------------------------------------
    # 7. Modell speichern
    # ---------------------------------------------------------
    model_path = os.path.join(model_dir, "gbt_volatility.pkl")
    joblib.dump(
        {
            "model": clf,
            "features": feature_cols,
            "target": target_col,
        },
        model_path,
    )
    print(f"\n Modell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()
