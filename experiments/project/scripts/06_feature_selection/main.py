"""
Feature-Selection / Feature-Ranking für das Volatilitätsprojekt (Random Forest).

Dieses Skript:
1) Lädt Projekt-Parameter aus ../../conf/params.yaml.
2) Lädt ein train-File (AAPL_train.parquet) aus dem PROCESSED_PATH.
3) Lädt die Liste der erzeugten Feature-Namen aus features.txt.
4) Behält nur diese Features + die Zielspalte (vol_label_30m).
5) Entfernt Zeilen mit fehlenden Werten.
6) Trainiert einen RandomForestClassifier.
7) Gibt die Feature-Importances sortiert aus.
"""

from __future__ import annotations

import os

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Projekt-Parameter & Pfade
# ---------------------------------------------------------
PARAMS_PATH = "../../conf/params.yaml"
FEATURE_LIST_PATH = "../03_pre_split_prep/features.txt"
IMPORTANCE_CSV = "../../experiments/project/data/feature_importances_rf.csv"

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
TARGET_COL = params["MODEL"]["TARGET"]

SYMBOL = "AAPL"
train_file = os.path.join(PROCESSED_PATH, f"{SYMBOL}_train.parquet")
print(f"Verwende Train-Datei: {train_file}")

# ---------------------------------------------------------
# 2. Featureliste laden
# ---------------------------------------------------------
with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"{len(feature_cols)} Features aus {FEATURE_LIST_PATH} geladen.")

# ---------------------------------------------------------
# 3. Trainingsdaten laden
# ---------------------------------------------------------
df = pd.read_parquet(train_file)
print("Rohshape:", df.shape)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target '{TARGET_COL}' fehlt in DataFrame.")

# Nur Features verwenden, die wirklich in den Daten existieren
cols_to_use = [c for c in feature_cols if c in df.columns]
print(f"{len(cols_to_use)} Feature-Spalten werden verwendet: {cols_to_use}")

# ---------------------------------------------------------
# 4. Teilmenge + NaNs entfernen
# ---------------------------------------------------------
df_sub = df[cols_to_use + [TARGET_COL]].dropna().reset_index(drop=True)
print("Shape nach Dropna:", df_sub.shape)

X = df_sub[cols_to_use]
y = df_sub[TARGET_COL]

# ---------------------------------------------------------
# 5. Random Forest trainieren
# ---------------------------------------------------------
print("Trainiere RandomForestClassifier ...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

rf.fit(X, y)
