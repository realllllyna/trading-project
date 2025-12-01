"""
Feature-Selection-Helfer für das Volatilitätsprojekt (Random Forest).

Dieses Skript:
1) Lädt Projekt-Parameter aus ../../conf/params.yaml.
2) Lädt einen geshuffelten Trainings-Shard (train_shard_0.parquet) von D:.
3) Lädt die Liste der erzeugten Feature-Namen aus features.txt.
4) Behält nur diese Features + die Zielspalte (vol_label_30m).
5) Entfernt Zeilen mit fehlenden Werten.
6) Trainiert einen RandomForestClassifier auf diesem Datensatz.
7) Sortiert die Features nach ihrer Modell-Importance (feature_importances_).
"""

import os

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Projekt-Parameter laden
# ---------------------------------------------------------
params = yaml.safe_load(open("../../conf/params.yaml", "r"))

# Pfad zu geshuffelten Shards (auf D:)
SHUFFLED_PATH = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_shuffled"

# Zielvariable
TARGET_COL = "vol_label_30m"

# Featureliste
FEATURE_LIST_PATH = "../../scripts/03_pre_split_prep/features.txt"

# ---------------------------------------------------------
# 2. Featureliste laden
# ---------------------------------------------------------
features: list[str] = []
with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if name:
            features.append(name)

print(f"{len(features)} Features geladen aus {FEATURE_LIST_PATH}")

# ---------------------------------------------------------
# 3. Einen geshuffelten Trainings-Shard laden
# ---------------------------------------------------------
shard_file = os.path.join(SHUFFLED_PATH, "train_shard_0.parquet")
print(f"Lade Daten aus: {shard_file}")

df = pd.read_parquet(shard_file)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target '{TARGET_COL}' fehlt in DataFrame.")

# Nur Features verwenden, die auch wirklich in den Daten existieren
cols_to_use = [c for c in features if c in df.columns]

# ---------------------------------------------------------
# 4. Teilmenge bilden + NaNs entfernen
# ---------------------------------------------------------
df_sub = df[cols_to_use + [TARGET_COL]].dropna().reset_index(drop=True)
print(f"Shape nach Dropna: {df_sub.shape}")

X = df_sub[cols_to_use]
y = df_sub[TARGET_COL]

# ---------------------------------------------------------
# 5. Random Forest trainieren
# ---------------------------------------------------------
print("Trainiere RandomForestClassifier ...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)

rf.fit(X, y)

# ---------------------------------------------------------
# 6. Feature Importances auslesen & sortieren
# ---------------------------------------------------------
importances = rf.feature_importances_
imp_series = pd.Series(importances, index=cols_to_use).sort_values(ascending=False)

print("\nRandom-Forest Feature Importances (absteigend):")
print(imp_series)
