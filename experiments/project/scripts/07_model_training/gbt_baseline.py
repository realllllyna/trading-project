"""
Baseline/Best Model: Gradient Boosted Trees (XGBoost) für Volatilitätsklassifikation.

- Nutzt eine begrenzte Anzahl an Shards (analog zum LSTM/RandomForest),
  um Speicher zu sparen und einen fairen Vergleich zu ermöglichen.
- Verwendet dieselben Features und dasselbe Target wie LSTM und Random Forest.
"""

import os
import random

import pandas as pd
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from dataset_utils import list_shard_files

# ---------------------------------------------------------
# Konfiguration laden
# ---------------------------------------------------------
params = yaml.safe_load(open("../../conf/params.yaml", "r"))

# Pfad zu den geshuffelten Shards (wie im LSTM-Skript)
SHUFFLED_PATH = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_shuffled"

# Target aus MODEL-Block (z.B. 'vol_label_30m')
TARGET_COL = params["MODEL"]["TARGET"]

# Featureliste laden
FEATURE_LIST_PATH = "../../scripts/03_pre_split_prep/features.txt"
feature_cols = [
    line.strip()
    for line in open(FEATURE_LIST_PATH, "r", encoding="utf-8")
    if line.strip()
]

print(f"Loaded {len(feature_cols)} features from {FEATURE_LIST_PATH}")
print(f"Using target column: {TARGET_COL}")

# ---------------------------------------------------------
# Shard-Auswahl (analog zum LSTM/RandomForest-Setup)
# ---------------------------------------------------------
N_TRAIN_SHARDS = 6   # wie dein aktuelles Setup
N_VAL_SHARDS = 2

train_shards = list_shard_files(SHUFFLED_PATH, "train")
val_shards = list_shard_files(SHUFFLED_PATH, "validation")

print(f"Found {len(train_shards)} train shards, {len(val_shards)} validation shards.")

train_selected = random.sample(train_shards, k=min(N_TRAIN_SHARDS, len(train_shards)))
val_selected = random.sample(val_shards, k=min(N_VAL_SHARDS, len(val_shards)))

print("Using train shards:")
for f in train_selected:
    print("  ", os.path.basename(f))

print("Using validation shards:")
for f in val_selected:
    print("  ", os.path.basename(f))


def load_flat_from_shards(shard_files):
    """Lädt eine Liste von Shards flach (ohne Sequenzen)."""
    dfs = []
    cols = feature_cols + [TARGET_COL]
    for f in shard_files:
        print(f"  Loading {os.path.basename(f)}")
        df = pd.read_parquet(f, columns=cols)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# Daten laden
# ---------------------------------------------------------
print("\nLoading training data...")
train_df = load_flat_from_shards(train_selected)
print(f"Train shape: {train_df.shape}")

print("\nLoading validation data...")
val_df = load_flat_from_shards(val_selected)
print(f"Validation shape: {val_df.shape}")

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]

X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL]

# ---------------------------------------------------------
# XGBoost / Gradient Boosted Trees trainieren
# ---------------------------------------------------------
print("\nTraining XGBClassifier (Gradient Boosted Trees) baseline...")

gbt = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
)

gbt.fit(X_train, y_train)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
print("\nEvaluating on validation data...")

val_prob = gbt.predict_proba(X_val)[:, 1]
val_pred = (val_prob >= 0.5).astype(int)

print("\nClassification report (XGBoost GBT):")
print(classification_report(y_val, val_pred, digits=4))

try:
    auc = roc_auc_score(y_val, val_prob)
    print(f"AUC: {auc:.4f}")
except ValueError:
    print("AUC konnte nicht berechnet werden (wahrscheinlich nur eine Klasse im Validation-Set).")
