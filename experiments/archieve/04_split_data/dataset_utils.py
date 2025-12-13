"""
dataset_utils.py – Hilfsfunktionen für Schritt 4 (Split Data / Build LSTM Sequences)

Enthält:
- Laden von params.yaml
- Laden der Featureliste
- Funktion build_sequences_from_df(), die aus einem DataFrame
  Sliding-Window-LSTM-Sequenzen baut.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------
# Parameter laden
# ---------------------------------------------------------
def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Featureliste laden
# ---------------------------------------------------------
def load_feature_list() -> list[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.join(this_dir, "../03_pre_split_prep/features.txt")

    features = []
    with open(feature_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                features.append(name)
    return features


# ---------------------------------------------------------
# Sequenzaufbau
# ---------------------------------------------------------
def build_sequences_from_df(
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
):
    """
    Wandelt einen DataFrame eines Symbols in LSTM-Sequenzen um.

    Output:
        X: [N, seq_len, F]
        y: [N]
        symbols: [N]
        timestamps: [N] → timestamp des TARGET-Zeitpunktes (τ)
    """

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Zielvektor
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' missing in DataFrame.")

    y_src = df[target_col].astype(float).to_numpy()

    # Featurematrix
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features in df: {missing}")

    X_src = df[feature_cols].to_numpy(dtype=np.float32)

    # Symbolspalte
    if "symbol" not in df.columns:
        raise KeyError("Column 'symbol' missing in df (required).")

    sym_src = df["symbol"].astype(str).to_numpy()

    # Timestamp
    ts_src = pd.to_datetime(df["timestamp"])
    if ts_src.dt.tz is not None:  # Zeitzone entfernen, falls vorhanden
        ts_src = ts_src.dt.tz_convert("UTC").dt.tz_localize(None)
    ts_src = ts_src.to_numpy()

    # Anzahl möglicher Sequenzen
    n_total = len(df)
    n_seq = n_total - seq_len

    if n_seq <= 0:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="<U10"),
            np.empty(0, dtype="datetime64[ns]"),
        )

    # Speicherplatz reservieren
    X = np.zeros((n_seq, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.zeros(n_seq, dtype=np.float32)
    symbols = np.empty(n_seq, dtype=object)
    timestamps = np.zeros(n_seq, dtype="datetime64[ns]")

    # Sliding Window
    for i in range(n_seq):
        X[i] = X_src[i:i + seq_len]
        y[i] = y_src[i + seq_len]  # Target am Zeitpunkt τ
        symbols[i] = sym_src[i + seq_len]
        timestamps[i] = ts_src[i + seq_len]

    return X, y, symbols, timestamps
