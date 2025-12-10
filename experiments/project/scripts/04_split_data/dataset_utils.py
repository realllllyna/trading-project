from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_feature_list() -> List[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(this_dir, "../03_pre_split_prep/features.txt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.txt not found at {feat_path}")
    with open(feat_path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    return features


def build_sequences_from_df(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Nur benötigte Spalten extrahieren
    X_src = df[feature_cols].to_numpy(dtype=np.float32)
    y_src = df[target_col].to_numpy(dtype=np.float32)
    symbols_src = df["symbol"].astype(str).to_numpy()

    # Timestamps: TZ-aware -> UTC -> TZ entfernen -> numpy
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    ts_src = ts.to_numpy(dtype="datetime64[ns]")

    n = len(df)
    n_feat = X_src.shape[1]
    if n < seq_len:
        return (
            np.empty((0, seq_len, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    X_list, y_list, sym_list, ts_list = [], [], [], []

    for i in range(seq_len - 1, n):
        start = i - seq_len + 1
        end = i + 1
        X_list.append(X_src[start:end, :])
        y_list.append(y_src[i])
        sym_list.append(symbols_src[i])
        ts_list.append(ts_src[i])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    symbols = np.array(sym_list, dtype=object)
    timestamps = np.array(ts_list, dtype="datetime64[ns]")

    return X, y, symbols, timestamps
