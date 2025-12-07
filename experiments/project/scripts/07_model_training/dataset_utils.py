"""
Dataset utilities for volatility LSTM training.

- Loads shuffled shards (train/val/test)
- Builds sequences of length SEQ_LEN
- Provides PyTorch dataset for LSTM training
"""

import glob
import os
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def list_shard_files(shuffled_path: str, split: str) -> List[str]:
    """Return sorted list of shard parquet files for a given split."""
    pattern = os.path.join(shuffled_path, f"{split}_shard_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No shard files found at {shuffled_path} for split '{split}'")

    return files


def load_shards(shuffled_path: str, split: str) -> pd.DataFrame:
    """Load and concatenate all shard parquet files for a split."""
    files = list_shard_files(shuffled_path, split)
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def build_sequences(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(df) < seq_len:
        raise ValueError(f"DataFrame has only {len(df)} rows, but seq_len={seq_len}")

    X, y = [], []

    feature_values = df[feature_cols].values
    target_values = df[target_col].values

    for i in range(len(df) - seq_len):
        X.append(feature_values[i:i + seq_len])
        y.append(target_values[i + seq_len - 1])  # last value in sequence

    return np.array(X), np.array(y)


class VolatilityDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
