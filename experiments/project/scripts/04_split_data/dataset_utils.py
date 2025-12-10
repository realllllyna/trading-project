"""
dataset_utils.py

Hilfsfunktionen für das Volatilitäts-LSTM:
- Sequenzen aus einem DataFrame in Chunks bauen (RAM-schonend)
- PyTorch-Dataset für Sequenzen
"""

from typing import Sequence, Tuple, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def generate_sequence_chunks(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        seq_len: int,
        chunk_num_sequences: int = 50_000,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator, der LSTM-Sequenzen in Chunks erzeugt.

    Statt alle Sequenzen auf einmal zu bauen (RAM-Killer),
    erzeugen wir z.B. immer nur 50k Sequenzen und geben sie zurück.

    Parameters
    ----------
    df : DataFrame
        Muss alle feature_cols und target_col enthalten, sortiert nach Zeit.
    feature_cols : Liste der Feature-Spalten
    target_col : Zielspalte (z.B. 'vol_label_30m')
    seq_len : Länge der LSTM-Sequenz (z.B. 30)
    chunk_num_sequences : Anzahl Sequenzen pro Chunk (default 50k)

    Yields
    ------
    X_chunk : np.ndarray, shape (n_chunk, seq_len, n_features), float32
    y_chunk : np.ndarray, shape (n_chunk,), float32
    """

    n = len(df)
    if n < seq_len:
        return

    feat_vals = df[feature_cols].values.astype(np.float32)
    target_vals = df[target_col].values.astype(np.float32)

    n_sequences = n - seq_len  # so viele Startpunkte möglich

    for start_seq in range(0, n_sequences, chunk_num_sequences):
        end_seq = min(start_seq + chunk_num_sequences, n_sequences)

        n_chunk = end_seq - start_seq
        n_features = len(feature_cols)

        X_chunk = np.empty((n_chunk, seq_len, n_features), dtype=np.float32)
        y_chunk = np.empty((n_chunk,), dtype=np.float32)

        for j, i in enumerate(range(start_seq, end_seq)):
            X_chunk[j] = feat_vals[i:i + seq_len]
            y_chunk[j] = target_vals[i + seq_len - 1]

        yield X_chunk, y_chunk


class VolatilityDataset(Dataset):
    """
    PyTorch Dataset:
    - X: (N, seq_len, n_features)
    - y: (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
