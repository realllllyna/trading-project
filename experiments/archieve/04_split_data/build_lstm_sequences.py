"""
Erzeugt LSTM-Sequenzen für Train/Validation/Test.
Jeder Shard enthält MAXIMAL 'SHARD_SIZE' Sequenzen (params.yaml).
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd

from dataset_utils import load_params, load_feature_list, build_sequences_from_df


# ---------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_single_shard(
        X: np.ndarray,
        y: np.ndarray,
        symbols: np.ndarray,
        timestamps: np.ndarray,
        seq_root: str,
        split: str,
        shard_idx: int,
):
    """
    Speichert EINEN Shard (max SHARD_SIZE Sequenzen).
    """
    if len(X) == 0:
        return

    out_dir = os.path.join(seq_root, split)
    ensure_dir(out_dir)

    fname = f"{split}_shard_{shard_idx:03d}.npz"
    out_path = os.path.join(out_dir, fname)

    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        symbols=symbols,
        timestamps=timestamps,
    )

    print(f"  → Saved {split} shard {shard_idx:03d} with {len(y)} sequences")


# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------
def main():
    params = load_params()

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    seq_len = params["FEATURE_ENGINEERING"]["SEQUENCE_LENGTH"]

    seq_cfg = params["SEQUENCE_BUILD"]
    seq_root = seq_cfg["SEQUENCE_PATH"]
    shard_size = seq_cfg["SHARD_SIZE"]
    target_col = params["MODEL"]["TARGET"]

    feature_cols = load_feature_list()

    print("\n=== BUILDING LSTM SEQUENCES (STEP 4) ===")
    print(f"Sequence length: {seq_len}")
    print(f"Shard size: {shard_size}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")

    splits = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split, suffix in splits.items():
        print(f"\n--- Processing split: {split} ---")

        pattern = os.path.join(processed_path, f"*_{suffix}.parquet")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"  No files found for: {pattern}")
            continue

        shard_idx = 0

        for fpath in files:
            symbol = os.path.basename(fpath).split("_")[0]
            print(f"  Reading {symbol} ...")

            df = pd.read_parquet(fpath)

            # Alle Sequenzen für dieses Symbol erzeugen
            X, y, symbols, timestamps = build_sequences_from_df(
                df, feature_cols, target_col, seq_len
            )

            n = len(y)
            print(f"    Built {n} sequences for {symbol}")

            # In viele kleine Shards schneiden
            start = 0
            while start < n:
                end = min(start + shard_size, n)

                save_single_shard(
                    X[start:end],
                    y[start:end],
                    symbols[start:end],
                    timestamps[start:end],
                    seq_root,
                    split,
                    shard_idx,
                )

                shard_idx += 1
                start = end


if __name__ == "__main__":
    main()
