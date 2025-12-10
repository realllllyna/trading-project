"""
Baut LSTM-Sequenzen aus den vorprozessierten Parquet-Dateien und speichert sie
als .npz-Shards, getrennt nach Train/Validation/Test.

Input (aus Schritt 3):
    <PROCESSED_PATH>/{SYMBOL}_train.parquet
    <PROCESSED_PATH>/{SYMBOL}_validation.parquet
    <PROCESSED_PATH>/{SYMBOL}_test.parquet

Output:
    <SEQUENCE_PATH>/
        train/train_shard_000.npz
        train/train_shard_001.npz
        validation/val_shard_000.npz
        test/test_shard_000.npz
    In jeder .npz:
        X: [N, seq_len, n_features]
        y: [N]
        symbols: [N]
        timestamps: [N]
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from dataset_utils import load_params, load_feature_list, build_sequences_from_df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_shard(
        X_list,
        y_list,
        sym_list,
        ts_list,
        out_dir: str,
        split: str,
        shard_idx: int,
) -> None:
    if not X_list:
        return

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    symbols = np.concatenate(sym_list, axis=0)
    timestamps = np.concatenate(ts_list, axis=0)

    fname = (
        f"{split}_shard_{shard_idx:03d}.npz"
        if split == "train"
        else f"{split}_shard_{shard_idx:03d}.npz"
    )
    out_path = os.path.join(out_dir, split, fname)
    ensure_dir(os.path.dirname(out_path))

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        symbols=symbols,
        timestamps=timestamps,
    )
    print(
        f"  Saved {split} shard {shard_idx:03d} with {len(y)} sequences "
        f"to {out_path}"
    )


def main():
    params = load_params()

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    seq_len = params["FEATURE_ENGINEERING"]["SEQUENCE_LENGTH"]
    seq_cfg = params["SEQUENCE_BUILD"]
    seq_out_root = seq_cfg["SEQUENCE_PATH"]
    shard_size = seq_cfg["SHARD_SIZE"]
    target_col = params["MODEL"]["TARGET"]

    feature_cols = load_feature_list()
    print("Using feature columns:", feature_cols)
    print("Target column:", target_col)
    print("Sequence length:", seq_len)
    print("Shard size:", shard_size)

    splits = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    for split, split_suffix in splits.items():
        print(f"\nBuilding sequences for split: {split}")

        pattern = os.path.join(processed_path, f"*_{split_suffix}.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  No files found for pattern: {pattern}")
            continue

        shard_idx = 0
        X_acc = []
        y_acc = []
        sym_acc = []
        ts_acc = []

        for fpath in files:
            symbol = os.path.basename(fpath).split("_")[0]
            print(f"  Reading {fpath} ({symbol})")
            df = pd.read_parquet(fpath)

            X, y, symbols, timestamps = build_sequences_from_df(
                df, feature_cols, target_col, seq_len
            )
            if len(y) == 0:
                print(f"    Not enough rows for sequences, skipping {symbol}.")
                continue

            X_acc.append(X)
            y_acc.append(y)
            sym_acc.append(symbols)
            ts_acc.append(timestamps)

            # Wenn genug gesammelt → Shard schreiben
            total_seqs = sum(len(chunk) for chunk in y_acc)
            if total_seqs >= shard_size:
                save_shard(X_acc, y_acc, sym_acc, ts_acc, seq_out_root, split, shard_idx)
                shard_idx += 1
                X_acc, y_acc, sym_acc, ts_acc = [], [], [], []

        # Reste schreiben
        if X_acc:
            save_shard(X_acc, y_acc, sym_acc, ts_acc, seq_out_root, split, shard_idx)


if __name__ == "__main__":
    main()
