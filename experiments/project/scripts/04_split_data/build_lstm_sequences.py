"""
build_lstm_sequences.py

Baut LSTM-Sequenzen aus den symbolweisen, vorverarbeiteten Parquet-Dateien
und speichert sie in vielen kleinen .npz-Chunks auf D:.

Wichtig:
- Es werden ALLE möglichen Sequenzen gebaut (kein Downsampling),
  aber nie alle gleichzeitig im RAM gehalten, sondern in Chunks.
"""

import os
from typing import List

import numpy as np
import pandas as pd
import yaml

from dataset_utils import generate_sequence_chunks


def list_symbol_files(processed_path: str, split: str) -> List[str]:
    """
    Liste aller Dateien wie AAPL_train.parquet, MSFT_train.parquet usw.
    """
    files: List[str] = []
    for fname in os.listdir(processed_path):
        if fname.endswith(f"_{split}.parquet"):
            files.append(os.path.join(processed_path, fname))
    files.sort()
    return files


def main():
    np.random.seed(42)

    # params.yaml laden
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]  # C:/.../Processed
    seq_len = params["FEATURE_ENGINEERING"]["SEQUENCE_LENGTH"]
    target_col = params["MODEL"]["TARGET"]

    # Featureliste
    feature_file = "../../scripts/03_pre_split_prep/features.txt"
    feature_cols = [
        line.strip()
        for line in open(feature_file, "r", encoding="utf-8")
        if line.strip()
    ]

    print(f"Processed path (per symbol): {processed_path}")
    print(f"Sequence length: {seq_len}")
    print(f"Target column: {target_col}")
    print(f"Number of features: {len(feature_cols)}")

    # Basis-Pfad auf D: für Sequenzen
    SEQ_BASE = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1"
    out_base = os.path.join(SEQ_BASE, "LSTM_sequences_chunks")
    os.makedirs(out_base, exist_ok=True)
    print(f"Sequence chunks will be saved to: {out_base}")

    # wie viele Sequenzen pro Chunk im RAM bauen?
    CHUNK_NUM_SEQS = 50_000  # kannst du bei Bedarf erhöhen/verkleinern

    global_chunk_id = 0

    for split in ["train", "validation", "test"]:
        print(f"\n=== BUILDING SEQUENCE CHUNKS FOR SPLIT: {split} ===")

        split_files = list_symbol_files(processed_path, split)
        if not split_files:
            print(f"  No *_{split}.parquet files found, skipping.")
            continue

        for path in split_files:
            symbol_name = os.path.basename(path).split("_")[0]
            print(f"  Processing {symbol_name} ({path})")

            df = pd.read_parquet(path)

            required_cols = set(feature_cols + [target_col, "timestamp"])
            missing = required_cols.difference(df.columns)
            if missing:
                print(f"    Skipping {symbol_name}: missing columns {missing}")
                continue

            df = df[list(required_cols)]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.dropna().reset_index(drop=True)

            n_rows = len(df)
            if n_rows < seq_len:
                print(f"    Skipping {symbol_name}: only {n_rows} rows (< seq_len={seq_len}).")
                continue

            chunk_idx = 0
            for X_chunk, y_chunk in generate_sequence_chunks(
                    df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    seq_len=seq_len,
                    chunk_num_sequences=CHUNK_NUM_SEQS,
            ):
                n_chunk = len(X_chunk)
                if n_chunk == 0:
                    continue

                out_file = os.path.join(
                    out_base,
                    f"{split}_{symbol_name}_chunk_{global_chunk_id:06d}.npz",
                )
                print(
                    f"    Saving chunk {chunk_idx} for {symbol_name} "
                    f"({n_chunk} sequences) → {out_file}"
                )

                np.savez_compressed(out_file, X=X_chunk, y=y_chunk)

                chunk_idx += 1
                global_chunk_id += 1

    print("\n✔ LSTM sequence chunk building complete.")


if __name__ == "__main__":
    main()
