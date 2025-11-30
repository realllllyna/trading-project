"""
Shuffling and sharding utility for preprocessed Parquet datasets.

This script reads the processed data path from the experiment's params.yaml
and produces fully shuffled, uniformly sharded Parquet files for each split
(train, validation, test) using DuckDB.

Input:
- Per-symbol preprocessed Parquet files under:
    <processed_path>/*_<split>.parquet
  (e.g. AAPL_train.parquet, MSFT_train.parquet, ...)

Output:
- Per-symbol shuffled files under:
    D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_indiv_shuffled/
- Global shard files under:
    D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_shuffled/

Requirements:
-------------
- DuckDB installed (pip install duckdb)
- Parquet files produced by Step 3
"""

import glob
import os

import duckdb
import yaml

# ---------------------------------------------------------
# Load experiment configuration
# ---------------------------------------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]

# Base path on D: for shuffled data
SHUFFLE_BASE = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1"

# Ensure the D: directories exist
INDIV_BASE_DIR = os.path.join(SHUFFLE_BASE, "Processed_indiv_shuffled")
SHARD_BASE_DIR = os.path.join(SHUFFLE_BASE, "Processed_shuffled")

os.makedirs(INDIV_BASE_DIR, exist_ok=True)
os.makedirs(SHARD_BASE_DIR, exist_ok=True)


# ---------------------------------------------------------
# Shuffle + Shard function
# ---------------------------------------------------------
def shuffle(split_type: str) -> None:
    # All *_<split>.parquet files from C: processed path
    temp_files = glob.glob(f"{processed_path}/*_{split_type}.parquet")

    if not temp_files:
        print(f"No files found for split '{split_type}' in {processed_path}")
        return

    # Output directories on D:
    indiv_dir = INDIV_BASE_DIR
    shard_dir = SHARD_BASE_DIR

    # ---------------------------------------------------------
    # Stage 1 — shuffle each file individually
    # ---------------------------------------------------------
    for in_file in temp_files:
        print(f"Shuffling (per-file): {in_file}")

        out_file = os.path.join(indiv_dir, os.path.basename(in_file))

        try:
            duckdb.sql(f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{in_file}')
                    ORDER BY random()
                )
                TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)
            """)
        except Exception as e:
            print(f"Error shuffling {in_file}: {e}")
            continue

    # ---------------------------------------------------------
    # Stage 2 — create shards across all files
    # ---------------------------------------------------------
    shuffled_files_path = f"{indiv_dir}/*_{split_type}.parquet"

    n_shards = len(temp_files)
    os.makedirs(shard_dir, exist_ok=True)

    for shard in range(n_shards):
        out_file = f"{shard_dir}/{split_type}_shard_{shard}.parquet"
        print(f"Creating shard {shard + 1}/{n_shards} → {out_file}")

        try:
            duckdb.sql(f"""
                COPY (
                    SELECT *
                    FROM (
                        SELECT *,
                               mod(row_number() OVER (), {n_shards}) AS shard_id
                        FROM read_parquet('{shuffled_files_path}')
                        USING SAMPLE 100%
                    )
                    WHERE shard_id = {shard}
                )
                TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)
            """)
        except Exception as e:
            print(f"Error generating shard {shard} for split '{split_type}': {e}")
            continue


# ---------------------------------------------------------
# Run full pipeline
# ---------------------------------------------------------
shuffle("train")
shuffle("validation")
shuffle("test")

print("\n✔ Shuffling & sharding complete!")
