"""
Shuffling and sharding utility for preprocessed Parquet datasets.

This script reads the processed data path from the experiment's params.yaml
and produces fully shuffled, uniformly sharded Parquet files for each split
(train, validation, test) using DuckDB.

Processing pipeline (for each split):
-------------------------------------

1) Per-symbol shuffle:
   - For each file matching `*_<split>.parquet`, create a fully shuffled copy
     under `<processed_path>_indiv_shuffled/` using `ORDER BY random()`.
   - This removes any chronological ordering inside each asset.

2) Global sharding:
   - Load all individually shuffled files.
   - Assign rows to shards using:
         mod(row_number() OVER (), n_shards)
   - Write uniform shard files:
         <processed_path>_shuffled/<split>_shard_<k>.parquet

Outputs:
--------
- <processed_path>_indiv_shuffled/*.parquet
- <processed_path>_shuffled/<split>_shard_*.parquet

Requirements:
-------------
- DuckDB installed (pip install duckdb)
- Parquet files produced by Step 3

Notes:
------
- Shuffling uses DuckDB’s non-deterministic random() → batches vary each run.
- The number of shards equals the number of per-symbol input files.
- Run this script from its own directory or adjust paths.
"""

import glob
import os

import duckdb
import yaml

# ---------------------------------------------------------
# Load experiment configuration
# ---------------------------------------------------------
params = yaml.safe_load(open("../../conf/params.yaml"))
processed_path = params["DATA_PREP"]["PROCESSED_PATH"]


# ---------------------------------------------------------
# Shuffle + Shard function
# ---------------------------------------------------------
def shuffle(split_type: str) -> None:
    # All *_<split>.parquet files
    temp_files = glob.glob(f"{processed_path}/*_{split_type}.parquet")

    # Output directories
    indiv_dir = f"{processed_path}_indiv_shuffled"
    os.makedirs(indiv_dir, exist_ok=True)

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
    shard_dir = f"{processed_path}_shuffled"
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
            print(f"Error generating shard {shard}: {e}")
            continue


# ---------------------------------------------------------
# Run full pipeline
# ---------------------------------------------------------
shuffle("train")
shuffle("validation")
shuffle("test")

print("\n✔ Shuffling & sharding complete!")
