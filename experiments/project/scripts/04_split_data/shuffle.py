"""
Memory-safe sharding utility (panel intraday data).

Instead of performing a global ORDER BY (which is very memory intensive),
this script assigns each row deterministically to a shard using a hash:

    shard_id = hash(symbol, timestamp, seed) % n_shards

This effectively "shuffles" the training data by distributing rows uniformly
across shards without needing a global sort. For GBT training this is typically
sufficient and avoids DuckDB out-of-memory issues.

Validation/Test are kept chronological by default (optional: also hash-shard).
"""

from __future__ import annotations

import glob
import os

import duckdb
import yaml


def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


params = load_params()
INPUT_PROCESSED_PATH = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]

OUTPUT_ROOT = r"D:/Data/Datasets/Financial_Trading/Experiment_2_1/Processed_sharded"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SEED = 42

N_SHARDS = {"train": 64, "validation": 16, "test": 16}

MODE_VALID_TEST = "hash"  # "hash" recommended for memory safety


def shard_split(split_type: str) -> None:
    pattern = os.path.join(INPUT_PROCESSED_PATH, f"*_{split_type}.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[{split_type}] No files found: {pattern}")
        return

    n_shards = N_SHARDS[split_type]
    split_dir = os.path.join(OUTPUT_ROOT, split_type)
    os.makedirs(split_dir, exist_ok=True)

    load_pattern = os.path.join(INPUT_PROCESSED_PATH, f"*_{split_type}.parquet")
    print(f"[{split_type}] {len(files)} files -> {n_shards} shards")

    # Tune DuckDB to avoid temp/memory issues (safe defaults)
    duckdb.sql("SET preserve_insertion_order=false;")
    duckdb.sql("SET threads=4;")  # reduce if still memory pressure

    # TRAIN: hash-based shard assignment (acts like shuffle via distribution)
    if split_type == "train":
        for shard in range(n_shards):
            out_file = os.path.join(split_dir, f"{split_type}_shard_{shard:03d}.parquet")
            query = f"""
                SELECT *
                FROM read_parquet('{load_pattern}')
                WHERE mod(hash(symbol, timestamp, {SEED}), {n_shards}) = {shard}
            """
            duckdb.sql(f"COPY ({query}) TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)")
            if shard % max(1, n_shards // 8) == 0:
                print(f"  wrote {out_file}")
        print(f"[{split_type}] done.\n")
        return

    # VALID/TEST:
    if MODE_VALID_TEST == "hash":
        # Low-memory deterministic sharding (no global ordering)
        for shard in range(n_shards):
            out_file = os.path.join(split_dir, f"{split_type}_shard_{shard:03d}.parquet")
            query = f"""
                SELECT *
                FROM read_parquet('{load_pattern}')
                WHERE mod(hash(symbol, timestamp, {SEED}), {n_shards}) = {shard}
            """
            duckdb.sql(f"COPY ({query}) TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)")
            if shard % max(1, n_shards // 8) == 0:
                print(f"  wrote {out_file}")
        print(f"[{split_type}] done.\n")
        return

    raise ValueError("MODE_VALID_TEST='time' is not implemented in the memory-safe variant.")


def main() -> None:
    print("Input:", INPUT_PROCESSED_PATH)
    print("Output:", OUTPUT_ROOT)

    for split in ["train", "validation", "test"]:
        shard_split(split)


if __name__ == "__main__":
    main()
