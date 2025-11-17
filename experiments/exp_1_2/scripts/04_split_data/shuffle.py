"""
Shuffling and sharding utility for preprocessed Parquet datasets.

This script reads the configured processed data path from the experiment's params.yaml
and produces fully shuffled, uniformly sharded Parquet files for each dataset split
(train, validation, test) using DuckDB. It works in two stages per split:

1) Individual file shuffle:
   - For each per-symbol Parquet file matching `*_<split>.parquet`, create a shuffled
     version of the same file under `<processed_path>_indiv_shuffled/` by ordering
     rows with DuckDB's random() function. This ensures within-file shuffling.

2) Global sharding across all symbols:
   - Read all individually shuffled files for the split, assign a shard_id based on
     `mod(row_number() OVER (), n_shards)`, and write out one file per shard under
     `<processed_path>_shuffled/` as `<split>_shard_<k>.parquet`.

Outputs:
- `<processed_path>_indiv_shuffled/*.parquet` per-symbol shuffled files
- `<processed_path}_shuffled/<split>_shard_*.parquet` shard files

Assumptions / Notes:
- DuckDB is available and can read/write Parquet files.
- Shuffling uses non-deterministic random() by default; reproducibility is not guaranteed
  unless a seed is configured at the DuckDB level (not done here).
- The script keeps the current behavior and paths; run it from its directory (or adapt paths).
"""

import duckdb
import yaml
import os, glob

# Load experiment parameters (relative path expected when running from this script's folder)
params = yaml.safe_load(open("../../conf/params.yaml"))
# Alternative (when executing from project root):
# params = yaml.safe_load(open("experiments/exp_1_2/conf/params.yaml"))

processed_path = params['DATA_PREP']['PROCESSED_PATH']


def shuffle(split_type: str) -> None:
    """Shuffle and shard all Parquet files for a given dataset split.

    Parameters
    ----------
    split_type : str
        The data split to process: one of {'train', 'validation', 'test'}.

    Behavior
    --------
    - Writes per-symbol shuffled files to `<processed_path>_indiv_shuffled/`.
    - Then writes uniformly distributed shard files to `<processed_path>_shuffled/`.
    - Prints status messages and continues on individual file/shard errors.
    """
    # Find all per-symbol Parquet files for the given split (e.g., *_train.parquet)
    temp_files = glob.glob(f"{processed_path}/*_{split_type}.parquet")
    indiv_dir = f"{processed_path}_indiv_shuffled"
    os.makedirs(indiv_dir, exist_ok=True)

    # Stage 1: Create per-symbol fully shuffled copies
    for in_file in temp_files:
        print(f"Shuffling {in_file}_{split_type} into individual shuffled files")
        base = os.path.basename(in_file)
        out_file = os.path.join(indiv_dir, base)
        try:
            duckdb.sql(f"""
                    COPY (
                        SELECT *
                        FROM read_parquet('{in_file}')
                        ORDER BY random()   -- full shuffle
                    )
                    TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)
                """)
        except Exception as e:
            print(f"Error processing {in_file}: {e}")
            continue

    # Stage 2: Shard uniformly across all shuffled per-symbol files
    load_path = f"{processed_path}_indiv_shuffled/*_{split_type}.parquet"

    n_shards = len(temp_files)
    shuffle_dir = f"{processed_path}_shuffled"
    os.makedirs(shuffle_dir, exist_ok=True)

    for shard in range(n_shards):
        print(f"Creating shard {shard} out of {n_shards} for {split_type}")
        out_file = f"{shuffle_dir}/{split_type}_shard_{shard}.parquet"
        try:
            duckdb.sql(f"""
                COPY (
                    SELECT *
                    FROM (
                        SELECT *,
                               mod(row_number() OVER (), {n_shards}) AS shard_id
                        FROM read_parquet('{load_path}')
                        USING SAMPLE 100%   -- shuffle once
                    )
                    WHERE shard_id = {shard}
                )
                TO '{out_file}' (FORMAT 'parquet', OVERWRITE_OR_IGNORE)
            """)
        except Exception as e:
            print(f"Error creating shard {shard} for {split_type}: {e}")
            continue


# Process all three splits sequentially
shuffle('train')
shuffle('validation')
shuffle('test')