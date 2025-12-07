"""
Hilfsfunktionen für das Volatilitäts-Experiment (GBT-Modell).

Aktuell:
- Laden der geshuffelten Shards (train / validation / test)
  mit optionaler Begrenzung der Anzahl Dateien (max_files),
  um Speicher zu sparen.
"""

from __future__ import annotations

import glob
import os
import random

import pandas as pd


def load_shards(shuffled_path: str, split: str, max_files: int | None = None) -> pd.DataFrame:
    pattern = os.path.join(shuffled_path, f"{split}_shard_*.parquet")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"Keine Shard-Dateien gefunden unter: {pattern}")

    print(f"Gefundene {split}-Shards: {len(files)} Dateien")

    if max_files is not None and len(files) > max_files:
        # zufällig max_files Dateien auswählen
        files = random.sample(files, max_files)
        print(f"→ Nutze nur {len(files)} Dateien für {split} (Speicherlimit).")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df
