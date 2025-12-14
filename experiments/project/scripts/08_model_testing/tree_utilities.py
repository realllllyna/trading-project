"""
Utility functions for model testing (loading params, features, model, and test files).
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import lightgbm as lgb
import yaml


def load_params() -> dict:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_list() -> list[str]:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(this_dir, "../03_pre_split_prep/features.txt")
    with open(feat_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_latest_model(model_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(model_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No model .txt found in: {model_dir}")
    return files[-1]


def load_model(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


def list_test_files(processed_path: str, max_symbols: Optional[int] = 50) -> list[str]:
    files = sorted(glob.glob(os.path.join(processed_path, "*_test.parquet")))
    if not files:
        raise FileNotFoundError(f"No *_test.parquet found in: {processed_path}")
    return files[:max_symbols] if max_symbols else files
