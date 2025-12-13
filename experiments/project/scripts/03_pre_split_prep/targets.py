"""
Target engineering utilities for the volatility experiment (LEAKAGE-FREE).

Für jede Minute τ und jeden Horizont t:
- zukünftige realisierte Volatilität RV_t(τ) über die nächsten t Minuten
- binäres High-Volatility-Label basierend auf TRAIN-fit Quantil-Schwelle

WICHTIG:
- KEINE Tagesmittel-Normalisierung (intraday Lookahead)
- Quantil-Schwelle wird NUR auf dem Trainingsset gefittet
- RV wird pro Handelstag berechnet (keine Overnight-Rollings)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

EPS = 1e-12


def _prepare_df(bars_df: pd.DataFrame) -> pd.DataFrame:
    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def add_realized_volatility(
        bars_df: pd.DataFrame,
        vola_windows: List[int],
        price_col: str = "Close",
) -> pd.DataFrame:
    """
    Adds RV_{t}m columns:
    RV_t(τ) = sqrt( sum_{k=τ+1}^{τ+t} r_k^2 )

    Computed per trading day (no overnight leakage).
    """
    df = _prepare_df(bars_df)

    def per_day(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        log_ret = np.log(d[price_col] / d[price_col].shift(1))

        for t in vola_windows:
            sq_ret = (log_ret ** 2).rolling(t).sum().shift(-t)
            d[f"RV_{t}m"] = np.sqrt(sq_ret)

        return d

    df = df.groupby("date", group_keys=False).apply(per_day)
    return df


def fit_volatility_thresholds(
        train_df: pd.DataFrame,
        vola_windows: List[int],
        high_vol_quantile: float = 0.7,
) -> Dict[int, float]:
    """
    Fit quantile thresholds ONLY on training data.
    """
    thresholds: Dict[int, float] = {}
    for t in vola_windows:
        thresholds[t] = float(train_df[f"RV_{t}m"].quantile(high_vol_quantile))
    return thresholds


def apply_volatility_labels(
        df: pd.DataFrame,
        vola_windows: List[int],
        thresholds: Dict[int, float],
) -> pd.DataFrame:
    """
    Apply pre-fitted thresholds to create vol_label_{t}m.
    """
    out = df.copy()
    for t in vola_windows:
        out[f"vol_label_{t}m"] = (out[f"RV_{t}m"] >= thresholds[t]).astype(int)
    return out
