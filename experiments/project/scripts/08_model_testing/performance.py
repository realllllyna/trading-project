"""
Performance metrics for backtest (minimal but sufficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 390
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def sharpe_annualized(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    mu, sig = r.mean(), r.std(ddof=1)
    if sig <= 0:
        return float("nan")
    return float((mu / sig) * np.sqrt(MINUTES_PER_YEAR))


def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def turnover(w: pd.Series) -> float:
    return float(w.diff().abs().fillna(0.0).sum())
