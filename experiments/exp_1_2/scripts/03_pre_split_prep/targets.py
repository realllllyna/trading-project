"""
Target engineering utilities for pre-split data preparation.

This module augments a bars DataFrame with forward-looking trend targets over
multiple prediction horizons. For each configured horizon (in bars), it computes:
- A normalized linear regression slope of the selected price column over the
  next `period` bars (including the current bar), normalized by the mean price
  in that window to make it scale-invariant across symbols.
- The simple percentage change from the first to the last bar in the window.

Inputs/assumptions:
- Input DataFrame must at least contain the price column specified by `price_col`
  (default: 'vwap').
- The data should be ordered chronologically and have no gaps for the intended
  interpretation.

Outputs:
- A copy of the original DataFrame with two new columns per horizon:
  'trend_slope_{period}m' and 'vwap_pctchg_{period}m'.

Notes:
- This function preserves alignment by padding the tail with NaNs for each horizon
  to maintain the original length and avoid index shift.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress
import pandas as pd


def add_normalized_trend_direction(
    bars_df: pd.DataFrame,
    prediction_periods: list[int],
    price_col: str = 'vwap',
) -> pd.DataFrame:
    """Add normalized trend slope and percentage change targets.

    For each `period` in `prediction_periods`, computes a linear regression slope over
    the window of length `period + 1` starting at each row (i) and normalizes it by the
    mean of the window's prices. Also computes the simple percentage change between the
    first and last element in the same window.

    Parameters
    ----------
    bars_df : pd.DataFrame
        Input DataFrame containing at least the `price_col`.
    prediction_periods : list[int]
        Horizons (in bars) for which to compute targets.
    price_col : str, default 'vwap'
        Column name to use as the price series.

    Returns
    -------
    pd.DataFrame
        A copy of `bars_df` with additional target columns per horizon.
    """
    # Work on a copy to avoid mutating the caller's DataFrame.
    df = bars_df.copy()

    for period in prediction_periods:
        trends: list[float] = []
        pct_changes: list[float] = []

        # Iterate over all starting indices where a full window of length (period + 1) fits.
        for i in range(len(df) - period - 1):
            # y: price values over the window, x: 0..period (time index)
            y = df[price_col].iloc[i:i + period + 1].values
            x = np.arange(period + 1)

            # Univariate linear regression slope; intercept and stats are not used.
            slope, _, _, _, _ = linregress(x, y)

            # Normalize by mean price in the window to make slope scale-invariant across symbols
            # or price levels (e.g., 50 USD vs 500 USD).
            normalized_slope = slope / np.mean(y)
            trends.append(normalized_slope)

            # Percentage change from first to last observation in the window.
            pct_change = (y[-1] - y[0]) / y[0]
            pct_changes.append(pct_change)

        # Pad the tail with NaNs so that the new columns align with the original index length.
        trends += [np.nan] * (period + 1)
        pct_changes += [np.nan] * (period + 1)

        # Store the targets using a consistent naming scheme.
        df[f'trend_slope_{period}m'] = trends
        df[f'vwap_pctchg_{period}m'] = pct_changes

    return df

