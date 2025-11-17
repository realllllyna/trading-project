"""
Minimal adaptation of plotter.py to plot engineered features around a specific row index.

This script:
- Loads configuration and selects a default S&P 500 symbol (like plotter.py)
- Loads that symbol's training feature data from experiments/exp_1_2/data/{SYMBOL}_train.parquet
- Plots the following feature columns on a single axis in a window around a chosen row index:
  'VWAP_norm', 'EMA_10', 'EMA_50', 'Slope_EMA_10_1'
- Highlights the target row with a red dashed line and each new trading day with green dashed lines
- Formats the x-axis with timestamp labels at regular intervals

Assumptions:
- The DataFrame contains a 'timestamp' column (datetime-like/parseable).
- The four feature columns may or may not all be present; missing ones are skipped.

Usage (from project root):
- Run this file directly to produce a plot for the configured symbol.
"""

import matplotlib
import pandas as pd
import yaml

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

FEATURES = ['VWAP_norm', 'EMA_10', 'EMA_60', 'Slope_EMA_10_1']


def plot_features(df: pd.DataFrame, index: int, window_before: int = 100, window_after: int = 100, symbol: str | None = None):
    """Plot selected feature series around a given row index with day separators.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'timestamp' and feature columns.
    index : int
        Row index (positional) around which to center the window and highlight with a vertical line.
    window_before : int, optional
        Number of rows to include before the target index (default: 100).
    window_after : int, optional
        Number of rows to include after the target index (default: 100).
    symbol : str | None, optional
        Optional symbol string used in labels and title for clarity.
    """
    # Normalize types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify the target timestamp
    target_date = df.loc[index, 'timestamp']

    # Window and subset
    start = max(0, index - window_before)
    end = min(len(df), index + window_after)
    subset = df.iloc[start:end+1].copy().reset_index(drop=True)

    # Create figure/axis
    fig, ax = plt.subplots(figsize=(18, 9))

    # Determine which features are available
    present = [c for c in FEATURES if c in subset.columns]
    if not present:
        raise KeyError(f"None of the expected features are present. Expected any of: {', '.join(FEATURES)}")

    # Plot each available feature
    for feat in present:
        line_label = f"{feat} ({symbol})" if symbol else feat
        ax.plot(subset.index, subset[feat], label=line_label, linewidth=1.5)

    # Locate closest timestamp index in subset
    target_idx = (subset['timestamp'] - pd.Timestamp(target_date)).abs().idxmin()
    ax.axvline(x=target_idx, color='red', linestyle='--', linewidth=1.5)

    # Day separators
    day_changes = subset['timestamp'].dt.date.ne(subset['timestamp'].dt.date.shift()).fillna(False)
    for i, is_new_day in enumerate(day_changes):
        if is_new_day and i != 0:
            ax.axvline(x=i, color='green', linestyle='--', linewidth=1.2, alpha=0.7)

    # X ticks
    tick_positions = subset.index[::max(1, len(subset)//10)]
    tick_labels = subset.loc[tick_positions, 'timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_xlabel('Index')
    ax.set_ylabel('Feature values')

    title_prefix = f"{symbol} " if symbol else ""
    ax.set_title(f"{title_prefix}Features ±{window_before} rows around {target_date.date()}")

    ax.legend()
    ax.grid(True)

    plt.show()


# Load config and symbol list (same approach as plotter.py)
params = yaml.safe_load(open("experiments/exp_1_2/conf/params.yaml"))

ticker_list_df = pd.read_csv('experiments/exp_1_2/data/sp500_companies_as_of_jan_2025.csv')

# Defaults from config (kept for context)
START_DATE = params['DATA_ACQUISITON']['START_DATE']
END_DATE = params['DATA_ACQUISITON']['END_DATE']
processed_path = params['DATA_PREP']['PROCESSED_PATH']

# Choose a default symbol (e.g., Apple)
SYMBOL = ticker_list_df['Symbol'][0]

# Load feature data; expects a Parquet file with 'timestamp' and feature columns
features_file = f"{processed_path}/{SYMBOL}_train.parquet"
features_df = pd.read_parquet(features_file)

# Produce the diagnostic plot
default_index = 2500
plot_features(features_df, index=default_index, window_before=500, window_after=500, symbol=SYMBOL)
