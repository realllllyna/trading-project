"""
Plotting utility for inspecting intraday open prices around a specific row index.

This script:
- Loads configuration (date range and data path) from experiments/exp_1_2/conf/params.yaml
- Reads the S&P 500 symbol list and chooses one symbol to plot
- Loads that symbol's 1-minute bar data from a Parquet file
- Plots the 'open' price in a window around a chosen row index
- Highlights the target row with a red dashed line and each new trading day with green dashed lines
- Formats the x-axis with timestamp labels at regular intervals

Inputs/assumptions:
- The bar DataFrame must contain at least these columns: 'timestamp' (datetime-like/parseable) and 'open' (numeric/parseable).
- Timestamps are assumed to be monotonically increasing.
- The repository layout matches the paths in the code (relative to project root).

Output:
- A Matplotlib window showing the open price series in the specified window, with day separators and a target marker.

Usage (from project root):
- Run this file directly to produce a plot for the configured symbol.

Dependencies:
- pandas, matplotlib, pyyaml. The Matplotlib backend is set to 'TkAgg'. Ensure Tk is available on your system.
"""

import matplotlib
import pandas as pd
import yaml

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def plot_open(df: pd.DataFrame, index: int, window_before: int = 100, window_after: int = 100, symbol: str | None = None):
    """Plot the 'open' price around a given row index with day separators.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'timestamp' and 'open' columns.
    index : int
        Row index (positional) around which to center the window and highlight with a vertical line.
    window_before : int, optional
        Number of rows to include before the target index (default: 100).
    window_after : int, optional
        Number of rows to include after the target index (default: 100).
    symbol : str | None, optional
        Optional symbol string used in labels and title for clarity.

    Side Effects
    ------------
    Displays a Matplotlib figure window; does not return a value.
    """
    # Normalize types to avoid downstream plotting/parsing issues.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['open'] = pd.to_numeric(df['open'], errors='coerce')

    # Identify the exact target timestamp corresponding to the given row label (assumes default RangeIndex).
    target_date = df.loc[index, 'timestamp']

    # Compute window bounds in positional terms and extract a copy for plotting.
    start = max(0, index - window_before)
    end = min(len(df), index + window_after)
    subset = df.iloc[start:end+1].copy().reset_index(drop=True)

    # Create the figure/axes and plot the open price. Include the symbol in the legend if provided.
    fig, ax = plt.subplots(figsize=(18, 9))
    line_label = f"Open ({symbol})" if symbol else 'Open'
    ax.plot(subset.index, subset['open'], label=line_label, color='blue', linewidth=1.5)

    # Robustly locate the closest timestamp index in the subset window.
    # Using idxmin on absolute timedeltas avoids static-analysis issues with boolean.idxmax()
    target_label = f"Target ({symbol})" if symbol else 'Target'
    target_idx = (subset['timestamp'] - pd.Timestamp(target_date)).abs().idxmin()
    ax.axvline(x=target_idx, color='red', linestyle='--', linewidth=1.5, label=target_label)

    # Draw separators at the start of each new day (skip first row). This helps visually group trading sessions.
    day_changes = subset['timestamp'].dt.date.ne(subset['timestamp'].dt.date.shift()).fillna(False)
    for i, is_new_day in enumerate(day_changes):
        if is_new_day and i != 0:  # skip the very first row
            ax.axvline(x=i, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    # Place at most ~10 tick labels across the x-axis for readability, formatting as date-time strings.
    tick_positions = subset.index[::max(1, len(subset)//10)]
    tick_labels = subset.loc[tick_positions, 'timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_xlabel('Index')
    ax.set_ylabel('Open Price')

    # Title reflects the symbol (if provided), the window size, and the target calendar date.
    title_prefix = f"{symbol} " if symbol else ""
    ax.set_title(f'{title_prefix}Open Prices ±{window_before} rows around {target_date.date()}')

    ax.legend()
    ax.grid(True)

    plt.show()


# params = yaml.safe_load(open("../conf/params.yaml"))
# ticker_list_df = pd.read_csv('../data/sp500_companies_as_of_jan_2025.csv')
# The following paths assume execution from the project root. Adjust if running from a different CWD.
params = yaml.safe_load(open("experiments/exp_1_2/conf/params.yaml"))

# Load the reference list of S&P 500 constituents; we select one by row index below.
# Note: Column casing ('Symbol') must match the CSV header.
ticker_list_df = pd.read_csv('experiments/exp_1_2/data/sp500_companies_as_of_jan_2025.csv')

# Extract date bounds (not directly used here, but kept for context/reference).
START_DATE = params['DATA_ACQUISITON']['START_DATE']
END_DATE = params['DATA_ACQUISITON']['END_DATE']

# Choose a default symbol; index 1 corresponds to the second row (example: Apple at the time of writing).
SYMBOL = ticker_list_df['Symbol'][0]  # Apple

# Build the path to 1-minute adjusted bars for the chosen symbol.
data_path = params['DATA_ACQUISITON']['DATA_PATH']

# Load bar data; expects a Parquet file with 'timestamp' and 'open' columns.
bars_file = f"{data_path}/Bars_1m_adj/{SYMBOL}.parquet"
bars_df = pd.read_parquet(bars_file)

# Produce the diagnostic plot with a wider window to provide context around the target row.
plot_open(bars_df, index=2500, window_before=500, window_after=500, symbol=SYMBOL)
