"""
This script visualizes intraday 1-minute open prices for two S&P 500 symbols.
It loads configuration parameters (data path, symbol list) from the YAML file,
reads the corresponding Parquet files, and generates two separate diagnostic
charts — one window for each selected symbol.

For each symbol, the script:
- extracts a window of 1-minute bars around a chosen row index,
- plots the 'Open' price over the selected window,
- marks the target row with a red dashed line,
- marks the beginning of each new trading day with green dashed lines,
- labels timestamps at regular intervals for readability.

Inputs:
- YAML config: ../../conf/params.yaml
- Symbols CSV (path specified in YAML)
- 1-minute bar Parquet files in <DATA_PATH>/Bars_1m/

Outputs:
- Two separate Matplotlib windows, one per symbol, displaying the selected
  open-price windows around the target index.

Dependencies:
- pandas, matplotlib, pyyaml, pyarrow (for reading Parquet), Tk (for Matplotlib backend 'TkAgg')
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

matplotlib.use("TkAgg")


def plot_open_window(
        df: pd.DataFrame,
        index: int,
        window_before: int,
        window_after: int,
        symbol: str,
):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")

    target_time = df.loc[index, "timestamp"]

    start = max(0, index - window_before)
    end = min(len(df) - 1, index + window_after)
    subset = df.iloc[start:end + 1].copy().reset_index(drop=True)

    plt.figure(figsize=(18, 8))
    ax = plt.gca()

    # Plot Open prices
    ax.plot(subset.index, subset["Open"], label=f"Open ({symbol})", linewidth=1.5)

    # Target line
    target_idx = (subset["timestamp"] - target_time).abs().idxmin()
    ax.axvline(target_idx, color="red", linestyle="--", linewidth=1.5, label=f"Target ({symbol})")

    # Day change markers
    day_changes = subset["timestamp"].dt.date.ne(subset["timestamp"].dt.date.shift()).fillna(False)
    for i, is_new_day in enumerate(day_changes):
        if is_new_day and i != 0:
            ax.axvline(i, color="green", linestyle="--", alpha=0.7)

    # Ticks
    tick_positions = subset.index[::max(1, len(subset) // 10)]
    tick_labels = subset.loc[tick_positions, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Index (1-minute bars)")
    ax.set_ylabel("Open Price")
    ax.set_title(f"{symbol} – Open ±{window_before} rows around {target_time.strftime('%Y-%m-%d')}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    data_path = params["DATA_ACQUISITION"]["DATA_PATH"]
    symbols_csv = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]

    ticker_list_df = pd.read_csv(symbols_csv)
    symbols = ticker_list_df["Symbol"].dropna().tolist()

    symbol1 = symbols[0]
    symbol2 = symbols[1]

    file1 = os.path.join(data_path, "Bars_1m", f"{symbol1}.parquet")
    file2 = os.path.join(data_path, "Bars_1m", f"{symbol2}.parquet")

    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    target_index = 2500
    window_before = 500
    window_after = 500

    # Ensure within range
    target_index = min(target_index, len(df1) - 1, len(df2) - 1)

    # **Two separate plots**
    plot_open_window(df1, target_index, window_before, window_after, symbol1)
    plot_open_window(df2, target_index, window_before, window_after, symbol2)


if __name__ == "__main__":
    main()
