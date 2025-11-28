"""
Diagnostic plotting utility for inspecting engineered features and volatility
targets around a specific row index for a single S&P 500 symbol.

This script:
- Loads configuration (data paths and symbol list) from ../../conf/params.yaml
- Reads the S&P 500 symbol list and selects one default symbol (first row)
- Loads the symbol's *training* set from <PROCESSED_PATH>/{SYMBOL}_train.parquet
- Extracts a window of rows around a chosen index
- Plots a selection of engineered features and the normalized volatility target
  in that window, with:
    - multiple feature curves on one axis,
    - a red dashed line at the target row,
    - green dashed lines at the beginning of each trading day,
    - formatted timestamp labels on the x-axis.

Assumptions:
- The training Parquet file contains at least:
    'timestamp',
    'log_ret_1m',
    'roll_vol_15m',
    'vwap_dev',
    'volume_z_30m',
    'RV_norm_30m'    (example volatility target for t=30)
- Timestamps are sorted and represent 1-minute bars.

Output:
- A Matplotlib window showing the selected feature series and volatility target
  around the chosen index.

Dependencies:
- pandas, matplotlib, pyyaml, pyarrow (for reading Parquet).
"""

import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Features, die geplottet werden sollen
FEATURES = [
    "log_ret_1m",
    "roll_vol_15m",
    "vwap_dev",
    "volume_z_30m",
    "RV_norm_30m",  # normalisierte Volatilität für t=30
]


def plot_features_window(
        df: pd.DataFrame,
        index: int,
        window_before: int = 100,
        window_after: int = 100,
        symbol: str | None = None,
):
    # timestamp in Datetime wandeln
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sicherstellen, dass numerische Spalten auch wirklich numerisch sind
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Index prüfen
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of bounds for DataFrame of length {len(df)}")

    target_ts = df.loc[index, "timestamp"]

    # Fenster bestimmen
    start = max(0, index - window_before)
    end = min(len(df) - 1, index + window_after)
    subset = df.iloc[start: end + 1].copy().reset_index(drop=True)

    # Plot aufsetzen
    fig, ax = plt.subplots(figsize=(18, 9))

    # Nur Features plotten, die auch vorhanden sind
    present = [c for c in FEATURES if c in subset.columns]
    if not present:
        raise KeyError(
            f"None of the expected features are present. Expected any of: {', '.join(FEATURES)}"
        )

    for feat in present:
        label = f"{feat} ({symbol})" if symbol else feat
        ax.plot(subset.index, subset[feat], linewidth=1.5, label=label)

    # Zielzeile im Fenster markieren (rote Linie)
    target_idx = (subset["timestamp"] - target_ts).abs().idxmin()
    ax.axvline(target_idx, color="red", linestyle="--", linewidth=1.5, label="Target row")

    # Beginn neuer Handelstage markieren (grüne Linien)
    day_changes = subset["timestamp"].dt.date.ne(
        subset["timestamp"].dt.date.shift()
    ).fillna(False)
    for i, is_new_day in enumerate(day_changes):
        if is_new_day and i != 0:
            ax.axvline(i, color="green", linestyle="--", linewidth=1.0, alpha=0.7)

    # X-Achse: Ticks und Labels
    tick_positions = subset.index[:: max(1, len(subset) // 10)]
    tick_labels = subset.loc[tick_positions, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Index (1-minute bars)")
    ax.set_ylabel("Feature / target values")

    title_prefix = f"{symbol} " if symbol else ""
    ax.set_title(f"{title_prefix}Features ±{window_before} rows around {target_ts.date()}")

    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # Konfiguration laden
    # -----------------------------
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    symbols_csv = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]
    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]

    ticker_list_df = pd.read_csv(symbols_csv)
    symbols = ticker_list_df["Symbol"].dropna().tolist()

    if not symbols:
        raise ValueError("No symbols found in symbol CSV.")

    symbol = symbols[0]  # z.B. AAPL

    train_file = os.path.join(processed_path, f"{symbol}_train.parquet")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    df = pd.read_parquet(train_file)

    target_index = 2500
    target_index = min(target_index, len(df) - 1)

    plot_features_window(
        df,
        index=target_index,
        window_before=500,
        window_after=500,
        symbol=symbol,
    )


if __name__ == "__main__":
    main()
