"""
Plots zur Diagnose der Feature- und Target-Qualität.
Öffnet interaktive Matplotlib-Fenster (TkAgg), sodass man Plots speichern kann.

Plots:
1. Zeitreihe eines Features
2. Histogramm eines Targets
3. Scatter Feature vs Target
4. Intraday-Volatilitätsmuster (Boxplot über Minuten)
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Interaktive GUI-Fenster aktivieren
matplotlib.use("TkAgg")


def load_config():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(this_dir, "../../conf/params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    processed_path = cfg["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    symbols_csv = cfg["DATA_ACQUISITION"]["SYMBOLS_CSV"]

    symbols = pd.read_csv(symbols_csv)["Symbol"].dropna().tolist()
    symbol = symbols[0]  # Erstes Symbol verwenden (AAPL)

    df = pd.read_parquet(os.path.join(processed_path, f"{symbol}_train.parquet"))

    print(f"Loaded {len(df)} rows for {symbol}")

    feature = "roll_vol_15m"
    target = "RV_30m"

    # ---------- Plot 1: Feature-Zeitreihe ----------
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df[feature], linewidth=0.8)
    plt.title(f"{symbol} – Time Series of {feature}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: Histogramm des Targets ----------
    plt.figure(figsize=(10, 5))
    plt.hist(df[target].dropna(), bins=60, alpha=0.7)
    plt.title(f"{symbol} – Histogram of {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 3: Scatter Feature vs Target ----------
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df[target], alpha=0.3, s=8)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{symbol}: {feature} vs {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Plot 4: Intraday Volatility Pattern ----------
    df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    plt.figure(figsize=(12, 6))
    df.boxplot(column=target, by="minute", showfliers=False, grid=True)
    plt.title(f"{symbol} – Intraday Volatility Pattern ({target})")
    plt.suptitle("")  # Entfernt "Boxplot grouped by minute"
    plt.xlabel("Minute of Day")
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
