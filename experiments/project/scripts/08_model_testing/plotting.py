"""
Plotting utilities:
- Equity curve (strategy vs benchmark)
- Example plots (price, p(t), w(t))
- Trading points distribution (per day, per hour)
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity(eq_strat: pd.Series, eq_bench: pd.Series, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(eq_strat.index, eq_strat.values, label="Strategy")
    plt.plot(eq_bench.index, eq_bench.values, label="Benchmark (Buy&Hold)")
    plt.legend()
    plt.title("Equity Curve")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_example(df_day: pd.DataFrame, out_dir: str, symbol: str, date_str: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(df_day.index, df_day["Close"].values)
    plt.title(f"{symbol} {date_str} - Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example_{symbol}_{date_str}_price.png"))
    plt.close()

    plt.figure()
    plt.plot(df_day.index, df_day["p"].values)
    plt.title(f"{symbol} {date_str} - p(t) HighVol")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example_{symbol}_{date_str}_prob.png"))
    plt.close()

    plt.figure()
    plt.plot(df_day.index, df_day["w"].values)
    plt.title(f"{symbol} {date_str} - Exposure w(t)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"example_{symbol}_{date_str}_exposure.png"))
    plt.close()


def plot_trading_points_distribution(w: pd.Series, out_dir: str) -> None:
    """
    Trading points are proxied by exposure changes: |Δw| > 0.
    """
    os.makedirs(out_dir, exist_ok=True)

    changes = (w.diff().abs().fillna(0.0) > 0).astype(int)
    df = pd.DataFrame({"changes": changes}, index=w.index)
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    per_day = df.groupby("date")["changes"].sum()
    plt.figure()
    plt.plot(per_day.index, per_day.values)
    plt.title("Trading points per day (|Δw|>0)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trading_points_per_day.png"))
    plt.close()

    per_hour = df.groupby("hour")["changes"].sum()
    plt.figure()
    plt.bar(per_hour.index.astype(int), per_hour.values)
    plt.title("Trading points by hour (|Δw|>0)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trading_points_by_hour.png"))
    plt.close()
