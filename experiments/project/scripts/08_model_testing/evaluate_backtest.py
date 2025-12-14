"""
Backtesting trading algorithm derived from the volatility model.

Derivation:
- Model outputs p(t) = P(high volatility next horizon)
- Trading algorithm: risk timing via exposure w(t) = 1 - p(t) (clipped)
- Execution delay: signal at t executed at t+1 minute (no lookahead)

Outputs:
- performance_summary.csv
- portfolio_timeseries.csv
- equity_curve.png
- example plots
- trading points distribution plots
"""

from __future__ import annotations

import os
import pandas as pd

from tree_utilities import load_params, load_feature_list, find_latest_model, load_model, list_test_files
from performance import equity_curve, sharpe_annualized, max_drawdown, turnover
from plotting import plot_equity, plot_example, plot_trading_points_distribution


def exposure_rule(p: pd.Series) -> pd.Series:
    return (1.0 - p.clip(0.0, 1.0)).clip(0.0, 1.0)


def apply_delay(w: pd.Series, steps: int = 1) -> pd.Series:
    return w.shift(steps).fillna(0.0)


def backtest_symbol(df: pd.DataFrame, model, feature_cols: list[str], delay_steps: int, cost_per_turnover: float) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    r = df["Close"].astype(float).pct_change().fillna(0.0)

    X = df[["symbol"] + feature_cols].copy()
    X["symbol"] = X["symbol"].astype("category")

    p = pd.Series(model.predict(X), index=df.index, name="p")
    w = apply_delay(exposure_rule(p), delay_steps)

    to = w.diff().abs().fillna(0.0)
    costs = cost_per_turnover * to
    strat_r = (w * r) - costs

    return pd.DataFrame({"Close": df["Close"], "r": r, "p": p, "w": w, "strat_r": strat_r}, index=df.index)


def main():
    params = load_params()
    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    model_dir = params["MODEL"]["SAVE_PATH"]

    feature_cols = load_feature_list()
    model_path = find_latest_model(model_dir)
    model = load_model(model_path)

    out_dir = os.path.join("experiments", "project", "results", "backtest")
    os.makedirs(out_dir, exist_ok=True)

    files = list_test_files(processed_path, max_symbols=50)

    per_symbol = {}
    for fp in files:
        sym = os.path.basename(fp).replace("_test.parquet", "")
        df = pd.read_parquet(fp)
        try:
            per_symbol[sym] = backtest_symbol(df, model, feature_cols, delay_steps=1, cost_per_turnover=0.0002)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    # Portfolio = equal-weight mean return across symbols per minute
    strat_mat = pd.concat([v["strat_r"].rename(k) for k, v in per_symbol.items()], axis=1).sort_index()
    bench_mat = pd.concat([v["r"].rename(k) for k, v in per_symbol.items()], axis=1).sort_index()
    w_mat = pd.concat([v["w"].rename(k) for k, v in per_symbol.items()], axis=1).sort_index()

    port_strat = strat_mat.mean(axis=1, skipna=True)
    port_bench = bench_mat.mean(axis=1, skipna=True)
    port_w = w_mat.mean(axis=1, skipna=True)

    eq_strat = equity_curve(port_strat)
    eq_bench = equity_curve(port_bench)

    perf = {
        "final_equity": float(eq_strat.iloc[-1]),
        "sharpe_ann": sharpe_annualized(port_strat),
        "max_drawdown": max_drawdown(eq_strat),
        "turnover": turnover(port_w),
    }
    perf_bench = {
        "final_equity": float(eq_bench.iloc[-1]),
        "sharpe_ann": sharpe_annualized(port_bench),
        "max_drawdown": max_drawdown(eq_bench),
        "turnover": 0.0,
    }

    pd.DataFrame([perf, perf_bench], index=["strategy", "benchmark"]).to_csv(
        os.path.join(out_dir, "performance_summary.csv")
    )

    pd.DataFrame({"port_strat_r": port_strat, "port_bench_r": port_bench, "port_w": port_w}).to_csv(
        os.path.join(out_dir, "portfolio_timeseries.csv")
    )

    plot_equity(eq_strat, eq_bench, os.path.join(out_dir, "equity_curve.png"))

    # Example plots (one symbol, one date)
    ex_symbol = "AAPL" if "AAPL" in per_symbol else list(per_symbol.keys())[0]
    ex_df = per_symbol[ex_symbol]
    ex_date = str(ex_df.index.date[0])  # first available date
    day_df = ex_df[ex_df.index.date.astype(str) == ex_date]
    plot_example(day_df, out_dir, ex_symbol, ex_date)

    # Distribution of trading points over time
    plot_trading_points_distribution(port_w, out_dir)

    print("Saved backtest results to:", out_dir)


if __name__ == "__main__":
    main()
