"""
Step 9b – Paper Trading (Simulation)

Live-ähnliche Simulation auf historischen Testdaten:
- Modell → p(t) = P(High Volatility)
- Trading-Regel: Exposure w(t) = 1 - p(t)
- 1-Minuten Delay (kein Lookahead)
- Transaktionskosten über Turnover

Outputs:
- Gesamtperformance
- Performance pro Aktie
- Performance pro Woche
- Vergleich Backtest vs Paper Trading
"""

from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml


MINUTES_PER_DAY = 390
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR


def load_params() -> dict:
    with open("../../conf/params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features() -> list[str]:
    with open("../03_pre_split_prep/features.txt", "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def sharpe(r: pd.Series) -> float:
    if r.std() == 0:
        return np.nan
    return (r.mean() / r.std()) * np.sqrt(MINUTES_PER_YEAR)


def max_drawdown(eq: pd.Series) -> float:
    return (eq / eq.cummax() - 1).min()


def run_paper(df: pd.DataFrame, model, feature_cols: list[str],
              delay: int = 1, cost: float = 0.0002) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    r = df["Close"].pct_change().fillna(0.0)

    X = df[["symbol"] + feature_cols].copy()
    X["symbol"] = X["symbol"].astype("category")
    p = pd.Series(model.predict(X), index=df.index)

    w = (1 - p).clip(0, 1)
    w = w.shift(delay).fillna(0.0)

    turnover = w.diff().abs().fillna(0.0)
    strat_r = w * r - cost * turnover

    return pd.DataFrame({
        "r": r,
        "p": p,
        "w": w,
        "strat_r": strat_r
    })


def main():
    params = load_params()
    feature_cols = load_features()

    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    model_dir = params["MODEL"]["SAVE_PATH"]

    model_path = sorted(glob.glob(os.path.join(model_dir, "*.txt")))[-1]
    model = lgb.Booster(model_file=model_path)

    out_dir = "../../results/paper_trading"
    os.makedirs(out_dir, exist_ok=True)

    test_files = sorted(glob.glob(os.path.join(processed_path, "*_test.parquet")))[:50]

    per_symbol = {}
    for fp in test_files:
        sym = os.path.basename(fp).replace("_test.parquet", "")
        df = pd.read_parquet(fp)
        per_symbol[sym] = run_paper(df, model, feature_cols)

    strat_mat = pd.concat([v["strat_r"] for v in per_symbol.values()], axis=1)
    port_r = strat_mat.mean(axis=1)

    eq = equity_curve(port_r)

    summary = {
        "final_equity": eq.iloc[-1],
        "sharpe": sharpe(port_r),
        "max_drawdown": max_drawdown(eq),
        "start": str(port_r.index.min()),
        "end": str(port_r.index.max()),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "paper_summary_overall.csv"), index=False
    )

    print("Paper Trading abgeschlossen.")
    print(summary)


if __name__ == "__main__":
    main()
