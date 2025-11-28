"""
Target engineering utilities for the volatility experiment.

Für jede Minute und jeden Prognosehorizont t in VOLA_WINDOWS werden berechnet:
- zukünftige realisierte Volatilität RV_t(τ) über die nächsten t Minuten
- tages-normalisierte Volatilität (RV_t / Tagesmittel)
- binäres Label: High Volatility (1) / Low/Normal Volatility (0)
  basierend auf einem Quantils-Schwellenwert (Standard: oberste 30 %)

Inputs:
- DataFrame mit 'timestamp' und einem Preisfeld (standard: 'Close')

Outputs:
- DataFrame mit zusätzlichen Spalten:
  RV_{t}m, RV_norm_{t}m, vol_label_{t}m
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_targets(
        bars_df: pd.DataFrame,
        vola_windows: list[int],
        price_col: str = "Close",
        high_vol_quantile: float = 0.7,
) -> pd.DataFrame:
    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1-Minuten-Log-Returns für das Target (kann parallel zu Feature-Version existieren)
    log_ret = np.log(df[price_col] / df[price_col].shift(1))

    # Datumsspalte (für Tagesmittel)
    date = df["timestamp"].dt.date
    df["date"] = date

    for t in vola_windows:
        # Realisierte Volatilität über die nächsten t Minuten:
        # RV_t(τ) = sqrt( sum_{k=τ+1}^{τ+t} r_k^2 )
        # Implementiert über Rolling-Summe, nach vorne verschoben.
        sq_ret = (log_ret ** 2).rolling(t).sum().shift(-t)
        rv = np.sqrt(sq_ret)
        col_rv = f"RV_{t}m"
        df[col_rv] = rv

        # Tagesmittel pro Symbol (hier pro Datei) und Periode
        daily_mean = df.groupby("date")[col_rv].transform("mean")
        col_rv_norm = f"RV_norm_{t}m"
        df[col_rv_norm] = df[col_rv] / daily_mean

        # Globale Schwelle für High Volatility (oberes Quantil)
        threshold = df[col_rv_norm].quantile(high_vol_quantile)
        col_label = f"vol_label_{t}m"
        df[col_label] = (df[col_rv_norm] >= threshold).astype(int)

    return df
