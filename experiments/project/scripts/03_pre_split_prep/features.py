"""
Feature engineering utilities for volatility experiment.

Aus 1-Minuten-Bars werden Eingangsfeatures für das LSTM gebaut.
Verwendet werden nur Informationen aus der Vergangenheit (bis Zeitpunkt τ).

Berechnete Features (pro Symbol, pro Minute):
- 1-Minuten-Log-Return
- Rolling Returns (5 Minuten)
- Rolling Volatilität (15 Minuten)
- Abweichung vom VWAP
- VWAP-Z-Score (30 Minuten)
- Volume-Z-Score (30 Minuten) + Rolling Volume (15 Minuten)
- High-Low-Spread (aktuelle Minute)
- Rolling High-Low-Range (15 Minuten)
- Zeitmerkmale: Minute im Handelstag (sin/cos), Dummy für Eröffnung/Schlussphase

Inputs:
- DataFrame mit Spalten: 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'vwap'

Outputs:
- DataFrame mit zusätzlichen Feature-Spalten
- Liste der neuen Feature-Namen (für features.txt)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def generate_features(
        bars_df: pd.DataFrame,
        sequence_length: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    df = bars_df.copy()

    # Sicherstellen, dass timestamp Datetime ist
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # 1. Preisbasierte Features
    # -----------------------------
    # 1-Minuten-Log-Return
    df["log_ret_1m"] = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling Return (letzte 5 Minuten)
    df["roll_ret_5m"] = df["log_ret_1m"].rolling(5).sum()

    # Rolling Volatilität (letzte 15 Minuten)
    df["roll_vol_15m"] = (df["log_ret_1m"] ** 2).rolling(15).mean().pow(0.5)

    # -----------------------------
    # 2. VWAP & Abweichung
    # -----------------------------
    df["vwap_dev"] = (df["Close"] - df["vwap"]) / df["vwap"]

    # VWAP-Z-Score (30 Minuten)
    vwap_roll_mean = df["vwap"].rolling(30).mean()
    vwap_roll_std = df["vwap"].rolling(30).std(ddof=0)
    df["vwap_z_30m"] = (df["vwap"] - vwap_roll_mean) / (vwap_roll_std + EPS)

    # -----------------------------
    # 3. Volumen & Liquidität
    # -----------------------------
    vol_roll_mean = df["Volume"].rolling(30).mean()
    vol_roll_std = df["Volume"].rolling(30).std(ddof=0)
    df["volume_z_30m"] = (df["Volume"] - vol_roll_mean) / (vol_roll_std + EPS)

    df["roll_vol_15m_volume"] = df["Volume"].rolling(15).mean()

    # -----------------------------
    # 4. Handelsbereich
    # -----------------------------
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

    # Rolling-Range (15 Minuten)
    high_max_15 = df["High"].rolling(15).max()
    low_min_15 = df["Low"].rolling(15).min()
    df["hl_range_15m"] = (high_max_15 - low_min_15) / df["Close"]

    # -----------------------------
    # 5. Zeitliche Merkmale
    # -----------------------------
    ts = df["timestamp"]
    # Falls Zeitzone gesetzt ist, nach US/Eastern konvertieren, sonst unverändert
    if ts.dt.tz is not None:
        ts_local = ts.dt.tz_convert("US/Eastern")
    else:
        ts_local = ts

    minute_of_day = ts_local.dt.hour * 60 + ts_local.dt.minute  # 0..389 typischerweise
    df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 390.0)
    df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 390.0)

    df["is_open30"] = (minute_of_day < 30).astype(int)
    df["is_close30"] = (minute_of_day >= 390 - 30).astype(int)

    # -----------------------------
    # Featureliste
    # -----------------------------
    original_cols = bars_df.columns.tolist()
    features_added = [c for c in df.columns if c not in original_cols]

    return df, features_added
