"""
Feature engineering utilities for volatility experiment.

Aus 1-Minuten-Bars werden Eingangsfeatures gebaut.
Verwendet werden nur Informationen aus der Vergangenheit (bis Zeitpunkt τ).

WICHTIG:
- Rolling-Features werden pro Handelstag berechnet (keine Vermischung über Nacht).
- Zeitfeatures werden in US/Eastern berechnet (DST-safe, falls tz-aware Timestamps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def generate_features(bars_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = bars_df.copy()

    # Sicherstellen, dass timestamp Datetime ist
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lokale Zeit (US/Eastern) für Session-Gruppierung & Zeitfeatures
    ts = df["timestamp"]
    if ts.dt.tz is not None:
        ts_local = ts.dt.tz_convert("US/Eastern")
    else:
        # Falls doch mal naive timestamps auftauchen: best-effort (keine Annahme über UTC)
        ts_local = ts

    df["_ts_local"] = ts_local
    df["_date"] = df["_ts_local"].dt.date

    def per_day(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()

        # -----------------------------
        # 1. Preisbasierte Features
        # -----------------------------
        d["log_ret_1m"] = np.log(d["Close"] / d["Close"].shift(1))
        d["roll_ret_5m"] = d["log_ret_1m"].rolling(5).sum()
        d["roll_vol_15m"] = (d["log_ret_1m"] ** 2).rolling(15).mean().pow(0.5)

        # -----------------------------
        # 2. VWAP & Abweichung
        # -----------------------------
        d["vwap_dev"] = (d["Close"] - d["vwap"]) / (d["vwap"] + EPS)

        vwap_roll_mean = d["vwap"].rolling(30).mean()
        vwap_roll_std = d["vwap"].rolling(30).std(ddof=0)
        d["vwap_z_30m"] = (d["vwap"] - vwap_roll_mean) / (vwap_roll_std + EPS)

        # -----------------------------
        # 3. Volumen & Liquidität
        # -----------------------------
        vol_roll_mean = d["Volume"].rolling(30).mean()
        vol_roll_std = d["Volume"].rolling(30).std(ddof=0)
        d["volume_z_30m"] = (d["Volume"] - vol_roll_mean) / (vol_roll_std + EPS)

        d["roll_vol_15m_volume"] = d["Volume"].rolling(15).mean()

        # -----------------------------
        # 4. Handelsbereich
        # -----------------------------
        d["hl_spread"] = (d["High"] - d["Low"]) / (d["Close"] + EPS)

        high_max_15 = d["High"].rolling(15).max()
        low_min_15 = d["Low"].rolling(15).min()
        d["hl_range_15m"] = (high_max_15 - low_min_15) / (d["Close"] + EPS)

        # -----------------------------
        # 5. Zeitliche Merkmale (pro Session)
        # -----------------------------
        ts_loc = d["_ts_local"]
        minute_of_session = (ts_loc.dt.hour - 9) * 60 + (ts_loc.dt.minute - 30)

        # Sicherheit: Clip (falls doch Pre/Post-Market in Daten landet)
        minute_clip = minute_of_session.clip(lower=0, upper=389)

        d["time_sin"] = np.sin(2 * np.pi * minute_clip / 390.0)
        d["time_cos"] = np.cos(2 * np.pi * minute_clip / 390.0)

        d["is_open30"] = (minute_of_session < 30).astype(int)
        d["is_close30"] = (minute_of_session >= 390 - 30).astype(int)

        return d

    # pro Handelstag berechnen (keine overnight Rollings)
    df = df.groupby("_date", group_keys=False).apply(per_day)

    # Featureliste
    original_cols = bars_df.columns.tolist()

    # Hilfsspalten entfernen
    df = df.drop(columns=["_ts_local", "_date"])

    features_added = [c for c in df.columns if c not in original_cols]
    return df, features_added
