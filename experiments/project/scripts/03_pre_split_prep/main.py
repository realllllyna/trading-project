"""
End-to-end pre-split data preparation for volatility experiment.

Für jedes S&P-500-Symbol werden:
- 1-Minuten-Bars aus Parquet geladen
- Volatilitäts-Targets (realisierte RV + High/Low-Labels) berechnet
- Eingangsfeatures (Preis, VWAP, Volumen, Zeitmerkmale) berechnet
- Zeilen mit NaNs (von Rolling/Shift) entfernt
- Die Daten zeitlich in Train / Validation / Test aufgeteilt
- Je Split in einem eigenen Parquet pro Symbol gespeichert

Außerdem wird einmalig eine features.txt erzeugt, die alle Feature-Spalten listet.
"""

from __future__ import annotations

import os

import pandas as pd
import yaml

import features
import targets


def main():
    # -----------------------------
    # 1. Konfiguration laden
    # -----------------------------
    params = yaml.safe_load(open("../../conf/params.yaml", "r"))

    data_path = params["DATA_ACQUISITION"]["DATA_PATH"]
    symbols_csv = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]

    vola_windows = params["FEATURE_ENGINEERING"]["VOLA_WINDOWS"]
    seq_len = params["FEATURE_ENGINEERING"]["SEQUENCE_LENGTH"]
    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    os.makedirs(processed_path, exist_ok=True)

    train_end = pd.to_datetime(params["DATA_SPLIT"]["TRAIN_END"])
    valid_end = pd.to_datetime(params["DATA_SPLIT"]["VALID_END"])
    test_end = pd.to_datetime(params["DATA_SPLIT"]["TEST_END"])

    ticker_list_df = pd.read_csv(symbols_csv)
    symbols = ticker_list_df["Symbol"].dropna().tolist()

    # -----------------------------
    # 2. Über alle Symbole iterieren
    # -----------------------------
    counter = 1
    feature_file_written = False

    for symbol in symbols:
        print(f"{counter}. Processing {symbol}")
        counter += 1

        bars_file = os.path.join(data_path, "Bars_1m", f"{symbol}.parquet")
        if not os.path.exists(bars_file):
            print(f"  File not found: {bars_file}, skipping.")
            continue

        df = pd.read_parquet(bars_file)
        df["symbol"] = symbol

        # Targets hinzufügen
        df = targets.add_volatility_targets(
            df,
            vola_windows=vola_windows,
            price_col="Close",
        )

        # Features hinzufügen
        df, feat_cols = features.generate_features(df, sequence_length=seq_len)

        # NaNs entfernen (von Rolling/Shift)
        df = df.dropna().reset_index(drop=True)

        # features.txt nur einmal schreiben
        if not feature_file_written:
            feat_path = os.path.join(os.path.dirname(__file__), "features.txt")
            with open(feat_path, "w", encoding="utf-8") as f:
                for col in feat_cols:
                    f.write(col + "\n")
            feature_file_written = True

        # -----------------------------
        # 3. Zeitliche Splits
        # -----------------------------
        # timestamp in Datetime umwandeln
        ts = pd.to_datetime(df["timestamp"])

        # Wenn Zeitzone vorhanden ist (z.B. US/Eastern), entfernen wir sie für den Vergleich
        if ts.dt.tz is not None:
            # Zeitzone entfernen → nur "normale" Datumswerte
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

        df["timestamp"] = ts

        train = df[df["timestamp"] <= train_end]
        valid = df[(df["timestamp"] > train_end) & (df["timestamp"] <= valid_end)]
        test = df[(df["timestamp"] > valid_end) & (df["timestamp"] <= test_end)]

        # Speichern
        train.to_parquet(os.path.join(processed_path, f"{symbol}_train.parquet"), index=False)
        valid.to_parquet(os.path.join(processed_path, f"{symbol}_validation.parquet"), index=False)
        test.to_parquet(os.path.join(processed_path, f"{symbol}_test.parquet"), index=False)

        print(
            f"  Saved: {len(train)} train, {len(valid)} val, {len(test)} test rows "
            f"for {symbol}"
        )


if __name__ == "__main__":
    main()
