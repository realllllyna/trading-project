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
    with open("../../conf/params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_path = params["DATA_ACQUISITION"]["DATA_PATH"]
    symbols_csv = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]

    vola_windows = params["FEATURE_ENGINEERING"]["VOLA_WINDOWS"]
    processed_path = params["FEATURE_ENGINEERING"]["PROCESSED_PATH"]
    os.makedirs(processed_path, exist_ok=True)

    train_end = pd.to_datetime(params["DATA_SPLIT"]["TRAIN_END"]).date()
    valid_end = pd.to_datetime(params["DATA_SPLIT"]["VALID_END"]).date()
    test_end = pd.to_datetime(params["DATA_SPLIT"]["TEST_END"]).date()

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

        # -----------------------------
        # 2a) Targets: NUR RV berechnen (noch keine Labels!)
        # -----------------------------
        df = targets.add_realized_volatility(
            df,
            vola_windows=vola_windows,
            price_col="Close",
        )

        # -----------------------------
        # 2b) Features hinzufügen
        # -----------------------------
        df, feat_cols = features.generate_features(df)

        # NaNs entfernen (von Rolling/Shift / RV)
        df = df.dropna().reset_index(drop=True)

        # features.txt nur einmal schreiben
        if not feature_file_written:
            feat_path = os.path.join(os.path.dirname(__file__), "features.txt")
            with open(feat_path, "w", encoding="utf-8") as f:
                for col in feat_cols:
                    f.write(col + "\n")
            feature_file_written = True

        # -----------------------------
        # 3) Zeitliche Splits (auf Tagesebene)
        # -----------------------------
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ts_date = df["timestamp"].dt.date

        train = df[ts_date <= train_end].copy()
        valid = df[(ts_date > train_end) & (ts_date <= valid_end)].copy()
        test = df[(ts_date > valid_end) & (ts_date <= test_end)].copy()

        # -----------------------------
        # 4) Labels leakage-frei: Thresholds NUR auf Train fitten
        # -----------------------------
        thresholds = targets.fit_volatility_thresholds(
            train,
            vola_windows=vola_windows,
            high_vol_quantile=0.7,  # falls du es config-gesteuert willst: aus params lesen
        )

        train = targets.apply_volatility_labels(train, vola_windows=vola_windows, thresholds=thresholds)
        valid = targets.apply_volatility_labels(valid, vola_windows=vola_windows, thresholds=thresholds)
        test = targets.apply_volatility_labels(test, vola_windows=vola_windows, thresholds=thresholds)

        # -----------------------------
        # 5) Speichern
        # -----------------------------
        train.to_parquet(os.path.join(processed_path, f"{symbol}_train.parquet"), index=False)
        valid.to_parquet(os.path.join(processed_path, f"{symbol}_validation.parquet"), index=False)
        test.to_parquet(os.path.join(processed_path, f"{symbol}_test.parquet"), index=False)

        print(
            f"  Saved: {len(train)} train, {len(valid)} val, {len(test)} test rows for {symbol}"
        )


if __name__ == "__main__":
    main()
