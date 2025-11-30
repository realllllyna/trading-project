"""
This script downloads historical 1-minute adjusted bar data for S&P 500 symbols from the Alpaca Market Data API
(https://docs.alpaca.markets/docs/sdks-and-tools) for a configured date range. It reads API credentials and parameters
(data path, start/end dates, ticker list) from YAML files, retrieves 1-minute adjusted bars for each symbol, filters the
data to regular US trading hours (US/Eastern), computes intraday VWAP, and writes one cleaned Parquet file per symbol
to <DATA_PATH>/Bars_1m/.

Inputs:
- YAML config: ../../conf/params.yaml
- YAML keys:   ../../conf/keys.yaml

Outputs:
- One Parquet file per symbol under: <DATA_PATH>/Bars_1m/{TICKER}.parquet

Requirements:
- Packages: alpaca-py, pandas, pyyaml, pyarrow
"""

import os
from datetime import datetime

import pandas as pd
import yaml
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# 1. Load configuration
params = yaml.safe_load(open("../../conf/params.yaml", "r"))

PATH_BARS = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE = datetime.strptime(params["DATA_ACQUISITION"]["START_DATE"], "%Y-%m-%d")
END_DATE = datetime.strptime(params["DATA_ACQUISITION"]["END_DATE"], "%Y-%m-%d")
SYMBOLS_CSV = params["DATA_ACQUISITION"]["SYMBOLS_CSV"]

# Load Alpaca API keys from keys.yaml
keys = yaml.safe_load(open("../../conf/keys.yaml", "r"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PATH_BARS, "Bars_1m")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ticker symbols
ticker_list_df = pd.read_csv(SYMBOLS_CSV)
symbols = ticker_list_df["Symbol"].dropna().unique().tolist()

print(f"Found {len(symbols)} symbols.")


# 2. Filter all bars to regular US trading hours (09:30–16:00 ET)
def filter_regular_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Ensure timezone-aware UTC timestamps
    if df.index.tz is None:
        df = df.tz_localize("UTC")

    # Convert to US/Eastern to check clock time
    df_et = df.tz_convert("US/Eastern")
    df_et = df_et.between_time("09:30", "16:00")
    return df_et


# 3. Add intraday VWAP
def add_intraday_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["vwap"] = pd.NA
        return df

    df = df.copy()
    df["date"] = df.index.date

    df["dollar_volume"] = df["Close"] * df["Volume"]
    df["cum_dollar_volume"] = df.groupby("date")["dollar_volume"].cumsum()
    df["cum_volume"] = df.groupby("date")["Volume"].cumsum()

    df["vwap"] = df["cum_dollar_volume"] / df["cum_volume"]

    return df.drop(columns=["date", "dollar_volume", "cum_dollar_volume", "cum_volume"])


# 4. Main download loop
counter = 1

for symbol in symbols:
    print(f"{counter}. Fetching 1m bars for {symbol}")
    counter += 1

    try:
        # Build request to Alpaca Market Data
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            adjustment=Adjustment.ALL,
            start=START_DATE,
            end=END_DATE,
        )

        # Retrieve dataset
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print(f"  No data for {symbol}. Skipping.")
            continue

        # Alpaca returns MultiIndex (symbol, timestamp) -> keep only timestamp level
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values("timestamp")

        # Standardize column names
        df.index.name = "timestamp"
        df = df[["open", "high", "low", "close", "volume"]]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Filter to regular trading hours
        df = filter_regular_trading_hours(df)
        if df.empty:
            print(f"  No RTH data for {symbol}. Skipping.")
            continue

        # Add VWAP
        df = add_intraday_vwap(df)

        # Reset index and save
        df_out = df.reset_index()
        out_path = os.path.join(OUTPUT_DIR, f"{symbol}.parquet")
        df_out.to_parquet(out_path, index=False)

        print(f"  Saved {len(df_out)} rows to {out_path}")

    except Exception as e:
        print(f"  Error processing {symbol}: {e}")
        continue
