# Python
"""
This script downloads historical 1-minute adjusted bar data for S&P 500 symbols from the Alpaca Market Data API
(https://docs.alpaca.markets/docs/sdks-and-tools) for a configured date range. It reads API credentials and parameters
(data path, start/end dates) from YAML files, retrieves the official US trading calendar, marks bars that occur during
regular market hours (US/Eastern),filters out non-regular trading hours, and writes one cleaned Parquet file per symbol.

Inputs:
- YAML configs: ../../conf/keys.yaml (API keys), ../../conf/params.yaml (data path and date range)
- Symbols CSV: ../../data/sp500_companies_as_of_jan_2025.csv

Outputs:
- One Parquet per symbol under <DATA_PATH>/Bars_1m_adj/

Requirements:
- Packages: alpaca-py, pandas, pytz, pyyaml, pyarrow
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
import pandas as pd
from datetime import datetime
import pytz
import yaml

# Load API credentials from YAML configuration file
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys['KEYS']['APCA-API-KEY-ID-Data']
SECRET_KEY = keys['KEYS']['APCA-API-SECRET-KEY-Data']

# Load data acquisition parameters from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params['DATA_ACQUISITON']['DATA_PATH']
START_DATE = datetime.strptime(params['DATA_ACQUISITON']['START_DATE'], "%Y-%m-%d")
END_DATE = datetime.strptime(params['DATA_ACQUISITON']['END_DATE'], "%Y-%m-%d")

# Initialize the Alpaca client with API credentials
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY)

# Load the list of ticker symbols from a CSV file
ticker_list_df = pd.read_csv('../../data/sp500_companies_as_of_jan_2025.csv')

# Get market calendar for that period
cal_request = GetCalendarRequest(start=START_DATE, end=END_DATE)
calendar = trading_client.get_calendar(cal_request)

# Build lookup table (date → open_dt, close_dt)
cal_map = {}
eastern = pytz.timezone("US/Eastern")

for c in calendar:
    # c.open and c.close are datetime.datetime (naive, in ET)
    open_dt = eastern.localize(c.open)  # set tzinfo to US/Eastern
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

# Add market open flag
def check_open(ts):
    ts_eastern = ts.tz_convert(eastern) if ts.tzinfo else ts.tz_localize("UTC").astimezone(eastern)
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

counter = 448

# Iterate over each ticker symbol and fetch 1-minute bar data
for symbol in ticker_list_df['Symbol'][counter:]:
    print(f"{counter}. Fetching 1m bars for {symbol} from {START_DATE} to {END_DATE}")
    counter += 1

    # Create a request object for historical bar data
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        adjustment=Adjustment.ALL, # adjusts for splits and dividends
        start=START_DATE,
        end=END_DATE
    )

    # Retrieve bar data from Alpaca API
    bars = client.get_stock_bars(request)
    df = bars.df
    df.reset_index(inplace=True)
    # Remove the 'symbol' column if it exists
    if 'symbol' in df.columns:
        df.drop(columns=['symbol'], inplace=True)

    df["is_open"] = df["timestamp"].map(check_open)

    # Filter the DataFrame to include only rows where the market was open
    df = df[df['is_open']]

    df.drop(columns=['is_open'], inplace=True)

    # Save the DataFrame as a Parquet file for efficient storage
    df.to_parquet(f'{PATH_BARS}/Bars_1m_adj/{symbol}.parquet', index=False)
