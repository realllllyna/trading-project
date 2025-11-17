"""
End-to-end pre-split data preparation script.

This script loads configuration from the experiment's params.yaml, iterates over a
list of S&P 500 ticker symbols, and for each symbol performs the following steps:
- Load 1-minute adjusted bars from Parquet
- Compute forward-looking targets (normalized trend slope and percentage change)
- Engineer normalized technical analysis features (EMA, slopes, second-order slopes)
- Drop rows with NaNs introduced by rolling/slope operations
- Split the data into train/validation/test partitions by timestamp boundaries
- Persist each split to Parquet under the configured processed data path

Artifacts:
- A plain-text file `features.txt` listing the engineered feature columns (written once).
- Per-symbol Parquet files: `<symbol>_train.parquet`, `<symbol>_validation.parquet`, `<symbol>_test.parquet`.

Assumptions:
- The configuration file and data files are accessible via the relative paths below.
- Bar data Parquet files include at least: 'timestamp', 'vwap', 'volume'.
- Timestamp is comparable to the configured TRAIN/VALIDATION/TEST date cutoffs.

Usage:
- Run this script from its directory (or project root if paths are adjusted accordingly).
"""

import os
import pandas as pd
import yaml
import targets, features  # local modules in the same folder

# Alternative import form if running from project root:
# from experiments.exp_1_2.data_prep_pre_split import targets, features

# Load run configuration and universe list.
params = yaml.safe_load(open("../../conf/params.yaml"))
# S&P 500 symbols reference file
ticker_list_df = pd.read_csv('../../data/sp500_companies_as_of_jan_2025.csv')

# Alternative absolute paths (when executing from project root):
# params = yaml.safe_load(open("experiments/exp_1_2/conf/params.yaml"))
# ticker_list_df = pd.read_csv('experiments/exp_1_2/data/sp500_companies_as_of_jan_2025.csv')

# Unpack relevant parameters for feature calculation.
prediction_periods = params['DATA_PREP']['PREDICTION_PERIODS']
ema_periods = params['DATA_PREP']['EMA_PERIODS']
slope_periods = params['DATA_PREP']['SLOPE_PERIODS']
z_norm_window = params['DATA_PREP']['Z_NORM_WINDOW']

# Unpack data paths and ensure processed data directory exists.
data_path = params['DATA_ACQUISITON']['DATA_PATH']
processed_path = params['DATA_PREP']['PROCESSED_PATH']
os.makedirs(processed_path, exist_ok=True)

# Unpack date boundaries for train/validation/test splits.
train_date = params['DATA_PREP']['TRAIN_DATE']
validation_date = params['DATA_PREP']['VALIDATION_DATE']
test_date = params['DATA_PREP']['TEST_DATE']

# Start processing from a given offset within the symbol list (useful for chunking runs).
counter = 76

# Iterate over each ticker symbol starting from 'counter'
for symbol in ticker_list_df['Symbol'][counter:]:

    print(f"{counter}. Processing features for {symbol}")
    counter += 1

    # Load 1-minute bar data for the current symbol
    bars_file = f"{data_path}/Bars_1m_adj/{symbol}.parquet"
    bars_df = pd.read_parquet(bars_file)
    bars_df['symbol'] = symbol

    # Add trend direction targets to bar data for multiple prediction periods
    bars_df = targets.add_normalized_trend_direction(bars_df, prediction_periods=prediction_periods)

    # Add normalized technical analysis features to the bar data
    bars_df, features_added = features.generate_features(
        bars_df,
        ema_periods=ema_periods,
        slope_periods=slope_periods,
        z_norm_window=z_norm_window,
    )

    # Drop rows with NaNs created by feature engineering to finalize the dataset
    bars_df = bars_df.dropna().reset_index(drop=True)

    # Save list of features added for reference if features.txt does not exist yet
    if not os.path.exists("features.txt"):
        with open("features.txt", "w") as f:
            for feat in features_added:
                f.write(f"{feat}\n")

    # Split into train, validation, and test sets and save the processed data to Parquet files
    train = bars_df[bars_df['timestamp'] <= train_date]
    train.to_parquet(f"{processed_path}/{symbol}_train.parquet", index=False)

    validation = bars_df[(bars_df['timestamp'] > train_date) & (bars_df['timestamp'] <= validation_date)]
    validation.to_parquet(f"{processed_path}/{symbol}_validation.parquet", index=False)

    test = bars_df[(bars_df['timestamp'] > validation_date) & (bars_df['timestamp'] <= test_date)]
    test.to_parquet(f"{processed_path}/{symbol}_test.parquet", index=False)
