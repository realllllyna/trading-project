"""
Feature selection helper: compute and inspect correlations of features with a target.

This script loads project parameters and a predefined feature list, then:
1) Loads a preprocessed training shard parquet (shuffled) from the configured path.
2) Subsets the DataFrame to the selected feature columns plus the target column.
3) Drops rows with missing values to ensure valid correlation computation.
4) Computes the full Pearson correlation matrix.
5) Extracts and sorts correlations w.r.t. the configured target variable, descending.

Intended usage
- Run from the project root so the relative paths in this script resolve correctly.
- Inspect `sorted_corr` in an interactive session (e.g., print head, visualize, or export to CSV).

Notes
- The feature list is read from a text file with one feature name per line.
- Correlation uses pandas' default method (Pearson). Adjust as needed for other measures.
- For large datasets, consider sampling to accelerate iterations.
- To persist results, you can add: `sorted_corr.to_csv('correlations_by_target.csv')`.
"""

import pandas as pd
import yaml

# Load parameters for the experiment. When executing from project root, this relative path should resolve.
# Alternative path (when executing from inside the folder) is kept commented above for reference.
# params = yaml.safe_load(open("../../conf/params.yaml"))
params = yaml.safe_load(open("experiments/exp_1_2/conf/params.yaml"))

# The processed (pre-split) data path; here we use the shuffled variant as source for correlation analysis.
processed_path = params['DATA_PREP']['PROCESSED_PATH'] + "_shuffled"

# The target variable name used to select the label column and to sort correlations.
target = params['MODELING']['TARGET']

# Path to the list of feature names (one per line). The alternative path from config is kept for reference.
# feature_path = params['DATA_PREP'].get('FEATURE_PATH')
feature_path = 'experiments/exp_1_2/scripts/03_pre_split_prep/features.txt'

# Read features from txt file (one per line), stripping whitespace/newlines.
features = []
with open(feature_path, "r") as f:
    for line in f:
        features.append(line.strip())

# Load a single training shard parquet as the source DataFrame for correlation analysis.
# You can iterate shards or concatenate them if you want a larger sample.
df = pd.read_parquet(f"{processed_path}/train_shard_0.parquet")

# Keep only the selected features and the target, and drop rows with any missing values to avoid NaNs in correlations.
df = df[features + [target]].dropna().reset_index(drop=True)

# Compute the full correlation matrix (Pearson by default).
corr_matrix = df.corr()

# Extract and sort the correlations with respect to the target (highest first) for quick inspection.
sorted_corr = corr_matrix[target].sort_values(ascending=False)