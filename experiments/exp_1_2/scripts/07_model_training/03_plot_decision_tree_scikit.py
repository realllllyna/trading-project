"""
Load a trained scikit-learn DecisionTree (decision_tree_scikit.pkl) and plot it using tree_utilities.py.
Also exports a basic node statistics CSV for quick inspection.

Outputs (default under MODEL_PATH from params.yaml):
- decision_tree_scikit.png (tree plot)
- tree_stats.csv (per-node stats)
"""

import os
import sys
import pickle
import yaml

# Ensure we can import sibling module tree_utilities.py
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from tree_utilities import plot_tree, get_tree_stats  # noqa: E402


def main():
    # Load config
    params = yaml.safe_load(open(os.path.join(THIS_DIR, "../../conf/params.yaml")))
    model_path = params['MODELING']['MODEL_PATH']

    # Model file
    model_file = os.path.join(model_path, "decision_tree_scikit.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Train it first with 02_decision_tree_scikit.py.")

    # Load model bundle
    with open(model_file, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle.get("model")
    embed_dim = int(bundle.get("embed_dim", getattr(clf, 'n_features_in_', 0)))

    # Build feature names for embeddings (emb_0 .. emb_{embed_dim-1})
    feature_cols = [f"emb_{i}" for i in range(embed_dim)]

    # Output paths
    os.makedirs(model_path, exist_ok=True)
    png_path = os.path.join(model_path, "decision_tree_scikit.png")
    csv_stats_path = os.path.join(model_path, "tree_stats.csv")

    # Plot and save
    print(f"[PLOT] Saving tree plot to: {png_path}")
    plot_tree(clf, feature_cols=feature_cols, path=png_path)

    # Export tree stats
    print(f"[STATS] Computing per-node stats -> {csv_stats_path}")
    stats_df = get_tree_stats(clf, feature_cols=feature_cols)
    stats_df.to_csv(csv_stats_path, index=False)

    print("[DONE] Tree plot and stats exported.")


if __name__ == "__main__":
    main()
