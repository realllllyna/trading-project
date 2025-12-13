"""
metrics.py - Minimal evaluation metrics for volatility classification.

Metrics:
- ROC-AUC
- LogLoss
- Balanced Accuracy
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, balanced_accuracy_score


def evaluate(
        y_true,
        y_prob,
        threshold: float = 0.5,
) -> dict:
    """
    Evaluate binary classification predictions.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1)
    y_prob : array-like
        Predicted probabilities for class 1
    threshold : float
        Probability threshold for hard predictions

    Returns
    -------
    dict with keys:
        - auc
        - logloss
        - balanced_accuracy
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
