from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
) -> dict:
    """
    Berechnet einige Standardmetriken für binäre Klassifikation.
    y_true: array der Labels (0/1)
    y_prob: array der vorhergesagten Wahrscheinlichkeiten (0..1)
    threshold: Schwelle zur Umwandlung in Klassenlabels
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics
