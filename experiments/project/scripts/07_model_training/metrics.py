"""
Metriken für das Volatilitäts-Experiment.

Berechnet:
- Accuracy
- F1-Score
- ROC-AUC

Input:
- y_true: echte Labels (0/1)
- y_pred_probs: vorhergesagte Wahrscheinlichkeiten für Klasse 1
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred_probs):
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    y_pred = (y_pred_probs >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_probs),
    }
