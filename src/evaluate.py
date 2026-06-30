"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute common binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def selection_score(metrics: dict) -> float:
    """Select by ROC-AUC when available, otherwise F1."""
    roc_auc = metrics.get("roc_auc")
    if roc_auc is not None and not np.isnan(roc_auc):
        return float(roc_auc)
    return float(metrics.get("f1", 0.0))

