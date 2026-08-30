from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_prob, dtype=np.float64)
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("labels and probabilities must be non-empty and equally sized")
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "mcc": matthews_corrcoef(truth, prediction),
        "pr_auc": average_precision_score(truth, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }

