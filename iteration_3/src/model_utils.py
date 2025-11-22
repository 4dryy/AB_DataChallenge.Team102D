# iteration_3/src/model_utils.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def train_isolation_forest(
    X_train: np.ndarray,
    random_state: int = 42,
    contamination: float | str = "auto",
) -> IsolationForest:
    """
    Train an IsolationForest on preprocessed features.

    Parameters
    ----------
    X_train : np.ndarray
    random_state : int
    contamination : float or 'auto'

    Returns
    -------
    model : IsolationForest
    """
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores_iforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Convert IsolationForest scores to anomaly scores (higher = more anomalous).
    """
    # score_samples: higher = more normal → invert
    raw = model.score_samples(X)
    return -raw


def evaluate_anomaly_model(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float | None = None,
) -> Dict:
    """
    Evaluate anomaly detection given scores and binary labels.

    If threshold is None, uses median of scores as threshold.

    Returns
    -------
    metrics : dict
    """
    if threshold is None:
        threshold = np.median(scores)

    y_pred = (scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, scores)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
