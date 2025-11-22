# iteration_3/src/selection.py
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def filter_by_missing_and_variance(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_missing_ratio: float = 0.4,
) -> List[str]:
    """
    Drop features with too much missingness or zero variance.

    Returns
    -------
    kept : List[str]
    """
    kept = []
    for c in feature_cols:
        col = df[c]
        missing_ratio = col.isna().mean()
        if missing_ratio > max_missing_ratio:
            continue
        if col.std(skipna=True) == 0:
            continue
        kept.append(c)
    return kept


def prune_by_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_corr: float = 0.95,
) -> List[str]:
    """
    Greedy correlation-based pruning: keep one feature in each highly
    correlated group.

    Returns
    -------
    pruned : List[str]
        Subset of feature_cols.
    """
    if not feature_cols:
        return []

    corr = df[feature_cols].corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        highly_corr = upper.index[upper[col] > max_corr].tolist()
        for hc in highly_corr:
            to_drop.add(hc)

    pruned = [c for c in feature_cols if c not in to_drop]
    return pruned


def rank_features_with_rf(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 8,
) -> pd.DataFrame:
    """
    Train a simple RandomForestClassifier and rank features by importance.

    Returns
    -------
    rankings : pd.DataFrame
        Columns: ['feature', 'importance', 'mean', 'std', 'auc_single']
    """
    X = df[feature_cols].values
    y = df[label_col].values

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    # single-feature AUC as additional signal (quick & dirty)
    aucs = []
    for i, c in enumerate(feature_cols):
        try:
            auc = roc_auc_score(y, X[:, i])
        except Exception:
            auc = np.nan
        aucs.append(auc)

    rankings = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
            "mean": df[feature_cols].mean().values,
            "std": df[feature_cols].std().values,
            "auc_single": aucs,
        }
    ).sort_values("importance", ascending=False)

    return rankings


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    max_missing_ratio: float = 0.4,
    max_corr: float = 0.95,
    top_k: int | None = 50,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Full simple pipeline:
      1) remove high-missing / zero-variance features
      2) correlation-based pruning
      3) RF ranking
      4) take top_k

    Returns
    -------
    selected : List[str]
    rankings : pd.DataFrame
    """
    step1 = filter_by_missing_and_variance(df, feature_cols, max_missing_ratio)
    step2 = prune_by_correlation(df, step1, max_corr=max_corr)
    rankings = rank_features_with_rf(df, step2, label_col)

    if top_k is not None:
        selected = rankings["feature"].head(top_k).tolist()
    else:
        selected = rankings["feature"].tolist()

    return selected, rankings
