# iteration_3/src/preprocessing.py
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_preprocessor(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[SimpleImputer, StandardScaler]:
    """
    Fit median imputer and standard scaler on feature_cols.

    Returns
    -------
    imputer, scaler
    """
    X = df[feature_cols].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    scaler.fit(X_imp)

    return imputer, scaler


def apply_preprocessor(
    df: pd.DataFrame,
    feature_cols: List[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply fitted imputer and scaler to df[feature_cols].

    Returns
    -------
    X_proc : np.ndarray
    """
    X = df[feature_cols].values
    X_imp = imputer.transform(X)
    X_proc = scaler.transform(X_imp)
    return X_proc


def save_preprocessor(
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_cols: List[str],
    out_dir: str,
    base_name: str = "preproc",
) -> None:
    """
    Save preprocessor parameters (using numpy & json).
    """
    os.makedirs(out_dir, exist_ok=True)

    # For simplicity, store means & stds instead of pickling objects
    params = {
        "feature_cols": feature_cols,
        "imputer_strategy": "median",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    json_path = os.path.join(out_dir, f"{base_name}_params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"[ok] Saved preprocessor params to: {json_path}")
