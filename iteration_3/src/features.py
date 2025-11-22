# iteration_3/src/features.py
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def define_y_anom(df: pd.DataFrame, anomaly_col: str | None = None) -> pd.Series:
    """
    Define the anomaly label y_anom from the anomaly code column
    and/or flags.

    By default: y_anom = 1 if anomaly present (32768 / 163840 / others),
                 0 otherwise.

    Parameters
    ----------
    df : pd.DataFrame
    anomaly_col : str or None
        Column with anomaly code (e.g. 'codi_anomalia').
        If None, tries to infer from column names.

    Returns
    -------
    y : pd.Series of int (0/1)
    """
    if anomaly_col is None:
        for c in df.columns:
            if "anom" in c.lower():
                anomaly_col = c
                break

    if anomaly_col is None:
        raise ValueError("Could not find anomaly column; please specify anomaly_col.")

    codes = df[anomaly_col]

    # basic binary label: any non-zero / non-null anomaly → 1
    y = (codes.fillna(0) != 0).astype(int)
    return y


def build_features(df_clean: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Build feature matrix from cleaned Dataset 2.

    Here you should replicate and slightly generalize what you did in
    02_iter2_feature_engineering: per-polissa statistics, temporal aggregates, etc.

    Parameters
    ----------
    df_clean : pd.DataFrame
    id_col : str
        Column identifying the supply / polissa.

    Returns
    -------
    feat_df : pd.DataFrame
        One row per id_col (or per anomaly event) with engineered features
        and y_anom label.
    """
    df = df_clean.copy()

    # Ensure id column exists
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found in dataframe")

    # Example: simple grouping by id (adapt to your real logic!)
    # – mean, std, max, min consumption, etc.
    if "consumption" in df.columns:
        agg = df.groupby(id_col)["consumption"].agg(
            ["mean", "std", "min", "max"]
        ).rename(
            columns={
                "mean": "consumption_mean",
                "std": "consumption_std",
                "min": "consumption_min",
                "max": "consumption_max",
            }
        )
    else:
        agg = df.groupby(id_col).size().to_frame("n_rows")

    # Example: anomaly label at id level = any anomaly in that id
    anomaly_col = None
    for c in df.columns:
        if "anom" in c.lower():
            anomaly_col = c
            break

    if anomaly_col is None:
        raise ValueError("No anomaly column found for y_anom creation")

    df["y_anom"] = define_y_anom(df, anomaly_col=anomaly_col)
    y_per_id = df.groupby(id_col)["y_anom"].max()

    feat_df = agg.join(y_per_id, how="left")

    # Optional: add other categorical / technical features by aggregation or mode
    # (e.g., municipality code, type of use, meter brand...)
    # Example stub:
    # if "MUNICIPI" in df.columns:
    #     muni_mode = df.groupby(id_col)["MUNICIPI"].agg(lambda x: x.mode().iloc[0])
    #     feat_df["municipi_mode"] = muni_mode

    return feat_df.reset_index()


def get_feature_columns(df: pd.DataFrame, id_col: str, label_col: str) -> List[str]:
    """
    Return the list of candidate feature columns (numeric only, by default),
    excluding id and label columns.
    """
    exclude = {id_col, label_col}
    num_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return num_cols
