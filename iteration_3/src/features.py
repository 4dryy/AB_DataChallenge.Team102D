# iteration_3/src/features.py
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helper detectors adapted to your schema
# -------------------------------------------------------------------
def _detect_id_col(df: pd.DataFrame) -> str:
    """
    Detect a suitable meter/policy identifier column.

    Priority for your dataset:
    1) NUMEROSERIECONTADOR (meter serial)
    2) POLISSA_SUBM (policy)
    Fallbacks (generic names) in case future datasets differ.
    """
    if "NUMEROSERIECONTADOR" in df.columns:
        return "NUMEROSERIECONTADOR"
    if "POLISSA_SUBM" in df.columns:
        return "POLISSA_SUBM"
    if "num_serie_contador" in df.columns:
        return "num_serie_contador"
    if "polissa_id" in df.columns:
        return "polissa_id"
    raise ValueError(
        "No meter/policy identifier found. Expected one of "
        "['NUMEROSERIECONTADOR','POLISSA_SUBM','num_serie_contador','polissa_id']."
    )


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'datetime' column used for temporal ordering.

    For your dataset, we use FECHA_HORA. We keep generic fallbacks for future reuse.
    """
    df = df.copy()

    if "datetime" in df.columns:
        return df

    if "FECHA_HORA" in df.columns:
        df["datetime"] = pd.to_datetime(df["FECHA_HORA"], errors="coerce")
        return df

    # Generic fallback: pick first date-like column
    candidate = None
    for c in df.columns:
        lc = c.lower()
        if "fecha_hora" in lc or "fecha" in lc or "data" in lc or "date" in lc:
            candidate = c
            break

    if candidate is not None:
        df["datetime"] = pd.to_datetime(df[candidate], errors="coerce")

    return df


def _group_sort(dfin: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Sort by id and time, and return (sorted_df, id_col).
    """
    dfin = _ensure_datetime_column(dfin)
    key = _detect_id_col(dfin)

    if "datetime" in dfin.columns:
        dfin = dfin.sort_values([key, "datetime"])
    else:
        dfin = dfin.sort_values([key]).copy()

    return dfin, key


def _detect_consumption_col(df: pd.DataFrame) -> str:
    """
    Detect the consumption column.

    For your dataset: CONSUMO_REAL is the main one.
    """
    if "CONSUMO_REAL" in df.columns:
        return "CONSUMO_REAL"
    if "consumption" in df.columns:
        return "consumption"

    for c in df.columns:
        if "consum" in c.lower():
            return c

    raise ValueError(
        "No consumption column found. Expected 'CONSUMO_REAL', 'consumption' or a column containing 'consum'."
    )


def _detect_anomaly_col(df: pd.DataFrame) -> str:
    """
    Detect the anomaly code column.

    For your dataset: CODI_ANOMALIA.
    """
    if "CODI_ANOMALIA" in df.columns:
        return "CODI_ANOMALIA"

    for c in df.columns:
        if "anom" in c.lower():
            return c

    raise ValueError(
        "No anomaly column found. Expected 'CODI_ANOMALIA' or a column containing 'anom'."
    )


# -------------------------------------------------------------------
# Label definition
# -------------------------------------------------------------------
def define_y_anom(df: pd.DataFrame, anomaly_col: Optional[str] = None) -> pd.Series:
    """
    Define the anomaly label y_anom from the anomaly code column.

    Basic rule (can be refined):
    - y_anom = 1 if anomaly code is non-zero / non-null
    - y_anom = 0 otherwise
    """
    if anomaly_col is None:
        anomaly_col = _detect_anomaly_col(df)

    codes = df[anomaly_col]
    y = (codes.fillna(0) != 0).astype(int)
    return y


# -------------------------------------------------------------------
# Feature engineering (lightweight & scalable)
# -------------------------------------------------------------------
def make_features(dfin: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level features (per reading, per meter context).

    For scalability on ~21M rows, we keep only cheap, vectorized features:

    - Lag 1 of consumption: cons_lag1
    - First-order difference: delta1 = cons - cons_lag1
    - Meter-level mean and std across all readings: meter_mean, meter_std
    - Z-score vs meter history: cons_z_meter
    - Period duration in hours: period_hours (if START_DATE & END_DATE exist)
    """
    dfout = dfin.copy()
    dfout, key = _group_sort(dfout)

    cons_col = _detect_consumption_col(dfout)
    dfout[cons_col] = pd.to_numeric(dfout[cons_col], errors="coerce")

    # --- Lag 1 & delta1 (fast, groupby.shift is vectorized) ---
    dfout["cons_lag1"] = dfout.groupby(key)[cons_col].shift(1)
    dfout["delta1"] = dfout[cons_col] - dfout["cons_lag1"]

    # --- Meter-level stats & z-score (one groupby.agg + merge) ---
    meter_stats = dfout.groupby(key)[cons_col].agg(["mean", "std"]).rename(
        columns={"mean": "meter_mean", "std": "meter_std"}
    )
    dfout = dfout.merge(meter_stats, left_on=key, right_index=True, how="left")
    dfout["cons_z_meter"] = (dfout[cons_col] - dfout["meter_mean"]) / dfout[
        "meter_std"
    ].replace(0, np.nan)

    # --- Period duration in hours (START_DATE / END_DATE) ---
    if {"START_DATE", "END_DATE"}.issubset(dfout.columns):
        start = pd.to_datetime(dfout["START_DATE"], errors="coerce")
        end = pd.to_datetime(dfout["END_DATE"], errors="coerce")
        dfout["period_hours"] = (end - start).dt.total_seconds() / 3600.0

    return dfout


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    High-level function: adds y_anom and engineered features.

    Returns a row-level feature table (same granularity as df_clean)
    with additional columns and y_anom.
    """
    df = df_clean.copy()

    # Add y_anom label
    y = define_y_anom(df)
    df["y_anom"] = y

    # Build engineered features
    feat_df = make_features(df)

    return feat_df


def get_feature_columns(df: pd.DataFrame, id_col: str, label_col: str) -> List[str]:
    """
    Return the list of candidate feature columns (numeric only),
    excluding id and label columns.
    """
    exclude = {id_col, label_col}
    num_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return num_cols
