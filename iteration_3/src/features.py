# iteration_3/src/features.py
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

ROLL_WINDOWS = [3, 12, 24]  # last-N readings


def _detect_id_col(df: pd.DataFrame) -> str:
    """
    Detect a suitable meter/policy identifier column.

    Priority:
    1) num_serie_contador
    2) polissa_id
    """
    if "num_serie_contador" in df.columns:
        return "num_serie_contador"
    if "polissa_id" in df.columns:
        return "polissa_id"
    raise ValueError(
        "No meter/policy identifier found. Expect one of ['num_serie_contador','polissa_id']."
    )


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'datetime' column used for temporal ordering.

    Tries common column names:
    - 'datetime'
    - 'FECHA_HORA'
    - any column with 'data' / 'fecha' in its name.
    """
    df = df.copy()

    if "datetime" in df.columns:
        return df

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
    elif "date" in dfin.columns:
        dfin = dfin.sort_values([key, "date"])
    else:
        dfin = dfin.sort_values([key]).copy()

    return dfin, key


def _detect_consumption_col(df: pd.DataFrame) -> str:
    """
    Detect the consumption column.

    Priority:
    - 'consumption'
    - first column containing 'consum' in its name
    """
    if "consumption" in df.columns:
        return "consumption"

    for c in df.columns:
        if "consum" in c.lower():
            return c

    raise ValueError(
        "No consumption column found. Expected 'consumption' or a column containing 'consum'."
    )


def define_y_anom(df: pd.DataFrame, anomaly_col: Optional[str] = None) -> pd.Series:
    """
    Define the anomaly label y_anom from the anomaly code column.

    Basic rule (can be refined):
    - y_anom = 1 if anomaly code is non-zero / non-null
    - y_anom = 0 otherwise
    """
    if anomaly_col is None:
        for c in df.columns:
            if "anom" in c.lower():
                anomaly_col = c
                break

    if anomaly_col is None:
        raise ValueError("Could not find anomaly column; please specify anomaly_col.")

    codes = df[anomaly_col]
    y = (codes.fillna(0) != 0).astype(int)
    return y


def make_features(dfin: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level features (per reading, per meter context).

    Features:
    - Lags & deltas: cons_lag1, cons_lag2, delta1, delta2
    - Rolling stats on last N readings (N ∈ ROLL_WINDOWS):
        mean, std, min, max, zero_ratio, neg_ratio
    - Meter-level z-score: cons_z_meter
    - Period duration in hours: period_hours (if data_inici & data_fi exist)
    """
    dfout = dfin.copy()
    dfout, key = _group_sort(dfout)

    cons_col = _detect_consumption_col(dfout)
    dfout[cons_col] = pd.to_numeric(dfout[cons_col], errors="coerce")

    # --- Lags & deltas ---
    dfout["cons_lag1"] = dfout.groupby(key)[cons_col].shift(1)
    dfout["cons_lag2"] = dfout.groupby(key)[cons_col].shift(2)
    dfout["delta1"] = dfout[cons_col] - dfout["cons_lag1"]
    dfout["delta2"] = dfout[cons_col] - dfout["cons_lag2"]

    # --- Rolling stats ---
    g = dfout.groupby(key)[cons_col]

    for win in ROLL_WINDOWS:
        roll_mean = g.transform(lambda s: s.rolling(win, min_periods=1).mean())
        roll_std = g.transform(lambda s: s.rolling(win, min_periods=1).std())
        roll_min = g.transform(lambda s: s.rolling(win, min_periods=1).min())
        roll_max = g.transform(lambda s: s.rolling(win, min_periods=1).max())
        roll_zero_ratio = g.transform(
            lambda s: s.rolling(win, min_periods=1).apply(
                lambda x: (x == 0).mean(), raw=False
            )
        )
        roll_neg_ratio = g.transform(
            lambda s: s.rolling(win, min_periods=1).apply(
                lambda x: (x < 0).mean(), raw=False
            )
        )

        dfout[f"cons_roll{win}_mean"] = roll_mean
        dfout[f"cons_roll{win}_std"] = roll_std
        dfout[f"cons_roll{win}_min"] = roll_min
        dfout[f"cons_roll{win}_max"] = roll_max
        dfout[f"cons_roll{win}_zero_ratio"] = roll_zero_ratio
        dfout[f"cons_roll{win}_neg_ratio"] = roll_neg_ratio

    # --- Meter-level stats & z-score ---
    meter_stats = dfout.groupby(key)[cons_col].agg(["mean", "std"]).rename(
        columns={"mean": "meter_mean", "std": "meter_std"}
    )
    dfout = dfout.merge(meter_stats, left_on=key, right_index=True, how="left")
    dfout["cons_z_meter"] = (dfout[cons_col] - dfout["meter_mean"]) / dfout[
        "meter_std"
    ].replace(0, np.nan)

    # --- Period duration in hours ---
    if {"data_inici", "data_fi"}.issubset(dfout.columns):
        di = pd.to_datetime(dfout["data_inici"], errors="coerce")
        df_ = pd.to_datetime(dfout["data_fi"], errors="coerce")
        dfout["period_hours"] = (df_ - di).dt.total_seconds() / 3600.0

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
