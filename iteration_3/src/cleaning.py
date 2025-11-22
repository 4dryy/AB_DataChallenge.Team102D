# iteration_3/src/cleaning.py
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import pandas as pd


def load_dataset2(path: str) -> pd.DataFrame:
    """
    Load 'Repte Consums Anòmals' dataset (Dataset 2).

    Parameters
    ----------
    path : str
        Path to the raw CSV/Parquet for Dataset 2.

    Returns
    -------
    df : pd.DataFrame
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def apply_basic_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply basic cleaning rules to Dataset 2.

    This is where you replicate/refine what you did in 01_iter2_data_analysis:
    - drop obvious junk rows
    - parse dates
    - create anomaly flags (32768, 163840)
    - handle negative / zero consumption

    Returns
    -------
    clean_df : pd.DataFrame
    changes_summary : dict
        A small dict describing what was done (for JSON export).
    """
    clean_df = df.copy()
    changes = {}

    # Example: ensure date columns are parsed (adapt names to your schema)
    date_cols = [c for c in clean_df.columns if "data" in c.lower() or "fecha" in c.lower()]
    for col in date_cols:
        try:
            clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")
        except Exception:
            pass

    # Example: numeric conversion for consumption column
    # TODO: adapt column name (e.g. 'consumption', 'CONSUM', etc.)
    if "consumption" in clean_df.columns:
        clean_df["consumption"] = pd.to_numeric(clean_df["consumption"], errors="coerce")

    # Flags for negative / zero consumption (if applicable)
    if "consumption" in clean_df.columns:
        clean_df["flag_negative_consumption"] = clean_df["consumption"] < 0
        clean_df["flag_zero_consumption"] = clean_df["consumption"] == 0

        changes["negative_consumption_count"] = int(clean_df["flag_negative_consumption"].sum())
        changes["zero_consumption_count"] = int(clean_df["flag_zero_consumption"].sum())

    # Example anomaly flags from 'codi_anomalia' (adapt to exact column name)
    anomaly_col = None
    for c in clean_df.columns:
        if "anom" in c.lower():
            anomaly_col = c
            break

    if anomaly_col is not None:
        clean_df["flag_anom_32768"] = clean_df[anomaly_col] == 32768
        clean_df["flag_anom_163840"] = clean_df[anomaly_col] == 163840

        changes["anom_32768_count"] = int(clean_df["flag_anom_32768"].sum())
        changes["anom_163840_count"] = int(clean_df["flag_anom_163840"].sum())

    # Drop rows with completely missing essential fields (example)
    before = len(clean_df)
    essential_cols = [anomaly_col] if anomaly_col is not None else []
    essential_cols += ["consumption"] if "consumption" in clean_df.columns else []
    if essential_cols:
        clean_df = clean_df.dropna(subset=essential_cols, how="all")
    after = len(clean_df)
    changes["dropped_fully_missing_essential"] = int(before - after)

    return clean_df, changes


def save_cleaned_outputs(
    clean_df: pd.DataFrame,
    changes: Dict,
    out_dir: str,
    base_name: str = "dataset2_cleaned",
) -> None:
    """
    Save cleaned dataset and changes summary into out_dir.

    Produces:
    - {out_dir}/{base_name}.csv
    - {out_dir}/{base_name}.parquet (if possible)
    - {out_dir}/{base_name}_cleaning_changes.json
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    parquet_path = os.path.join(out_dir, f"{base_name}.parquet")
    json_path = os.path.join(out_dir, f"{base_name}_cleaning_changes.json")

    clean_df.to_csv(csv_path, index=False)
    try:
        clean_df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[warn] Could not write parquet: {e}")

    with open(json_path, "w") as f:
        json.dump(changes, f, indent=2)

    print(f"[ok] Saved cleaned dataset to: {csv_path}")
    print(f"[ok] Saved cleaning summary to: {json_path}")
