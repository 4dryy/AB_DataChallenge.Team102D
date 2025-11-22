# iteration_3/src/splitting.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_splits(
    df: pd.DataFrame,
    id_col: str,
    train_size: float = 0.7,
    valid_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create group-aware train/valid/test indices using id_col as group.

    Returns
    -------
    train_idx, valid_idx, test_idx : np.ndarray
    """
    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found in dataframe")

    groups = df[id_col].values
    n = len(df)

    # First: train vs holdout (valid+test)
    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, hold_idx = next(gss1.split(np.zeros(n), groups=groups))

    # Second: valid vs test on holdout
    hold_groups = groups[hold_idx]
    valid_prop_in_hold = valid_size / (1.0 - train_size)
    gss2 = GroupShuffleSplit(
        n_splits=1, train_size=valid_prop_in_hold, random_state=random_state + 1
    )
    valid_rel, test_rel = next(gss2.split(np.zeros(len(hold_idx)), groups=hold_groups))

    valid_idx = hold_idx[valid_rel]
    test_idx = hold_idx[test_rel]

    return train_idx, valid_idx, test_idx


def add_split_column(
    df: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray, test_idx: np.ndarray
) -> pd.DataFrame:
    """
    Add a 'split' column to df with values 'train', 'valid', 'test'.
    """
    df = df.copy()
    df["split"] = "unk"
    df.loc[train_idx, "split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx, "split"] = "test"
    return df
