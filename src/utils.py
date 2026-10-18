"""Utility functions for Electricity Forecasting: seeds, datetime normalization, metrics."""

import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and environment for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_and_sort_chronological(
    df: pd.DataFrame,
    date_col: str = "date",
    nsm_col: str = "NSM"
) -> pd.DataFrame:
    """
    Parse timestamps and sort observations chronologically.

    The supplied live telemetry already contains actual timestamps such as:

        2018-01-08 00:15:00
        2018-01-08 00:30:00

    Therefore timestamps must NOT be shifted based on NSM.

    The original dataset may contain midnight records whose calendar date
    representation requires special handling, but applying that transformation
    to live telemetry would move valid observations into the wrong day.

    This function therefore preserves the timestamp exactly as supplied.
    """

    df = df.copy()

    if date_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{date_col}' was not found in the dataframe."
        )

    # Parse timestamp without changing the supplied date/time.
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        parsed_dt = pd.to_datetime(df[date_col], errors="raise")
    else:
        parsed_dt = pd.to_datetime(
            df[date_col],
            errors="raise",
            format="mixed",
            dayfirst=True
        )

    # Preserve the actual timestamp exactly.
    df["datetime"] = parsed_dt

    # Sort chronologically.
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate RMSE, MAE, and R2 regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }