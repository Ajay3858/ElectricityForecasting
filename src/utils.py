
"""Utility functions for electricity forecasting."""

import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_and_sort_chronological(
    df: pd.DataFrame,
    date_col: str = "date",
    nsm_col: str = "NSM",
) -> pd.DataFrame:
    """
    Parse timestamps and sort chronologically.

    Correct the historical dataset's known midnight convention only when
    the source-order signature is unambiguous:
      - previous row is 23:45 with NSM=85500
      - current row is 00:00 with NSM=0
      - both source timestamps carry the same calendar date

    The correction is applied to a copy; the input dataframe is unchanged.
    Correctly dated midnight records are not shifted.
    """
    result = df.copy()

    if date_col not in result.columns:
        if date_col == "date" and "datetime" in result.columns:
            date_col = "datetime"
        else:
            raise ValueError(
                f"Timestamp column '{date_col}' was not found in the dataframe."
            )

    if pd.api.types.is_datetime64_any_dtype(result[date_col]):
        parsed = pd.to_datetime(result[date_col], errors="raise")
    else:
        parsed = pd.to_datetime(
            result[date_col],
            errors="raise",
            format="mixed",
            dayfirst=True,
        )

    # Detect the documented historical convention using source row order.
    if nsm_col in result.columns and len(result) > 1:
        nsm = pd.to_numeric(result[nsm_col], errors="coerce")
        previous_dt = parsed.shift(1)

        same_date = (
            parsed.dt.normalize().eq(previous_dt.dt.normalize())
        )
        previous_2345 = (
            previous_dt.dt.hour.eq(23)
            & previous_dt.dt.minute.eq(45)
        )
        current_midnight = (
            parsed.dt.hour.eq(0)
            & parsed.dt.minute.eq(0)
        )
        previous_nsm_85500 = nsm.shift(1).eq(85500)
        current_nsm_zero = nsm.eq(0)

        rollover = (
            same_date
            & previous_2345
            & current_midnight
            & previous_nsm_85500
            & current_nsm_zero
        )

        parsed.loc[rollover] += pd.Timedelta(days=1)

    result["datetime"] = parsed

    result = result.sort_values(
        "datetime",
        kind="stable",
    ).reset_index(drop=True)

    return result


def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate RMSE, MAE, and R² regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same non-zero length."
        )

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics inputs must contain only finite values.")

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
