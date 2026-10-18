"""Feature engineering module for Electricity Consumption Forecasting.

Features use observations at or before the current interval t.
The target represents consumption at the next interval.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import parse_and_sort_chronological


# Deterministic mappings for categorical features
LOAD_TYPE_MAP = {
    "Light_Load": 0,
    "Medium_Load": 1,
    "Maximum_Load": 2,
}

WEEK_STATUS_MAP = {
    "Weekend": 0,
    "Weekday": 1,
}

DAY_OF_WEEK_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


# Standard feature list used for model input
FEATURE_COLUMNS = [
    # Raw telemetry at interval t
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",

    # Encoded categorical features
    "Load_Type_enc",
    "WeekStatus_enc",
    "Day_of_week_enc",

    # Derived electrical telemetry
    "total_reactive_kVarh",
    "apparent_power_kVAh",
    "power_factor_diff",
    "q_p_ratio",

    # Cyclical time encodings
    "sin_nsm",
    "cos_nsm",
    "sin_dow",
    "cos_dow",
    "hour",
    "minute",
    "month",

    # Next-interval calendar context
    "next_minute",
    "is_next_top_of_hour",

    # Autoregressive usage lags
    "usage_lag_1",
    "usage_lag_2",
    "usage_lag_3",
    "usage_lag_4",
    "usage_lag_96",
    "usage_lag_672",

    # Rolling usage statistics
    "usage_roll_mean_4",
    "usage_roll_std_4",
    "usage_roll_min_4",
    "usage_roll_max_4",
    "usage_roll_range_4",
    "usage_roll_mean_16",
    "usage_roll_std_16",
    "usage_roll_mean_96",
    "usage_roll_std_96",
    "usage_ewma_4",

    # Consumption momentum and acceleration
    "usage_diff_1",
    "usage_diff_96",
    "usage_accel",

    # Electrical dynamics
    "q_diff_1",
    "co2_diff_1",
]

TARGET_COLUMN = "next_interval_Usage_kWh"


def engineer_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Generate time, electrical, lag, rolling, and target features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or previously parsed dataframe containing telemetry columns.
    drop_na : bool
        Drop rows with missing feature/target values caused by warm-up or
        missing observations. Set to False when preparing inference data.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe, including the target when present
        in the input or when it is created by the preprocessing pipeline.
    """
    df = df.copy()

    # 1. Parse and sort timestamps chronologically.
    if "datetime" not in df.columns:
        df = parse_and_sort_chronological(df)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    # 2. Encode categorical features.
    if "Load_Type" in df.columns:
        df["Load_Type_enc"] = (
            df["Load_Type"].map(LOAD_TYPE_MAP).fillna(0).astype(int)
        )
    else:
        df["Load_Type_enc"] = 0

    # Prefer timestamp-derived calendar values so these features agree
    # with the normalized chronological timestamps.
    if "WeekStatus" in df.columns:
        df["WeekStatus_enc"] = (
            df["WeekStatus"].map(WEEK_STATUS_MAP).fillna(1).astype(int)
        )
    else:
        df["WeekStatus_enc"] = (
            (df["datetime"].dt.dayofweek >= 5).astype(int).map(
                {1: 0, 0: 1}
            )
        )

    # Day of week is derived from the corrected timestamp.
    df["Day_of_week_enc"] = df["datetime"].dt.dayofweek.astype(int)

    # 3. Calendar features.
    dt = df["datetime"]

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["month"] = dt.dt.month

    # Next-interval calendar context is deterministic.
    next_dt = dt + pd.Timedelta(minutes=15)
    df["next_minute"] = next_dt.dt.minute
    df["is_next_top_of_hour"] = (
        next_dt.dt.minute == 0
    ).astype(int)

    # Cyclical encoding of NSM (seconds from midnight).
    nsm = df["NSM"].astype(float)
    df["sin_nsm"] = np.sin(2 * np.pi * nsm / 86400.0)
    df["cos_nsm"] = np.cos(2 * np.pi * nsm / 86400.0)

    # Cyclical encoding of day of week (Monday=0, Sunday=6).
    dow = df["Day_of_week_enc"].astype(float)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    # 4. Electrical features at interval t.
    p_kw = df["Usage_kWh"].astype(float)
    q_lag = df["Lagging_Current_Reactive.Power_kVarh"].astype(float)
    q_lead = df["Leading_Current_Reactive_Power_kVarh"].astype(float)

    df["total_reactive_kVarh"] = q_lag + q_lead

    df["apparent_power_kVAh"] = np.sqrt(
        p_kw**2 + df["total_reactive_kVarh"] ** 2
    )

    df["power_factor_diff"] = (
        df["Lagging_Current_Power_Factor"].astype(float)
        - df["Leading_Current_Power_Factor"].astype(float)
    )

    df["q_p_ratio"] = df["total_reactive_kVarh"] / (p_kw + 0.1)

    # 5. Autoregressive usage lags.
    df["usage_lag_1"] = p_kw.shift(1)
    df["usage_lag_2"] = p_kw.shift(2)
    df["usage_lag_3"] = p_kw.shift(3)
    df["usage_lag_4"] = p_kw.shift(4)
    df["usage_lag_96"] = p_kw.shift(96)
    df["usage_lag_672"] = p_kw.shift(672)

    # 6. Rolling statistics.
    # These windows include the current interval t. That is appropriate
    # when Usage_kWh at t is available before predicting interval t+1.
    df["usage_roll_mean_4"] = p_kw.rolling(
        window=4, min_periods=4
    ).mean()

    df["usage_roll_std_4"] = p_kw.rolling(
        window=4, min_periods=4
    ).std().fillna(0.0)

    df["usage_roll_min_4"] = p_kw.rolling(
        window=4, min_periods=4
    ).min()

    df["usage_roll_max_4"] = p_kw.rolling(
        window=4, min_periods=4
    ).max()

    df["usage_roll_range_4"] = (
        df["usage_roll_max_4"] - df["usage_roll_min_4"]
    )

    df["usage_roll_mean_16"] = p_kw.rolling(
        window=16, min_periods=16
    ).mean()

    df["usage_roll_std_16"] = p_kw.rolling(
        window=16, min_periods=16
    ).std().fillna(0.0)

    df["usage_roll_mean_96"] = p_kw.rolling(
        window=96, min_periods=96
    ).mean()

    df["usage_roll_std_96"] = p_kw.rolling(
        window=96, min_periods=96
    ).std().fillna(0.0)

    df["usage_ewma_4"] = p_kw.ewm(
        span=4, adjust=True
    ).mean()

    # 7. Consumption momentum and acceleration.
    df["usage_diff_1"] = p_kw - df["usage_lag_1"]
    df["usage_diff_96"] = p_kw - df["usage_lag_96"]

    df["usage_accel"] = (
        df["usage_diff_1"]
        - (df["usage_lag_1"] - df["usage_lag_2"])
    )

    # 8. Electrical changes from the previous interval.
    df["q_diff_1"] = (
        df["total_reactive_kVarh"]
        - df["total_reactive_kVarh"].shift(1).fillna(0.0)
    )

    df["co2_diff_1"] = (
        df["CO2(tCO2)"]
        - df["CO2(tCO2)"].shift(1).fillna(0.0)
    )

    # 9. Verify/create the next-interval target after chronological sorting.
    # The final observation has no known next-interval target.
    if TARGET_COLUMN in df.columns:
        expected_target = p_kw.shift(-1)
        supplied_target = pd.to_numeric(
            df[TARGET_COLUMN], errors="coerce"
        )

        # Keep only supplied targets that match the next chronological usage.
        valid_target = np.isclose(
            supplied_target.to_numpy(dtype=float),
            expected_target.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
            equal_nan=False,
        )

        valid_target[-1] = False
        df[TARGET_COLUMN] = supplied_target.where(valid_target)
    else:
        df[TARGET_COLUMN] = p_kw.shift(-1)
    # 10. Remove rows with incomplete lag/rolling/target values.
    if drop_na:
        cols_to_check = FEATURE_COLUMNS.copy()

        if TARGET_COLUMN in df.columns:
            cols_to_check.append(TARGET_COLUMN)

        df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    return df


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    purge: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature-engineered data chronologically.

    The final `purge` rows before the validation boundary are excluded
    from training. Validation begins at the original split index.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataframe."
        )

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not isinstance(purge, int) or purge < 0:
        raise ValueError("purge must be a non-negative integer.")

    split_idx = int(len(df) * train_ratio)

    if split_idx <= purge or split_idx >= len(df):
        raise ValueError("Insufficient rows for this split.")

    train_df = df.iloc[: split_idx - purge]
    val_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df[TARGET_COLUMN].copy()

    return X_train, X_val, y_train, y_val