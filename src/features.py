"""Feature engineering module for Electricity Consumption Forecasting.

All features are strictly causal: they only rely on observations at or before
the current interval t. No information from t+1 or the future target is used.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from src.utils import parse_and_sort_chronological

# Deterministic mappings for categorical features
LOAD_TYPE_MAP = {
    "Light_Load": 0,
    "Medium_Load": 1,
    "Maximum_Load": 2
}

WEEK_STATUS_MAP = {
    "Weekend": 0,
    "Weekday": 1
}

DAY_OF_WEEK_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Standard feature list used for model input
FEATURE_COLUMNS = [
    # Raw telemetry at t
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    # Encoded categoricals
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
    # Next interval calendar context (deterministic, zero leakage)
    "next_minute",
    "is_next_top_of_hour",
    # Autoregressive past lags (Usage_kWh)
    "usage_lag_1",
    "usage_lag_2",
    "usage_lag_3",
    "usage_lag_4",
    "usage_lag_96",   # 24 hours prior (daily seasonality)
    "usage_lag_672",  # 7 days prior (weekly seasonality)
    # Rolling past statistics (Usage_kWh)
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
    # Consumption momentum & acceleration
    "usage_diff_1",   # Usage_t - Usage_{t-1}
    "usage_diff_96",  # Usage_t - Usage_{t-96}
    "usage_accel",    # (Usage_t - Usage_{t-1}) - (Usage_{t-1} - Usage_{t-2})
    # Electrical dynamics
    "q_diff_1",       # Q_t - Q_{t-1}
    "co2_diff_1"      # CO2_t - CO2_{t-1}
]

TARGET_COLUMN = "next_interval_Usage_kWh"


def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Generate all time-series, electrical, cyclical, lag, and rolling features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw or sorted dataframe containing required telemetry columns.
    drop_na : bool
        Whether to drop rows with NaN values resulting from lag/rolling warm-up.
        Set to True for training/evaluation; False for inference buffering.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features and target (if present).
    """
    df = df.copy()
    
    # 1. Parse and sort chronologically if not already parsed
    if "datetime" not in df.columns:
        df = parse_and_sort_chronological(df)
    else:
        df = df.sort_values("datetime").reset_index(drop=True)
    
    # 2. Categorical Encodings
    if "Load_Type" in df.columns:
        df["Load_Type_enc"] = df["Load_Type"].map(LOAD_TYPE_MAP).fillna(0).astype(int)
    else:
        df["Load_Type_enc"] = 0
        
    if "WeekStatus" in df.columns:
        df["WeekStatus_enc"] = df["WeekStatus"].map(WEEK_STATUS_MAP).fillna(1).astype(int)
    else:
        df["WeekStatus_enc"] = 1
        
    if "Day_of_week" in df.columns:
        df["Day_of_week_enc"] = df["Day_of_week"].map(DAY_OF_WEEK_MAP).fillna(0).astype(int)
    else:
        df["Day_of_week_enc"] = df["datetime"].dt.dayofweek
        
    # 3. Calendar & Cyclical Time Features
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["month"] = dt.dt.month
    
    # Next interval calendar context (completely deterministic from current timestamp, zero target leakage)
    next_dt = dt + pd.Timedelta(minutes=15)
    df["next_minute"] = next_dt.dt.minute
    df["is_next_top_of_hour"] = (next_dt.dt.minute == 0).astype(int)
    
    # Cyclical encoding of NSM (0 to 86400 seconds)
    nsm = df["NSM"].astype(float)
    df["sin_nsm"] = np.sin(2 * np.pi * nsm / 86400.0)
    df["cos_nsm"] = np.cos(2 * np.pi * nsm / 86400.0)
    
    # Cyclical encoding of Day of Week (0 to 6)
    dow = df["Day_of_week_enc"].astype(float)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)
    
    # 4. Electrical Derived Telemetry at interval t
    p_kw = df["Usage_kWh"].astype(float)
    q_lag = df["Lagging_Current_Reactive.Power_kVarh"].astype(float)
    q_lead = df["Leading_Current_Reactive_Power_kVarh"].astype(float)
    
    df["total_reactive_kVarh"] = q_lag + q_lead
    df["apparent_power_kVAh"] = np.sqrt(p_kw**2 + df["total_reactive_kVarh"]**2)
    df["power_factor_diff"] = (
        df["Lagging_Current_Power_Factor"].astype(float) - 
        df["Leading_Current_Power_Factor"].astype(float)
    )
    df["q_p_ratio"] = df["total_reactive_kVarh"] / (p_kw + 0.1)
    
    # 5. Autoregressive Past Lag Features of Usage_kWh
    df["usage_lag_1"] = p_kw.shift(1)
    df["usage_lag_2"] = p_kw.shift(2)
    df["usage_lag_3"] = p_kw.shift(3)
    df["usage_lag_4"] = p_kw.shift(4)
    df["usage_lag_96"] = p_kw.shift(96)    # Same interval yesterday
    df["usage_lag_672"] = p_kw.shift(672)  # Same interval 1 week ago
    
    # 6. Past-Only Rolling Statistics of Usage_kWh
    df["usage_roll_mean_4"] = p_kw.rolling(window=4, min_periods=4).mean()
    df["usage_roll_std_4"] = p_kw.rolling(window=4, min_periods=4).std().fillna(0.0)
    df["usage_roll_min_4"] = p_kw.rolling(window=4, min_periods=4).min()
    df["usage_roll_max_4"] = p_kw.rolling(window=4, min_periods=4).max()
    df["usage_roll_range_4"] = df["usage_roll_max_4"] - df["usage_roll_min_4"]
    
    df["usage_roll_mean_16"] = p_kw.rolling(window=16, min_periods=16).mean()
    df["usage_roll_std_16"] = p_kw.rolling(window=16, min_periods=16).std().fillna(0.0)
    
    df["usage_roll_mean_96"] = p_kw.rolling(window=96, min_periods=96).mean()
    df["usage_roll_std_96"] = p_kw.rolling(window=96, min_periods=96).std().fillna(0.0)
    
    df["usage_ewma_4"] = p_kw.ewm(span=4).mean()
    
    # 7. Consumption Momentum, Acceleration, & Electrical Rate of Change
    df["usage_diff_1"] = p_kw - df["usage_lag_1"]
    df["usage_diff_96"] = p_kw - df["usage_lag_96"]
    df["usage_accel"] = df["usage_diff_1"] - (df["usage_lag_1"] - df["usage_lag_2"])
    
    df["q_diff_1"] = df["total_reactive_kVarh"] - df["total_reactive_kVarh"].shift(1).fillna(0.0)
    df["co2_diff_1"] = df["CO2(tCO2)"] - df["CO2(tCO2)"].shift(1).fillna(0.0)
    
    # Drop warm-up rows if requested
    if drop_na:
        cols_to_check = FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])
        df = df.dropna(subset=cols_to_check).reset_index(drop=True)
        
    return df


def split_chronological(
    df: pd.DataFrame, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature-engineered dataframe chronologically into Train and Validation sets."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
        
    n = len(df)
    split_idx = int(n * train_ratio)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    
    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    
    return X_train, X_val, y_train, y_val
