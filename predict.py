"""Production Inference Pipeline for Electricity Consumption Forecasting.

Given historical electricity consumption data up to the current 15-minute interval t,
predicts electricity consumption for interval t+15 minutes (next_interval_Usage_kWh).

Uses models/best_model.joblib and strictly causal feature engineering logic from src/features.py.
Guarantees ZERO target or future lookahead leakage.

Usage:
    python predict.py --input train.csv
    python predict.py --input path/to/telemetry.csv --output custom_prediction.csv
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from src.features import engineer_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.utils import parse_and_sort_chronological

MODEL_PATH = os.path.join("models", "best_model.joblib")
MIN_REQUIRED_ROWS = 673  # 7 days of 15-minute intervals (672 lags) + 1 current interval

REQUIRED_TELEMETRY_COLUMNS = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM"
]


def validate_input_data(df: pd.DataFrame, file_path: str):
    """
    Validate presence of timestamp column, required telemetry columns,
    and adequate history length for lag/rolling feature computation.
    """
    # 1. Check timestamp column
    time_col = None
    for candidate in ["date", "datetime", "timestamp", "Date", "Timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break
            
    if time_col is None:
        raise ValueError(
            f"Validation Error in '{file_path}': Missing timestamp column. "
            "Input CSV must contain a 'date', 'datetime', or 'timestamp' column."
        )
        
    # 2. Check required electrical telemetry columns
    missing_cols = [col for col in REQUIRED_TELEMETRY_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Validation Error in '{file_path}': Missing required telemetry columns:\n"
            f"  {missing_cols}\n"
            f"The model requires all 7 foundational electrical telemetry features."
        )
        
    # 3. Check historical row count
    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(
            f"Validation Error in '{file_path}': Insufficient historical observations.\n"
            f"  Rows provided: {len(df):,}\n"
            f"  Rows required: at least {MIN_REQUIRED_ROWS:,} (7 days of 15-minute readings)\n"
            "A minimum 7-day lookback window is required to calculate the weekly seasonal "
            "lag feature (usage_lag_672) and past rolling statistics without missing data."
        )
        
    return time_col


def load_production_model(model_path: str = MODEL_PATH):
    """Load the trained production model with verification."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model Loading Error: Production model not found at '{model_path}'.\n"
            "Please ensure 'python train.py' has been executed to generate the model artifact."
        )
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Model Loading Error: Failed to load model artifact from '{model_path}': {e}")


def predict_next_interval(input_data, output_path: str = "prediction.csv"):
    """
    Load historical CSV or take a validated telemetry DataFrame, engineer features 
    strictly up to interval t, and predict consumption for interval t+15 minutes.
    
    Parameters
    ----------
    input_data : str or pd.DataFrame
        Path to CSV file or a DataFrame containing historical observations up to interval t.
    output_path : str
        Destination path for output prediction.csv summary.
    """
    if isinstance(input_data, pd.DataFrame):
        raw_df = input_data.copy()
        time_col = validate_input_data(raw_df, "In-Memory Telemetry")
    elif isinstance(input_data, str):
        csv_path = input_data
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input Error: CSV file '{csv_path}' does not exist.")
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as e:
            raise IOError(f"Read Error: Could not parse CSV file '{csv_path}': {e}")
        time_col = validate_input_data(raw_df, csv_path)
    else:
        raise TypeError(f"Invalid input type: Expected str or pd.DataFrame, got {type(input_data)}")

    # ANTI-LEAKAGE ENFORCEMENT:
    # Explicitly drop the future target if present in the input file so it can NEVER be used
    clean_df = raw_df.copy()
    if TARGET_COLUMN in clean_df.columns:
        clean_df = clean_df.drop(columns=[TARGET_COLUMN])
        
    # 4. Chronological parsing, validation, and sorting
    try:
        sorted_df = parse_and_sort_chronological(clean_df, date_col=time_col)
    except Exception as e:
        raise ValueError(f"Timestamp Parse Error in '{time_col}': {e}")
        
    # 5. Feature Engineering (Strictly causal, past-only)
    features_df = engineer_features(sorted_df, drop_na=False)
    
    # 6. Load Production Model
    model = load_production_model(MODEL_PATH)
    
    # 7. Select latest available timestamp (interval t)
    latest_row = features_df.iloc[-1]
    current_timestamp = str(latest_row["datetime"])
    current_usage = float(latest_row["Usage_kWh"])
    
    # 8. Extract feature vector for latest timestamp
    X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]])
    
    # Check for any unexpected NaNs in the latest feature vector
    nan_cols = X_latest.columns[X_latest.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"Feature Computation Error: NaN encountered in features: {nan_cols}")
        
    # 9. Predict consumption at t+15 minutes
    prediction_raw = float(model.predict(X_latest)[0])
    # Enforce non-negativity bound based on physical consumption constraints
    prediction_kwh = max(prediction_raw, 0.0)
    
    # 10. Output Formatted Results to Console
    print("=" * 50)
    print("ELECTRICITY CONSUMPTION FORECAST")
    print("=" * 50)
    print("\nCurrent timestamp:")
    print(f"{current_timestamp}")
    print("\nCurrent Usage:")
    print(f"{current_usage:.2f} kWh")
    print("\nPredicted next 15-minute Usage:")
    print(f"{prediction_kwh:.2f} kWh")
    print("\nPrediction horizon:")
    print("15 minutes")
    print("\n" + "=" * 50)
    
    # 11. Save output prediction.csv
    pred_summary = pd.DataFrame([{
        "timestamp": current_timestamp,
        "current_usage_kWh": round(current_usage, 2),
        "predicted_next_15min_usage_kWh": round(prediction_kwh, 2)
    }])
    pred_summary.to_csv(output_path, index=False)
    print(f"Prediction saved to '{output_path}'.")
    return prediction_kwh


def main():
    parser = argparse.ArgumentParser(
        description="Predict industrial electricity consumption for interval t+15 minutes."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to CSV containing historical observations up to interval t."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="prediction.csv", 
        help="Destination path for prediction CSV (default: prediction.csv)."
    )
    
    args = parser.parse_args()
    
    try:
        predict_next_interval(input_data=args.input, output_path=args.output)
    except Exception as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
