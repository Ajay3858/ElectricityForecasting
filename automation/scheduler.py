"""Automated Local Electricity Forecasting Scheduler.

Runs the validated production inference pipeline every 15 minutes synchronized
to real clock quarter-hour boundaries (00:00, 00:15, 00:30, 00:45).

Ingests observations through a pluggable TelemetrySource interface (CSV, API, MQTT),
maintains historical logs and prediction records without duplicate entries, and
reuses the production model in models/best_model.joblib without retraining.

Usage:
    # Run once for testing or manual execution:
    python automation/scheduler.py --once

    # Run continuously every 15 minutes:
    python automation/scheduler.py
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Ensure parent project root is in sys.path so predict and src modules are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import predict_next_interval, MIN_REQUIRED_ROWS
from automation.telemetry import TelemetrySource, create_telemetry_source

DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config.json")
DEFAULT_LOG_PATH = os.path.join(CURRENT_DIR, "automation.log")


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration parameters from config.json without hard-coding."""
    if not os.path.exists(config_path):
        alt_path = os.path.join("automation", "config.json")
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'.")
            
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    telemetry_cfg = config.get("telemetry", {})
    if not telemetry_cfg and "input_file" in config:
        telemetry_cfg = {"source": "csv", "file": config["input_file"]}
        
    output_file = os.path.join(PROJECT_ROOT, config.get("output_file", "prediction.csv"))
    history_file = os.path.join(PROJECT_ROOT, config.get("history_file", "automation/prediction_history.csv"))
    interval_minutes = int(config.get("interval_minutes", 15))
    
    return {
        "telemetry": telemetry_cfg,
        "output_file": output_file,
        "history_file": history_file,
        "interval_minutes": interval_minutes,
        "config_path": config_path
    }


def log_message(message: str, log_file: str = DEFAULT_LOG_PATH):
    """Write timestamped log message to console and automation.log."""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp_str}] {message}"
    print(formatted_msg)
    
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


def execute_prediction_cycle(config: dict, telemetry_source: TelemetrySource) -> bool:
    """
    Execute one complete prediction cycle:
    1. Ingest historical/current telemetry window through TelemetrySource.
    2. Check that at least 673 historical rows exist.
    3. Run inference via predict.py pipeline (without modifying model).
    4. Read generated prediction.csv.
    5. Enforce duplicate protection and append to prediction_history.csv.
    6. Log all milestones.
    """
    output_path = config["output_file"]
    history_path = config["history_file"]
    
    log_message("Starting prediction")
    
       # 1. Retrieve telemetry window from TelemetrySource
    telemetry_df = telemetry_source.get_historical_data(
        min_rows=MIN_REQUIRED_ROWS
    )

    # Safety check: identify the actual latest telemetry timestamp
    latest_telemetry = telemetry_source.get_latest_data()

    timestamp_col = None
    for candidate in ["date", "datetime", "timestamp", "Date", "Timestamp"]:
        if candidate in latest_telemetry.index:
            timestamp_col = candidate
            break

    if timestamp_col is None:
        raise ValueError(
            "Could not identify timestamp column in latest telemetry."
        )

    latest_timestamp = pd.to_datetime(
        latest_telemetry[timestamp_col]
    )

    print(f"Latest telemetry timestamp: {latest_timestamp}")

    # 2. Execute existing production prediction pipeline
    predict_next_interval(
        input_data=telemetry_df,
        output_path=output_path
    )   
    # 3. Read Generated prediction.csv
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Expected output file '{output_path}' was not generated.")
        
    pred_df = pd.read_csv(output_path)
    if pred_df.empty:
        raise ValueError(f"Output file '{output_path}' is empty.")
        
    current_ts = str(pred_df.iloc[-1]["timestamp"])
    current_usage = float(pred_df.iloc[-1]["current_usage_kWh"])
    predicted_usage = float(pred_df.iloc[-1]["predicted_next_15min_usage_kWh"])
    gen_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_message(f"Current usage: {current_usage:.2f} kWh")
    log_message(f"Predicted next 15-min usage: {predicted_usage:.2f} kWh")
    
    # 4. Append to History with Duplicate Protection
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    
    existing_timestamps = set()
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        try:
            hist_df = pd.read_csv(history_path)
            if "timestamp" in hist_df.columns:
                existing_timestamps = set(hist_df["timestamp"].astype(str))
        except Exception:
            existing_timestamps = set()
    else:
        init_df = pd.DataFrame(columns=[
            "timestamp", 
            "current_usage_kWh", 
            "predicted_next_15min_usage_kWh", 
            "prediction_generated_at"
        ])
        init_df.to_csv(history_path, index=False)
        
    if current_ts in existing_timestamps:
        log_message(f"Duplicate entry check: Timestamp '{current_ts}' already recorded in history. Skipping append.")
    else:
        new_record = pd.DataFrame([{
            "timestamp": current_ts,
            "current_usage_kWh": current_usage,
            "predicted_next_15min_usage_kWh": predicted_usage,
            "prediction_generated_at": gen_time_str
        }])
        new_record.to_csv(history_path, mode="a", header=not os.path.exists(history_path) or os.path.getsize(history_path) == 0, index=False)
        
    log_message("Prediction saved successfully")
    return True


def calculate_seconds_to_next_boundary(interval_minutes: int = 15) -> float:
    """
    Calculate exact seconds to sleep until the next real clock boundary.
    Boundaries: :00, :15, :30, :45 past the hour.
    Guarantees zero clock drift over extended operational runs.
    """
    now = datetime.now()
    minutes_past = now.minute % interval_minutes
    minutes_to_add = interval_minutes - minutes_past
    
    next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
    
    if next_boundary <= now:
        next_boundary += timedelta(minutes=interval_minutes)
        
    sleep_seconds = (next_boundary - now).total_seconds()
    return max(sleep_seconds, 1.0), next_boundary


def run_scheduler(once: bool = False):
    """
    Scheduler main loop:
    - Instantiates TelemetrySource from configuration.
    - In --once mode, executes one prediction cycle and exits.
    - In continuous mode, calculates next clock boundary, sleeps, and repeats.
    - Catches all exceptions to ensure scheduler resilience without crashing.
    """
    try:
        config = load_config()
    except Exception as e:
        log_message(f"[CRITICAL ERROR] Failed to load configuration: {e}")
        sys.exit(1)
        
    try:
        telemetry_source = create_telemetry_source(config, base_dir=PROJECT_ROOT)
    except Exception as e:
        log_message(f"[CRITICAL ERROR] Failed to initialize telemetry source: {e}")
        sys.exit(1)
        
    log_message(f"Automated Forecasting Scheduler initialized (Interval: {config['interval_minutes']} min)")
    log_message(f"Telemetry Source: {telemetry_source.__class__.__name__} ({config['telemetry']})")
    log_message(f"History archive:  '{config['history_file']}'")
    
    if once:
        log_message("Running in single-execution test mode (--once)...")
        try:
            execute_prediction_cycle(config, telemetry_source)
            log_message("Single prediction execution completed successfully.")
            sys.exit(0)
        except Exception as e:
            log_message(f"[ERROR] Prediction cycle failed: {e}")
            sys.exit(1)
            
    # Continuous 15-minute loop
    log_message("Entering continuous 15-minute scheduled forecasting loop...")
    while True:
        try:
            execute_prediction_cycle(config, telemetry_source)
        except Exception as e:
            log_message(f"[ERROR] Prediction cycle failed: {e}")
            log_message("Scheduler will remain active and retry at next boundary.")
            
        sleep_sec, next_time = calculate_seconds_to_next_boundary(config["interval_minutes"])
        log_message(f"Next forecast scheduled at {next_time.strftime('%Y-%m-%d %H:%M:%S')} (sleeping {sleep_sec:.1f}s)")
        
        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            log_message("Scheduler interrupted by user. Shutting down gracefully.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Automated 15-Minute Local Electricity Forecasting Scheduler"
    )
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Execute exactly one prediction cycle and exit (test mode)."
    )
    args = parser.parse_args()
    
    run_scheduler(once=args.once)


if __name__ == "__main__":
    main()
