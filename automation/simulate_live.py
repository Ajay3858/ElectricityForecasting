"""Live Telemetry Stream Simulator for Real-Time Electricity Forecasting.

Simulates real-time telemetry arrival from smart electricity meters by reading
historical observations from train.csv one 15-minute interval at a time and
feeding them into the rolling buffer (automation/live_telemetry.csv).

Strict Anti-Leakage Guarantee:
- The target column 'next_interval_Usage_kWh' is strictly excluded.
- The simulator only provides telemetry up to current interval t.
- Downstream models predict interval t+15 min without access to actual future usage.

Usage:
    # Release exactly one 15-minute telemetry reading and exit (testing):
    python automation/simulate_live.py --once

    # Continuously stream one observation every 15 minutes:
    python automation/simulate_live.py
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automation.live_buffer import LiveTelemetryBuffer, DEFAULT_BUFFER_PATH, DEFAULT_LOG_PATH
from automation.scheduler import calculate_seconds_to_next_boundary

DEFAULT_SOURCE_PATH = os.path.join(PROJECT_ROOT, "train.csv")


def get_next_simulation_row(
    source_df: pd.DataFrame, 
    date_col: str, 
    last_timestamp: Optional[str]
) -> Tuple[int, pd.Series]:
    """
    Determine the next chronological row from source data to release.
    """
    if last_timestamp is None:
        return 0, source_df.iloc[0]

    # Find the row matching last_timestamp
    matches = source_df.index[source_df[date_col].astype(str).str.strip() == last_timestamp.strip()].tolist()
    if matches:
        last_idx = matches[-1]
        next_idx = last_idx + 1
        if next_idx >= len(source_df):
            raise IndexError("All historical records from source dataset have been streamed.")
        return next_idx, source_df.iloc[next_idx]

    # If exact timestamp string not matched, find by buffer length
    raise ValueError(f"Could not locate last buffer timestamp '{last_timestamp}' in source dataset.")


def release_single_observation(
    buffer: LiveTelemetryBuffer,
    source_path: str = DEFAULT_SOURCE_PATH
) -> dict:
    """
    Extract the next sequential historical observation from source_path,
    strictly strip any target column, and append to the live buffer.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Simulation source file '{source_path}' does not exist.")

    # Ensure buffer has initial warm-up history if empty
    if buffer.get_record_count() < 673:
        buffer.initialize_with_history(source_csv=source_path, initial_rows=700)

    last_ts = buffer.get_latest_timestamp()

    # Load source dataset
    source_df = pd.read_csv(source_path)

    # Identify date column
    date_col = None
    for cand in ["date", "datetime", "timestamp", "Date", "Timestamp"]:
        if cand in source_df.columns:
            date_col = cand
            break
    if not date_col:
        raise ValueError(f"Source file '{source_path}' has no valid timestamp column.")

    next_idx, next_row = get_next_simulation_row(source_df, date_col, last_ts)

    # Build clean reading dictionary strictly omitting any target column
    reading = {
        "timestamp": str(next_row[date_col]).strip(),
        "Usage_kWh": float(next_row["Usage_kWh"]),
        "Lagging_Current_Reactive.Power_kVarh": float(next_row["Lagging_Current_Reactive.Power_kVarh"]),
        "Leading_Current_Reactive_Power_kVarh": float(next_row["Leading_Current_Reactive_Power_kVarh"]),
        "CO2(tCO2)": float(next_row["CO2(tCO2)"]),
        "Lagging_Current_Power_Factor": float(next_row["Lagging_Current_Power_Factor"]),
        "Leading_Current_Power_Factor": float(next_row["Leading_Current_Power_Factor"]),
        "NSM": float(next_row["NSM"])
    }

    # Strict assertion: Ensure next_interval_Usage_kWh is NEVER passed
    assert "next_interval_Usage_kWh" not in reading, "FATAL: Target column detected in simulator reading!"

    # Append to live buffer
    buffer.append_reading(reading)

    return {
        "source_index": next_idx,
        "timestamp": reading["timestamp"],
        "usage_kwh": reading["Usage_kWh"],
        "total_buffer_records": buffer.get_record_count()
    }


def run_simulator(once: bool = False, source_path: str = DEFAULT_SOURCE_PATH):
    """
    Execute live meter telemetry simulation.
    """
    buffer = LiveTelemetryBuffer()
    print("=" * 60)
    print("LIVE TELEMETRY STREAM SIMULATOR")
    print("=" * 60)
    print(f"Source dataset: {source_path}")
    print(f"Live buffer:    {buffer.buffer_path}")
    print(f"Current buffer: {buffer.get_record_count()} records")
    print("=" * 60)

    if once:
        print("\nExecuting single telemetry release (--once)...")
        result = release_single_observation(buffer, source_path)
        print("\nObservation Released Successfully:")
        print(f"  Source Row Index: {result['source_index']}")
        print(f"  Timestamp:        {result['timestamp']}")
        print(f"  Current Usage:    {result['usage_kwh']:.2f} kWh")
        print(f"  Live Buffer Size: {result['total_buffer_records']} records")
        print("  Target Leakage:   STRICTLY ZERO (next_interval_Usage_kWh excluded)")
        print("=" * 60)
        return

    print("\nEntering continuous 15-minute live streaming loop...")
    while True:
        try:
            result = release_single_observation(buffer, source_path)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{now_str}] Released interval {result['timestamp']} | "
                f"Usage: {result['usage_kwh']:.2f} kWh | Buffer: {result['total_buffer_records']} rows"
            )
        except IndexError as e:
            print(f"\n[STREAM COMPLETE] {e}")
            break
        except Exception as e:
            print(f"[STREAM ERROR] Failed to release observation: {e}")

        sleep_sec, next_time = calculate_seconds_to_next_boundary(interval_minutes=15)
        print(f"Next live telemetry release scheduled at {next_time.strftime('%Y-%m-%d %H:%M:%S')} (sleeping {sleep_sec:.1f}s)")
        try:
            time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\nLive telemetry simulator stopped by user.")
            break


def main():
    parser = argparse.ArgumentParser(description="Live Electricity Telemetry Simulator")
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Release exactly one new 15-minute observation and exit."
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=DEFAULT_SOURCE_PATH, 
        help="Source historical CSV dataset (default: train.csv)."
    )
    args = parser.parse_args()

    run_simulator(once=args.once, source_path=args.source)


if __name__ == "__main__":
    main()
