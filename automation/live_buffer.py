"""Live Rolling Telemetry Buffer Management for Real-Time Electricity Forecasting.

Maintains a rolling CSV buffer (automation/live_telemetry.csv) that receives
new electricity telemetry readings from external streams, sensors, or simulators.

Enforces strict input validation:
1. Valid, parseable timestamp.
2. Unique timestamp (no duplicates).
3. Monotonic chronological ordering (t > t_previous).
4. All required foundational electrical telemetry fields are valid numerics.
5. Anti-leakage: Strictly rejects any target column (next_interval_Usage_kWh).
6. Robust logging of invalid or rejected readings without crashing.
7. Maintains complete historical buffer (retaining at least 673 intervals for 7-day lags).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Dict, Any, Optional

# Ensure project root is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BUFFER_PATH = os.path.join(CURRENT_DIR, "live_telemetry.csv")
DEFAULT_LOG_PATH = os.path.join(CURRENT_DIR, "automation.log")

REQUIRED_TELEMETRY_COLUMNS = [
    "timestamp",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM"
]

NUMERIC_FIELDS = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM"
]

FORBIDDEN_TARGET_KEYS = [
    "next_interval_Usage_kWh",
    "target",
    "actual_next_usage",
    "next_usage"
]

MIN_REQUIRED_RECORDS = 673


def _log_event(message: str, log_file: str = DEFAULT_LOG_PATH):
    """Log formatted timestamped message to automation log and console."""
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts_str}] [LIVE_BUFFER] {message}"
    print(formatted)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception as e:
        sys.stderr.write(f"Failed to write to log file '{log_file}': {e}\n")


class LiveTelemetryBuffer:
    """
    Manages the live rolling telemetry buffer for electricity forecasting.
    Enforces data integrity, chronological causality, and anti-leakage constraints.
    """

    def __init__(
        self, 
        buffer_path: str = DEFAULT_BUFFER_PATH, 
        log_file: str = DEFAULT_LOG_PATH
    ):
        self.buffer_path = os.path.abspath(buffer_path)
        self.log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)

    def _parse_effective_datetime(self, ts_val: Any, nsm_val: float) -> pd.Timestamp:
        """
        Parse timestamp and adjust for midnight (NSM==0) rollover consistent with
        the project's canonical chronological parsing logic.
        """
        if isinstance(ts_val, pd.Timestamp):
            dt = ts_val
        elif isinstance(ts_val, datetime):
            dt = pd.Timestamp(ts_val)
        elif isinstance(ts_val, str):
            try:
                dt = pd.to_datetime(ts_val, format="%d/%m/%Y %H:%M")
            except Exception:
                try:
                    dt = pd.to_datetime(ts_val, format="%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = pd.to_datetime(ts_val, format="mixed", dayfirst=True)
        else:
            raise ValueError(f"Unsupported timestamp value type '{type(ts_val)}': {ts_val}")

        if pd.isna(dt):
            raise ValueError(f"Unable to parse timestamp '{ts_val}' into a valid datetime.")

        # Midnight rollover handling (NSM==0 maps to 24:00 of that day, i.e., start of next day)
        if int(round(nsm_val)) == 0:
            dt = dt + pd.Timedelta(days=1)
            
        return dt

    def get_dataframe(self) -> pd.DataFrame:
        """Load current buffer dataframe from CSV."""
        if not os.path.exists(self.buffer_path) or os.path.getsize(self.buffer_path) == 0:
            return pd.DataFrame(columns=REQUIRED_TELEMETRY_COLUMNS)
        return pd.read_csv(self.buffer_path)

    def get_record_count(self) -> int:
        """Return total number of records currently in buffer."""
        if not os.path.exists(self.buffer_path) or os.path.getsize(self.buffer_path) == 0:
            return 0
        try:
            with open(self.buffer_path, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return len(self.get_dataframe())

    def get_latest_reading(self) -> Optional[pd.Series]:
        """Return the latest single observation in the buffer."""
        df = self.get_dataframe()
        if df.empty:
            return None
        return df.iloc[-1]

    def get_latest_timestamp(self) -> Optional[str]:
        """Return the timestamp string of the latest observation."""
        latest = self.get_latest_reading()
        if latest is not None and "timestamp" in latest:
            return str(latest["timestamp"])
        return None

    def append_reading(self, reading: Union[Dict[str, Any], pd.Series]) -> bool:
        """
        Validate and append a single live telemetry observation to the buffer.
        
        Validations performed:
        1. Target leakage check: Rejects any reading containing target columns.
        2. Schema check: Ensures all 8 foundational columns are present.
        3. Numeric validation: Ensures all 7 electrical values are valid non-NaN numbers.
        4. Timestamp parsing: Validates timestamp can be parsed to valid datetime.
        5. Uniqueness & Chronology: Ensures incoming timestamp is strictly greater
           than the previous timestamp in the buffer and not duplicated.
        
        Returns True on successful append; raises ValueError on validation failure.
        """
        if isinstance(reading, pd.Series):
            reading_dict = reading.to_dict()
        elif isinstance(reading, dict):
            reading_dict = reading.copy()
        else:
            err_msg = f"Invalid reading type: Expected dict or pd.Series, got {type(reading)}"
            _log_event(f"REJECTED: {err_msg}", self.log_file)
            raise ValueError(err_msg)

        # 1. Anti-leakage check: Reject any target column
        for forbidden in FORBIDDEN_TARGET_KEYS:
            for k in reading_dict.keys():
                if forbidden.lower() in k.lower():
                    err_msg = (
                        f"Target Leakage Rejected: Reading contains forbidden target column '{k}'. "
                        "Live telemetry buffer must never receive future target values."
                    )
                    _log_event(f"REJECTED: {err_msg}", self.log_file)
                    raise ValueError(err_msg)

        # 2. Check for missing required columns
        # Support 'date' as an alias for 'timestamp' in input dict if passed
        if "timestamp" not in reading_dict and "date" in reading_dict:
            reading_dict["timestamp"] = reading_dict["date"]

        missing = [c for c in REQUIRED_TELEMETRY_COLUMNS if c not in reading_dict]
        if missing:
            err_msg = f"Schema Validation Failed: Missing required columns: {missing}"
            _log_event(f"REJECTED: {err_msg}", self.log_file)
            raise ValueError(err_msg)

        # 3. Numeric validation for telemetry values
        cleaned_row = {}
        for num_field in NUMERIC_FIELDS:
            raw_val = reading_dict[num_field]
            try:
                flt_val = float(raw_val)
                if np.isnan(flt_val) or np.isinf(flt_val):
                    raise ValueError(f"Value is {flt_val}")
                cleaned_row[num_field] = flt_val
            except Exception as e:
                err_msg = f"Numeric Validation Failed: Field '{num_field}' has invalid value '{raw_val}': {e}"
                _log_event(f"REJECTED: {err_msg}", self.log_file)
                raise ValueError(err_msg)

        raw_ts = str(reading_dict["timestamp"]).strip()
        cleaned_row["timestamp"] = raw_ts

        # 4. Timestamp validity check
        try:
            incoming_eff_dt = self._parse_effective_datetime(raw_ts, cleaned_row["NSM"])
        except Exception as e:
            err_msg = f"Timestamp Validation Failed: Invalid timestamp '{raw_ts}': {e}"
            _log_event(f"REJECTED: {err_msg}", self.log_file)
            raise ValueError(err_msg)

        # 5. Chronological and Uniqueness Validation against Buffer
        if os.path.exists(self.buffer_path) and os.path.getsize(self.buffer_path) > 0:
            existing_df = pd.read_csv(self.buffer_path)
            if not existing_df.empty:
                # Check duplicate timestamp
                existing_timestamps = set(existing_df["timestamp"].astype(str).str.strip())
                if raw_ts in existing_timestamps:
                    err_msg = f"Duplicate Timestamp Rejected: Timestamp '{raw_ts}' already exists in buffer."
                    _log_event(f"REJECTED: {err_msg}", self.log_file)
                    raise ValueError(err_msg)

                # Check chronological monotonicity against latest record
                latest_row = existing_df.iloc[-1]
                latest_raw_ts = str(latest_row["timestamp"]).strip()
                latest_nsm = float(latest_row["NSM"])
                try:
                    last_eff_dt = self._parse_effective_datetime(latest_raw_ts, latest_nsm)
                    if incoming_eff_dt <= last_eff_dt:
                        err_msg = (
                            f"Non-Chronological Timestamp Rejected: Incoming effective timestamp '{incoming_eff_dt}' "
                            f"is not strictly after latest buffer timestamp '{last_eff_dt}' (raw: '{latest_raw_ts}')."
                        )
                        _log_event(f"REJECTED: {err_msg}", self.log_file)
                        raise ValueError(err_msg)
                except Exception as e:
                    if "Rejected" in str(e):
                        raise
                    _log_event(f"Warning: Failed to parse last record timestamp '{latest_raw_ts}': {e}", self.log_file)

        # 6. Append valid row to buffer CSV
        df_to_append = pd.DataFrame([cleaned_row])[REQUIRED_TELEMETRY_COLUMNS]
        file_exists = os.path.exists(self.buffer_path) and os.path.getsize(self.buffer_path) > 0
        df_to_append.to_csv(
            self.buffer_path, 
            mode="a", 
            header=not file_exists, 
            index=False
        )

        _log_event(
            f"Successfully appended live observation: Timestamp='{raw_ts}', "
            f"Usage={cleaned_row['Usage_kWh']:.2f} kWh, Total records={self.get_record_count()}",
            self.log_file
        )
        return True

    def initialize_with_history(
        self, 
        source_csv: str = "train.csv", 
        initial_rows: int = 700, 
        overwrite: bool = False
    ) -> int:
        """
        Initialize the live telemetry buffer with initial historical observations
        to satisfy the 673-record history requirement for weekly lag features.
        
        Strictly strips any target column and conforms to live telemetry schema.
        """
        if os.path.exists(self.buffer_path) and os.path.getsize(self.buffer_path) > 0 and not overwrite:
            count = self.get_record_count()
            _log_event(f"Buffer already exists with {count} records. Skipping initial seeding.", self.log_file)
            return count

        if not os.path.isabs(source_csv):
            source_csv = os.path.join(PROJECT_ROOT, source_csv)

        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Source CSV '{source_csv}' not found for seeding buffer.")

        _log_event(f"Initializing live buffer from '{source_csv}' with {initial_rows} records...", self.log_file)
        raw_df = pd.read_csv(source_csv, nrows=initial_rows)

        # Identify date column
        time_col = None
        for cand in ["date", "datetime", "timestamp", "Date", "Timestamp"]:
            if cand in raw_df.columns:
                time_col = cand
                break
        if not time_col:
            raise ValueError(f"No timestamp column found in source CSV '{source_csv}'.")

        # Map to required schema and drop any target column
        seed_df = pd.DataFrame()
        seed_df["timestamp"] = raw_df[time_col].astype(str).str.strip()
        for col in NUMERIC_FIELDS:
            if col not in raw_df.columns:
                raise ValueError(f"Required telemetry column '{col}' missing from seed source '{source_csv}'.")
            seed_df[col] = raw_df[col].astype(float)

        seed_df = seed_df[REQUIRED_TELEMETRY_COLUMNS]
        seed_df.to_csv(self.buffer_path, index=False)
        _log_event(
            f"Live buffer initialized at '{self.buffer_path}' with {len(seed_df)} records. "
            f"Zero target columns included.",
            self.log_file
        )
        return len(seed_df)


def main():
    parser = argparse.ArgumentParser(description="Live Telemetry Buffer Management")
    parser.add_argument("--info", action="store_true", help="Print current buffer stats and exit.")
    parser.add_argument("--init", type=int, nargs="?", const=700, default=None,
                        help="Initialize or reset buffer with N historical warm-up rows (default 700).")
    args = parser.parse_args()

    buffer = LiveTelemetryBuffer()

    if args.init is not None:
        buffer.initialize_with_history(initial_rows=args.init, overwrite=True)
        print(f"Initialized live buffer with {buffer.get_record_count()} records.")
        return

    count = buffer.get_record_count()
    print("=" * 50)
    print("LIVE TELEMETRY BUFFER STATUS")
    print("=" * 50)
    print(f"Buffer File:    {buffer.buffer_path}")
    print(f"Total Records:  {count}")
    if count > 0:
        df = buffer.get_dataframe()
        print(f"Earliest TS:    {df.iloc[0]['timestamp']}")
        print(f"Latest TS:      {df.iloc[-1]['timestamp']}")
        print(f"Latest Usage:   {df.iloc[-1]['Usage_kWh']:.2f} kWh")
        print(f"History Status: {'ADEQUATE (>= 673)' if count >= MIN_REQUIRED_RECORDS else 'INSUFFICIENT (< 673)'}")
    else:
        print("Buffer is currently empty.")
    print("=" * 50)


if __name__ == "__main__":
    main()
