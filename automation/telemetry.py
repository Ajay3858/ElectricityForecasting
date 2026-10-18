"""Telemetry ingestion abstraction for the electricity forecasting system."""

import os
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional

from src.utils import parse_and_sort_chronological

MIN_REQUIRED_ROWS = 673

REQUIRED_TELEMETRY_COLUMNS = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]

TARGET_COLUMN = "next_interval_Usage_kWh"


class TelemetrySource(ABC):

    @abstractmethod
    def get_historical_data(self, min_rows: int = MIN_REQUIRED_ROWS) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_latest_data(self) -> pd.Series:
        pass

    @staticmethod
    def validate_telemetry(
        df: pd.DataFrame,
        source_identifier: str = "Telemetry"
    ) -> str:

        time_col = None

        for candidate in [
            "date",
            "datetime",
            "timestamp",
            "Date",
            "Timestamp",
        ]:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            raise ValueError(
                f"Telemetry Validation Error ({source_identifier}): "
                "Missing timestamp column."
            )

        missing_cols = [
            c for c in REQUIRED_TELEMETRY_COLUMNS
            if c not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Telemetry Validation Error ({source_identifier}): "
                f"Missing required telemetry columns: {missing_cols}"
            )

        return time_col


class CSVTelemetrySource(TelemetrySource):

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_historical_data(
        self,
        min_rows: int = MIN_REQUIRED_ROWS
    ) -> pd.DataFrame:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Telemetry Error: File '{self.file_path}' does not exist."
            )

        raw_df = pd.read_csv(self.file_path)

        time_col = self.validate_telemetry(
            raw_df,
            source_identifier=self.file_path
        )

        if len(raw_df) < min_rows:
            raise ValueError(
                f"Telemetry History Error: "
                f"Available={len(raw_df)}, Required={min_rows}"
            )

        clean_df = raw_df.copy()

        if TARGET_COLUMN in clean_df.columns:
            clean_df = clean_df.drop(columns=[TARGET_COLUMN])

        return parse_and_sort_chronological(
            clean_df,
            date_col=time_col
        )

    def get_latest_data(self) -> pd.Series:

        hist_df = self.get_historical_data(min_rows=1)

        return hist_df.iloc[-1]


class LiveCSVTelemetrySource(TelemetrySource):

    """
    Live telemetry source.

    IMPORTANT:
    The simulator_state.json file determines how many source rows
    have actually been released. The scheduler must NEVER use rows
    beyond that release position.
    """

    def __init__(
        self,
        file_path: str = os.path.join(
            "automation",
            "live_telemetry.csv"
        ),
        state_file: str = os.path.join(
            "automation",
            "simulator_state.json"
        ),
    ):

        self.file_path = file_path
        self.state_file = state_file

    def _get_released_row_count(self) -> Optional[int]:
        """
        Read simulator_state.json.

        Returns:
            Number of released rows, or None if no simulator state exists.
        """

        if not os.path.exists(self.state_file):
            return None

        try:

            with open(
                self.state_file,
                "r",
                encoding="utf-8"
            ) as f:

                state = json.load(f)

            value = state.get("released_rows")

            if value is None:
                value = state.get("current_row")

            if value is None:
                value = state.get("next_source_row")

            if value is None:
                return None

            return int(value)

        except Exception as e:

            raise ValueError(
                f"Live Telemetry State Error: "
                f"Could not read '{self.state_file}': {e}"
            )

    def get_historical_data(
        self,
        min_rows: int = MIN_REQUIRED_ROWS
    ) -> pd.DataFrame:

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Live Telemetry Error: "
                f"Buffer file '{self.file_path}' does not exist."
            )

        raw_df = pd.read_csv(self.file_path)

        time_col = self.validate_telemetry(
            raw_df,
            source_identifier=f"LiveCSV:{self.file_path}"
        )

        # ---------------------------------------------------------
        # IMPORTANT FIX
        # ---------------------------------------------------------
        # If simulator_state.json exists, only rows that have
        # actually been released by the simulator are visible
        # to the forecasting system.
        # ---------------------------------------------------------

        released_rows = self._get_released_row_count()

        if released_rows is not None:

            if released_rows < min_rows:

                raise ValueError(
                    f"Live Telemetry History Error: "
                    f"Only {released_rows:,} released observations "
                    f"are available; {min_rows:,} are required."
                )

            if released_rows > len(raw_df):
                raise ValueError(
                    f"Live Telemetry State Error: "
                    f"State says {released_rows:,} rows are released, "
                    f"but CSV contains only {len(raw_df):,} rows."
                )

            # CRITICAL:
            # Do not use unreleased rows.
            clean_df = raw_df.iloc[:released_rows].copy()

        else:

            if len(raw_df) < min_rows:

                raise ValueError(
                    f"Live Telemetry History Error: "
                    f"Available={len(raw_df):,}, "
                    f"Required={min_rows:,}"
                )

            clean_df = raw_df.copy()

        # ---------------------------------------------------------
        # ANTI-LEAKAGE
        # ---------------------------------------------------------

        if TARGET_COLUMN in clean_df.columns:

            clean_df = clean_df.drop(
                columns=[TARGET_COLUMN]
            )

        # ---------------------------------------------------------
        # Chronological ordering
        # ---------------------------------------------------------

        sorted_df = parse_and_sort_chronological(
            clean_df,
            date_col=time_col
        )

        if len(sorted_df) < min_rows:

            raise ValueError(
                f"Live Telemetry History Error: "
                f"Only {len(sorted_df):,} usable observations "
                f"after processing; {min_rows:,} required."
            )

        return sorted_df

    def get_latest_data(self) -> pd.Series:

        hist_df = self.get_historical_data(
            min_rows=1
        )

        return hist_df.iloc[-1]


def create_telemetry_source(
    config: dict,
    base_dir: Optional[str] = None
) -> TelemetrySource:

    telemetry_cfg = config.get(
        "telemetry",
        {}
    )

    source_type = telemetry_cfg.get(
        "source",
        "csv"
    ).lower()

    if source_type == "csv":

        file_path = telemetry_cfg.get(
            "file",
            "train.csv"
        )

        if base_dir and not os.path.isabs(file_path):

            file_path = os.path.join(
                base_dir,
                file_path
            )

        return CSVTelemetrySource(
            file_path=file_path
        )

    elif source_type == "live_csv":

        file_path = telemetry_cfg.get(
            "file",
            os.path.join(
                "automation",
                "live_telemetry.csv"
            )
        )

        if base_dir and not os.path.isabs(file_path):

            file_path = os.path.join(
                base_dir,
                file_path
            )

        state_file = os.path.join(
            base_dir or os.getcwd(),
            "automation",
            "simulator_state.json"
        )

        return LiveCSVTelemetrySource(
            file_path=file_path,
            state_file=state_file
        )

    else:

        raise NotImplementedError(
            f"Unsupported telemetry source type "
            f"'{source_type}'."
        )