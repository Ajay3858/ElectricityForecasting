import pandas as pd
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_FILE = BASE_DIR / "automation" / "config.json"
SOURCE_FILE = BASE_DIR / "train.csv"
LIVE_FILE = BASE_DIR / "automation" / "live_telemetry.csv"
STATE_FILE = BASE_DIR / "automation" / "simulator_state.json"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def find_timestamp_column(df):
    possible_names = [
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "date",
        "Date",
        "time",
        "Time"
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find timestamp column. Columns: {list(df.columns)}"
    )


def prepare_source():

    df = pd.read_csv(SOURCE_FILE)

    timestamp_column = find_timestamp_column(df)

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    df = df.dropna(subset=[timestamp_column])

    df = df.sort_values(timestamp_column).reset_index(drop=True)

    # Never expose the future target to the live system.
    if "next_interval_Usage_kWh" in df.columns:
        df = df.drop(columns=["next_interval_Usage_kWh"])

    return df, timestamp_column


def initialize():

    df, timestamp_column = prepare_source()

    # We need 673 rows for the production feature pipeline.
    initial_rows = 673

    live_df = df.iloc[:initial_rows].copy()

    live_df.to_csv(LIVE_FILE, index=False)

    state = {
        "next_row": initial_rows,
        "total_rows": len(df),
        "timestamp_column": timestamp_column
    }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    print("=" * 60)
    print("LIVE TELEMETRY SIMULATOR INITIALIZED")
    print("=" * 60)

    print(f"Initial telemetry rows: {initial_rows}")
    print(f"Total source rows:      {len(df)}")
    print(f"Latest live timestamp:  {live_df[timestamp_column].iloc[-1]}")
    print()
    print("Next run will release ONE new telemetry reading.")


def release_next():

    df, timestamp_column = prepare_source()

    if not STATE_FILE.exists():

        print("Simulator has not been initialized.")
        print("Run:")
        print("python automation\\simulate_telemetry.py --init")
        return

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    next_row = state["next_row"]

    if next_row >= len(df):

        print("All simulated telemetry has been released.")
        return

    new_row = df.iloc[[next_row]]

    if LIVE_FILE.exists():
        live_df = pd.read_csv(LIVE_FILE)
        live_df[timestamp_column] = pd.to_datetime(
            live_df[timestamp_column]
        )
    else:
        live_df = df.iloc[:673].copy()

    live_df = pd.concat(
        [live_df, new_row],
        ignore_index=True
    )

    live_df.to_csv(LIVE_FILE, index=False)

    state["next_row"] = next_row + 1

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

    released = new_row.iloc[0]

    print("=" * 60)
    print("NEW TELEMETRY READING RELEASED")
    print("=" * 60)

    print(
        f"Timestamp: {released[timestamp_column]}"
    )

    print(
        f"Usage:     {released['Usage_kWh']:.2f} kWh"
    )

    print(
        f"Live rows: {len(live_df)}"
    )

    print(
        f"Next source row: {state['next_row']}"
    )


if __name__ == "__main__":

    import sys

    if "--init" in sys.argv:
        initialize()
    else:
        release_next()