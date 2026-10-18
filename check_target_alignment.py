
import numpy as np
import pandas as pd

from src.utils import parse_and_sort_chronological

# Load the original dataset.
raw = pd.read_csv("train.csv")

# Preserve the supplied target and normalize the timestamps.
ordered = parse_and_sort_chronological(raw)

# Compare supplied targets with the next chronological consumption.
expected = ordered["Usage_kWh"].shift(-1)
actual = ordered["next_interval_Usage_kWh"]

# The final row has no next observation in this dataset.
valid = expected.notna()

matches = np.isclose(
    actual.loc[valid].to_numpy(dtype=float),
    expected.loc[valid].to_numpy(dtype=float),
    rtol=0,
    atol=1e-8,
)

print("Rows after chronological sorting:", len(ordered))
print("Target comparisons:", int(valid.sum()))
print("Matches:", int(matches.sum()))
print("Mismatches:", int((~matches).sum()))

# Check the corrected timestamp intervals.
gaps = ordered["datetime"].diff().dropna()
bad_gaps = gaps != pd.Timedelta(minutes=15)

print("Non-15-minute intervals:", int(bad_gaps.sum()))

if (~matches).any():
    mismatch_positions = np.flatnonzero(valid.to_numpy())[~matches]
    print("\nFirst 10 target mismatches:")

    for i in mismatch_positions[:10]:
        print(
            ordered.iloc[i]["datetime"],
            "current =", ordered.iloc[i]["Usage_kWh"],
            "supplied target =", ordered.iloc[i]["next_interval_Usage_kWh"],
            "next chronological usage =", expected.iloc[i],
        )
