"""Comprehensive Data Quality Audit for train.csv"""
import pandas as pd
import numpy as np
import os
import sys

DATA_PATH = r"c:\Users\AJAY R\ElectricityForecasting\train.csv"
REPORT_DIR = r"c:\Users\AJAY R\ElectricityForecasting\reports"
REPORT_PATH = os.path.join(REPORT_DIR, "data_quality_audit.txt")

os.makedirs(REPORT_DIR, exist_ok=True)

lines = []
def log(msg=""):
    lines.append(str(msg))
    print(msg)

df = pd.read_csv(DATA_PATH)

# ============================================================
# 1. Dataset shape, column names, data types
# ============================================================
log("=" * 70)
log("1. DATASET SHAPE, COLUMNS, DATA TYPES")
log("=" * 70)
log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Columns: {list(df.columns)}")
log("\nData types:")
for col in df.columns:
    log(f"  {col}: {df[col].dtype}")

# ============================================================
# 2. Target distribution
# ============================================================
log("\n" + "=" * 70)
log("2. TARGET DISTRIBUTION")
log("=" * 70)
for col in ["Usage_kWh", "next_interval_Usage_kWh"]:
    desc = df[col].describe()
    log(f"\n{col}:")
    log(f"  count  = {desc['count']}")
    log(f"  mean   = {desc['mean']:.6f}")
    log(f"  std    = {desc['std']:.6f}")
    log(f"  min    = {desc['min']:.6f}")
    log(f"  25%    = {desc['25%']:.6f}")
    log(f"  50%    = {desc['50%']:.6f}")
    log(f"  75%    = {desc['75%']:.6f}")
    log(f"  max    = {desc['max']:.6f}")
    log(f"  median = {df[col].median():.6f}")

# ============================================================
# 3. Missing values per column
# ============================================================
log("\n" + "=" * 70)
log("3. MISSING VALUES PER COLUMN")
log("=" * 70)
missing = df.isnull().sum()
total_missing = missing.sum()
for col in df.columns:
    log(f"  {col}: {missing[col]}")
log(f"  TOTAL MISSING: {total_missing}")

# ============================================================
# 4. Duplicate rows
# ============================================================
log("\n" + "=" * 70)
log("4. DUPLICATE ROWS")
log("=" * 70)
dup_count = df.duplicated().sum()
log(f"  Fully duplicate rows: {dup_count}")
if dup_count > 0:
    log(f"  First duplicate indices: {list(df[df.duplicated()].index[:10])}")

# ============================================================
# 5. Duplicate timestamps
# ============================================================
log("\n" + "=" * 70)
log("5. DUPLICATE TIMESTAMPS")
log("=" * 70)
dt = pd.to_datetime(df["date"], dayfirst=True)
dup_ts = dt.duplicated().sum()
log(f"  Duplicate timestamps: {dup_ts}")
if dup_ts > 0:
    dup_vals = dt[dt.duplicated(keep=False)]
    log(f"  Example duplicate timestamps (first 10):")
    for v in dup_vals.head(10):
        log(f"    {v}")

# ============================================================
# 6. Infinite values
# ============================================================
log("\n" + "=" * 70)
log("6. INFINITE VALUES IN NUMERIC COLUMNS")
log("=" * 70)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    inf_count = np.isinf(df[col]).sum()
    log(f"  {col}: {inf_count} infinite values")

# ============================================================
# 7. Values outside plausible ranges
# ============================================================
log("\n" + "=" * 70)
log("7. VALUES OUTSIDE PLAUSIBLE RANGES")
log("=" * 70)
neg_usage = (df["Usage_kWh"] < 0).sum()
neg_target = (df["next_interval_Usage_kWh"] < 0).sum()
log(f"  Negative Usage_kWh: {neg_usage}")
log(f"  Negative next_interval_Usage_kWh: {neg_target}")

for col in ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
            "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]:
    if col in df.columns:
        below = (df[col] < 0).sum()
        above = (df[col] > 100).sum()
        log(f"  {col}: {below} below 0, {above} above 100")

if "CO2(tCO2)" in df.columns:
    neg_co2 = (df["CO2(tCO2)"] < 0).sum()
    log(f"  Negative CO2(tCO2): {neg_co2}")

if "NSM" in df.columns:
    nsm_neg = (df["NSM"] < 0).sum()
    nsm_over = (df["NSM"] >= 86400).sum()
    log(f"  NSM < 0: {nsm_neg}")
    log(f"  NSM >= 86400 (over 24h): {nsm_over}")
    log(f"  NSM range: [{df['NSM'].min()}, {df['NSM'].max()}]")

# ============================================================
# 8. Categorical labels
# ============================================================
log("\n" + "=" * 70)
log("8. CATEGORICAL LABELS")
log("=" * 70)
for col in ["WeekStatus", "Day_of_week", "Load_Type"]:
    if col in df.columns:
        uniq = df[col].unique()
        log(f"  {col}: {len(uniq)} unique values: {sorted(uniq)}")

# ============================================================
# 9. Date parsing & range
# ============================================================
log("\n" + "=" * 70)
log("9. DATE PARSING & RANGE")
log("=" * 70)
df["datetime"] = pd.to_datetime(df["date"], dayfirst=True)
log(f"  Raw date first 5: {list(df['date'].head())}")
log(f"  Parsed datetime first 5: {list(df['datetime'].head())}")
log(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
log(f"  Number of unique dates (day-level): {df['datetime'].dt.date.nunique()}")
log(f"  Number of unique months: {df['datetime'].dt.to_period('M').nunique()}")

# ============================================================
# 10. Chronological ordering
# ============================================================
log("\n" + "=" * 70)
log("10. CHRONOLOGICAL ORDERING")
log("=" * 70)
is_sorted = df["datetime"].is_monotonic_increasing
log(f"  Strictly increasing: {is_sorted}")
diffs = df["datetime"].diff()
neg_diffs = (diffs < pd.Timedelta(0)).sum()
log(f"  Rows where timestamp goes backward: {neg_diffs}")
if neg_diffs > 0:
    bad_idx = df.index[diffs < pd.Timedelta(0)]
    for i in bad_idx[:10]:
        log(f"    Row {i}: {df['datetime'].iloc[i-1]} -> {df['datetime'].iloc[i]}")

# ============================================================
# 11. 15-minute interval consistency
# ============================================================
log("\n" + "=" * 70)
log("11. 15-MINUTE INTERVAL CONSISTENCY")
log("=" * 70)
diffs_min = diffs.dt.total_seconds() / 60.0
diffs_min_valid = diffs_min.dropna()
expected_15 = (diffs_min_valid == 15.0).sum()
not_15 = (diffs_min_valid != 15.0).sum()
log(f"  Total consecutive pairs: {len(diffs_min_valid)}")
log(f"  Pairs with exactly 15-min gap: {expected_15}")
log(f"  Pairs with non-15-min gap: {not_15}")
if not_15 > 0:
    log(f"  Unique gap values (minutes): {sorted(diffs_min_valid.unique())}")
    log(f"\n  Distribution of gap values:")
    gap_counts = diffs_min_valid.value_counts().sort_index()
    for gap_val, cnt in gap_counts.items():
        log(f"    {gap_val:.1f} min: {cnt} occurrences")
    anomalous = df.index[diffs_min != 15.0].tolist()
    anomalous = [i for i in anomalous if i > 0]
    log(f"\n  First 15 non-15-min gap locations:")
    for i in anomalous[:15]:
        log(f"    Row {i}: {df['datetime'].iloc[i-1]} -> {df['datetime'].iloc[i]}  (gap={diffs_min.iloc[i]:.1f} min)")

# ============================================================
# 12. NSM consistency
# ============================================================
log("\n" + "=" * 70)
log("12. NSM (Number of Seconds from Midnight) CONSISTENCY")
log("=" * 70)
computed_nsm = df["datetime"].dt.hour * 3600 + df["datetime"].dt.minute * 60 + df["datetime"].dt.second
nsm_match = (df["NSM"] == computed_nsm).sum()
nsm_mismatch = (df["NSM"] != computed_nsm).sum()
log(f"  NSM matches time-of-day from parsed datetime: {nsm_match}")
log(f"  NSM mismatches: {nsm_mismatch}")
if nsm_mismatch > 0:
    mismatch_idx = df.index[df["NSM"] != computed_nsm]
    log(f"  First 10 mismatches:")
    for i in mismatch_idx[:10]:
        log(f"    Row {i}: datetime={df['datetime'].iloc[i]}, NSM={df['NSM'].iloc[i]}, computed={computed_nsm.iloc[i]}")

# ============================================================
# 13. Day_of_week consistency
# ============================================================
log("\n" + "=" * 70)
log("13. DAY_OF_WEEK CONSISTENCY")
log("=" * 70)
day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
computed_dow = df["datetime"].dt.dayofweek.map(day_map)
dow_match = (df["Day_of_week"] == computed_dow).sum()
dow_mismatch = (df["Day_of_week"] != computed_dow).sum()
log(f"  Day_of_week matches parsed datetime: {dow_match}")
log(f"  Day_of_week mismatches: {dow_mismatch}")
if dow_mismatch > 0:
    mm_idx = df.index[df["Day_of_week"] != computed_dow]
    log(f"  First 15 mismatches:")
    for i in mm_idx[:15]:
        log(f"    Row {i}: datetime={df['datetime'].iloc[i]} (actual={computed_dow.iloc[i]}), recorded={df['Day_of_week'].iloc[i]}, NSM={df['NSM'].iloc[i]}")
    log(f"\n  Mismatch breakdown by NSM:")
    nsm_of_mismatches = df.loc[mm_idx, "NSM"].value_counts().sort_index()
    for nsm_val, cnt in nsm_of_mismatches.head(20).items():
        log(f"    NSM={nsm_val}: {cnt}")

# ============================================================
# 14. WeekStatus consistency
# ============================================================
log("\n" + "=" * 70)
log("14. WEEKSTATUS CONSISTENCY")
log("=" * 70)
computed_ws = computed_dow.apply(lambda d: "Weekend" if d in ["Saturday", "Sunday"] else "Weekday")
ws_match = (df["WeekStatus"] == computed_ws).sum()
ws_mismatch = (df["WeekStatus"] != computed_ws).sum()
log(f"  WeekStatus matches parsed datetime: {ws_match}")
log(f"  WeekStatus mismatches: {ws_mismatch}")
if ws_mismatch > 0:
    mm_idx2 = df.index[df["WeekStatus"] != computed_ws]
    log(f"  First 15 mismatches:")
    for i in mm_idx2[:15]:
        log(f"    Row {i}: datetime={df['datetime'].iloc[i]} ({computed_dow.iloc[i]}), recorded WeekStatus={df['WeekStatus'].iloc[i]}, expected={computed_ws.iloc[i]}")

# ============================================================
# 15. Midnight handling
# ============================================================
log("\n" + "=" * 70)
log("15. MIDNIGHT HANDLING")
log("=" * 70)
nsm_900 = df[df["NSM"] == 900]
log(f"  Rows with NSM=900 (00:15:00): {len(nsm_900)}")
if len(nsm_900) > 0:
    log(f"  First 5 rows with NSM=900:")
    for i in nsm_900.index[:5]:
        log(f"    Row {i}: date={df['date'].iloc[i]}, datetime={df['datetime'].iloc[i]}, Day_of_week={df['Day_of_week'].iloc[i]}")

log(f"\n  First row: date={df['date'].iloc[0]}, datetime={df['datetime'].iloc[0]}, NSM={df['NSM'].iloc[0]}")
log(f"  Second row: date={df['date'].iloc[1]}, datetime={df['datetime'].iloc[1]}, NSM={df['NSM'].iloc[1]}")
log(f"  Last row: date={df['date'].iloc[-1]}, datetime={df['datetime'].iloc[-1]}, NSM={df['NSM'].iloc[-1]}")

# ============================================================
# 16. Target alignment
# ============================================================
log("\n" + "=" * 70)
log("16. TARGET ALIGNMENT: next_interval_Usage_kWh[i] == Usage_kWh[i+1]")
log("=" * 70)
n = len(df)
target_vals = df["next_interval_Usage_kWh"].values[:-1]
next_usage = df["Usage_kWh"].values[1:]
matches = np.sum(np.isclose(target_vals, next_usage, atol=1e-9))
mismatches = len(target_vals) - matches
log(f"  Pairs checked: {len(target_vals)}")
log(f"  Matches: {matches}")
log(f"  Mismatches: {mismatches}")
if mismatches > 0:
    mismatch_mask = ~np.isclose(target_vals, next_usage, atol=1e-9)
    mismatch_indices = np.where(mismatch_mask)[0]
    log(f"  First 10 mismatch locations:")
    for idx in mismatch_indices[:10]:
        log(f"    Row {idx}: next_interval={target_vals[idx]}, Usage_kWh[{idx+1}]={next_usage[idx]}, datetime[{idx}]={df['datetime'].iloc[idx]}, datetime[{idx+1}]={df['datetime'].iloc[idx+1]}")

# ============================================================
# 17. Last row
# ============================================================
log("\n" + "=" * 70)
log("17. LAST ROW ANALYSIS")
log("=" * 70)
last_idx = n - 1
log(f"  Last row index: {last_idx}")
log(f"  Last row date: {df['date'].iloc[last_idx]}")
log(f"  Last row datetime: {df['datetime'].iloc[last_idx]}")
log(f"  Last row Usage_kWh: {df['Usage_kWh'].iloc[last_idx]}")
log(f"  Last row next_interval_Usage_kWh: {df['next_interval_Usage_kWh'].iloc[last_idx]}")
log(f"  Is there a following observation? NO (last row in dataset)")

# ============================================================
# 18. Midnight rows
# ============================================================
log("\n" + "=" * 70)
log("18. MIDNIGHT ROWS (NSM=0)")
log("=" * 70)
nsm_0 = df[df["NSM"] == 0]
log(f"  Rows with NSM=0 (exactly midnight): {len(nsm_0)}")
if len(nsm_0) > 0:
    log(f"  First 5:")
    for i in nsm_0.index[:5]:
        log(f"    Row {i}: date={df['date'].iloc[i]}, datetime={df['datetime'].iloc[i]}")
else:
    log("  No rows at exactly midnight (NSM=0).")
    log(f"  Minimum NSM value: {df['NSM'].min()}")
    min_nsm_rows = df[df["NSM"] == df["NSM"].min()]
    log(f"  Rows with minimum NSM ({df['NSM'].min()}):")
    for i in min_nsm_rows.index[:5]:
        log(f"    Row {i}: date={df['date'].iloc[i]}, datetime={df['datetime'].iloc[i]}")

# ============================================================
log("\n" + "=" * 70)
log("AUDIT COMPLETE")
log("=" * 70)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved to: {REPORT_PATH}")
