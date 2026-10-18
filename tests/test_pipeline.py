"""Comprehensive automated test suite for Electricity Consumption Forecasting pipeline.

Covers:
- Timestamp parsing and midnight handling
- Chronological sorting
- Target alignment verification
- Feature column consistency
- Lag and rolling feature correctness
- Leakage detection (target not in features)
- Train/validation boundary checks
- Model loading and inference
- Invalid input handling
- Prediction output schema
- Reproducibility with fixed seed
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features import engineer_features, split_chronological, FEATURE_COLUMNS, TARGET_COLUMN
from src.utils import parse_and_sort_chronological, calculate_metrics, set_seed


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def raw_df():
    """Load the real training dataset."""
    path = os.path.join(PROJECT_ROOT, "train.csv")
    if not os.path.exists(path):
        pytest.skip("train.csv not found")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def parsed_df(raw_df):
    """Parse and sort the dataset chronologically."""
    return parse_and_sort_chronological(raw_df, date_col="date")


@pytest.fixture(scope="module")
def features_df(raw_df):
    """Full feature-engineered dataset."""
    return engineer_features(raw_df, drop_na=True)


@pytest.fixture(scope="module")
def features_df_no_drop(raw_df):
    """Feature-engineered dataset without dropping NaN warmup rows."""
    return engineer_features(raw_df, drop_na=False)


def make_synthetic_df(n_rows=700, start_date="2018-01-01 00:15:00"):
    """Create a small synthetic dataset for fast testing."""
    dates = pd.date_range(start_date, periods=n_rows, freq="15min")
    np.random.seed(42)
    usage = np.random.uniform(3.0, 80.0, size=n_rows)
    df = pd.DataFrame({
        "date": dates.strftime("%d/%m/%Y %H:%M"),
        "Usage_kWh": usage,
        "Lagging_Current_Reactive.Power_kVarh": np.random.uniform(0, 50, n_rows),
        "Leading_Current_Reactive_Power_kVarh": np.random.uniform(0, 5, n_rows),
        "CO2(tCO2)": np.random.uniform(0, 0.1, n_rows),
        "Lagging_Current_Power_Factor": np.random.uniform(60, 100, n_rows),
        "Leading_Current_Power_Factor": np.random.uniform(90, 100, n_rows),
        "NSM": [(d.hour * 3600 + d.minute * 60 + d.second) for d in dates],
        "WeekStatus": ["Weekday" if d.weekday() < 5 else "Weekend" for d in dates],
        "Day_of_week": [d.strftime("%A") for d in dates],
        "Load_Type": np.random.choice(["Light_Load", "Medium_Load", "Maximum_Load"], n_rows),
        "next_interval_Usage_kWh": np.append(usage[1:], [np.random.uniform(3.0, 80.0)]),
    })
    return df


@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_df(n_rows=700)


# ============================================================================
# Test: Timestamp Parsing
# ============================================================================

class TestTimestampParsing:
    def test_date_column_parsed(self, parsed_df):
        assert "datetime" in parsed_df.columns

    def test_datetime_dtype(self, parsed_df):
        assert pd.api.types.is_datetime64_any_dtype(parsed_df["datetime"])

    def test_date_range(self, parsed_df):
        assert parsed_df["datetime"].min().year == 2018
        assert parsed_df["datetime"].min().month == 1
        assert parsed_df["datetime"].max().year == 2018
        assert parsed_df["datetime"].max().month == 10

    def test_dayfirst_parsing(self, parsed_df):
        first_date = parsed_df["datetime"].iloc[0]
        assert first_date.month == 1
        assert first_date.day == 1

    def test_no_nat_values(self, parsed_df):
        assert parsed_df["datetime"].isna().sum() == 0


# ============================================================================
# Test: Chronological Sorting
# ============================================================================

class TestChronologicalSorting:
    def test_sorted_ascending(self, parsed_df):
        diffs = parsed_df["datetime"].diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all()

    def test_no_duplicate_timestamps(self, parsed_df):
        assert parsed_df["datetime"].duplicated().sum() == 0

    def test_15min_intervals(self, parsed_df):
        diffs = parsed_df["datetime"].diff().dropna()
        expected = pd.Timedelta(minutes=15)
        matches = (diffs == expected).sum()
        assert matches == len(diffs), f"Only {matches}/{len(diffs)} intervals are exactly 15 min"


# ============================================================================
# Test: Target Alignment
# ============================================================================

class TestTargetAlignment:
    def test_target_matches_next_row(self, parsed_df):
        target = parsed_df["next_interval_Usage_kWh"].values[:-1]
        next_usage = parsed_df["Usage_kWh"].values[1:]
        mismatches = np.sum(~np.isclose(target, next_usage, atol=1e-6))
        assert mismatches == 0, f"{mismatches} target mismatches found"

    def test_target_column_exists(self, raw_df):
        assert TARGET_COLUMN in raw_df.columns

    def test_engineered_final_row_is_excluded(self, raw_df):
        df = engineer_features(raw_df, drop_na=True)

        assert df[TARGET_COLUMN].notna().all()
        assert df["datetime"].max() < parse_and_sort_chronological(
            raw_df
        )["datetime"].max()
# ============================================================================
# Test: Feature Column Consistency
# ============================================================================

class TestFeatureColumns:
    def test_all_feature_columns_present(self, features_df):
        missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
        assert len(missing) == 0, f"Missing feature columns: {missing}"

    def test_target_not_in_features(self):
        assert TARGET_COLUMN not in FEATURE_COLUMNS, "TARGET LEAKAGE: target in feature list!"

    def test_feature_count(self):
        assert len(FEATURE_COLUMNS) == 44

    def test_no_nan_in_features_after_drop(self, features_df):
        nan_counts = features_df[FEATURE_COLUMNS].isna().sum()
        total_nans = nan_counts.sum()
        assert total_nans == 0, f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}"

    def test_no_inf_in_features(self, features_df):
        for col in FEATURE_COLUMNS:
            if features_df[col].dtype in [np.float64, np.float32]:
                assert not np.isinf(features_df[col]).any(), f"Inf found in {col}"


# ============================================================================
# Test: Lag Feature Correctness
# ============================================================================

class TestLagFeatures:
    def test_lag_1_correct(self, features_df_no_drop):
        df = features_df_no_drop
        for i in [1, 100, 500, 1000, 5000, 10000]:
            if i < len(df):
                expected = df["Usage_kWh"].iloc[i - 1]
                actual = df["usage_lag_1"].iloc[i]
                assert np.isclose(actual, expected, atol=1e-6), \
                    f"Lag-1 mismatch at row {i}: expected {expected}, got {actual}"

    def test_lag_96_correct(self, features_df_no_drop):
        df = features_df_no_drop
        for i in [96, 200, 1000, 5000]:
            if i < len(df):
                expected = df["Usage_kWh"].iloc[i - 96]
                actual = df["usage_lag_96"].iloc[i]
                assert np.isclose(actual, expected, atol=1e-6), \
                    f"Lag-96 mismatch at row {i}: expected {expected}, got {actual}"

    def test_lag_672_correct(self, features_df_no_drop):
        df = features_df_no_drop
        for i in [672, 1000, 5000, 10000]:
            if i < len(df):
                expected = df["Usage_kWh"].iloc[i - 672]
                actual = df["usage_lag_672"].iloc[i]
                assert np.isclose(actual, expected, atol=1e-6), \
                    f"Lag-672 mismatch at row {i}: expected {expected}, got {actual}"

    def test_lag_1_is_nan_for_first_row(self, features_df_no_drop):
        assert pd.isna(features_df_no_drop["usage_lag_1"].iloc[0])


# ============================================================================
# Test: Rolling Feature Correctness
# ============================================================================

class TestRollingFeatures:
    def test_rolling_mean_4(self, features_df_no_drop):
        df = features_df_no_drop
        for i in [10, 100, 500, 1000]:
            if i < len(df):
                expected = df["Usage_kWh"].iloc[i-3:i+1].mean()
                actual = df["usage_roll_mean_4"].iloc[i]
                assert np.isclose(actual, expected, atol=1e-6), \
                    f"Rolling mean mismatch at row {i}: expected {expected}, got {actual}"

    def test_rolling_uses_current_and_past_only(self, features_df_no_drop):
        df = features_df_no_drop
        i = 500
        usage_values = df["Usage_kWh"].values
        window = usage_values[i-3:i+1]
        assert np.isclose(df["usage_roll_mean_4"].iloc[i], window.mean(), atol=1e-6)

    def test_ewma_no_future(self, features_df_no_drop):
        df = features_df_no_drop
        i = 100
        truncated_ewma = df["Usage_kWh"].iloc[:i+1].ewm(span=4).mean().iloc[-1]
        actual = df["usage_ewma_4"].iloc[i]
        assert np.isclose(actual, truncated_ewma, atol=1e-6), \
            f"EWMA may use future values: truncated={truncated_ewma}, full={actual}"


# ============================================================================
# Test: Leakage Detection
# ============================================================================

class TestLeakageDetection:
    def test_target_not_used_as_feature(self):
        assert "next_interval_Usage_kWh" not in FEATURE_COLUMNS

    def test_usage_kwh_is_current_not_future(self, features_df):
        usage_vals = features_df["Usage_kWh"].values
        target_vals = features_df[TARGET_COLUMN].values
        correlation = np.corrcoef(usage_vals, target_vals)[0, 1]
        assert correlation < 0.99, f"Usage_kWh suspiciously correlated with target: r={correlation:.6f}"


# ============================================================================
# Test: Train/Validation Split
# ============================================================================

class TestTrainValSplit:
    def test_chronological_split(self, features_df):
        X_train, X_val, y_train, y_val = split_chronological(features_df, train_ratio=0.8)
        train_last_idx = X_train.index[-1]
        val_first_idx = X_val.index[0]
        assert train_last_idx < val_first_idx

    def test_no_overlap(self, features_df):
        X_train, X_val, y_train, y_val = split_chronological(features_df, train_ratio=0.8)
        overlap = set(X_train.index) & set(X_val.index)
        assert len(overlap) == 0

    def test_split_ratio(self, features_df):
        X_train, X_val, y_train, y_val = split_chronological(features_df, train_ratio=0.8)
        ratio = len(X_train) / (len(X_train) + len(X_val))
        assert abs(ratio - 0.8) < 0.01

    def test_no_future_leakage_at_boundary(self, features_df):
        X_train, X_val, y_train, y_val = split_chronological(features_df, train_ratio=0.8)
        train_end = features_df.loc[X_train.index[-1], "datetime"]
        val_start = features_df.loc[X_val.index[0], "datetime"]
        assert train_end < val_start


# ============================================================================
# Test: Model Loading
# ============================================================================

class TestModelLoading:
    def test_model_file_exists(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
        assert os.path.exists(model_path), f"Model not found at {model_path}"

    def test_metadata_file_exists(self):
        meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        assert os.path.exists(meta_path)

    def test_metadata_valid_json(self):
        meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert "best_model_name" in meta
        assert "feature_columns" in meta
        assert "target_column" in meta

    def test_metadata_features_match_code(self):
        meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["feature_columns"] == FEATURE_COLUMNS

    def test_model_loads_and_predicts(self, features_df):
        import joblib
        model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
        model = joblib.load(model_path)
        X_sample = features_df[FEATURE_COLUMNS].iloc[:5]
        preds = model.predict(X_sample)
        assert len(preds) == 5
        assert all(np.isfinite(preds))


# ============================================================================
# Test: Inference Pipeline
# ============================================================================

class TestInference:
    def test_predict_from_csv(self):
        from predict import predict_next_interval
        csv_path = os.path.join(PROJECT_ROOT, "train.csv")
        if not os.path.exists(csv_path):
            pytest.skip("train.csv not found")
        output_path = os.path.join(PROJECT_ROOT, "prediction_test_auto.csv")
        result = predict_next_interval(input_data=csv_path, output_path=output_path)
        assert isinstance(result, float)
        assert result >= 0.0
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_predict_from_dataframe(self, raw_df):
        from predict import predict_next_interval
        output_path = os.path.join(PROJECT_ROOT, "prediction_test_df.csv")
        result = predict_next_interval(input_data=raw_df, output_path=output_path)
        assert isinstance(result, float)
        assert result >= 0.0
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_prediction_output_schema(self):
        from predict import predict_next_interval
        csv_path = os.path.join(PROJECT_ROOT, "train.csv")
        if not os.path.exists(csv_path):
            pytest.skip("train.csv not found")
        output_path = os.path.join(PROJECT_ROOT, "prediction_schema_test.csv")
        predict_next_interval(input_data=csv_path, output_path=output_path)
        pred_df = pd.read_csv(output_path)
        expected_cols = {"timestamp", "current_usage_kWh", "predicted_next_15min_usage_kWh"}
        assert set(pred_df.columns) == expected_cols
        if os.path.exists(output_path):
            os.remove(output_path)


# ============================================================================
# Test: Invalid Input Handling
# ============================================================================

class TestInvalidInput:
    def test_missing_timestamp_column(self):
        from predict import validate_input_data
        df = pd.DataFrame({"Usage_kWh": [1.0], "foo": ["bar"]})
        with pytest.raises(ValueError, match="Missing timestamp column"):
            validate_input_data(df, "test")

    def test_missing_telemetry_columns(self):
        from predict import validate_input_data
        df = pd.DataFrame({"date": ["01/01/2018"], "Usage_kWh": [1.0]})
        with pytest.raises(ValueError, match="Missing required telemetry"):
            validate_input_data(df, "test")

    def test_insufficient_history(self):
        from predict import validate_input_data
        df = pd.DataFrame({
            "date": pd.date_range("2018-01-01", periods=10, freq="15min"),
            "Usage_kWh": [1.0] * 10,
            "Lagging_Current_Reactive.Power_kVarh": [1.0] * 10,
            "Leading_Current_Reactive_Power_kVarh": [0.0] * 10,
            "CO2(tCO2)": [0.0] * 10,
            "Lagging_Current_Power_Factor": [70.0] * 10,
            "Leading_Current_Power_Factor": [100.0] * 10,
            "NSM": [900] * 10,
        })
        with pytest.raises(ValueError, match="Insufficient historical"):
            validate_input_data(df, "test")

    def test_nonexistent_csv(self):
        from predict import predict_next_interval
        with pytest.raises(FileNotFoundError):
            predict_next_interval(input_data="nonexistent_file.csv")

    def test_invalid_input_type(self):
        from predict import predict_next_interval
        with pytest.raises(TypeError):
            predict_next_interval(input_data=12345)


# ============================================================================
# Test: Reproducibility
# ============================================================================

class TestReproducibility:
    def test_feature_engineering_deterministic(self, raw_df):
        df1 = engineer_features(raw_df.copy(), drop_na=True)
        df2 = engineer_features(raw_df.copy(), drop_na=True)
        for col in FEATURE_COLUMNS:
            assert np.allclose(df1[col].values, df2[col].values, equal_nan=True), \
                f"Non-deterministic feature: {col}"

    def test_metrics_calculation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        m1 = calculate_metrics(y_true, y_pred)
        m2 = calculate_metrics(y_true, y_pred)
        assert m1 == m2


# ============================================================================
# Test: Synthetic Dataset (fast, no real data needed)
# ============================================================================

class TestSyntheticPipeline:
    def test_engineer_features_on_synthetic(self, synthetic_df):
        df = engineer_features(synthetic_df, drop_na=True)
        assert len(df) > 0
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_split_on_synthetic(self, synthetic_df):
        df = engineer_features(synthetic_df, drop_na=True)
        if TARGET_COLUMN not in df.columns:
            pytest.skip("Target not in synthetic engineered data")
        X_train, X_val, y_train, y_val = split_chronological(df, train_ratio=0.8)
        assert len(X_train) > 0
        assert len(X_val) > 0


# ============================================================================
# Test: Midnight Records
# ============================================================================

class TestMidnightRecords:
    def test_first_interval_is_0015(self, parsed_df):
        """First interval should be 00:15 (NSM=900), not 00:00."""
        first = parsed_df.iloc[0]
        assert first["NSM"] == 900, f"First NSM is {first['NSM']}, expected 900"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
