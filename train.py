"""Training and Model Selection Pipeline for Electricity Consumption Forecasting.

Trains CatBoost, XGBoost, LightGBM, and a Weighted Ensemble using causal,
past-only features and a chronological train/validation/test split.

Workflow:
1. Load and engineer features.
2. Split data chronologically into 70% train, 15% validation, and 15% test.
3. Train candidate models and select the best by validation RMSE.
4. Refit the selected model, or ensemble components, on train + validation.
5. Evaluate once on the held-out test set.
6. Save the model, metadata, feature importance, and model report.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.base import clone

from src.utils import set_seed, calculate_metrics
from src.features import (
    engineer_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.models import WeightedEnsembleRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
MODELS_DIR = "models"
DATA_PATH = "train.csv"
REPORT_PATH = os.path.join(MODELS_DIR, "model_report.txt")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

ENSEMBLE_WEIGHTS = {
    "CatBoost": 0.50,
    "LightGBM": 0.35,
    "XGBoost": 0.15,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_metrics(y_true, predictions):
    """Calculate metrics and convert values to ordinary Python floats."""
    metrics = calculate_metrics(
        np.asarray(y_true),
        np.asarray(predictions),
    )

    return {
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "r2": float(metrics["r2"]),
    }


def print_metrics(model_name, metrics, prefix=""):
    """Print a consistent model-metrics summary."""
    print(
        f"{prefix}{model_name:<22} | "
        f"RMSE: {metrics['rmse']:.4f} kWh | "
        f"MAE: {metrics['mae']:.4f} kWh | "
        f"R2: {metrics['r2']:.4f}"
    )


def get_feature_importance(model, feature_names):
    """Return CatBoost feature importance as a sorted DataFrame."""
    importance_values = model.get_feature_importance()

    importance_df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": importance_values,
    })

    return importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def generate_report(
    total_raw_rows,
    total_processed_rows,
    train_rows,
    val_rows,
    test_rows,
    train_range,
    val_range,
    test_range,
    results,
    best_model_name,
    feat_importance_df,
    test_metrics,
    persistence_metrics,
):
    """Write the evaluation report to models/model_report.txt."""

    os.makedirs(MODELS_DIR, exist_ok=True)

    lines = []

    def add(line=""):
        lines.append(str(line))

    # --------------------------------------------------------
    # 1. Dataset summary
    # --------------------------------------------------------
    add("=" * 85)
    add("ELECTRICITY CONSUMPTION FORECASTING")
    add("MODEL TRAINING AND EVALUATION REPORT")
    add("=" * 85)
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Random seed: {RANDOM_SEED}")
    add()

    add("1. DATASET SUMMARY")
    add("-" * 85)
    add(f"Source file: {DATA_PATH}")
    add(f"Raw observations: {total_raw_rows:,}")
    add(f"Processed observations: {total_processed_rows:,}")
    add(f"Training rows: {train_rows:,}")
    add(f"Validation rows: {val_rows:,}")
    add(f"Test rows: {test_rows:,}")
    add("Sampling interval: 15 minutes (as expected by this project)")
    add(f"Target column: {TARGET_COLUMN}")
    add("Forecast horizon: Next chronological interval")
    add()

    # --------------------------------------------------------
    # 2. Chronological split
    # --------------------------------------------------------
    add("2. CHRONOLOGICAL DATA SPLIT")
    add("-" * 85)
    add(
        "Configured proportions: "
        f"{TRAIN_RATIO:.0%} train, "
        f"{VALIDATION_RATIO:.0%} validation, "
        f"{TEST_RATIO:.0%} test"
    )
    add("The split is chronological; rows are not randomly shuffled.")
    add("One row is omitted at each partition boundary.")
    add()

    add(f"Training:   {train_range[0]} to {train_range[1]}")
    add(f"Validation: {val_range[0]} to {val_range[1]}")
    add(f"Test:       {test_range[0]} to {test_range[1]}")
    add()

    # --------------------------------------------------------
    # 3. Validation results
    # --------------------------------------------------------
    add("3. VALIDATION RESULTS - MODEL SELECTION")
    add("-" * 85)
    add(
        f"{'Model':<24}"
        f"{'RMSE (kWh)':>16}"
        f"{'MAE (kWh)':>16}"
        f"{'R2':>12}"
    )
    add("-" * 68)

    for name, metrics in results.items():
        add(
            f"{name:<24}"
            f"{metrics['rmse']:>16.4f}"
            f"{metrics['mae']:>16.4f}"
            f"{metrics['r2']:>12.4f}"
        )

    add()
    add(f"Selected model: {best_model_name}")
    add("Selection criterion: Lowest validation RMSE")
    add(
        "The validation results are used to choose the model. "
        "They are not the final held-out test results."
    )
    add()

    # --------------------------------------------------------
    # 4. Held-out test results
    # --------------------------------------------------------
    add("4. FINAL HELD-OUT TEST RESULTS")
    add("-" * 85)
    add(f"Selected model: {best_model_name}")
    add(f"Test RMSE: {test_metrics['rmse']:.4f} kWh")
    add(f"Test MAE:  {test_metrics['mae']:.4f} kWh")
    add(f"Test R2:   {test_metrics['r2']:.4f}")
    add()
    add(
        "The selected model was refitted using training and validation "
        "data before its final evaluation on the held-out test set."
    )
    add()

    # --------------------------------------------------------
    # 5. Persistence baseline
    # --------------------------------------------------------
    add("5. PERSISTENCE BASELINE - TEST SET")
    add("-" * 85)
    add(
        "Baseline: predict next-interval consumption using "
        "current-interval Usage_kWh."
    )
    add(f"Test RMSE: {persistence_metrics['rmse']:.4f} kWh")
    add(f"Test MAE:  {persistence_metrics['mae']:.4f} kWh")
    add(f"Test R2:   {persistence_metrics['r2']:.4f}")
    add()

    baseline_rmse = persistence_metrics["rmse"]

    if baseline_rmse > 0:
        improvement = (
            (baseline_rmse - test_metrics["rmse"])
            / baseline_rmse
        ) * 100.0

        add(
            "Test RMSE improvement over persistence baseline: "
            f"{improvement:.2f}%"
        )
    else:
        add("RMSE improvement was not calculated because baseline RMSE is zero.")

    if test_metrics["rmse"] < baseline_rmse:
        add("The selected model has lower test RMSE than the baseline.")
    elif test_metrics["rmse"] > baseline_rmse:
        add("The persistence baseline has lower test RMSE.")
    else:
        add("The model and persistence baseline have equal test RMSE.")

    add()

    # --------------------------------------------------------
    # 6. Feature importance
    # --------------------------------------------------------
    add("6. CATBOOST FEATURE IMPORTANCE")
    add("-" * 85)
    add(
        "Feature importance below is from the final CatBoost model. "
        "It is not necessarily the importance ranking of the selected model."
    )
    add()

    if (
        feat_importance_df is not None
        and not feat_importance_df.empty
    ):
        for rank, (_, row) in enumerate(
            feat_importance_df.head(15).iterrows(),
            start=1,
        ):
            add(
                f"{rank:>2}. {row['feature']:<40} "
                f"Importance: {row['importance']:.6f}"
            )
    else:
        add("Feature importance is unavailable.")

    add()

    # --------------------------------------------------------
    # 7. Metric definitions and limitations
    # --------------------------------------------------------
    add("7. METRIC INTERPRETATION")
    add("-" * 85)
    add("RMSE: Root Mean Squared Error; lower is better.")
    add("MAE: Mean Absolute Error; lower is better.")
    add("R2: Coefficient of determination; higher is generally better.")
    add()
    add(
        "Test metrics estimate performance on the held-out chronological "
        "period. Future performance may differ."
    )
    add(
        "The test results should not be used repeatedly to tune model "
        "parameters or select features."
    )
    add()
    add("=" * 85)
    add("END OF REPORT")
    add("=" * 85)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"\nModel report saved to '{REPORT_PATH}'.")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_and_evaluate():
    """Train candidate models, select the best, and evaluate on held-out data."""

    set_seed(RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 85)
    print(" ELECTRICITY CONSUMPTION FORECASTING")
    print(" TRAINING, MODEL SELECTION, AND FINAL TEST EVALUATION")
    print("=" * 85)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    print(f"\n[1/5] Loading dataset '{DATA_PATH}'...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset '{DATA_PATH}' was not found in the current directory."
        )

    raw_df = pd.read_csv(DATA_PATH)
    total_raw_rows = len(raw_df)

    if total_raw_rows == 0:
        raise ValueError("The input dataset is empty.")

    print(f"      Loaded {total_raw_rows:,} raw records.")

    # --------------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------------
    print("\n[2/5] Engineering causal, past-only features...")

    df_features = engineer_features(raw_df, drop_na=True)
    total_processed_rows = len(df_features)

    if total_processed_rows < 10:
        raise ValueError(
            "Not enough usable rows remain after feature engineering."
        )

    if "datetime" not in df_features.columns:
        raise KeyError(
            "The engineered dataset must contain a 'datetime' column "
            "for chronological splitting."
        )

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df_features.columns
    ]

    if missing_features:
        raise KeyError(
            f"Missing required feature columns: {missing_features}"
        )

    if TARGET_COLUMN not in df_features.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    if df_features[TARGET_COLUMN].isna().any():
        raise ValueError(
            "The engineered target contains missing values. "
            "Check engineer_features() before training."
        )

    df_features = (
        df_features
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print(f"      Input features: {len(FEATURE_COLUMNS)}")
    print(f"      Usable rows: {total_processed_rows:,}")
    print(
        f"      Rows removed during preprocessing: "
        f"{total_raw_rows - total_processed_rows:,}"
    )

    # --------------------------------------------------------
    # 3. Chronological train/validation/test split
    # --------------------------------------------------------
    print(
        "\n[3/5] Splitting chronologically "
        "(70% train, 15% validation, 15% test)..."
    )

    n = len(df_features)

    train_boundary = int(n * TRAIN_RATIO)
    validation_boundary = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    # Leave one unused row at each boundary. This prevents
    # adjacent next-interval target labels from crossing partitions.
    train_df = df_features.iloc[:train_boundary - 1].copy()

    validation_df = df_features.iloc[
        train_boundary:validation_boundary - 1
    ].copy()

    test_df = df_features.iloc[validation_boundary:].copy()

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One of the data partitions is empty. "
            "Check the dataset size and split boundaries."
        )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_val = validation_df[FEATURE_COLUMNS].copy()
    y_val = validation_df[TARGET_COLUMN].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    train_start = train_df["datetime"].iloc[0]
    train_end_time = train_df["datetime"].iloc[-1]

    val_start = validation_df["datetime"].iloc[0]
    val_end_time = validation_df["datetime"].iloc[-1]

    test_start = test_df["datetime"].iloc[0]
    test_end_time = test_df["datetime"].iloc[-1]

    print(
        f"      Training:   {len(X_train):,} rows "
        f"({train_start} to {train_end_time})"
    )
    print(
        f"      Validation: {len(X_val):,} rows "
        f"({val_start} to {val_end_time})"
    )
    print(
        f"      Test:       {len(X_test):,} rows "
        f"({test_start} to {test_end_time})"
    )

    # --------------------------------------------------------
    # 4. Train candidate models
    # --------------------------------------------------------
    print("\n[4/5] Training candidate models...")

    results = {}
    candidate_models = {}

    # Persistence baseline on validation data
    print("\n      Evaluating persistence baseline...")

    persistence_val_predictions = X_val["Usage_kWh"].to_numpy()
    persistence_val_metrics = get_metrics(
        y_val,
        persistence_val_predictions,
    )

    results["Naive_Persistence"] = persistence_val_metrics

    print_metrics(
        "Naive_Persistence",
        persistence_val_metrics,
        prefix="      ",
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------
    print("\n      Training LightGBM...")
    start_time = time.time()

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    lgb_model.fit(X_train, y_train)

    lgb_predictions = lgb_model.predict(X_val)
    lgb_metrics = get_metrics(y_val, lgb_predictions)

    results["LightGBM"] = lgb_metrics
    candidate_models["LightGBM"] = lgb_model

    print(
        f"      Training time: {time.time() - start_time:.2f}s"
    )
    print_metrics("LightGBM", lgb_metrics, prefix="      ")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------
    print("\n      Training XGBoost...")
    start_time = time.time()

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    xgb_model.fit(X_train, y_train)

    xgb_predictions = xgb_model.predict(X_val)
    xgb_metrics = get_metrics(y_val, xgb_predictions)

    results["XGBoost"] = xgb_metrics
    candidate_models["XGBoost"] = xgb_model

    print(
        f"      Training time: {time.time() - start_time:.2f}s"
    )
    print_metrics("XGBoost", xgb_metrics, prefix="      ")

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------
    print("\n      Training CatBoost...")
    start_time = time.time()

    cb_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )

    cb_model.fit(X_train, y_train)

    cb_predictions = cb_model.predict(X_val)
    cb_metrics = get_metrics(y_val, cb_predictions)

    results["CatBoost"] = cb_metrics
    candidate_models["CatBoost"] = cb_model

    print(
        f"      Training time: {time.time() - start_time:.2f}s"
    )
    print_metrics("CatBoost", cb_metrics, prefix="      ")

    # --------------------------------------------------------
    # Weighted ensemble
    # --------------------------------------------------------
    print("\n      Evaluating weighted ensemble...")

    ensemble_model = WeightedEnsembleRegressor(
        models={
            "CatBoost": cb_model,
            "LightGBM": lgb_model,
            "XGBoost": xgb_model,
        },
        weights=ENSEMBLE_WEIGHTS,
    )

    ensemble_predictions = ensemble_model.predict(X_val)
    ensemble_metrics = get_metrics(
        y_val,
        ensemble_predictions,
    )

    results["Weighted_Ensemble"] = ensemble_metrics
    candidate_models["Weighted_Ensemble"] = ensemble_model

    print_metrics(
        "Weighted_Ensemble",
        ensemble_metrics,
        prefix="      ",
    )

    # --------------------------------------------------------
    # 5. Select, refit, test, and save
    # --------------------------------------------------------
    print("\n" + "=" * 85)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 85)

    print(
        f"{'Model':<24}"
        f"{'RMSE (kWh)':>16}"
        f"{'MAE (kWh)':>16}"
        f"{'R2':>12}"
    )
    print("-" * 68)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<24}"
            f"{metrics['rmse']:>16.4f}"
            f"{metrics['mae']:>16.4f}"
            f"{metrics['r2']:>12.4f}"
        )

    # Do not allow the persistence baseline to win model selection.
    ml_candidates = {
        name: metrics
        for name, metrics in results.items()
        if name != "Naive_Persistence"
    }

    best_model_name = min(
        ml_candidates,
        key=lambda name: ml_candidates[name]["rmse"],
    )

    best_validation_metrics = results[best_model_name]

    print("=" * 85)
    print(f"Selected model: {best_model_name}")
    print(
        f"Validation RMSE: "
        f"{best_validation_metrics['rmse']:.4f} kWh"
    )

    # Refit models using train + validation only.
    # The held-out test data is not used for fitting or selection.
    print("\nRefitting model components on training + validation data...")

    X_train_val = pd.concat(
        [X_train, X_val],
        axis=0,
    )

    y_train_val = pd.concat(
        [y_train, y_val],
        axis=0,
    )

    final_lgb = clone(lgb_model)
    final_lgb.fit(X_train_val, y_train_val)

    final_xgb = clone(xgb_model)
    final_xgb.fit(X_train_val, y_train_val)

    final_cb = clone(cb_model)
    final_cb.fit(X_train_val, y_train_val)

    final_models = {
        "LightGBM": final_lgb,
        "XGBoost": final_xgb,
        "CatBoost": final_cb,
        "Weighted_Ensemble": WeightedEnsembleRegressor(
            models={
                "CatBoost": final_cb,
                "LightGBM": final_lgb,
                "XGBoost": final_xgb,
            },
            weights=ENSEMBLE_WEIGHTS,
        ),
    }

    best_model = final_models[best_model_name]

    # Final held-out test evaluation
    print("\nEvaluating selected model on the held-out test set...")

    test_predictions = best_model.predict(X_test)
    test_metrics = get_metrics(y_test, test_predictions)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"Selected model: {best_model_name}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f} kWh")
    print(f"Test MAE:  {test_metrics['mae']:.4f} kWh")
    print(f"Test R2:   {test_metrics['r2']:.4f}")

    # Test-set persistence baseline
    persistence_test_predictions = X_test["Usage_kWh"].to_numpy()

    persistence_test_metrics = get_metrics(
        y_test,
        persistence_test_predictions,
    )

    print("\nPERSISTENCE BASELINE - TEST SET")
    print("=" * 60)
    print(f"Test RMSE: {persistence_test_metrics['rmse']:.4f} kWh")
    print(f"Test MAE:  {persistence_test_metrics['mae']:.4f} kWh")
    print(f"Test R2:   {persistence_test_metrics['r2']:.4f}")

    # Feature importance from final CatBoost model.
    feat_importance_df = get_feature_importance(
        final_cb,
        FEATURE_COLUMNS,
    )

    print("\nTop 15 CatBoost Features")
    print("-" * 60)

    for idx, row in feat_importance_df.head(15).iterrows():
        print(
            f"{idx + 1:2d}. "
            f"{row['feature']:<35} "
            f"Importance: {row['importance']:.4f}"
        )

    # Save selected final model.
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nSaved selected model to '{BEST_MODEL_PATH}'.")

    # Save metadata.
    metadata = {
        "best_model_name": best_model_name,
        "validation_rmse": best_validation_metrics["rmse"],
        "validation_mae": best_validation_metrics["mae"],
        "validation_r2": best_validation_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "persistence_test_rmse": persistence_test_metrics["rmse"],
        "persistence_test_mae": persistence_test_metrics["mae"],
        "persistence_test_r2": persistence_test_metrics["r2"],
        "all_validation_results": results,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "train_start": str(train_start),
        "train_end": str(train_end_time),
        "validation_start": str(val_start),
        "validation_end": str(val_end_time),
        "test_start": str(test_start),
        "test_end": str(test_end_time),
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved metadata to '{METADATA_PATH}'.")

    # Generate the report.
    generate_report(
        total_raw_rows=total_raw_rows,
        total_processed_rows=total_processed_rows,
        train_rows=len(X_train),
        val_rows=len(X_val),
        test_rows=len(X_test),
        train_range=(train_start, train_end_time),
        val_range=(val_start, val_end_time),
        test_range=(test_start, test_end_time),
        results=results,
        best_model_name=best_model_name,
        feat_importance_df=feat_importance_df,
        test_metrics=test_metrics,
        persistence_metrics=persistence_test_metrics,
    )

    print("\n" + "=" * 85)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 85)
    print(f"Selected model: {best_model_name}")
    print(f"Saved model:    {BEST_MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved report:   {REPORT_PATH}")


if __name__ == "__main__":
    train_and_evaluate()