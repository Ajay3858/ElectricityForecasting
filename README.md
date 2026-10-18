# Electricity Consumption Forecasting for a Steel Plant

A machine-learning pipeline that forecasts **electricity consumption one
15-minute interval ahead** from historical usage, electrical telemetry,
and time-based features. The pipeline compares LightGBM, XGBoost,
CatBoost, and a weighted ensemble using chronological validation and
held-out testing.

> **Latest held-out test result:** Weighted Ensemble --- RMSE **8.6425
> kWh**, MAE **4.0018 kWh**, R² **0.9220**. Test RMSE was approximately
> **31.19% lower** than the persistence baseline.

## Contents

-   [Overview](#project-overview)
-   [Dataset and Target](#dataset-and-prediction-target)
-   [Workflow](#workflow)
-   [Results](#latest-results)
-   [Feature Importance](#important-features)
-   [Project Structure](#project-structure)
-   [Installation and Usage](#installation)
-   [Tests and Data Quality](#data-quality-and-tests)
-   [Limitations](#notes-and-limitations)

## Project Overview

Industrial electricity demand varies with operating conditions and time
of day. This project predicts the next interval's electricity usage
using supervised machine learning.

The pipeline includes: - Chronological timestamp parsing and sorting. -
Feature engineering for electrical measurements, cyclical time
encodings, usage lags, and rolling statistics. - Chronological
train/validation/test splits. - LightGBM, XGBoost, CatBoost, and
weighted-ensemble comparison. - Validation-based model selection and
held-out test evaluation. - Saved model, metadata, and text reports. -
Data-quality and target-alignment checks.

## Dataset and Prediction Target

The project uses `train.csv`, with **28,031 raw rows** of steel-plant
electricity consumption and telemetry. The columns include `date`,
`Usage_kWh`, electrical telemetry fields, `NSM`, `WeekStatus`,
`Day_of_week`, `Load_Type`, and `next_interval_Usage_kWh`.

The target, `next_interval_Usage_kWh`, represents electricity usage at
the **next 15-minute interval**. When a supplied target is unavailable,
the feature pipeline can derive it by shifting `Usage_kWh` by one row.

The latest feature-engineering run retained **27,358 usable rows** and
**44 features**.

## Workflow

1.  Parse and sort timestamps chronologically.
2.  Engineer time, cyclical, electrical, lag, and rolling-window
    features.
3.  Split data by time into training, validation, and test partitions.
4.  Train LightGBM, XGBoost, CatBoost, and a weighted ensemble.
5.  Select the machine-learning candidate with the lowest validation
    RMSE.
6.  Refit component models on training plus validation data and evaluate
    on the held-out test set.
7.  Save model artifacts and reports.

### Chronological split from the latest run

  Partition        Rows Start              End
  ------------ -------- ------------------ ------------------
  Training       19,149 2018-01-08 00:15   2018-07-26 11:15
  Validation      4,103 2018-07-26 11:45   2018-09-07 05:15
  Test            4,104 2018-09-07 05:45   2018-10-19 23:30

The pipeline omits a row at partition boundaries to reduce the risk of
adjacent feature/target rows crossing between partitions.

## Models and Selection

The models compared are: - LightGBM - XGBoost - CatBoost - Weighted
Ensemble

The ensemble weights are **CatBoost 0.50**, **LightGBM 0.35**, and
**XGBoost 0.15**. The latest run selected the weighted ensemble because
it had the lowest validation RMSE.

A persistence baseline predicts next-interval usage using
current-interval usage, providing a simple comparison point.

## Latest Results

### Validation performance

  Model                           RMSE          MAE           R²
  ----------------------- ------------ ------------ ------------
  Persistence baseline         12.7861       5.4752       0.8235
  LightGBM                      8.4233       3.9329       0.9234
  XGBoost                       8.4042       3.8926       0.9237
  CatBoost                      8.3978       4.0337       0.9239
  **Weighted Ensemble**     **8.3208**   **3.9068**   **0.9252**

### Held-out test performance

  Metric     Weighted Ensemble   Persistence baseline
  -------- ------------------- ----------------------
  RMSE          **8.6425 kWh**                12.5593
  MAE           **4.0018 kWh**                 5.5686
  R²                **0.9220**                 0.8353

The weighted ensemble's test RMSE is approximately **31.19% lower** than
persistence. Test RMSE is approximately **3.87% higher** than validation
RMSE. Results reflect this historical dataset and are not a guarantee of
future performance.

## Important Features

CatBoost feature importance from the latest run:

    Rank Feature                   Importance
  ------ ----------------------- ------------
       1 `Usage_kWh`                  19.8949
       2 `cos_nsm`                    12.1806
       3 `apparent_power_kVAh`        11.4777
       4 `CO2(tCO2)`                   8.9267
       5 `q_p_ratio`                   3.2134
       6 `usage_roll_std_16`           3.1583
       7 `sin_nsm`                     3.0878
       8 `usage_roll_max_4`            2.3983
       9 `usage_diff_1`                2.1147
      10 `minute`                      2.0833
      11 `usage_roll_min_4`            2.0422
      12 `NSM`                         1.9946
      13 `usage_lag_1`                 1.9111
      14 `month`                       1.7579
      15 `usage_roll_range_4`          1.6478

Feature importance is model-specific and does not establish causality.

## Project Structure

``` text
ElectricityForecasting/
├── src/
│   ├── features.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
├── models/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   └── model_report.txt
├── reports/
│   └── data_quality_audit.txt
├── audit_data_quality.py
├── check_target_alignment.py
├── train.py
├── predict.py
├── model_report.txt
└── train.csv  # if included in the repository
```

Your local repository may contain additional files. Avoid committing
credentials, private data, virtual environments, or unnecessary
generated training logs.

## Installation

Use a supported Python environment. From the project root, create and
activate a virtual environment in Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the main dependencies:

``` powershell
pip install pandas numpy scikit-learn joblib lightgbm xgboost catboost pytest
```

If the repository has a pinned `requirements.txt`, use it instead:

``` powershell
pip install -r requirements.txt
```

Place `train.csv` at the path expected by the data-loading code. Check
`src/utils.py` if the dataset is not found.

## Train the Models

``` powershell
python train.py
```

The training pipeline performs feature engineering, chronological
splitting, candidate model training, validation selection, test
evaluation, and artifact saving.

Expected artifacts include: - `models/best_model.joblib` -
`models/model_metadata.json` - `models/model_report.txt` -
`model_report.txt`

## Generate Predictions

``` powershell
python predict.py
```

Review `predict.py` for its expected input schema, file path, and output
behavior before using it with new data.

## Data Quality and Tests

Run automated tests:

``` powershell
python -m pytest -q
```

The latest recorded test run passed **47 tests**. The repository also
includes:

``` powershell
python audit_data_quality.py
python check_target_alignment.py
```

The pipeline sorts timestamps and checks a supplied next-interval target
against the following usage observation. For a new dataset, verify
timestamp duplicates and gaps so that a following row really represents
the next 15-minute interval.

## Notes and Limitations

-   This is a **one-step-ahead, 15-minute forecast**, not a daily or
    weekly forecast.
-   The reported data is historical and from 2018; performance can
    change with different plants or operating conditions.
-   Prediction-time feature engineering must match the training feature
    schema.
-   Avoid repeated tuning against the held-out test set if you want to
    preserve it as a final evaluation.
-   Confirm the input paths and schema in `predict.py` and
    `src/utils.py` before deployment.

## Future Improvements

-   Add a pinned `requirements.txt` for reproducible setup.
-   Add prediction-versus-actual plots and residual analysis.
-   Provide a simple web interface or API for forecasts.
-   Monitor performance and retrain if operating conditions change.

------------------------------------------------------------------------

**Project focus:** industrial energy forecasting · time-series feature
engineering · gradient boosting · ensemble learning · chronological
evaluation
