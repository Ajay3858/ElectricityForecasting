# Industrial Electricity Consumption Forecasting

A production-grade machine learning project for **15-minute ahead electricity consumption forecasting** (`next_interval_Usage_kWh`). Built using **CatBoost**, **LightGBM**, **XGBoost**, and an optimal **Weighted Ensemble** with spike-resilient feature engineering, past-only causal rolling aggregates, and chronological time-series validation.

---

## 📌 Project Overview

- **Dataset**: Steel Industry Energy Consumption time-series (`train.csv`).
- **Resolution**: 15-minute continuous sampling intervals (96 intervals per day).
- **Temporal Range**: January 1, 2018 (`00:15`) to October 19, 2018 (`23:45`) — 292 days, 28,031 contiguous records.
- **Target**: `next_interval_Usage_kWh` (Active power consumption in kWh during interval $t+1$).
- **Active Production Model**: **Weighted Ensemble Regressor** (50% CatBoost + 35% LightGBM + 15% XGBoost).
  - **Validation RMSE**: **`8.2281` kWh**
  - **Validation MAE**: **`3.8281` kWh**
  - **Validation $R^2$**: **`0.9244`**

---

## 📁 Repository Structure

```text
ElectricityForecasting/
├── train.csv                               # Raw time-series dataset (28,031 intervals)
├── requirements.txt                        # Core Python dependencies
├── train.py                                # Production model training & selection pipeline
├── train_regime_experiment.py              # Experiment 1: Two-stage regime-switching pipeline
├── train_residual_experiment.py            # Experiment 2: Conservative residual-correction pipeline
├── predict.py                              # Standalone production inference pipeline
├── prediction.csv                          # Latest prediction output
├── model_report.txt                        # Production model evaluation report
├── automation/                             # Automated local forecasting subsystem
│   ├── config.json                         # Telemetry source, output, and history config
│   ├── telemetry.py                        # Pluggable TelemetrySource abstraction (CSV, API, MQTT)
│   ├── scheduler.py                        # Automated scheduler running on clock quarter-hours
│   ├── prediction_history.csv              # Cumulative history of predictions with duplicate protection
│   └── automation.log                      # Timestamped execution and error logging
├── reports/
│   ├── regime_model_report.txt             # Experiment 1 evaluation report
│   └── residual_correction_report.txt      # Experiment 2 evaluation report
├── models/
│   ├── best_model.joblib                   # Active production model (Weighted Ensemble)
│   ├── model_metadata.json                 # Production metadata & feature specifications
│   └── experiments/                        # Isolated experimental model artifacts
└── src/
    ├── __init__.py
    ├── utils.py                            # Seed management, datetime repair, and evaluation metrics
    ├── features.py                         # Past-only causal lag, rolling stats, dynamics, and cyclical features
    ├── models.py                           # Weighted ensemble regressor wrapper class
    └── regime_model.py                     # Two-stage regime forecaster class
```

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
```

---

## 📡 Telemetry Source Ingestion Architecture

Data ingestion is decoupled via the [`TelemetrySource`](file:///c:/Users/AJAY%20R/ElectricityForecasting/automation/telemetry.py) abstraction in `automation/telemetry.py`:

```text
TelemetrySource (CSV, API, MQTT, Database, IoT)
       ↓
historical / current telemetry (min 673 intervals)
       ↓
predict.py (causal features strictly up to t)
       ↓
models/best_model.joblib (Fixed Production Model)
       ↓
next 15-minute prediction (Usage t+1)
       ↓
prediction.csv & automation/prediction_history.csv
```

Configuration in [`automation/config.json`](file:///c:/Users/AJAY%20R/ElectricityForecasting/automation/config.json):
```json
{
    "telemetry": {
        "source": "csv",
        "file": "train.csv"
    },
    "output_file": "prediction.csv",
    "history_file": "automation/prediction_history.csv",
    "interval_minutes": 15
}
```
This architecture makes it seamless to swap `CSVTelemetrySource` with an `APITelemetrySource` or `MQTTTelemetrySource` without touching any ML model or feature code.

---

## 🚀 Execution Guide

### 1. Run a Single Forecast Manually
```bash
python predict.py --input train.csv
```

### 2. Test Automated Forecasting Once (Test Mode)
```bash
python automation/scheduler.py --once
```

### 3. Run Continuous 15-Minute Automated Forecasting
```bash
python automation/scheduler.py
```
*(Wakes up automatically at `:00`, `:15`, `:30`, `:45` past every hour, pulls from the telemetry source, runs the forecast, updates the history archive, logs progress, and sleeps until the next quarter-hour).*
