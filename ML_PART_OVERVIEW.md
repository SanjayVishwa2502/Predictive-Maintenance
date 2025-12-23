# ML Part Overview (Project Reality Check)

This document describes how the **ML (Predictions) part** of this repo works end-to-end, based on the code and model artifacts currently present in the workspace.

## 1) Big Picture

At runtime, the ML dashboard is a **single-machine** monitoring and prediction UI.

- **Frontend UI (React)** polls machine status and can trigger predictions.
- **Backend API (FastAPI)** exposes `/api/ml/*` endpoints.
- **MLManager** is the backend service-layer singleton that:
  - Loads machine metadata (from `GAN/metadata/`)
  - Discovers available trained models (from `ml_models/models/`)
  - Runs predictions via `IntegratedPredictionSystem` (LLM + model inference)

## 2) Key Runtime Data Sources

### 2.1 Machine metadata (used for machine list + sensor inventory)

- Folder: `GAN/metadata/`
- Files consumed by ML: `*_metadata.json`

The ML backend uses these metadata files to build:
- `machine_id`
- `display_name` (derived from `machine_id`)
- `category`, `manufacturer`, `model` (parsed heuristically from `machine_id`)
- `sensor_count` and sensor list (derived from metadata columns)

Important: that folder also contains GAN upload staging artifacts like `*_profile_temp.json`. Those are **not real machines** and should not appear in the ML machine selector.

### 2.2 Synthetic datasets (used for training + as inference fallback)

- Folder: `GAN/data/synthetic/<machine_id>/`
- Split files: `train.parquet`, `val.parquet`, `test.parquet`

The integrated prediction layer can load a sample row from `train.parquet` when no sensor_data is provided (mostly useful for offline testing).

## 3) Model Artifacts on Disk (`ml_models/models/`)

### 3.1 Observed structure

`ml_models/models/` contains separate subfolders by model family:
- `classification/`
- `regression/`
- `anomaly/`
- `timeseries/`

In this workspace, the trained per-machine folders exist at least for:
- `ml_models/models/classification/<machine_id>/...`
- `ml_models/models/regression/<machine_id>/...`

Example trained machine folders (both classification + regression):
- `motor_siemens_1la7_001`
- `motor_abb_m3bp_002`
- `pump_grundfos_cr3_004`
- `pump_flowserve_ansi_005`
- `compressor_atlas_copco_ga30_001`
- `compressor_ingersoll_rand_2545_009`
- `cooling_tower_bac_vti_018`
- `hydraulic_beckwood_press_011`
- `cnc_dmg_mori_nlx_010`
- `motor_weg_w22_003`

There are also non-machine-specific folders like:
- `generic_all_machines/`
- `pooled_test_3_machines/`

### 3.2 AutoGluon model format

Inside an AutoGluon model directory (classification or regression), you’ll typically see:
- `predictor.pkl`
- `learner.pkl`
- `models/`
- `utils/`
- `metadata.json`
- `version.txt`

The `metadata.json` shows the training environment and confirms AutoGluon is used (example: `autogluon` version `1.4.0`).

## 4) Backend: ML API (`/api/ml/*`)

### 4.1 Where the endpoints live

- Routes: [frontend/server/api/routes/ml.py](frontend/server/api/routes/ml.py)
- Schemas: [frontend/server/api/models/ml.py](frontend/server/api/models/ml.py)

### 4.2 Core endpoints and what they do

- `GET /api/ml/machines`
  - Purpose: populate the dashboard machine selector
  - Backend source: `ml_manager.get_machines()`

- `GET /api/ml/machines/{machine_id}/status`
  - Purpose: provide latest sensor readings for the real-time dashboard
  - Current implementation reality: returns a basic stub status object.

- `POST /api/ml/predict/classification`
  - Purpose: run a health/failure-type classifier

- `POST /api/ml/predict/rul`
  - Purpose: run an RUL (Remaining Useful Life) regressor

- `GET /api/ml/machines/{machine_id}/history`
  - Purpose: prediction history
  - Current implementation reality: returns an empty list (stub).

- `GET /api/ml/health`
  - Purpose: report service readiness + model counts + GPU/LLM status

## 5) Backend: Service Layer (`MLManager`)

- Implementation: [frontend/server/services/ml_manager.py](frontend/server/services/ml_manager.py)

### 5.1 What MLManager does

On backend start, the singleton `ml_manager`:

1) Sets key paths:
- Models dir: `ml_models/models/`
- Metadata dir: `GAN/metadata/`

2) Loads the integrated prediction system:
- `LLM/api/ml_integration.py:IntegratedPredictionSystem`

3) Loads machine metadata:
- Scans `GAN/metadata/*_metadata.json`
- Extracts a sensor list from `metadata['columns']`
- Excludes typical non-sensor fields (e.g. `timestamp`, `rul`, ids, labels)

4) Computes model availability flags:
- Classification model exists if `ml_models/models/classification/<machine_id>` exists
- Regression model exists if `ml_models/models/regression/<machine_id>` exists

### 5.2 Prediction execution path

`MLManager.predict_classification()` and `MLManager.predict_rul()` delegate to:

- `IntegratedPredictionSystem.predict_with_explanation(machine_id, sensor_data, model_type=...)`

That call:
- Runs the model inference (AutoGluon predictors)
- Then runs the LLM-based explainer (RAG + LLM) to produce a human-oriented explanation

## 6) Integrated Prediction System (Models + LLM)

- Implementation: [LLM/api/ml_integration.py](LLM/api/ml_integration.py)

### 6.1 How models are loaded

The integrated system lazy-loads models into an in-memory cache:
- Classification: `ClassificationPredictor(machine_id)`
- Regression (RUL): `RULPredictor(machine_id)`
- Anomaly: `AnomalyPredictor(machine_id)`
- Timeseries: `TimeSeriesPredictor(machine_id)`

These predictors are defined in:
- [ml_models/scripts/inference/predict_classification.py](ml_models/scripts/inference/predict_classification.py)
- [ml_models/scripts/inference/predict_rul.py](ml_models/scripts/inference/predict_rul.py)
- [ml_models/scripts/inference/predict_anomaly.py](ml_models/scripts/inference/predict_anomaly.py)
- [ml_models/scripts/inference/predict_timeseries.py](ml_models/scripts/inference/predict_timeseries.py)

### 6.2 Input feature handling

The predictors load the AutoGluon model with `TabularPredictor.load(model_path)` and obtain the expected feature list:

- `feature_names = predictor.feature_metadata.get_features()`

When predicting:
- Missing features are filled with `0.0`
- Extra fields are ignored by selecting only the required columns

This means your inference layer is robust to:
- Missing sensors
- Extra columns (like `timestamp`)

…but prediction quality depends on the correctness/availability of the real sensor values.

## 7) Frontend: ML Dashboard Wiring

- Main page: [frontend/client/src/pages/MLDashboardPage.tsx](frontend/client/src/pages/MLDashboardPage.tsx)

### 7.1 The frontend calls

`MLDashboardPage` uses these endpoints:
- Machines list: `GET /api/ml/machines`
- Machine status polling (30s): `GET /api/ml/machines/{id}/status`
- Classification prediction: `POST /api/ml/predict/classification`
- RUL prediction (available, but the UI mainly triggers classification right now): `POST /api/ml/predict/rul`

If the backend calls fail, the frontend has fallbacks:
- machine list fallback: `getMockMachines()`
- prediction fallback: `generateMockPrediction(...)`

### 7.2 Real-Time Sensor Monitoring in UI

The Real-Time Sensor Monitoring section is rendered by:
- [frontend/client/src/modules/ml/components/SensorDashboard.tsx](frontend/client/src/modules/ml/components/SensorDashboard.tsx)

It displays `latest_sensors` returned by the backend status endpoint.

## 8) Current “Reality vs Intended Design” Notes

Based on the current code:

- **Classification + RUL**: intended to be “real” (AutoGluon inference + LLM explanation) via `IntegratedPredictionSystem`.
- **Anomaly + Timeseries**: described as not fully implemented / may be mock depending on model availability.
- **Machine Status (`/machines/{id}/status`)**: currently returns a stub payload; `latest_sensors` is empty in the backend route.
- **Prediction History (`/machines/{id}/history`)**: currently returns an empty list (stub).

So the ML part’s most complete, end-to-end flow today is:
- machine list discovery (metadata + model folder existence)
- prediction inference (classification/regression) when valid `sensor_data` is provided
- LLM explanation generation

## 9) Where to look for deeper model/training details

The most detailed training design documentation is:
- [ml_models/PHASE_2_ML_DETAILED_APPROACH.md](ml_models/PHASE_2_ML_DETAILED_APPROACH.md)

This covers:
- why per-machine models were chosen
- training scripts + features
- expected folder layout
- evaluation/metrics approach

---

If you want, I can also write a second markdown doc that focuses only on the **exact trained machines and their model artifact contents** (per folder inventory, model versions, and which predictors are available per machine).
