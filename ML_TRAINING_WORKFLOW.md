# ML Model Training Workflow
**Complete Guide: From Synthetic Data → Trained Models**

---

## Overview

This document describes how to train ML models for newly created machines after synthetic data generation via the GAN workflow.

### Complete End-to-End Flow

```
1. GAN Wizard (New Machine Profile)
   ↓
2. Generate Seed Data (Physics-based simulation)
   ↓
3. Train TVAE Model (Synthetic data generator)
   ↓
4. Generate Synthetic Datasets (train/val/test splits)
   ↓
5. **ML MODEL TRAINING** ← This Document
   ↓
6. Trained Models Ready for Predictions
```

---

## What Gets Trained

For each machine, **4 different model types** are trained:

| Model Type | Purpose | Target Variable | Example Use Case |
|------------|---------|-----------------|------------------|
| **Classification** | Health state prediction | `failure_status` (binary: normal/failure) | "Is this machine about to fail?" |
| **Regression (RUL)** | Remaining Useful Life | `rul` (hours until failure) | "How many hours until maintenance needed?" |
| **Anomaly Detection** | Unusual behavior detection | Unsupervised | "Is this sensor pattern abnormal?" |
| **Time-Series Forecast** | Future sensor predictions | Sensor values (24h ahead) | "What will temperature be tomorrow?" |

---

## Prerequisites

Before training models for a machine:

✅ **Synthetic data must exist:**
- Location: `GAN/data/synthetic/<machine_id>/`
- Required files:
  - `train.parquet` (70% of data)
  - `val.parquet` (15% of data)
  - `test.parquet` (15% of data)

✅ **Machine metadata must exist:**
- Location: `GAN/metadata/<machine_id>_metadata.json`
- Contains sensor definitions and column mappings

✅ **System requirements:**
- Python environment with AutoGluon installed
- At least 4GB RAM per training job
- ~15-30 minutes per model type
- ~250-500 MB disk space per trained model

---

## Training Scripts (Existing Implementation)

Training scripts are located in: `ml_models/scripts/training/`

### 1. Classification Training

**Script:** `train_classification_fast.py`

**What it does:**
- Creates realistic failure labels from sensor thresholds
- Trains binary classifier (normal vs. failure)
- Uses AutoGluon with Pi-optimized presets
- Outputs: Trained model saved to `ml_models/models/classification/<machine_id>/`

**Usage:**
```bash
python ml_models/scripts/training/train_classification_fast.py --machine_id motor_siemens_1la7_001
```

**Training Parameters:**
- Time limit: 15 minutes (900 seconds)
- Preset: `medium_quality_faster_train`
- Models: LightGBM, RandomForest, XGBoost, CatBoost
- Excluded: Neural networks (too heavy for Raspberry Pi)
- Target model size: ~250 MB

**Output Metrics:**
- Accuracy
- Precision/Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### 2. RUL Regression Training

**Script:** `train_regression_fast.py`

**What it does:**
- Trains regressor to predict remaining useful life (hours)
- Uses AutoGluon with regression-optimized presets
- Validates against holdout test set
- Outputs: Trained model saved to `ml_models/models/regression/<machine_id>/`

**Usage:**
```bash
python ml_models/scripts/training/train_regression_fast.py --machine_id motor_siemens_1la7_001
```

**Training Parameters:**
- Time limit: 15 minutes (900 seconds)
- Preset: `medium_quality_faster_train`
- Models: LightGBM, RandomForest, XGBoost
- Target model size: ~250 MB

**Output Metrics:**
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### 3. Anomaly Detection Training

**Script:** `train_anomaly_comprehensive.py`

**What it does:**
- Trains ensemble of anomaly detectors:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor
  - Z-Score based detection
- Unsupervised learning (no labels required)
- Outputs: Trained detectors saved to `ml_models/models/anomaly/<machine_id>/`

**Usage:**
```bash
python ml_models/scripts/training/train_anomaly_comprehensive.py --machine_id motor_siemens_1la7_001
```

**Training Parameters:**
- Contamination rate: 5-10% (expected anomaly percentage)
- Ensemble voting: Majority consensus across detectors
- Target model size: ~50 MB (lightweight)

**Output Metrics:**
- Precision/Recall on synthetic anomalies
- False Positive Rate
- Anomaly detection accuracy

### 4. Time-Series Forecasting Training

**Script:** `train_timeseries.py`

**What it does:**
- Trains Prophet models for each critical sensor
- 24-hour ahead forecasting
- Seasonal/trend decomposition
- Outputs: Trained models saved to `ml_models/models/timeseries/<machine_id>/`

**Usage:**
```bash
python ml_models/scripts/training/train_timeseries.py --machine_id motor_siemens_1la7_001
```

**Training Parameters:**
- Forecast horizon: 24 hours
- Uncertainty intervals: 80% and 95%
- Seasonality: Daily, weekly patterns
- Target model size: ~100 MB

**Output Metrics:**
- MAPE (Mean Absolute Percentage Error)
- Coverage of prediction intervals
- R² Score for trend accuracy

---

## Batch Training (Train All 4 Models at Once)

For convenience, batch scripts train all 4 model types sequentially:

**Location:** `ml_models/scripts/training/batch_train_*.py`

**Example: Train all models for one machine**
```bash
# Navigate to training scripts
cd ml_models/scripts/training

# Train all 4 model types for motor_siemens_1la7_001
python batch_train_classification.py --machine_id motor_siemens_1la7_001
python batch_train_regression.py --machine_id motor_siemens_1la7_001
python batch_train_anomaly_comprehensive.py --machine_id motor_siemens_1la7_001
python batch_train_timeseries.py --machine_id motor_siemens_1la7_001
```

**Total Time:** ~60 minutes (4 models × 15 min each)

---

## Model Storage Structure

After training, models are stored in:

```
ml_models/models/
├── classification/
│   └── motor_siemens_1la7_001/
│       ├── predictor.pkl          # AutoGluon predictor
│       ├── learner.pkl             # Model weights
│       ├── models/                 # Individual model files
│       ├── metadata.json           # Training config
│       └── version.txt             # AutoGluon version
├── regression/
│   └── motor_siemens_1la7_001/
│       └── (same structure)
├── anomaly/
│   └── motor_siemens_1la7_001/
│       ├── all_detectors.pkl       # Ensemble detectors
│       ├── preprocessing.pkl       # Scaler pipeline
│       └── features.json           # Feature names
└── timeseries/
    └── motor_siemens_1la7_001/
        ├── prophet_models/         # Per-sensor Prophet models
        └── metadata.json
```

---

## Integration with Dashboard

### Current State (Manual Training)

Currently, training is done via command-line scripts manually by developers.

### Proposed Integration (This Workflow)

Add **"Model Training"** to the ML Dashboard navigation:

```
ML Dashboard Navigation:
├─ Predictions (existing)
├─ New Machine Wizard (GAN - existing)
├─ **Model Training** (NEW - this feature)
│  └─ Train Classification
│  └─ Train Regression (RUL)
│  └─ Train Anomaly Detection
│  └─ Train Time-Series Forecast
│  └─ Train All Models (batch)
├─ Prediction History
├─ Downloads
└─ Settings
```

### Backend API Design (New Endpoints)

Proposed endpoints for `/api/ml/train/*`:

```
POST /api/ml/train/classification
  Body: { machine_id, time_limit (optional) }
  Response: { job_id, status, estimated_time }

POST /api/ml/train/regression
  Body: { machine_id, time_limit (optional) }
  Response: { job_id, status, estimated_time }

POST /api/ml/train/anomaly
  Body: { machine_id, contamination_rate (optional) }
  Response: { job_id, status, estimated_time }

POST /api/ml/train/timeseries
  Body: { machine_id, forecast_hours (optional) }
  Response: { job_id, status, estimated_time }

POST /api/ml/train/batch
  Body: { machine_id, model_types: ['classification', 'regression', 'anomaly', 'timeseries'] }
  Response: { job_id, status, estimated_time }

GET /api/ml/train/status/{job_id}
  Response: { 
    status: 'pending' | 'running' | 'completed' | 'failed',
    progress: 0-100,
    current_step: string,
    metrics: {...},
    model_path: string (if completed)
  }
```

### Frontend UI Design

**Training Workflow Page:**

1. **Machine Selector** (dropdown of machines with synthetic data)
2. **Model Type Selection**:
   - [ ] Classification (Failure Prediction)
   - [ ] Regression (RUL Estimation)
   - [ ] Anomaly Detection
   - [ ] Time-Series Forecast
   - [x] **Train All Models** (recommended)

3. **Training Configuration**:
   - Time Limit: [15 min] (default)
   - Quality Preset: [Medium (Pi-compatible)] (dropdown)
   - Advanced Options (collapsible):
     - Excluded models
     - Ensemble settings
     - Cross-validation folds

4. **Action Button**: `[Start Training]`

5. **Progress Monitor**:
   - Current step: "Loading data... ✓"
   - Progress bar: 45%
   - Estimated time remaining: 8 minutes
   - Live training logs (scrollable terminal-style output)

6. **Results Panel** (after completion):
   - Model Performance Metrics
   - Leaderboard (best models ranked)
   - Model Size: 247 MB
   - Download Model button
   - Test Prediction button (redirects to Predictions page)

---

## Training Best Practices

### 1. Data Quality Checks

Before training, verify:
- Synthetic data has realistic distributions
- No missing values in critical sensors
- Class balance for classification (avoid 99% normal)
- RUL values are sensible (0-500 hours range)

### 2. Time Budgets

Recommended time limits:
- **Fast (Pi-compatible):** 15 minutes
- **Standard:** 30 minutes
- **High Quality:** 60 minutes

### 3. Model Selection

For **Raspberry Pi deployment:**
- ✅ Use: LightGBM, RandomForest, XGBoost
- ❌ Avoid: Neural Networks, FASTAI, KNN

For **Server deployment:**
- ✅ Use: All models including neural networks
- Increase time limit to 60 minutes

### 4. Validation Strategy

- Use holdout test set (15% of data)
- Never train on test data (data leakage!)
- Monitor for overfitting (train vs. test metrics)

### 5. MLflow Tracking

All training runs are logged to MLflow:
- Location: `ml_models/scripts/training/mlruns/`
- View UI: `mlflow ui --port 5000`
- Metrics tracked: Accuracy, F1, RMSE, training time, model size

---

## Troubleshooting

### Issue: "RUL column not found in data"

**Cause:** Synthetic data doesn't have `rul` column

**Solution:**
1. Check `GAN/metadata/<machine_id>_metadata.json`
2. Ensure RUL configuration was set during profile creation
3. Regenerate synthetic data with RUL enabled

### Issue: "Model training takes too long (>30 min)"

**Cause:** Dataset too large or preset too slow

**Solution:**
- Reduce dataset size (use 50K samples max)
- Use `medium_quality_faster_train` preset
- Exclude neural network models
- Reduce `num_bag_folds` to 3

### Issue: "Model size >1GB (too large for Pi)"

**Cause:** Ensemble stacking enabled

**Solution:**
- Set `num_stack_levels=0` (no stacking)
- Use `excluded_model_types=['NN_TORCH', 'FASTAI']`
- Reduce `num_bag_folds` to 3

### Issue: "Low F1 score (<0.5) on test set"

**Cause:** Class imbalance or data leakage

**Solution:**
1. Check class distribution (aim for 10-20% failure rate)
2. Use realistic failure thresholds (not too extreme)
3. Add label noise to prevent overfitting
4. Validate label generation logic

---

## Next Steps After Training

Once models are trained:

1. ✅ **Verify models exist** in `ml_models/models/<type>/<machine_id>/`
2. ✅ **Check model size** (<500 MB for Pi deployment)
3. ✅ **Test inference** using prediction scripts in `ml_models/scripts/inference/`
4. ✅ **Deploy to dashboard** (models auto-detected by MLManager)
5. ✅ **Run predictions** via ML Dashboard → Predictions page

---

## Summary

The ML Training Workflow bridges the gap between synthetic data generation (GAN) and real-time predictions:

```
GAN Wizard → Synthetic Data → ML Training → Trained Models → Predictions Dashboard
```

**Key Points:**
- 4 model types per machine (classification, regression, anomaly, timeseries)
- 15-30 minutes per model (Pi-optimized)
- AutoGluon handles hyperparameter tuning automatically
- Models stored in `ml_models/models/<type>/<machine_id>/`
- Integrated with MLflow for experiment tracking
- Dashboard integration via new `/api/ml/train/*` endpoints (to be implemented)

---

**Document Version:** 1.0  
**Last Updated:** December 19, 2025  
**Status:** ✅ Training scripts exist | ⏳ Dashboard UI pending implementation
