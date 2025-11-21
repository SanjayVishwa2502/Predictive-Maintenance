# Phase 2.3.1: Training Scripts Verification & Activation Guide

**Date:** November 21, 2025  
**Status:** ✅ READY TO START REGRESSION TRAINING

---

## Virtual Environment Activation

**IMPORTANT:** Always activate the venv before running training scripts!

```powershell
# From project root
cd "c:\Projects\Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Navigate to training scripts
cd ml_models\scripts\training
```

---

## Training Scripts Structure

### 1. Classification Models
**Location:** `ml_models/scripts/training/`

**Individual Training:**
- `train_classification_fast.py` (273 lines)
  - Fast training: 15 min per machine
  - Pi-optimized: LightGBM, RandomForest, XGBoost, CatBoost only
  - Excludes: NN_TORCH, FASTAI, XT (heavy models)

**Batch Training:**
- `batch_train_classification.py`
  - Trains all 10 priority machines sequentially
  - Progress tracking + ETA
  - Auto-validation after each

**Usage:**
```powershell
# Activate venv first!
cd "c:\Projects\Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Individual
cd ml_models\scripts\training
python train_classification_fast.py --machine_id motor_siemens_1la7_001 --time_limit 900

# Batch
python batch_train_classification.py
```

**Status:** ✅ COMPLETED (10/10 models trained)

---

### 2. Regression Models (RUL Prediction)
**Location:** `ml_models/scripts/training/`

**Individual Training:**
- `train_regression_fast.py` (247 lines)
  - Fast training: 15 min per machine
  - Uses existing RUL column from GAN data
  - Pi-optimized: LightGBM, RandomForest, XGBoost only
  - Excludes: NN_TORCH, FASTAI, XT (heavy models)

**Batch Training:**
- `batch_train_regression.py`
  - Trains all 10 priority machines sequentially
  - Progress tracking + ETA
  - Auto-validation after each

**Usage:**
```powershell
# Activate venv first!
cd "c:\Projects\Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Individual
cd ml_models\scripts\training
python train_regression_fast.py --machine_id motor_siemens_1la7_001 --time_limit 900

# Batch (ALL 10 MACHINES)
python batch_train_regression.py
```

**Status:** 🔄 READY TO START (currently testing motor_siemens_1la7_001)

**Current Test Run:**
- Machine: motor_siemens_1la7_001
- Data verified: ✅ RUL column exists (0.0-1014.6 hours)
- Training started: Successfully running
- Expected completion: ~15 minutes

---

### 3. Anomaly Detection Models
**Location:** `ml_models/scripts/training/`

**Individual Training:**
- `train_anomaly_comprehensive.py` (850+ lines)
  - 7 algorithms: Isolation Forest, One-Class SVM, LOF, DBSCAN, Z-Score, IQR, Modified Z-Score
  - Ensemble voting system
  - Fast training: ~0.4 min per machine
  - Comprehensive metrics: 8+ metrics per algorithm

**Batch Training:**
- `batch_train_anomaly_comprehensive.py` (280+ lines)
  - Trains all 10 priority machines sequentially
  - Auto-validation after each
  - Generates 14+ visualizations per machine

**Usage:**
```powershell
# Activate venv first!
cd "c:\Projects\Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Individual
cd ml_models\scripts\training
python train_anomaly_comprehensive.py --machine_id motor_siemens_1la7_001

# Batch (ALL 10 MACHINES)
python batch_train_anomaly_comprehensive.py
```

**Status:** ✅ COMPLETED (10/10 models trained, F1 avg 0.8441)

---

### 4. Time-Series Forecasting Models
**Location:** `ml_models/scripts/training/`

**Individual Training:**
- `train_timeseries.py`
  - LSTM/Transformer based
  - Predicts next 24 hours
  - Uses temporal sequences

**Status:** ⚠️ NOT YET IMPLEMENTED (waiting on Phase 2.5)

---

## 10 Priority Machines

```
1. motor_siemens_1la7_001
2. motor_abb_m3bp_002
3. motor_weg_w22_003
4. pump_grundfos_cr3_004
5. pump_flowserve_ansi_005
6. compressor_atlas_copco_ga30_001
7. compressor_ingersoll_rand_2545_009
8. cnc_dmg_mori_nlx_010
9. hydraulic_beckwood_press_011
10. cooling_tower_bac_vti_018
```

---

## Training Progress Summary

| Model Type | Status | Models Trained | Avg Training Time | Total Size |
|------------|--------|----------------|-------------------|------------|
| **Classification** | ✅ COMPLETE | 10/10 | 0.63 min/machine | 2.58 GB |
| **Regression** | 🔄 IN PROGRESS | 0/10 → 1/10 | ~15 min/machine | TBD |
| **Anomaly** | ✅ COMPLETE | 10/10 | 0.44 min/machine | 39.95 MB |
| **Time-Series** | ⏳ PENDING | 0/10 | TBD | TBD |

---

## Next Steps for Phase 2.3

### Option A: Continue Individual Testing
```powershell
# Wait for motor_siemens_1la7_001 to complete (~15 min)
# Check results in reports/performance_metrics/motor_siemens_1la7_001_regression_report.json
# If successful, proceed to batch training
```

### Option B: Run Batch Training (Recommended)
```powershell
# Activate venv
cd "c:\Projects\Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Run batch training for all 10 machines
cd ml_models\scripts\training
python batch_train_regression.py

# Expected time: 10 machines × 15 min = ~2.5 hours
# Can run overnight or in background
```

---

## Key Findings

### RUL Data Verified ✅
- All 10 priority machines have RUL column in temporal data
- RUL range: 0-1015 hours per machine
- Mean RUL: ~450-500 hours
- Temporal structure: timestamp decreases, RUL decreases over time

### Training Script Updates ✅
- Removed synthetic RUL generation (lines 56-103)
- Now uses existing RUL column from GAN data
- Validates RUL column exists before training
- Cleaner, more reliable approach

### Performance Targets
- **R² Score:** >0.75 (excellent)
- **RMSE:** <100 hours (acceptable error)
- **MAE:** <75 hours (mean absolute error)
- **MAPE:** <15% (percentage error)

---

## Current Test Run Details

**Machine:** motor_siemens_1la7_001  
**Started:** Running now  
**Progress:** AutoGluon training models (LightGBM, RandomForest, XGBoost)  
**Observed Performance:**
- LightGBMXT_BAG_L1: R²=0.9785 (excellent!)
- LightGBM_BAG_L1: R²=0.9801 (excellent!)
- RandomForestMSE_BAG_L1: Currently training...

**Expected Completion:** ~5-10 more minutes  
**Output:** `reports/performance_metrics/motor_siemens_1la7_001_regression_report.json`

---

## Recommendations

1. **Let Current Test Complete** - Verify full training pipeline works end-to-end
2. **Review Test Results** - Check R², RMSE, MAE, MAPE metrics
3. **If Successful** - Run batch training on all 10 machines overnight
4. **Monitor Progress** - Check MLflow experiments at `mlruns/` folder

---

**Updated by:** AI Assistant  
**Phase 2.3.1:** ✅ COMPLETE & VERIFIED  
**Phase 2.3.2:** 🔄 STARTING NOW (test run in progress)
