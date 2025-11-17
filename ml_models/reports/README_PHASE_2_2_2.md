# Phase 2.2.2: Per-Machine Classification Training

## Overview
Train dedicated classification models for 10 priority machines (one model per machine).

**Date:** November 17, 2025  
**Status:** Ready to Start  
**Approach:** Per-machine models (not generic)  
**Total Models:** 10 classification models

---

## Priority Machines (10 Total)

1. `motor_siemens_1la7_001` - High-priority motor
2. `motor_abb_m3bp_002` - High-priority motor
3. `motor_weg_w22_003` - High-priority motor
4. `pump_grundfos_cr3_004` - Critical pump
5. `pump_flowserve_ansi_005` - Critical pump
6. `compressor_atlas_copco_ga30_001` - Critical compressor
7. `compressor_ingersoll_rand_2545_009` - Critical compressor
8. `cnc_dmg_mori_nlx_010` - High-value CNC
9. `hydraulic_beckwood_press_011` - Critical hydraulic
10. `cooling_tower_bac_vti_018` - Facility-critical

---

## Prerequisites

✅ Phase 1 Complete (Synthetic data generated)  
✅ Virtual environment activated  
✅ AutoGluon installed  
✅ 21 machines with synthetic data in `../GAN/data/synthetic/`

---

## Step-by-Step Instructions

### Step 1: Validate Setup

```powershell
# Navigate to ml_models folder
cd ml_models

# Run validation script
python scripts/validate_setup_phase_2_2_2.py
```

**Expected Output:**
```
✅ ALL VALIDATION CHECKS PASSED
Ready to start training!
```

If validation fails, fix reported issues before proceeding.

---

### Step 2: Training Options

You have **two options** for training:

#### Option A: Single Machine Training (Recommended for Testing)

Train one machine at a time to verify everything works:

```powershell
# Train first machine
python scripts/train_classification_per_machine.py --machine_id motor_siemens_1la7_001

# Train second machine
python scripts/train_classification_per_machine.py --machine_id motor_abb_m3bp_002

# ... and so on
```

**When to use:**
- Testing setup before batch training
- Training specific machines only
- Debugging issues with individual machines

**Parameters:**
- `--machine_id`: Machine identifier (required)
- `--time_limit`: Training time in seconds (default: 3600 = 1 hour)
- `--presets`: AutoGluon presets (default: 'best_quality')

---

#### Option B: Batch Training (Recommended for Production)

Train all 10 machines automatically in sequence:

```powershell
# Train all 10 priority machines
python scripts/batch_train_classification.py --machines_file config/priority_10_machines.txt
```

**When to use:**
- Training all machines at once
- Unattended overnight training
- Production pipeline

**Parameters:**
- `--machines_file`: File with machine IDs (default: config/priority_10_machines.txt)
- `--time_limit`: Training time per machine in seconds (default: 3600)
- `--presets`: AutoGluon presets (default: 'best_quality')

---

### Step 3: Monitor Training

**Single Machine Training:**
- Watch terminal output for progress
- Training takes ~15-60 minutes per machine
- Look for "✅ Training completed successfully!"

**Batch Training:**
- Progress shown after each machine
- Total time: ~2.5-10 hours (10 machines)
- Check `reports/batch_training_classification_10_machines.json` for status

**What to Watch:**
```
Training time: X.XX minutes
Metrics:
  Accuracy:  0.XXXX
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1 Score:  0.XXXX  ← Should be >0.85
  ROC-AUC:   0.XXXX
```

---

### Step 4: Verify Results

After training completes, check:

#### 4.1 Model Files
```powershell
# Check if models were created
ls models/classification/

# Expected output:
# motor_siemens_1la7_001/
# motor_abb_m3bp_002/
# ... (10 directories total)
```

#### 4.2 Performance Reports
```powershell
# Check performance reports
ls reports/performance_metrics/*classification_report.json

# View a specific report
cat reports/performance_metrics/motor_siemens_1la7_001_classification_report.json | ConvertFrom-Json | ConvertTo-Json
```

#### 4.3 Batch Report (if used batch training)
```powershell
# View batch training summary
cat reports/batch_training_classification_10_machines.json | ConvertFrom-Json | ConvertTo-Json
```

---

## Expected Results

### Per Machine:
- **F1 Score:** >0.85 (target: >0.85)
- **Accuracy:** >0.90
- **Training Time:** 15-60 minutes per machine
- **Model Size:** ~50 MB per model

### All 10 Machines:
- **Total Models:** 10
- **Total Storage:** ~500 MB
- **Total Training Time:** ~2.5-10 hours (sequential)
- **Success Rate:** 100% (all 10 models trained)

---

## Troubleshooting

### Issue: Validation Failed - Data Not Found
**Solution:**
```powershell
# Check if GAN data exists
cd ../GAN
ls data/synthetic/motor_siemens_1la7_001/

# If missing, regenerate synthetic data (Phase 1)
```

### Issue: Out of Memory
**Solution:**
```powershell
# Reduce time_limit to use fewer models
python scripts/train_classification_per_machine.py --machine_id motor_siemens_1la7_001 --time_limit 1800

# Or use lower quality preset
python scripts/train_classification_per_machine.py --machine_id motor_siemens_1la7_001 --presets good_quality
```

### Issue: Training Too Slow
**Solution:**
```powershell
# Use faster preset
python scripts/batch_train_classification.py --presets high_quality --time_limit 1800
```

### Issue: Low F1 Score (<0.85)
**Possible Causes:**
1. Class imbalance (check class distribution in output)
2. Insufficient training time (increase --time_limit)
3. Data quality issues (verify synthetic data)

**Solution:**
```powershell
# Retrain with longer time limit
python scripts/train_classification_per_machine.py --machine_id <machine_id> --time_limit 7200
```

---

## Output Files

### Models (per machine):
```
models/classification/<machine_id>/
├── models/
│   ├── XGBoost_BAG_L1/
│   ├── LightGBM_BAG_L1/
│   └── ... (other models)
├── predictor.pkl
└── trainer.pkl
```

### Reports (per machine):
```
reports/performance_metrics/<machine_id>_classification_report.json
```

**Report Structure:**
```json
{
  "machine_id": "motor_siemens_1la7_001",
  "task_type": "classification",
  "training_time_minutes": 15.5,
  "metrics": {
    "accuracy": 0.9876,
    "precision": 0.9234,
    "recall": 0.9567,
    "f1_score": 0.9398,
    "roc_auc": 0.9987
  },
  "best_model": "XGBoost_BAG_L1",
  "class_distribution": { ... },
  "feature_importance": { ... },
  "confusion_matrix": [ ... ]
}
```

### Batch Report (if batch training used):
```
reports/batch_training_classification_10_machines.json
```

---

## Next Steps After Phase 2.2.2

Once all 10 classification models are trained:

1. ✅ **Phase 2.2.3:** Validate models (check F1 >0.85 for all)
2. ⏭️ **Phase 2.3:** Train regression models (RUL prediction)
3. ⏭️ **Phase 2.4:** Train anomaly detection models
4. ⏭️ **Phase 2.5:** Train time-series forecasting models

---

## Quick Reference Commands

```powershell
# Validate setup
python scripts/validate_setup_phase_2_2_2.py

# Train single machine
python scripts/train_classification_per_machine.py --machine_id motor_siemens_1la7_001

# Train all 10 machines (recommended)
python scripts/batch_train_classification.py

# Check results
ls models/classification/
ls reports/performance_metrics/

# View batch report
cat reports/batch_training_classification_10_machines.json | ConvertFrom-Json
```

---

## Time Estimates

| Configuration | Time per Machine | Total Time (10 machines) |
|---------------|------------------|--------------------------|
| best_quality (recommended) | 15-60 min | 2.5-10 hours |
| high_quality | 10-30 min | 1.7-5 hours |
| good_quality | 5-15 min | 0.8-2.5 hours |
| medium_quality | 3-10 min | 0.5-1.7 hours |

**Recommendation:** Start with `best_quality` for best performance.

---

## Hardware Utilization

- **CPU:** 6 cores (i7-14700HX)
- **GPU:** RTX 4070 (used for NN_TORCH, FASTAI only)
- **RAM:** ~4-6 GB per training session
- **Storage:** ~50 MB per model (~500 MB total)

---

## Success Criteria

✅ All 10 models trained successfully  
✅ F1 Score >0.85 for each machine  
✅ No training errors  
✅ Model files saved in `models/classification/`  
✅ Reports saved in `reports/performance_metrics/`  

---

**Ready to Start Training? Run:**
```powershell
python scripts/validate_setup_phase_2_2_2.py
python scripts/batch_train_classification.py
```

Good luck! 🚀
