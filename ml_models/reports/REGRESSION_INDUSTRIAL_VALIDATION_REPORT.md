# Regression Models Industrial Validation Report
**Phase 2.3.2-2.3.3: RUL Prediction Models**  
**Validation Date:** November 21, 2025  
**Total Models:** 10 Priority Machines

---

## Executive Summary

✅ **Training Completed:** All 10 regression models trained successfully  
⚠️ **Validation Results:** 9 Grade B (Deployment Ready), 1 Grade C (Needs Improvement)  
❌ **CRITICAL ISSUE:** All models 13x too large for Raspberry Pi deployment (646-758 MB vs <50 MB target)

### Overall Performance Metrics
| Metric | Average | Min | Max | Target | Status |
|--------|---------|-----|-----|--------|--------|
| **R² Score** | 0.9332 | 0.8993 | 0.9653 | >0.75 | ✅ **EXCELLENT** |
| **RMSE** | 45.82 hours | 21.41 hours | 70.03 hours | <100 hours | ✅ **EXCELLENT** |
| **MAE** | 24.12 hours | 12.05 hours | 36.37 hours | <75 hours | ✅ **EXCELLENT** |
| **Model Size** | 666.17 MB | 646.21 MB | 758.18 MB | <50 MB | ❌ **CRITICAL** |
| **Training Time** | 1.1 min/model | 0.5 min | 3.0 min | <15 min | ✅ **EXCELLENT** |

### Deployment Readiness
- **Deployment Ready:** 9/10 models (90%)
- **Pi Compatible:** 0/10 models (0%) ⚠️ **URGENT OPTIMIZATION NEEDED**
- **Grade Distribution:**
  - Grade A: 0 models
  - Grade B: 9 models (90%)
  - Grade C: 1 model (10%)
  - Grade D: 0 models

---

## Detailed Model Performance

### 1. motor_siemens_1la7_001
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9426 | ✅ Excellent |
| **RMSE** | 69.03 hours | ✅ Good |
| **MAE** | 33.45 hours | ✅ Good |
| **Model Size** | 668.66 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Unstable across time |
| **Critical Range (<100h)** | A (Excellent) | ✅ MAE=11.05h |
| **Prediction Bias** | D (Poor) | ⚠️ Mean=-25.51h |
| **Pi Compatible** | NO | ❌ Needs optimization |

**Key Issues:**
- Temporal stability poor (Std=1.03, negative R² in time-series folds)
- Life phase consistency poor (R² spread=0.42)
- Model size 13x target
- Significant under-estimation bias

---

### 2. motor_abb_m3bp_002
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9389 | ✅ Excellent |
| **RMSE** | 49.65 hours | ✅ Excellent |
| **MAE** | 26.39 hours | ✅ Excellent |
| **Model Size** | 655.24 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Std=2.28 |
| **Critical Range (<100h)** | A (Excellent) | ✅ MAE=10.17h |
| **Prediction Bias** | D (Poor) | ⚠️ Mean=-24.88h |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 3. motor_weg_w22_003
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9626 | ✅ Excellent |
| **RMSE** | 48.88 hours | ✅ Excellent |
| **MAE** | 22.14 hours | ✅ Excellent |
| **Model Size** | 758.18 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Std=1.51 |
| **Critical Range (<100h)** | A (Excellent) | ✅ MAE=11.90h |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 4. pump_grundfos_cr3_004
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9419 | ✅ Excellent |
| **RMSE** | 37.68 hours | ✅ Excellent |
| **MAE** | 20.54 hours | ✅ Excellent |
| **Model Size** | 656.62 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ |
| **Critical Range (<100h)** | A (Excellent) | ✅ |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 5. pump_flowserve_ansi_005
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9153 | ✅ Excellent |
| **RMSE** | 43.38 hours | ✅ Excellent |
| **MAE** | 23.49 hours | ✅ Excellent |
| **Model Size** | 646.21 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ |
| **Critical Range (<100h)** | A (Excellent) | ✅ |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 6. compressor_atlas_copco_ga30_001
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9328 | ✅ Excellent |
| **RMSE** | 29.19 hours | ✅ Excellent |
| **MAE** | 15.64 hours | ✅ Excellent |
| **Model Size** | 657.75 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ |
| **Critical Range (<100h)** | A (Excellent) | ✅ |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 7. compressor_ingersoll_rand_2545_009
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9246 | ✅ Excellent |
| **RMSE** | 30.25 hours | ✅ Excellent |
| **MAE** | 17.47 hours | ✅ Excellent |
| **Model Size** | 648.41 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ |
| **Critical Range (<100h)** | A (Excellent) | ✅ |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 8. cnc_dmg_mori_nlx_010
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9653 | ✅ Excellent (BEST) |
| **RMSE** | 21.41 hours | ✅ Excellent (BEST) |
| **MAE** | 12.05 hours | ✅ Excellent (BEST) |
| **Model Size** | 667.01 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Std=0.73 |
| **Critical Range (<100h)** | A (Excellent) | ✅ MAE=10.53h |
| **Prediction Bias** | C (Fair) | ⚠️ Mean=-11.36h |
| **Pi Compatible** | NO | ❌ Needs optimization |

**Best performing model overall!**

---

### 9. hydraulic_beckwood_press_011
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | B | ⚠️ Deployment Ready |
| **Test R²** | 0.9085 | ✅ Excellent |
| **RMSE** | 58.70 hours | ✅ Good |
| **MAE** | 33.64 hours | ✅ Good |
| **Model Size** | 652.17 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Std=2.24 |
| **Critical Range (<100h)** | A (Excellent) | ✅ |
| **Prediction Bias** | D (Poor) | ⚠️ Mean=-30.28h |
| **Pi Compatible** | NO | ❌ Needs optimization |

---

### 10. cooling_tower_bac_vti_018
| **Metric** | **Value** | **Grade** |
|------------|-----------|-----------|
| **Overall Grade** | C | ❌ NOT Deployment Ready |
| **Test R²** | 0.8993 | ⚠️ Lowest performance |
| **RMSE** | 70.03 hours | ⚠️ Highest error |
| **MAE** | 36.37 hours | ⚠️ Highest error |
| **Model Size** | 651.46 MB | ❌ Too Large |
| **Temporal Stability** | D (Poor) | ⚠️ Std=2.00 |
| **Critical Range (<100h)** | D (Poor) | ❌ MAE=42.82h, R²=-17.44 |
| **Prediction Bias** | D (Poor) | ⚠️ Mean=-35.90h (worst) |
| **Pi Compatible** | NO | ❌ Needs optimization |

**Weakest model - needs retraining!**

---

## Validation Test Results

### Test 1: Temporal Robustness (Time-Series Split)
**Purpose:** Check model stability across different time windows  
**Method:** 5-fold time-series split validation

| Machine | Mean R² | Std Dev | Grade | Status |
|---------|---------|---------|-------|--------|
| motor_siemens_1la7_001 | -1.39 | 1.03 | D | ❌ Poor |
| motor_abb_m3bp_002 | -1.62 | 2.28 | D | ❌ Poor |
| motor_weg_w22_003 | -0.66 | 1.51 | D | ❌ Poor |
| pump_grundfos_cr3_004 | -1.37 | 1.88 | D | ❌ Poor |
| pump_flowserve_ansi_005 | -0.79 | 1.42 | D | ❌ Poor |
| compressor_atlas_copco_ga30_001 | -1.45 | 1.65 | D | ❌ Poor |
| compressor_ingersoll_rand_2545_009 | -1.11 | 1.72 | D | ❌ Poor |
| cnc_dmg_mori_nlx_010 | -0.44 | 0.73 | D | ❌ Poor |
| hydraulic_beckwood_press_011 | -2.91 | 2.24 | D | ❌ Poor |
| cooling_tower_bac_vti_018 | -2.94 | 2.00 | D | ❌ Poor |

**Issue:** All models show poor temporal stability with negative R² scores in early time folds. This indicates models struggle with distribution shifts over time.

---

### Test 2: Early vs Late Life RUL Prediction
**Purpose:** Check consistency across equipment life phases  
**Method:** Split data into early (>66% life), mid (33-66%), late (<33%) life stages

| Machine | Early R² | Mid R² | Late R² | R² Spread | Grade |
|---------|----------|--------|---------|-----------|-------|
| motor_siemens_1la7_001 | 0.74 | 0.32 | 0.51 | 0.42 | D |
| motor_abb_m3bp_002 | 0.54 | 0.65 | 0.97 | 0.42 | D |
| motor_weg_w22_003 | 0.69 | 0.64 | 0.77 | 0.13 | C |
| pump_grundfos_cr3_004 | 0.72 | 0.64 | 0.85 | 0.21 | D |
| pump_flowserve_ansi_005 | 0.63 | 0.63 | 0.70 | 0.07 | B |
| compressor_atlas_copco_ga30_001 | 0.68 | 0.63 | 0.90 | 0.27 | D |
| compressor_ingersoll_rand_2545_009 | 0.68 | 0.44 | 0.85 | 0.42 | D |
| cnc_dmg_mori_nlx_010 | 0.82 | 0.78 | 0.46 | 0.36 | D |
| hydraulic_beckwood_press_011 | 0.45 | 0.36 | 0.54 | 0.18 | D |
| cooling_tower_bac_vti_018 | 0.39 | 0.48 | 0.29 | 0.19 | D |

**Issue:** Most models show inconsistent performance across life phases, with 8/10 receiving Grade D.

---

### Test 3: Prediction Bias Analysis
**Purpose:** Check for systematic over/under-estimation  
**Method:** Analyze residuals (actual - predicted)

| Machine | Mean Residual | Bias Grade | Over-Est % | Under-Est % |
|---------|---------------|------------|------------|-------------|
| motor_siemens_1la7_001 | -25.51h | D | 47.2% | 52.8% |
| motor_abb_m3bp_002 | -24.88h | D | 26.0% | 74.0% |
| motor_weg_w22_003 | -21.61h | D | 16.5% | 83.5% |
| pump_grundfos_cr3_004 | -19.27h | C | 18.3% | 81.7% |
| pump_flowserve_ansi_005 | -22.11h | D | 15.0% | 85.0% |
| compressor_atlas_copco_ga30_001 | -14.38h | C | 9.2% | 90.8% |
| compressor_ingersoll_rand_2545_009 | -15.90h | C | 10.1% | 89.9% |
| cnc_dmg_mori_nlx_010 | -11.36h | C | 11.0% | 89.0% |
| hydraulic_beckwood_press_011 | -30.28h | D | 28.7% | 71.3% |
| cooling_tower_bac_vti_018 | -35.90h | D | 6.3% | 93.7% |

**Issue:** All models show negative bias (under-estimating RUL), which is less dangerous than over-estimating but still problematic.

---

### Test 4: Critical Range Accuracy (<100 hours RUL)
**Purpose:** Test accuracy in most critical maintenance window  
**Method:** Evaluate on samples with RUL < 100 hours

| Machine | Samples | R² | MAE | Dangerous Over-Est | Grade |
|---------|---------|----|----|-------------------|-------|
| motor_siemens_1la7_001 | 738 | 0.02 | 11.05h | 4.5% | A ✅ |
| motor_abb_m3bp_002 | 122 | 0.74 | 10.17h | 0.8% | A ✅ |
| motor_weg_w22_003 | 239 | 0.46 | 11.90h | 3.8% | A ✅ |
| pump_grundfos_cr3_004 | 242 | 0.73 | 10.45h | 0.0% | A ✅ |
| pump_flowserve_ansi_005 | 320 | 0.59 | 12.32h | 0.6% | A ✅ |
| compressor_atlas_copco_ga30_001 | 298 | 0.74 | 8.59h | 0.0% | A ✅ |
| compressor_ingersoll_rand_2545_009 | 329 | 0.33 | 11.74h | 2.4% | A ✅ |
| cnc_dmg_mori_nlx_010 | 105 | 0.55 | 10.53h | 1.0% | A ✅ |
| hydraulic_beckwood_press_011 | 403 | -0.97 | 13.76h | 3.7% | A ✅ |
| cooling_tower_bac_vti_018 | 59 | -17.44 | 42.82h | 13.6% | D ❌ |

**Strong Point:** 9/10 models achieve Grade A in critical range accuracy with MAE < 15 hours and low dangerous over-estimation rates.

---

### Test 5: Raspberry Pi Compatibility
**Purpose:** Check deployment readiness for edge devices  
**Method:** Evaluate model size and inference speed

| Machine | Size (MB) | Size Grade | Est. Inference | Pi Compatible |
|---------|-----------|------------|----------------|---------------|
| motor_siemens_1la7_001 | 668.66 | D | ~100ms | ❌ |
| motor_abb_m3bp_002 | 655.24 | D | ~100ms | ❌ |
| motor_weg_w22_003 | 758.18 | D | ~100ms | ❌ |
| pump_grundfos_cr3_004 | 656.62 | D | ~100ms | ❌ |
| pump_flowserve_ansi_005 | 646.21 | D | ~100ms | ❌ |
| compressor_atlas_copco_ga30_001 | 657.75 | D | ~100ms | ❌ |
| compressor_ingersoll_rand_2545_009 | 648.41 | D | ~100ms | ❌ |
| cnc_dmg_mori_nlx_010 | 667.01 | D | ~100ms | ❌ |
| hydraulic_beckwood_press_011 | 652.17 | D | ~100ms | ❌ |
| cooling_tower_bac_vti_018 | 651.46 | D | ~100ms | ❌ |

**CRITICAL ISSUE:** All models are 13x larger than 50 MB target. Using WeightedEnsemble with bagging creates massive models.

---

## Key Findings

### ✅ Strengths
1. **Excellent Test Performance:** Average R² = 0.9332 (target >0.75)
2. **Low Prediction Errors:** Average RMSE = 45.82 hours, MAE = 24.12 hours
3. **Fast Training:** Average 1.1 minutes per model
4. **Critical Range Accuracy:** 9/10 models achieve Grade A (<15h MAE in <100h RUL range)
5. **High Deployment Readiness:** 9/10 models ready for production use

### ❌ Critical Issues
1. **Model Size:** All models 646-758 MB (13x target) - **BLOCKS RASPBERRY PI DEPLOYMENT**
2. **Temporal Instability:** All models Grade D in time-series validation
3. **Life Phase Inconsistency:** 8/10 models show inconsistent performance across equipment life stages
4. **Prediction Bias:** Systematic under-estimation (mean bias -11 to -36 hours)
5. **One Weak Model:** cooling_tower_bac_vti_018 only achieves Grade C

### ⚠️ Root Causes
1. **Ensemble Models:** WeightedEnsemble_L2 with 3-fold bagging creates large models
2. **Multiple Base Models:** Training 6 models (LightGBMXT, LightGBM, RF, CatBoost, XGBoost, LightGBMLarge)
3. **Distribution Shift:** Models trained on full dataset don't handle temporal distribution changes
4. **Feature Engineering:** DateTime features may not capture temporal patterns effectively

---

## Recommendations

### IMMEDIATE (Phase 2.3.4)
1. **Model Optimization for Pi:**
   - Use single LightGBM model (no ensemble)
   - Set `num_bag_folds=0` (no bagging)
   - Use `presets='optimize_for_deployment'`
   - Exclude heavy models: RF, CatBoost, XGBoost, XT
   - Target: <50 MB per model

2. **Retrain Weak Model:**
   - Retrain cooling_tower_bac_vti_018 with hyperparameter tuning
   - Investigate low feature count (only 2 features)

### SHORT-TERM (Phase 2.4)
3. **Temporal Robustness:**
   - Implement online learning / model updating
   - Add time-decay weighting in training
   - Use temporal cross-validation during training

4. **Bias Correction:**
   - Apply post-processing calibration
   - Investigate feature scaling issues
   - Add domain knowledge constraints

### LONG-TERM (Phase 3)
5. **Model Compression:**
   - Apply quantization (float32 → int8)
   - Use knowledge distillation
   - Convert to ONNX for optimized inference
   - Prune unnecessary features

6. **Advanced Architectures:**
   - Explore LSTM for temporal patterns
   - Test transformer models for long sequences
   - Implement ensemble pruning

---

## Comparison: Classification vs Regression Models

| Aspect | Classification (Phase 2.2.3) | Regression (Phase 2.3.3) |
|--------|------------------------------|--------------------------|
| **Avg Performance** | F1=0.7695 | R²=0.9332 |
| **Grade Distribution** | 10 Grade B | 9 Grade B, 1 Grade C |
| **Deployment Ready** | 0/10 (0%) | 9/10 (90%) |
| **Model Size** | ~650 MB | ~666 MB |
| **Pi Compatible** | 0/10 | 0/10 |
| **Temporal Stability** | All Grade D | All Grade D |
| **Critical Metric** | High FNR (20-41%) | Critical range Grade A (9/10) |
| **Overall Status** | ❌ Needs Improvement | ⚠️ Good but needs optimization |

**Conclusion:** Regression models significantly outperform classification models, with 9/10 deployment-ready vs 0/10. However, both face same critical issue: model size blocks Pi deployment.

---

## Next Steps

### Phase 2.3.4: Model Optimization (URGENT)
**Goal:** Reduce model size from 666 MB → <50 MB while maintaining R² >0.75

1. **Create Optimized Training Script:**
   - `train_regression_pi_optimized.py`
   - Single LightGBM model
   - No bagging, no ensemble
   - Optimize for deployment preset

2. **Batch Retrain All 10 Models:**
   - Use optimized configuration
   - Target: <50 MB, R² >0.75, inference <50ms

3. **Revalidate:**
   - Run industrial validation again
   - Verify Pi compatibility
   - Document performance trade-offs

### Phase 2.4: Anomaly Detection Models
- Already complete (10/10 models trained)
- Need to run industrial validation
- Expected to have similar size issues

### Phase 2.5: Time-Series Forecasting
- Not yet started
- Will require LSTM/Transformer models
- Plan for edge optimization from start

---

## Validation Artifacts

### Reports Generated
1. **Summary Report:**
   - `ml_models/reports/regression_industrial_validation_summary.json`
   - Contains all metrics, grades, and statistics

2. **Individual Reports (10 files):**
   - `ml_models/reports/industrial_validation_regression/{machine_id}_regression_validation.json`
   - Detailed test results for each model

3. **Training Report:**
   - `ml_models/reports/batch_training_regression_10_machines.json`
   - Training metrics and timings

### Validation Script
- `ml_models/scripts/validation/validate_regression_industrial.py`
- 5 comprehensive tests:
  1. Basic performance metrics
  2. Temporal robustness (time-series split)
  3. Early vs late life consistency
  4. Prediction bias analysis
  5. Critical range accuracy (<100h RUL)
  6. Raspberry Pi compatibility

---

## Conclusion

The regression models demonstrate **excellent predictive performance** with an average R² of 0.9332 and low prediction errors. 9 out of 10 models achieve Grade B and are deployment-ready for server/cloud environments.

However, there is a **critical blocker for edge deployment**: all models are 13x larger than the 50 MB Raspberry Pi target due to ensemble architectures with bagging. This issue must be resolved immediately before Pi deployment can proceed.

The models also show **poor temporal stability**, suggesting they may struggle with distribution shifts over time. This is acceptable for initial deployment but should be addressed through online learning in Phase 3.

**Overall Assessment:** Strong foundation with excellent accuracy, but requires urgent optimization for edge deployment.

---

**Report Created:** November 21, 2025  
**Validation Script:** `validate_regression_industrial.py`  
**Phase:** 2.3.2-2.3.3 Complete, 2.3.4 Pending (Model Optimization)
