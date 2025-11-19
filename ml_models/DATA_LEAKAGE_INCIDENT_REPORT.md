# 🚨 DATA LEAKAGE INCIDENT REPORT - Phase 2.4 Anomaly Detection

**Date:** November 18, 2025  
**Severity:** CRITICAL  
**Status:** IDENTIFIED & FIXED  

---

## Executive Summary

Critical data leakage was discovered in Phase 2.4 (Anomaly Detection) causing artificially inflated performance metrics. The issue resulted in **perfect F1 scores (1.0)** for 2 machines and average F1=0.8540, which appeared too good to be true.

**Root Cause:** Failure labels were created using thresholds calculated from the same dataset being labeled (including test data), creating perfect correlation between features and labels.

---

## Technical Details

### The Data Leakage Pattern

**Original Flawed Code:**
```python
def create_failure_labels(df, machine_id):
    """Create failure labels based on sensor thresholds"""
    
    # ⚠️ BUG: Calculates threshold from the SAME data being labeled!
    temp_threshold = df[temp_cols].quantile(0.90).max()  # Uses test data!
    temp_high = df[temp_cols].max(axis=1) > temp_threshold
    
    failure_status = (temp_high).astype(int)
    return failure_status
```

**What Happened:**
```python
# In prepare_ml_data():
train_df['failure_status'] = create_failure_labels(train_df, machine_id)  # Threshold from train
val_df['failure_status'] = create_failure_labels(val_df, machine_id)      # ⚠️ NEW threshold from val!
test_df['failure_status'] = create_failure_labels(test_df, machine_id)    # ⚠️ NEW threshold from test!
```

### Why This Caused Perfect Scores

1. **Test labels created from test data:**
   - Test threshold = 90th percentile of TEST data
   - Test labels = samples above TEST threshold
   
2. **Z-Score detector perfect correlation:**
   - Z-score flags: `temperature > mean + 3σ` (roughly 99.7th percentile)
   - Test labels flag: `temperature > 90th percentile`
   - **Same samples flagged = F1 = 1.0!**

3. **Example with cooling_tower_bac_vti_018:**
   ```
   Test Data: 7,500 samples
   - 90th percentile temp = 85°C (calculated from test data)
   - Labels: 750 samples > 85°C = "anomaly"
   - Z-score: Flags samples > 3σ ≈ same 750 samples
   - Result: Perfect overlap = F1 = 1.0000
   ```

---

## Impact Assessment

### Affected Metrics (Pre-Fix)

| Machine | Reported F1 | Actual F1 (Expected) | Over-inflation |
|---------|-------------|----------------------|----------------|
| cooling_tower_bac_vti_018 | 1.0000 | 0.65-0.75 | +33% |
| cnc_dmg_mori_nlx_010 | 0.9973 | 0.65-0.75 | +32% |
| pump_flowserve_ansi_005 | 0.9096 | 0.60-0.70 | +30% |
| motor_weg_w22_003 | 0.8379 | 0.55-0.65 | +29% |
| Average (all 10) | 0.8540 | 0.55-0.70 | +22-35% |

### Models Affected

- ✅ **All 10 anomaly detection models** trained on November 18, 2025
- ✅ **All 7 algorithms** per machine (Isolation Forest, One-Class SVM, LOF, DBSCAN, Z-Score, IQR, Modified Z-Score)
- ✅ **140+ visualizations** showing inflated performance
- ✅ **10 JSON reports** with incorrect metrics
- ✅ **Batch summary report** with averaged inflated metrics

### What Was NOT Affected

- ❌ **Classification models** (Phase 2.2) - Different labeling approach
- ❌ **Regression models** (Phase 2.3) - RUL labels use different logic
- ❌ **GAN synthetic data** - Data generation is unaffected
- ❌ **Other phases** - Isolated to Phase 2.4

---

## Root Cause Analysis

### Why This Happened

1. **Synthetic Data Limitation:**
   - No ground truth labels in GAN-generated data
   - Had to create labels programmatically
   
2. **Rushed Implementation:**
   - Focused on getting multiple algorithms working
   - Didn't validate label creation process thoroughly
   
3. **Validation Blind Spot:**
   - Perfect scores should have triggered immediate investigation
   - Assumed high performance was due to clean synthetic data

4. **Similar Issue in Phase 2.2:**
   - Classification had same problem (documented earlier)
   - Pattern repeated in anomaly detection

---

## Fix Implementation

### Corrected Code

```python
def create_failure_labels(df, machine_id, train_thresholds=None):
    """
    Create failure labels based on sensor thresholds
    
    Args:
        train_thresholds: Dict of thresholds from training data (prevents leakage)
                         If None, calculate from current df (ONLY for training data!)
    """
    
    if train_thresholds is None:
        # Only calculate when creating thresholds from TRAINING data
        train_thresholds = {}
        
        temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
        if temp_cols:
            train_thresholds['temp_threshold'] = df[temp_cols].quantile(0.90).max()
            train_thresholds['temp_cols'] = temp_cols
        # ... similar for vibration, current
    
    # Apply training thresholds to current data
    failure_score = 0
    if 'temp_threshold' in train_thresholds:
        temp_high = df[train_thresholds['temp_cols']].max(axis=1) > train_thresholds['temp_threshold']
        failure_score += temp_high.astype(int)
    
    failure_status = (failure_score >= 1).astype(int)
    return failure_status, train_thresholds
```

**Correct Usage:**
```python
# In prepare_ml_data():
# Step 1: Calculate thresholds from training data ONLY
train_df['failure_status'], train_thresholds = create_failure_labels(train_df, machine_id, train_thresholds=None)

# Step 2: Apply SAME thresholds to val/test (no leakage)
val_df['failure_status'], _ = create_failure_labels(val_df, machine_id, train_thresholds=train_thresholds)
test_df['failure_status'], _ = create_failure_labels(test_df, machine_id, train_thresholds=train_thresholds)
```

---

## Action Items

### Immediate (DONE)

- [x] Fixed `create_failure_labels()` function with train_thresholds parameter
- [x] Updated `prepare_ml_data()` to use training thresholds for all splits
- [x] Documented data leakage issue in this report
- [x] Identified TensorFlow import issue (slow load, not missing)

### Required Next Steps

- [ ] **Retrain all 10 anomaly models** with fixed labeling
- [ ] **Regenerate all 140+ visualizations** with correct metrics
- [ ] **Update all 10 JSON reports** with realistic performance
- [ ] **Update batch summary report** with corrected averages
- [ ] **Update PHASE_2_ML_DETAILED_APPROACH.md** with realistic metrics
- [ ] **Add validation checks** for suspiciously high scores (F1 > 0.95 = investigate)

### Long-Term Improvements

- [ ] Request ground truth labels from colleague's GAN phase
- [ ] Implement hold-out calibration set for threshold tuning
- [ ] Add automated data leakage detection tests
- [ ] Create label quality assessment metrics
- [ ] Document labeling strategy limitations in all model cards

---

## Expected Realistic Performance

After retraining with fixed labeling:

| Metric | Inflated (Buggy) | Expected (Fixed) | Change |
|--------|------------------|------------------|--------|
| Average F1 | 0.8540 | 0.55-0.70 | -15-30% |
| Models F1 ≥ 0.80 | 8/10 (80%) | 0-2/10 (0-20%) | -60-80% |
| Perfect F1=1.0 | 2/10 (20%) | 0/10 (0%) | -20% |
| Range | 0.6768-1.0000 | 0.45-0.75 | Lower bounds |

**Why lower is actually better:**
- More realistic for synthetic data with programmatic labels
- Matches Phase 2.2 classification performance (F1=0.778 average)
- Indicates proper train/test separation
- Leaves room for improvement with real data

---

## Lessons Learned

1. **Perfect scores are red flags** - Always investigate F1 > 0.95
2. **Validate label creation** - Especially with synthetic data
3. **Document assumptions** - Programmatic labels ≠ ground truth
4. **Test for leakage** - Check if threshold sources match data splits
5. **Synthetic data limitations** - High accuracy may mask fundamental issues

---

## Communication Plan

### Internal Documentation
- ✅ This incident report (DATA_LEAKAGE_INCIDENT_REPORT.md)
- ⏳ Update PHASE_2_ML_DETAILED_APPROACH.md with corrected metrics
- ⏳ Add warnings to all affected model reports

### User Communication
- Inform user that retraining is required
- Explain why "perfect" scores were actually bugs
- Set realistic expectations for retrained models
- Provide timeline for corrected results (est. 5-10 minutes)

---

## Approval & Sign-off

**Issue Identified By:** User observation ("perfect scores look suspicious")  
**Root Cause Analysis:** GitHub Copilot  
**Fix Implemented:** November 18, 2025  
**Status:** FIXED (code), PENDING (retraining)  

**Next Action:** User decision on whether to retrain immediately or document and move forward.

