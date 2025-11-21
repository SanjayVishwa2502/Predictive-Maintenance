# Phase 2.5.1: Time-Series Data Analysis Report

**Date:** November 19, 2025  
**Phase:** 2.5.1 - Time-Series Pipeline Setup  
**Purpose:** Document timestamp analysis and data characteristics for time-series forecasting

---

## 1. Critical Finding: No Native Timestamps

### Issue
The synthetic data generated in Phase 1 (GAN) **does NOT contain timestamp columns**.

**Data Sources Analyzed:**
- `ml_models/data/processed/pooled_train.parquet` - ❌ No timestamp column
- `GAN/data/synthetic/<machine>/train.parquet` - ❌ No timestamp column

**Only time-related column:** `cycle_time_seconds` (for certain machines, not timestamps)

### Solution Implemented
**Synthetic timestamp generation** in `train_timeseries.py`:
```python
# Assume hourly data starting from arbitrary timestamp
train_machine['timestamp'] = pd.date_range(
    start='2024-01-01', periods=len(train_machine), freq='h'
)
```

**Assumption:** Sequential rows represent consecutive hourly measurements.

**Justification:**
- Industrial IoT sensors typically log at regular intervals (1 min - 1 hour)
- Hourly sampling is common for non-critical monitoring
- GAN data is sequential (row N+1 follows row N temporally)
- 35,000 samples @ 1 hour = ~4 years of data per machine

---

## 2. Machine Sensor Inventory

### Sensor Count Distribution

| Machine ID | Sensor Count | Category |
|-----------|-------------|----------|
| `cooling_tower_bac_vti_018` | 1 | ⚠️ Very Low |
| `cnc_dmg_mori_nlx_010` | 2 | Low |
| `hydraulic_parker_hpu_012` | 3 | Low |
| `cnc_haas_vf2_001` | 4 | Low |
| `conveyor_dorner_2200_013` | 4 | Low |
| `conveyor_hytrol_e24ez_014` | 4 | Low |
| `pump_ksb_etanorm_006` | 4 | Low |
| `transformer_square_d_017` | 4 | Low |
| `compressor_ingersoll_rand_2545_009` | 5 | Medium |
| `hydraulic_beckwood_press_011` | 5 | Medium |
| `pump_flowserve_ansi_005` | 5 | Medium |
| `robot_abb_irb6700_016` | 5 | Medium |
| `fan_ebm_papst_a3g710_007` | 6 | Medium |
| `fan_howden_buffalo_008` | 6 | Medium |
| `robot_fanuc_m20ia_015` | 8 | High |
| `turbofan_cfm56_7b_001` | 8 | High |
| `compressor_atlas_copco_ga30_001` | 10 | High |
| `motor_abb_m3bp_002` | 10 | High |
| `motor_weg_w22_003` | 10 | High |
| `pump_grundfos_cr3_004` | 10 | High |
| `motor_siemens_1la7_001` | 22 | Very High |

**Total:** 21 machines with sensor counts ranging from 1 to 22.

---

## 3. Adaptive Target Selection Logic

### Strategy

**Rule:** Based on sensor count threshold (7 sensors)

#### Case A: Machines with <7 sensors (14 machines)
**Action:** Predict ALL active sensors

**Rationale:**
- Limited sensor suite → all sensors are critical
- No redundancy → must model all available signals
- Example: `cooling_tower_bac_vti_018` has only 1 sensor → predict that 1 sensor

**Machines:**
```
cooling_tower_bac_vti_018          (1)
cnc_dmg_mori_nlx_010               (2)
hydraulic_parker_hpu_012           (3)
cnc_haas_vf2_001                   (4)
conveyor_dorner_2200_013           (4)
conveyor_hytrol_e24ez_014          (4)
pump_ksb_etanorm_006               (4)
transformer_square_d_017           (4)
compressor_ingersoll_rand_2545_009 (5)
hydraulic_beckwood_press_011       (5)
pump_flowserve_ansi_005            (5)
robot_abb_irb6700_016              (5)
fan_ebm_papst_a3g710_007           (6)
fan_howden_buffalo_008             (6)
```

#### Case B: Machines with ≥7 sensors (7 machines)
**Action:** Select 6 core sensors using keyword matching

**Core Sensor Categories (Priority Order):**
1. **Temperature** - Keywords: `temp`, `temperature`, `_C`, `_K`, `_R`
2. **Pressure** - Keywords: `pressure`, `psi`, `bar`, `psia`
3. **Vibration** - Keywords: `vibration`, `rms`, `mm_s`, `mm/s`
4. **Current** - Keywords: `current`, `_a`, `_amp`, `ampere`
5. **Voltage** - Keywords: `voltage`, `_v`, `_volt`
6. **Power** - Keywords: `power`, `_kw`, `_w`, `watt`

**Selection Algorithm:**
```python
for each category (temp, pressure, vibration, current, voltage, power):
    find first sensor matching category keywords
    add to selected list (max 6)
```

**Machines:**
```
robot_fanuc_m20ia_015              (8 sensors → select 6)
turbofan_cfm56_7b_001              (8 sensors → select 6)
compressor_atlas_copco_ga30_001    (10 sensors → select 6)
motor_abb_m3bp_002                 (10 sensors → select 6)
motor_weg_w22_003                  (10 sensors → select 6)
pump_grundfos_cr3_004              (10 sensors → select 6)
motor_siemens_1la7_001             (22 sensors → select 6)
```

---

## 4. Time-Series Configuration

### Model Architecture
**Type:** LSTM (Long Short-Term Memory)

**Layers:**
```
Input: (168 timesteps, n_features)
  ↓
LSTM(128 units, return_sequences=True) + Dropout(0.2)
  ↓
LSTM(64 units) + Dropout(0.2)
  ↓
Dense(64, relu) + Dropout(0.2)
  ↓
Dense(forecast_hours × n_targets) → Reshape to (24, n_targets)
  ↓
Output: (24 timesteps, n_features)
```

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Lookback Window** | 168 hours (7 days) | Standard week pattern, captures weekly cycles |
| **Forecast Horizon** | 24 hours (1 day) | Next-day prediction for maintenance planning |
| **Batch Size** | 32 | Balance between speed and stability |
| **Learning Rate** | 0.001 | Adam optimizer default, with ReduceLROnPlateau |
| **Epochs** | 50 (max) | Early stopping monitors validation loss |
| **Early Stopping Patience** | 10 epochs | Prevents overfitting |

### Data Split
- **Training:** 35,000 samples → ~34,808 sequences
- **Validation:** ~10,000 samples → ~9,808 sequences
- **Test:** ~10,000 samples → ~9,808 sequences

**Sequence Calculation:**
```
n_sequences = n_samples - lookback - forecast + 1
            = 35,000 - 168 - 24 + 1
            = 34,809 sequences
```

---

## 5. Evaluation Metrics

### Primary Metrics

#### 1. **RMSE (Root Mean Squared Error)**
- **Formula:** `sqrt(mean((y_true - y_pred)²))`
- **Unit:** Same as sensor unit (e.g., °C, bar, mm/s)
- **Use:** Primary metric - penalizes large errors

#### 2. **MAE (Mean Absolute Error)**
- **Formula:** `mean(|y_true - y_pred|)`
- **Unit:** Same as sensor unit
- **Use:** Robust to outliers, interpretable scale

#### 3. **MAPE (Mean Absolute Percentage Error)**
- **Formula:** `mean(|y_true - y_pred| / |y_true|) × 100`
- **Unit:** Percentage (%)
- **Use:** Scale-independent comparison across sensors

#### 4. **SMAPE (Symmetric Mean Absolute Percentage Error)**
- **Formula:** `mean(|y_pred - y_true| / ((|y_true| + |y_pred|) / 2)) × 100`
- **Unit:** Percentage (%)
- **Use:** Handles near-zero values better than MAPE

### Target Performance
- **MAPE < 15%** (per documentation)
- **SMAPE < 20%** (reasonable for industrial data)

---

## 6. Machine Metadata Approach

### Metadata Columns Available
From GAN synthetic data:
```python
metadata_cols = [
    'machine_id',           # Categorical identifier
    'machine_category',     # Type: motor, pump, fan, etc.
    'manufacturer',         # Brand/OEM
    'power_rating_kw',      # Installed power capacity
    'rated_speed_rpm',      # Design speed
    'operating_voltage',    # Voltage rating
    'equipment_age_years'   # Age of machine
]
```

### Usage in Time-Series Model

**Current Approach (Phase 2.5.1):**
- **Machine-specific models** - One model per machine
- Metadata NOT used as input features
- Each model learns patterns for its specific machine

**Future Enhancement (Phase 2.5.2 - if needed):**
- Train **single generic model** for all machines
- Include `machine_id` (one-hot encoded) as additional input feature
- Include numeric metadata (power_rating, age) as context
- Model architecture: `LSTM(features + metadata) → Forecast`

---

## 7. Data Quality Observations

### Missing Values
**Handled:** Each machine has NaN values in non-applicable sensors.

**Example:** 
- `motor_abb_m3bp_002` has values only in 10 specific motor sensors
- All pump/compressor sensors are NaN for this machine

**Solution:** 
```python
# Only select active sensors (non-null)
active_sensors = [col for col in all_cols if machine_data[col].notna().any()]

# Fill remaining NaNs with median
for i, col in enumerate(target_sensors):
    median_val = np.nanmedian(X_train[:, i])
    X_train[:, i] = np.nan_to_num(X_train[:, i], nan=median_val)
```

### Normalization
**Method:** StandardScaler (z-score normalization)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

**Important:** Scaler fit ONLY on training data to prevent data leakage.

---

## 8. Implementation Summary

### Files Created
- ✅ `ml_models/scripts/training/train_timeseries.py` - Main training script

### Key Functions
1. `identify_sensors()` - Extract active sensors for a machine
2. `select_target_sensors()` - Adaptive target selection (<7 → all; ≥7 → 6 core)
3. `create_sequences()` - Sliding window sequence generation
4. `build_lstm_model()` - LSTM architecture construction
5. `train_timeseries_model()` - Single machine training pipeline
6. `batch_train_all_machines()` - Batch training for all 21 machines

### Usage

**Train Single Machine:**
```bash
python ml_models/scripts/training/train_timeseries.py motor_abb_m3bp_002
```

**Batch Train All Machines:**
```bash
python ml_models/scripts/training/train_timeseries.py
```

### Output Artifacts

**Per Machine:**
- `ml_models/models/timeseries/<machine_id>/lstm_model.keras` - Trained model
- `ml_models/models/timeseries/<machine_id>/scaler.pkl` - Fitted StandardScaler
- `ml_models/models/timeseries/<machine_id>/config.json` - Model configuration
- `ml_models/reports/<machine_id>_timeseries_report.json` - Metrics report

**Batch Summary:**
- `ml_models/reports/timeseries_batch_summary.json` - Aggregate metrics

---

## 9. Next Steps (Phase 2.5.2)

### If Performance is Good (MAPE < 15%)
- ✅ Move to deployment/inference scripts
- ✅ Create visualization tools for forecasts
- ✅ Integrate with anomaly detection (flag abnormal forecasts)

### If Performance Needs Improvement
- 🔄 Try Transformer architecture instead of LSTM
- 🔄 Increase lookback window (168h → 336h)
- 🔄 Add attention mechanisms
- 🔄 Ensemble multiple architectures
- 🔄 Hyperparameter tuning (grid search)

### Blocked Items
- ⏸️ **Phase 2.3 Regression still blocked** - Waiting on RUL labels from GAN team

---

## 10. Assumptions and Limitations

### Assumptions Made
1. **Hourly sampling:** Rows are hourly sequential measurements
2. **Continuous operation:** No gaps or downtime in data
3. **Stationary patterns:** Machine behavior consistent across time
4. **Independent machines:** No cross-machine dependencies

### Known Limitations
1. **No real timestamps:** Using synthetic sequential timestamps
2. **No seasonal patterns:** Cannot model yearly/quarterly cycles
3. **No external factors:** Weather, load schedules, maintenance events not included
4. **Limited history:** Only 35,000 samples (~4 years at hourly sampling)

### Risks
- **Overfitting:** Models may memorize training patterns (mitigated by early stopping)
- **Distribution shift:** Real deployment data may differ from synthetic data
- **Computational cost:** 21 models × 50 epochs × LSTM training = significant GPU time

---

## Appendix: Example Sensor Selections

### Example 1: Low Sensor Machine
**Machine:** `hydraulic_parker_hpu_012`  
**Active Sensors:** 3  
**Selected Targets:** ALL 3 sensors
```
oil_temp_C
pump_vibration_mm_s
pump_temp_C
```

### Example 2: High Sensor Machine
**Machine:** `motor_siemens_1la7_001`  
**Active Sensors:** 22  
**Selected Targets:** 6 core sensors (selected by algorithm)
```
ambient_temp_C          (temperature)
discharge_pressure_bar  (pressure - if available)
rms_velocity_mm_s       (vibration)
current_25pct_load_A    (current)
voltage_phase_to_phase_V (voltage - if available)
power_factor_100pct     (power-related)
```

---

**Report Complete** ✅  
**Author:** Phase 2.5.1 Implementation  
**Date:** 2025-11-19
