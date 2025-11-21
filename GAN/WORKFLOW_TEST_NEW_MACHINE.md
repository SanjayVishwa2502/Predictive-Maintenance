# GAN Workflow Test: Adding a New Machine

This guide demonstrates the complete end-to-end workflow for adding a new machine to the GAN system.

## Overview

The workflow consists of 4 main steps:
1. **Create machine profile** (metadata + RUL configuration)
2. **Generate temporal seed data** (physics-based degradation patterns)
3. **Train TVAE model** (learn sensor correlations and temporal structure)
4. **Generate synthetic dataset** (35K train, 7.5K val, 7.5K test)

## Prerequisites

- Python environment with GAN dependencies installed
- All existing 26 machines validated (✅ confirmed)
- Temporal seed generation script updated to 10,000 samples (✅ confirmed)

## Step 1: Create Machine Profile

### 1.1 Create Metadata File

Location: `metadata/{machine_id}_metadata.json`

```json
{
    "machine_id": "NEW_MACHINE_ID",
    "machine_type": "motor|pump|cnc|fan|compressor|conveyor|robot|hydraulic|transformer|cooling_tower|turbofan",
    "manufacturer": "Manufacturer Name",
    "model": "Model Number",
    "sensors": [
        {
            "name": "sensor_1_name",
            "unit": "unit (e.g., C, mm/s, A)",
            "type": "numerical"
        },
        {
            "name": "sensor_2_name",
            "unit": "unit",
            "type": "numerical"
        }
    ],
    "operational_parameters": {
        "rated_power_kW": 100,
        "rated_speed_rpm": 1800,
        "rated_voltage_V": 480
    }
}
```

**Example for Motor:**
```json
{
    "machine_id": "motor_test_newmodel_001",
    "machine_type": "motor",
    "manufacturer": "TestMotor Corp",
    "model": "TM-5000",
    "sensors": [
        {
            "name": "bearing_de_temp_C",
            "unit": "C",
            "type": "numerical"
        },
        {
            "name": "winding_temp_C",
            "unit": "C",
            "type": "numerical"
        },
        {
            "name": "vibration_mm_s",
            "unit": "mm/s",
            "type": "numerical"
        },
        {
            "name": "current_A",
            "unit": "A",
            "type": "numerical"
        }
    ],
    "operational_parameters": {
        "rated_power_kW": 50,
        "rated_speed_rpm": 1500,
        "rated_voltage_V": 400
    }
}
```

### 1.2 Add RUL Configuration

Location: `config/rul_profiles.py`

Add your machine to the appropriate category:

```python
RUL_PROFILES = {
    'motor': {
        'machines': [
            'motor_siemens_1la7_001',
            'motor_abb_m3bp_002',
            'motor_weg_w22_003',
            'motor_test_newmodel_001'  # ADD YOUR MACHINE HERE
        ],
        'max_rul': 1000,  # Maximum RUL in hours
        'cycles_per_dataset': 3,  # Number of run-to-failure cycles
        'degradation_pattern': 'linear_slow',  # linear_slow|linear_medium|linear_fast|exponential
        'noise_std': 8.0,  # Standard deviation of RUL noise
        'sensor_correlation': {
            'bearing_de_temp_C': {'base': 40, 'range': 35, 'noise': 0.5},
            'winding_temp_C': {'base': 50, 'range': 40, 'noise': 0.8},
            'vibration_mm_s': {'base': 2.5, 'range': 5.0, 'noise': 0.15},
            'current_A': {'base': 8.0, 'range': 6.0, 'noise': 0.3}
        }
    }
}
```

**Degradation Pattern Options:**
- `linear_slow`: Gradual wear (motors, fans, robots)
- `linear_medium`: Moderate wear (pumps, conveyors, hydraulics)
- `linear_fast`: Rapid wear (compressors)
- `exponential`: Accelerating failure (CNC tools, turbofans)

**Sensor Correlation Structure:**
```python
'sensor_name': {
    'base': 50,      # Healthy baseline value
    'range': 30,     # Change from healthy to failure (temp increases by 30°C)
    'noise': 0.5     # Random noise standard deviation
}
```

## Step 2: Generate Temporal Seed Data

Run the seed generation script for your new machine:

```bash
cd "c:\Projects\Predictive Maintenance\GAN"
python scripts/create_temporal_seed_data.py motor_test_newmodel_001
```

**Expected Output:**
```
======================================================================
Creating temporal seed data: motor_test_newmodel_001
======================================================================
Configuration:
  Max RUL: 1000 hours
  Cycles: 3
  Pattern: linear_slow
  Total samples: 10000
  Generating cycle 1/3... OK 3333 samples
  Generating cycle 2/3... OK 3333 samples
  Generating cycle 3/3... OK 3334 samples

======================================================================
Seed data creation complete!
  Total samples: 10000
  Features: 6 (timestamp + rul + 4 sensors)
  RUL range: 1015 → 0
  Time range: 2024-01-01 00:00:00 to 2025-02-20 15:00:00
======================================================================

✅ Saved: motor_test_newmodel_001_temporal_seed.parquet
✅ RUL decreasing: 98.5%
```

**Validation Checks:**
- ✅ 10,000 samples generated
- ✅ RUL decreasing >90%
- ✅ Timestamp spans ~13 months (10K hours)
- ✅ Sensors correlate with RUL degradation

## Step 3: Train TVAE Model

Train the TVAE model on the temporal seed data:

```bash
python scripts/retrain_tvae_temporal.py motor_test_newmodel_001
```

**Expected Output:**
```
[1/7] Loading temporal seed data from: motor_test_newmodel_001_temporal_seed.parquet
✅ Loaded 10000 samples with 6 features

[2/7] Creating temporal metadata with RUL
✅ Metadata created: 4 numerical features + RUL

[3/7] Preparing training data
✅ Training data ready: (10000, 5)

[4/7] Training TVAE model with temporal seed data...
Epoch 1/300: 100%|██████████| 300/300 [Loss: 0.8234]
Epoch 50/300: 100%|██████████| 300/300 [Loss: 0.3421]
Epoch 100/300: 100%|██████████| 300/300 [Loss: 0.1892]
Epoch 300/300: 100%|██████████| 300/300 [Loss: 0.0534]
✅ Training complete

[5/7] Saving model
✅ Model saved: models/tvae/motor_test_newmodel_001_tvae.pkl

[6/7] Generating test samples
✅ Generated 1000 test samples

[7/7] Quality validation
✅ All sensors within expected ranges
✅ RUL correlations preserved (r=0.85-0.92)

Training time: 3.2 minutes
```

**Training Time Estimates:**
- Small model (4-8 sensors): 2-4 minutes
- Medium model (8-15 sensors): 4-6 minutes
- Large model (15+ sensors): 6-10 minutes

## Step 4: Generate Synthetic Dataset

Generate the full training/validation/test splits:

```bash
python scripts/generate_from_temporal_tvae.py motor_test_newmodel_001
```

**Expected Output:**
```
[1/5] Loading trained TVAE model
✅ Model loaded: motor_test_newmodel_001_tvae.pkl

[2/5] Generating training data (35,000 samples)
100%|██████████| 35000/35000 [00:45<00:00, 780.12it/s]
✅ Generated 35000 samples

[3/5] Generating validation data (7,500 samples)
100%|██████████| 7500/7500 [00:09<00:00, 785.34it/s]
✅ Generated 7500 samples

[4/5] Generating test data (7,500 samples)
100%|██████████| 7500/7500 [00:09<00:00, 782.91it/s]
✅ Generated 7500 samples

[5/5] Saving datasets
✅ Saved: data/synthetic_fixed/motor_test_newmodel_001/train.parquet
✅ Saved: data/synthetic_fixed/motor_test_newmodel_001/val.parquet
✅ Saved: data/synthetic_fixed/motor_test_newmodel_001/test.parquet

Generation time: 1.2 minutes
```

## Step 5: Validate Dataset Quality

Run the validation script to confirm temporal structure:

```bash
python validate_all_26_machines.py
```

Or validate just your new machine:

```python
import pandas as pd

machine = "motor_test_newmodel_001"
df = pd.read_parquet(f'data/synthetic_fixed/{machine}/train.parquet')

# Check 1: Timestamp sorted
print(f"Timestamp sorted: {df['timestamp'].is_monotonic_increasing}")

# Check 2: RUL decreasing
rul_dec = (df['rul'].diff()[1:] <= 0).sum()
rul_pct = rul_dec/(len(df)-1)*100
print(f"RUL decreasing: {rul_pct:.1f}%")

# Check 3: Range checks
print(f"RUL range: {df['rul'].max():.0f} → {df['rul'].min():.0f}")
print(f"Samples: {len(df)}")
```

**Expected Results:**
```
Timestamp sorted: True ✅
RUL decreasing: 100.0% ✅
RUL range: 1015 → 0 ✅
Samples: 35000 ✅
```

## Step 6: Ready for ML Training

Once validated, your new machine is ready for Phase 2.5 time-series training!

```bash
cd "c:\Projects\Predictive Maintenance\ml_models"
python scripts/train_timeseries.py --machine motor_test_newmodel_001
```

---

## Quick Reference: File Locations

```
GAN/
├── metadata/
│   └── {machine_id}_metadata.json          # Machine configuration
├── config/
│   └── rul_profiles.py                     # RUL settings (edit this)
├── seed_data/
│   └── temporal/
│       └── {machine_id}_temporal_seed.parquet  # Generated seed data
├── models/
│   └── tvae/
│       └── {machine_id}_tvae.pkl           # Trained TVAE model
└── data/
    └── synthetic_fixed/
        └── {machine_id}/
            ├── train.parquet               # 35K samples
            ├── val.parquet                 # 7.5K samples
            └── test.parquet                # 7.5K samples
```

## Troubleshooting

### Issue: Seed generation fails
- **Check:** Machine added to `config/rul_profiles.py`
- **Check:** Sensor names in metadata match RUL profile

### Issue: TVAE training fails
- **Check:** Temporal seed data exists in `seed_data/temporal/`
- **Check:** RUL column present in seed data
- **Check:** At least 2,000 samples in seed data

### Issue: Generated data quality poor
- **Check:** RUL correlations in `rul_profiles.py` are realistic
- **Check:** TVAE trained for 300 epochs (check loss < 0.1)
- **Check:** Seed data has >90% RUL decreasing

### Issue: Validation fails
- **Check:** Timestamps are chronologically sorted
- **Check:** RUL is >90% decreasing
- **Check:** Start RUL near max, end RUL near 0

---

## Summary Checklist

✅ **Step 1:** Create metadata JSON file
✅ **Step 2:** Add machine to `config/rul_profiles.py`
✅ **Step 3:** Generate temporal seed data (10K samples)
✅ **Step 4:** Train TVAE model (~3-5 minutes)
✅ **Step 5:** Generate synthetic dataset (~1-2 minutes)
✅ **Step 6:** Validate temporal structure
✅ **Step 7:** Ready for ML training!

**Total Time:** ~10-15 minutes per new machine

---

**Ready to test with your new machine profile!** 🚀

Just provide the machine details and I'll walk through the complete workflow.
