# 🔧 GAN Enhancement Tasks - Colleague Handoff Document
**Date:** November 18, 2025  
**Assigned To:** Colleague (Working on Copilot)  
**System:** i7-14650HX + RTX 4060 Laptop GPU  
**Estimated Time:** 3-5 days

---

## 📋 Executive Summary

You are assigned **TWO CRITICAL TASKS** to enhance the GAN synthetic data generation:

1. **RUL Label Generation** - Add Remaining Useful Life labels to all synthetic datasets
2. **Phase 1.5 Completion** - Implement new machine onboarding workflow

**IMPORTANT:** All changes must be made ONLY in the `GAN/` directory. Do NOT modify:
- ❌ `ml_models/` directory (ML training scripts)
- ❌ Existing trained models
- ❌ Project structure outside GAN/

---

## 🎯 Task 1: RUL Label Generation (Priority: CRITICAL)

### Background

**Problem Identified:** Current synthetic datasets lack RUL (Remaining Useful Life) labels, which are essential for regression-based predictive maintenance. Attempts to train regression models resulted in R² ≈ 0.0000 because no real RUL data exists.

**Current Dataset Structure:**
```
GAN/data/synthetic/<machine_id>/
├── train.parquet  (35,000 rows, 23 sensor columns, NO RUL)
├── val.parquet    (7,500 rows, 23 sensor columns, NO RUL)
└── test.parquet   (7,500 rows, 23 sensor columns, NO RUL)
```

**Target Dataset Structure:**
```
GAN/data/synthetic/<machine_id>/
├── train.parquet  (35,000 rows, 23 sensors + RUL column)
├── val.parquet    (7,500 rows, 23 sensors + RUL column)
└── test.parquet   (7,500 rows, 23 sensors + RUL column)
```

### What is RUL?

**RUL (Remaining Useful Life)** = Number of hours/cycles remaining until equipment failure or maintenance required.

**Examples from Industry:**
- NASA Turbofan: "Cycles until engine failure" (0-300 cycles)
- Medical: "Days until patient condition worsens" (0-30 days)
- Industrial: "Hours until bearing replacement needed" (0-1000 hours)

**Key Characteristics:**
- ✅ Decreases over time as equipment degrades
- ✅ Correlated with sensor readings (temperature ↑ → RUL ↓)
- ✅ Has realistic variance (not perfectly linear)
- ✅ Reaches 0 at failure point

---

## 📐 Phase 1.5.1: RUL Algorithm Design

### Recommended Approach: Time-Based Degradation with Sensor Correlation

**Algorithm Specification:**

```python
def generate_rul_labels(df, machine_metadata):
    """
    Generate RUL labels for synthetic dataset
    
    Args:
        df: DataFrame with sensor readings (35,000 rows for train)
        machine_metadata: Machine-specific parameters
    
    Returns:
        DataFrame with added 'rul' column
    """
    
    n_samples = len(df)
    
    # 1. Define equipment lifecycle
    max_rul = machine_metadata.get('max_operational_hours', 1000)
    
    # 2. Create time progression (0 = new equipment, 1 = end of life)
    time_index = np.linspace(0, 1, n_samples)
    
    # 3. Base RUL: Linear degradation from max_rul to 0
    base_rul = max_rul * (1 - time_index)
    
    # 4. Sensor-based adjustments (20-30% influence)
    sensor_factor = calculate_sensor_degradation(df, machine_metadata)
    
    # 5. Apply sensor adjustments
    rul = base_rul * (1 - 0.2 * sensor_factor)
    
    # 6. Add realistic noise (±10%)
    noise = np.random.normal(0, max_rul * 0.1, n_samples)
    rul = rul + noise
    
    # 7. Clip to valid range
    rul = np.clip(rul, 0, max_rul)
    
    return rul


def calculate_sensor_degradation(df, machine_metadata):
    """
    Calculate degradation factor from sensor readings
    
    Returns:
        Normalized degradation score (0 = healthy, 1 = degraded)
    """
    
    degradation = 0
    sensor_weights = {
        'temperature': 0.4,  # 40% weight
        'vibration': 0.4,    # 40% weight
        'current': 0.2       # 20% weight
    }
    
    # Temperature contribution
    temp_cols = [c for c in df.columns if 'temp' in c.lower()]
    if temp_cols:
        temp_mean = df[temp_cols].mean(axis=1)
        temp_norm = (temp_mean - temp_mean.min()) / (temp_mean.max() - temp_mean.min() + 1e-6)
        degradation += sensor_weights['temperature'] * temp_norm
    
    # Vibration contribution
    vib_cols = [c for c in df.columns if any(x in c.lower() for x in ['vib', 'velocity', 'rms'])]
    if vib_cols:
        vib_mean = df[vib_cols].mean(axis=1)
        vib_norm = (vib_mean - vib_mean.min()) / (vib_mean.max() - vib_mean.min() + 1e-6)
        degradation += sensor_weights['vibration'] * vib_norm
    
    # Current contribution
    current_cols = [c for c in df.columns if 'current' in c.lower()]
    if current_cols:
        current_mean = df[current_cols].mean(axis=1)
        current_norm = (current_mean - current_mean.min()) / (current_mean.max() - current_mean.min() + 1e-6)
        degradation += sensor_weights['current'] * current_norm
    
    return degradation
```

### Machine-Specific RUL Parameters

Add to each machine metadata file (`GAN/metadata/<machine_id>_metadata.json`):

```json
{
    "existing_fields": "...",
    "rul_parameters": {
        "max_operational_hours": 1000,
        "degradation_profile": "linear_with_sensor_correlation",
        "sensor_influence_percentage": 20,
        "noise_percentage": 10,
        "critical_sensors": ["temperature", "vibration"]
    }
}
```

**Machine Type Specific Values:**

| Machine Type | Max RUL (hours) | Degradation Profile | Critical Sensors |
|-------------|-----------------|---------------------|------------------|
| Electric Motors | 1000 | Linear | Temperature, Vibration |
| Pumps | 800 | Linear | Vibration, Pressure |
| Compressors | 1200 | Linear | Temperature, Pressure |
| CNC Machines | 500 | Accelerated | Vibration, Force |
| Hydraulic Systems | 600 | Linear | Pressure, Temperature |
| Fans | 1500 | Linear | Vibration, Temperature |
| Conveyors | 2000 | Linear | Vibration, Current |
| Robots | 800 | Accelerated | Joint Temp, Torque |
| Transformers | 5000 | Slow | Temperature, Load |
| Cooling Towers | 3000 | Linear | Temperature, Flow |
| Turbofan | 300 cycles | Accelerated | Multiple |

---

## 🛠️ Phase 1.5.2: Implementation Plan

### Step 1: Create RUL Generation Script

**File:** `GAN/scripts/add_rul_to_datasets.py`

```python
"""
Add RUL (Remaining Useful Life) labels to existing synthetic datasets
Phase 1.5.2: RUL Label Generation

This script adds RUL column to all synthetic datasets without regenerating sensor data
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_sensor_degradation(df, machine_type):
    """Calculate normalized degradation from sensors (0=healthy, 1=degraded)"""
    
    degradation = 0
    count = 0
    
    # Temperature contribution (40% weight)
    temp_cols = [c for c in df.columns if 'temp' in c.lower()]
    if temp_cols:
        temp_mean = df[temp_cols].mean(axis=1)
        temp_norm = (temp_mean - temp_mean.min()) / (temp_mean.max() - temp_mean.min() + 1e-6)
        degradation += 0.4 * temp_norm
        count += 1
    
    # Vibration contribution (40% weight)
    vib_cols = [c for c in df.columns if any(x in c.lower() for x in ['vib', 'velocity', 'rms', 'peak'])]
    if vib_cols:
        vib_mean = df[vib_cols].mean(axis=1)
        vib_norm = (vib_mean - vib_mean.min()) / (vib_mean.max() - vib_mean.min() + 1e-6)
        degradation += 0.4 * vib_norm
        count += 1
    
    # Current/Load contribution (20% weight)
    current_cols = [c for c in df.columns if any(x in c.lower() for x in ['current', 'load', 'power'])]
    if current_cols:
        current_mean = df[current_cols].mean(axis=1)
        current_norm = (current_mean - current_mean.min()) / (current_mean.max() - current_mean.min() + 1e-6)
        degradation += 0.2 * current_norm
        count += 1
    
    return degradation


def generate_rul_column(df, machine_type, max_rul=1000, sensor_influence=0.2, noise_pct=0.1):
    """
    Generate RUL labels with time-based degradation and sensor correlation
    
    Args:
        df: DataFrame with sensor data
        machine_type: Type of machine (e.g., 'motor', 'pump')
        max_rul: Maximum operational hours
        sensor_influence: Percentage of RUL influenced by sensors (0.2 = 20%)
        noise_pct: Noise percentage (0.1 = ±10%)
    
    Returns:
        Array of RUL values
    """
    
    n_samples = len(df)
    np.random.seed(42)  # For reproducibility
    
    # Time progression (0 = new, 1 = end of life)
    time_index = np.linspace(0, 1, n_samples)
    
    # Base RUL: Linear degradation
    base_rul = max_rul * (1 - time_index)
    
    # Sensor-based adjustments
    sensor_degradation = calculate_sensor_degradation(df, machine_type)
    
    # Apply sensor influence
    rul = base_rul * (1 - sensor_influence * sensor_degradation)
    
    # Add realistic noise
    noise = np.random.normal(0, max_rul * noise_pct, n_samples)
    rul = rul + noise
    
    # Clip to valid range
    rul = np.clip(rul, 0, max_rul)
    
    return rul


def get_machine_type(machine_id):
    """Extract machine type from machine ID"""
    if 'motor' in machine_id.lower():
        return 'motor', 1000
    elif 'pump' in machine_id.lower():
        return 'pump', 800
    elif 'compressor' in machine_id.lower():
        return 'compressor', 1200
    elif 'cnc' in machine_id.lower():
        return 'cnc', 500
    elif 'hydraulic' in machine_id.lower():
        return 'hydraulic', 600
    elif 'fan' in machine_id.lower():
        return 'fan', 1500
    elif 'conveyor' in machine_id.lower():
        return 'conveyor', 2000
    elif 'robot' in machine_id.lower():
        return 'robot', 800
    elif 'transformer' in machine_id.lower():
        return 'transformer', 5000
    elif 'cooling_tower' in machine_id.lower():
        return 'cooling_tower', 3000
    elif 'turbofan' in machine_id.lower():
        return 'turbofan', 300
    else:
        return 'generic', 1000


def add_rul_to_machine(machine_id, data_root, backup=True):
    """
    Add RUL column to all datasets (train, val, test) for a machine
    
    Args:
        machine_id: Machine identifier
        data_root: Path to GAN/data/synthetic
        backup: Whether to create backups before modifying
    
    Returns:
        dict: Statistics about RUL generation
    """
    
    machine_path = data_root / machine_id
    
    if not machine_path.exists():
        raise FileNotFoundError(f"Machine data not found: {machine_path}")
    
    machine_type, max_rul = get_machine_type(machine_id)
    
    print(f"\n{'='*70}")
    print(f"Processing: {machine_id}")
    print(f"Machine Type: {machine_type} | Max RUL: {max_rul} hours")
    print(f"{'='*70}")
    
    stats = {
        'machine_id': machine_id,
        'machine_type': machine_type,
        'max_rul': max_rul,
        'datasets': {}
    }
    
    # Process train, val, test
    for split in ['train', 'val', 'test']:
        file_path = machine_path / f"{split}.parquet"
        
        if not file_path.exists():
            print(f"⚠️  {split}.parquet not found, skipping...")
            continue
        
        print(f"\n[{split.upper()}] Loading data...")
        df = pd.read_parquet(file_path)
        original_shape = df.shape
        
        # Check if RUL already exists
        if 'rul' in df.columns:
            print(f"⚠️  RUL column already exists! Overwriting...")
            df = df.drop(columns=['rul'])
        
        # Backup if requested
        if backup:
            backup_path = machine_path / f"{split}_backup_no_rul.parquet"
            if not backup_path.exists():
                df.to_parquet(backup_path, index=False)
                print(f"✅ Backup created: {backup_path.name}")
        
        # Generate RUL
        print(f"[{split.upper()}] Generating RUL labels...")
        rul_values = generate_rul_column(df, machine_type, max_rul=max_rul)
        
        # Add RUL column
        df['rul'] = rul_values
        
        # Save updated dataset
        df.to_parquet(file_path, index=False)
        
        # Statistics
        rul_stats = {
            'min': float(rul_values.min()),
            'max': float(rul_values.max()),
            'mean': float(rul_values.mean()),
            'std': float(rul_values.std()),
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        stats['datasets'][split] = rul_stats
        
        print(f"[{split.upper()}] RUL Statistics:")
        print(f"  Min RUL:  {rul_stats['min']:.2f} hours")
        print(f"  Max RUL:  {rul_stats['max']:.2f} hours")
        print(f"  Mean RUL: {rul_stats['mean']:.2f} hours")
        print(f"  Std Dev:  {rul_stats['std']:.2f} hours")
        print(f"  Shape: {original_shape} → {df.shape}")
        print(f"✅ Saved: {file_path.name}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Add RUL labels to synthetic datasets')
    parser.add_argument('--machine_id', type=str, help='Specific machine ID (optional)')
    parser.add_argument('--no_backup', action='store_true', help='Skip backup creation')
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data' / 'synthetic'
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 1.5.2: RUL LABEL GENERATION")
    print("="*70)
    print(f"Data Directory: {data_root}")
    print(f"Backup: {'Disabled' if args.no_backup else 'Enabled'}")
    
    # Get list of machines
    if args.machine_id:
        machines = [args.machine_id]
    else:
        machines = [d.name for d in data_root.iterdir() if d.is_dir()]
        machines.sort()
    
    print(f"Machines to process: {len(machines)}")
    
    # Process all machines
    all_stats = []
    failed_machines = []
    
    for machine_id in tqdm(machines, desc="Processing machines"):
        try:
            stats = add_rul_to_machine(machine_id, data_root, backup=not args.no_backup)
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ ERROR processing {machine_id}: {e}")
            failed_machines.append({'machine_id': machine_id, 'error': str(e)})
    
    # Save report
    report = {
        'phase': '1.5.2',
        'task': 'RUL Label Generation',
        'total_machines': len(machines),
        'successful': len(all_stats),
        'failed': len(failed_machines),
        'machines': all_stats,
        'errors': failed_machines
    }
    
    report_path = reports_dir / 'rul_generation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("RUL GENERATION COMPLETED")
    print("="*70)
    print(f"✅ Successful: {len(all_stats)}/{len(machines)} machines")
    print(f"❌ Failed: {len(failed_machines)}/{len(machines)} machines")
    print(f"📊 Report saved: {report_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
```

### Step 2: Update Metadata Files

**File:** `GAN/scripts/add_rul_metadata.py`

```python
"""
Add RUL parameters to machine metadata files
Phase 1.5.2: RUL Metadata Enhancement
"""

import json
from pathlib import Path


RUL_PARAMETERS = {
    'motor': {
        'max_operational_hours': 1000,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['temperature', 'vibration']
    },
    'pump': {
        'max_operational_hours': 800,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['vibration', 'pressure']
    },
    'compressor': {
        'max_operational_hours': 1200,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['temperature', 'pressure']
    },
    'cnc': {
        'max_operational_hours': 500,
        'degradation_profile': 'accelerated',
        'sensor_influence_percentage': 30,
        'noise_percentage': 15,
        'critical_sensors': ['vibration', 'force']
    },
    'hydraulic': {
        'max_operational_hours': 600,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['pressure', 'temperature']
    },
    'fan': {
        'max_operational_hours': 1500,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['vibration', 'temperature']
    },
    'conveyor': {
        'max_operational_hours': 2000,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 15,
        'noise_percentage': 10,
        'critical_sensors': ['vibration', 'current']
    },
    'robot': {
        'max_operational_hours': 800,
        'degradation_profile': 'accelerated',
        'sensor_influence_percentage': 25,
        'noise_percentage': 12,
        'critical_sensors': ['joint_temp', 'torque']
    },
    'transformer': {
        'max_operational_hours': 5000,
        'degradation_profile': 'slow_degradation',
        'sensor_influence_percentage': 15,
        'noise_percentage': 8,
        'critical_sensors': ['temperature', 'load']
    },
    'cooling_tower': {
        'max_operational_hours': 3000,
        'degradation_profile': 'linear_with_sensor_correlation',
        'sensor_influence_percentage': 20,
        'noise_percentage': 10,
        'critical_sensors': ['temperature', 'flow']
    },
    'turbofan': {
        'max_operational_hours': 300,  # cycles
        'degradation_profile': 'accelerated',
        'sensor_influence_percentage': 30,
        'noise_percentage': 15,
        'critical_sensors': ['temperature', 'vibration', 'pressure']
    }
}


def get_machine_type(machine_id):
    """Extract machine type from machine ID"""
    for machine_type in RUL_PARAMETERS.keys():
        if machine_type in machine_id.lower():
            return machine_type
    return 'generic'


def add_rul_to_metadata(metadata_path):
    """Add RUL parameters to metadata file"""
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    machine_id = metadata_path.stem.replace('_metadata', '')
    machine_type = get_machine_type(machine_id)
    
    if machine_type in RUL_PARAMETERS:
        rul_params = RUL_PARAMETERS[machine_type]
    else:
        rul_params = RUL_PARAMETERS['motor']  # Default
    
    metadata['rul_parameters'] = rul_params
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Updated: {metadata_path.name} (Type: {machine_type})")


def main():
    project_root = Path(__file__).parent.parent
    metadata_dir = project_root / 'metadata'
    
    print("\n" + "="*70)
    print("ADDING RUL PARAMETERS TO METADATA FILES")
    print("="*70 + "\n")
    
    metadata_files = list(metadata_dir.glob('*_metadata.json'))
    
    for metadata_path in metadata_files:
        add_rul_to_metadata(metadata_path)
    
    print(f"\n✅ Updated {len(metadata_files)} metadata files\n")


if __name__ == '__main__':
    main()
```

---

## 🚀 Task 2: Phase 1.5 - New Machine Workflow

### Phase 1.5 Overview

**Goal:** Create a streamlined workflow for adding new machines to the system without requiring Phase 1 expertise.

**Files to Create:**

1. `GAN/PHASE_1.5_NEW_MACHINE_GUIDE.md` - User guide
2. `GAN/scripts/add_new_machine.py` - Automated onboarding script
3. `GAN/templates/machine_metadata_template.json` - Template for new machines
4. `GAN/scripts/validate_new_machine.py` - Validation script

### Step 1: Create New Machine Guide

**File:** `GAN/PHASE_1.5_NEW_MACHINE_GUIDE.md`

```markdown
# Phase 1.5: New Machine Onboarding Guide
**Simplified workflow for adding new machines to the predictive maintenance system**

---

## Quick Start (5 Steps)

### Step 1: Prepare Machine Information

Collect the following information:
- Machine ID (e.g., `motor_siemens_1la7_002`)
- Machine type (motor, pump, compressor, etc.)
- Sensor list with types (temperature, vibration, etc.)
- Operating specifications (voltage, speed, capacity)

### Step 2: Create Seed Data

Place small real dataset (100-1000 rows) in:
```
GAN/seed_data/<machine_id>_seed.parquet
```

Or generate from specifications:
```bash
python scripts/generate_seed_from_specs.py --machine_id motor_siemens_1la7_002 --type motor
```

### Step 3: Run Automated Onboarding

```bash
python scripts/add_new_machine.py --machine_id motor_siemens_1la7_002
```

This will:
1. Create metadata file
2. Train TVAE model
3. Generate synthetic datasets (train/val/test)
4. Add RUL labels
5. Validate quality

### Step 4: Validate Results

```bash
python scripts/validate_new_machine.py --machine_id motor_siemens_1la7_002
```

Expected output:
- ✅ Quality score > 0.85
- ✅ Train: 35,000 rows
- ✅ Val: 7,500 rows  
- ✅ Test: 7,500 rows
- ✅ RUL labels present

### Step 5: Train ML Models

After GAN work is complete, send datasets to colleague for ML training:
```
Send: GAN/data/synthetic/<machine_id>/
They will train classification and regression models
```

---

## Detailed Workflow

[Continue with detailed instructions...]
```

### Step 2: Create Automated Onboarding Script

**File:** `GAN/scripts/add_new_machine.py`

```python
"""
Phase 1.5: Automated New Machine Onboarding
Streamlined workflow for adding new machines
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import sys

# Import RUL generation functions
sys.path.append(str(Path(__file__).parent))
from add_rul_to_datasets import generate_rul_column, get_machine_type


def create_metadata(machine_id, seed_data_path):
    """Generate metadata from seed data"""
    
    print("[1/6] Creating metadata...")
    
    df = pd.read_parquet(seed_data_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Save metadata
    metadata_dir = Path(__file__).parent.parent / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    metadata_path = metadata_dir / f"{machine_id}_metadata.json"
    metadata.save_to_json(str(metadata_path))
    
    print(f"✅ Metadata created: {metadata_path.name}")
    return metadata


def train_tvae(machine_id, seed_data_path, metadata, epochs=300):
    """Train TVAE model"""
    
    print("[2/6] Training TVAE model...")
    
    df = pd.read_parquet(seed_data_path)
    
    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=500,
        cuda=True,  # Use RTX 4060 GPU
        verbose=True
    )
    
    start = time.time()
    synthesizer.fit(df)
    train_time = time.time() - start
    
    # Save model
    models_dir = Path(__file__).parent.parent / 'models' / 'tvae'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{machine_id}_tvae_{epochs}epochs.pkl"
    synthesizer.save(str(model_path))
    
    print(f"✅ Model trained in {train_time/60:.2f} minutes")
    return synthesizer


def generate_datasets(machine_id, synthesizer):
    """Generate train/val/test splits"""
    
    print("[3/6] Generating synthetic datasets...")
    
    output_dir = Path(__file__).parent.parent / 'data' / 'synthetic' / machine_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate splits
    splits = {
        'train': 35000,
        'val': 7500,
        'test': 7500
    }
    
    for split_name, n_rows in splits.items():
        print(f"  Generating {split_name}: {n_rows} rows...")
        data = synthesizer.sample(num_rows=n_rows)
        output_path = output_dir / f"{split_name}.parquet"
        data.to_parquet(output_path, index=False)
        print(f"  ✅ Saved: {output_path.name}")
    
    return output_dir


def add_rul_labels(machine_id, data_dir):
    """Add RUL labels to datasets"""
    
    print("[4/6] Adding RUL labels...")
    
    machine_type, max_rul = get_machine_type(machine_id)
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f"{split}.parquet"
        df = pd.read_parquet(file_path)
        
        rul_values = generate_rul_column(df, machine_type, max_rul=max_rul)
        df['rul'] = rul_values
        
        df.to_parquet(file_path, index=False)
        print(f"  ✅ RUL added to {split} (mean: {rul_values.mean():.1f} hours)")


def validate_quality(seed_data, synthetic_data, metadata):
    """Validate synthetic data quality"""
    
    print("[5/6] Validating quality...")
    
    quality_report = evaluate_quality(seed_data, synthetic_data, metadata)
    score = quality_report.get_score()
    
    print(f"  Quality Score: {score:.3f}")
    
    if score >= 0.85:
        print(f"  ✅ EXCELLENT - Meets quality standards")
        return True
    elif score >= 0.75:
        print(f"  ⚠️  ACCEPTABLE - Consider retraining with more epochs")
        return True
    else:
        print(f"  ❌ POOR - Retraining recommended")
        return False


def generate_report(machine_id, stats):
    """Generate completion report"""
    
    print("[6/6] Generating report...")
    
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / f"{machine_id}_onboarding_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Report saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Add new machine to system')
    parser.add_argument('--machine_id', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"PHASE 1.5: ONBOARDING NEW MACHINE - {args.machine_id}")
    print("="*70 + "\n")
    
    # Paths
    project_root = Path(__file__).parent.parent
    seed_path = project_root / 'seed_data' / f"{args.machine_id}_seed.parquet"
    
    if not seed_path.exists():
        print(f"❌ ERROR: Seed data not found at {seed_path}")
        return
    
    seed_data = pd.read_parquet(seed_path)
    
    # Execute workflow
    start_time = time.time()
    
    metadata = create_metadata(args.machine_id, seed_path)
    synthesizer = train_tvae(args.machine_id, seed_path, metadata, epochs=args.epochs)
    data_dir = generate_datasets(args.machine_id, synthesizer)
    add_rul_labels(args.machine_id, data_dir)
    
    # Validate
    test_data = pd.read_parquet(data_dir / 'test.parquet')
    quality_ok = validate_quality(seed_data, test_data, metadata)
    
    total_time = time.time() - start_time
    
    # Report
    stats = {
        'machine_id': args.machine_id,
        'status': 'completed' if quality_ok else 'needs_review',
        'total_time_minutes': round(total_time / 60, 2),
        'datasets_generated': {
            'train': 35000,
            'val': 7500,
            'test': 7500
        },
        'rul_added': True
    }
    
    generate_report(args.machine_id, stats)
    
    print("\n" + "="*70)
    print("ONBOARDING COMPLETED")
    print("="*70)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Next step: Send GAN/data/synthetic/{args.machine_id}/ to ML team")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
```

---

## 📊 Execution Plan

### Timeline

| Phase | Task | Duration | GPU Usage |
|-------|------|----------|-----------|
| **Week 1** | RUL Implementation | 2-3 days | Low (testing only) |
| **Week 1-2** | Run RUL generation on all 21 machines | 1-2 hours | None (CPU) |
| **Week 2** | Phase 1.5 scripts | 2-3 days | Medium (testing) |
| **Week 2** | Documentation & validation | 1 day | None |

### Detailed Schedule

**Day 1-2: RUL Implementation**
- ✅ Create `add_rul_to_datasets.py` (2 hours)
- ✅ Create `add_rul_metadata.py` (1 hour)
- ✅ Test on 2 machines (1 hour)
- ✅ Review and adjust algorithm (2 hours)

**Day 3: RUL Batch Processing**
- ✅ Update all 21 metadata files (10 minutes)
- ✅ Run RUL generation on all machines (1-2 hours)
- ✅ Validate results (30 minutes)
- ✅ Generate quality report (30 minutes)

**Day 4-5: Phase 1.5 Implementation**
- ✅ Create new machine guide (2 hours)
- ✅ Create `add_new_machine.py` (3 hours)
- ✅ Create validation scripts (2 hours)
- ✅ Create templates (1 hour)
- ✅ Test complete workflow (2 hours)

**Day 6: Documentation & Handoff**
- ✅ Final validation
- ✅ Document changes
- ✅ Prepare handoff package

---

## 🔒 Critical Rules - READ CAREFULLY

### ✅ DO:

1. **ONLY modify files in `GAN/` directory**
2. **Create backups before modifying datasets** (use `--no_backup` flag to skip)
3. **Test on 1-2 machines first** before batch processing all 21
4. **Validate quality scores** after RUL generation
5. **Use GPU acceleration** for TVAE training (`cuda=True`)
6. **Follow naming conventions** exactly as specified
7. **Generate reports** for all operations

### ❌ DO NOT:

1. **DO NOT modify `ml_models/` directory** - This contains trained models
2. **DO NOT delete existing synthetic data** without backups
3. **DO NOT change project structure** outside GAN/
4. **DO NOT modify trained TVAE models** - Only create new ones
5. **DO NOT change file paths** - Use absolute paths from project root
6. **DO NOT skip validation** - Always validate after generation

---

## 📦 Deliverables Checklist

When sending completed work back:

### RUL Task:
- [ ] All 21 machines have RUL column in train/val/test
- [ ] All metadata files updated with RUL parameters
- [ ] `GAN/reports/rul_generation_report.json` created
- [ ] Validation passed (no errors)
- [ ] Backups created (optional but recommended)

### Phase 1.5 Task:
- [ ] `PHASE_1.5_NEW_MACHINE_GUIDE.md` created
- [ ] `scripts/add_new_machine.py` created and tested
- [ ] `scripts/validate_new_machine.py` created
- [ ] `scripts/add_rul_metadata.py` created
- [ ] `templates/machine_metadata_template.json` created
- [ ] Successfully tested on 1 new test machine

### Package Structure:
```
GAN/
├── PHASE_1.5_NEW_MACHINE_GUIDE.md ✅ NEW
├── scripts/
│   ├── add_rul_to_datasets.py ✅ NEW
│   ├── add_rul_metadata.py ✅ NEW
│   ├── add_new_machine.py ✅ NEW
│   └── validate_new_machine.py ✅ NEW
├── metadata/
│   └── *_metadata.json (all updated with RUL params) ✅ MODIFIED
├── data/synthetic/
│   └── <all 21 machines>/ (all have RUL column) ✅ MODIFIED
└── reports/
    └── rul_generation_report.json ✅ NEW
```

---

## 💻 Hardware Optimization Tips

Your system: **i7-14650HX + RTX 4060 Laptop GPU**

### GPU Settings for TVAE Training:

```python
# Optimal settings for RTX 4060 Laptop
synthesizer = TVAESynthesizer(
    metadata=metadata,
    epochs=300,
    batch_size=500,  # Good for 8GB VRAM
    cuda=True,       # Enable GPU
    verbose=True
)
```

### Expected Performance:

| Task | Time (Single Machine) | GPU Usage |
|------|----------------------|-----------|
| TVAE Training (300 epochs) | 10-15 seconds | 60-80% |
| RUL Generation (50K rows) | 1-2 seconds | N/A (CPU) |
| Quality Validation | 5-10 seconds | N/A (CPU) |

### Batch Processing:

- **21 machines RUL generation:** ~1-2 hours (mostly I/O)
- **New machine onboarding:** ~2-3 minutes per machine
- **Can process in parallel** if needed

---

## 🆘 Support & Questions

### Common Issues:

**Issue: CUDA out of memory**
```python
# Solution: Reduce batch size
batch_size=250  # Instead of 500
```

**Issue: Quality score < 0.75**
```python
# Solution: Increase epochs
epochs=500  # Instead of 300
```

**Issue: RUL values look wrong**
```python
# Check: Verify sensor columns detected
# Look for: temp_cols, vib_cols in output
# Adjust: sensor_influence parameter
```

### Contact Points:

- **Your Original Copilot Session**: For ML models questions
- **This Copilot Session**: For GAN/RUL questions
- **Project Documentation**: `PHASE_1_GAN_DETAILED_APPROACH.md`

---

## ✅ Final Notes

1. **Test incrementally** - Don't process all 21 machines at once
2. **Keep backups** - Original datasets are valuable
3. **Document issues** - Note any anomalies in reports
4. **Validate thoroughly** - Quality > Speed
5. **Ask questions** - Better to clarify than assume

**Estimated Total Time:** 3-5 days  
**Critical Path:** RUL generation → Validation → Phase 1.5 scripts → Testing

**Success Criteria:**
- All 21 machines have RUL labels ✅
- Quality scores maintained (>0.85) ✅
- New machine workflow tested ✅
- Documentation complete ✅

---

**Good luck! 🚀**
```
