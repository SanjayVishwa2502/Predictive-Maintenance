"""
Feature Engineering Utilities for Phase 2.1.2
CRITICAL: Features must work for ALL machines (not machine-specific)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_machine_metadata(machine_id):
    """Load machine metadata from GAN metadata files"""
    # Use absolute path from project root (works from any subdirectory)
    project_root = Path(__file__).parent.parent.parent.parent
    metadata_path = project_root / 'GAN' / 'metadata' / f'{machine_id}_metadata.json'
    
    if not metadata_path.exists():
        print(f"Warning: Metadata not found for {machine_id}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def add_machine_metadata_features(df, machine_id):
    """
    Add machine metadata as features for generic model
    This allows model to differentiate between machine types
    """
    
    # Parse machine_id to extract info
    parts = machine_id.split('_')
    category = parts[0] if len(parts) > 0 else 'unknown'
    manufacturer = parts[1] if len(parts) > 1 else 'unknown'
    
    # Add categorical features (will be one-hot encoded by AutoGluon)
    df['machine_category'] = category  # motor, pump, compressor, etc.
    df['manufacturer'] = manufacturer
    
    # Map categories to typical power ratings (placeholder - can be enhanced)
    power_mapping = {
        'motor': 75.0,
        'pump': 50.0,
        'compressor': 100.0,
        'fan': 30.0,
        'cnc': 150.0,
        'hydraulic': 80.0,
        'conveyor': 25.0,
        'robot': 40.0,
        'transformer': 200.0,
        'cooling': 60.0,
        'turbofan': 500.0
    }
    
    # Add numerical metadata features
    df['power_rating_kw'] = power_mapping.get(category, 50.0)
    df['rated_speed_rpm'] = 1500.0 if category in ['motor', 'pump', 'fan'] else 0.0
    df['operating_voltage'] = 480.0
    df['equipment_age_years'] = 5.0
    
    return df

def add_normalized_sensor_features(df):
    """
    Create normalized features that work across machine types
    """
    
    # Generic sensor aggregations (works for any machine)
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    vib_cols = [col for col in df.columns if 'vib' in col.lower() or 'velocity' in col.lower()]
    current_cols = [col for col in df.columns if 'current' in col.lower()]
    
    # Temperature features
    if temp_cols:
        df['temp_mean_normalized'] = df[temp_cols].mean(axis=1)
        df['temp_max_normalized'] = df[temp_cols].max(axis=1)
        df['temp_std'] = df[temp_cols].std(axis=1)
        df['temp_range'] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)
    
    # Vibration features
    if vib_cols:
        df['vib_rms'] = np.sqrt((df[vib_cols] ** 2).mean(axis=1))
        df['vib_peak'] = df[vib_cols].max(axis=1)
        df['vib_mean'] = df[vib_cols].mean(axis=1)
        df['vib_std'] = df[vib_cols].std(axis=1)
    
    # Current features
    if current_cols:
        df['current_mean'] = df[current_cols].mean(axis=1)
        df['current_max'] = df[current_cols].max(axis=1)
        df['current_std'] = df[current_cols].std(axis=1)
    
    # Health score (0-100) - generic across machines
    df['health_score'] = calculate_health_score(df, temp_cols, vib_cols, current_cols)
    
    return df

def calculate_health_score(df, temp_cols, vib_cols, current_cols):
    """
    Calculate a generic health score (0-100) based on sensor readings
    Lower score = worse health
    OPTIMIZED: Vectorized operations instead of row-by-row loop
    """
    score = pd.Series(100.0, index=df.index)  # Start with perfect health
    
    # Temperature penalty (vectorized)
    if temp_cols:
        temp_max = df[temp_cols].max(axis=1)
        temp_threshold_warn = df[temp_cols].quantile(0.75).max()
        temp_threshold_critical = df[temp_cols].quantile(0.90).max()
        
        score = score - 30 * (temp_max > temp_threshold_critical).astype(int)
        score = score - 15 * ((temp_max > temp_threshold_warn) & (temp_max <= temp_threshold_critical)).astype(int)
    
    # Vibration penalty (vectorized)
    if vib_cols:
        vib_max = df[vib_cols].max(axis=1)
        vib_threshold_warn = df[vib_cols].quantile(0.75).max()
        vib_threshold_critical = df[vib_cols].quantile(0.90).max()
        
        score = score - 30 * (vib_max > vib_threshold_critical).astype(int)
        score = score - 15 * ((vib_max > vib_threshold_warn) & (vib_max <= vib_threshold_critical)).astype(int)
    
    # Current penalty (vectorized)
    if current_cols:
        current_max = df[current_cols].max(axis=1)
        current_threshold_warn = df[current_cols].quantile(0.75).max()
        current_threshold_critical = df[current_cols].quantile(0.90).max()
        
        score = score - 20 * (current_max > current_threshold_critical).astype(int)
        score = score - 10 * ((current_max > current_threshold_warn) & (current_max <= current_threshold_critical)).astype(int)
    
    # Ensure non-negative
    score = score.clip(lower=0)
    
    return score

def add_engineered_features(df, machine_id):
    """
    Add all engineered features to dataframe
    """
    df = add_machine_metadata_features(df.copy(), machine_id)
    df = add_normalized_sensor_features(df)
    return df

def prepare_ml_data(machine_id, task_type='classification'):
    """Prepare data for specific ML task"""
    
    # Use absolute path from project root (works from any subdirectory)
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    gan_data_path = project_root / 'GAN' / 'data' / 'synthetic' / machine_id
    
    # Load splits
    train_df = pd.read_parquet(gan_data_path / 'train.parquet')
    val_df = pd.read_parquet(gan_data_path / 'val.parquet')
    test_df = pd.read_parquet(gan_data_path / 'test.parquet')
    
    # Add engineered features
    train_df = add_engineered_features(train_df, machine_id)
    val_df = add_engineered_features(val_df, machine_id)
    test_df = add_engineered_features(test_df, machine_id)
    
    # Create target variable based on task type
    if task_type == 'classification':
        # Binary: normal vs failure
        if 'failure_status' not in train_df.columns:
            # ⚠️ CRITICAL: Calculate thresholds ONLY from training data to prevent data leakage
            train_df['failure_status'], train_thresholds = create_failure_labels(train_df, machine_id, train_thresholds=None)
            # Use training thresholds for val/test to prevent leakage
            val_df['failure_status'], _ = create_failure_labels(val_df, machine_id, train_thresholds=train_thresholds)
            test_df['failure_status'], _ = create_failure_labels(test_df, machine_id, train_thresholds=train_thresholds)
    
    elif task_type == 'regression':
        # RUL prediction
        if 'rul' not in train_df.columns:
            train_df['rul'] = create_rul_labels(train_df, machine_id)
            val_df['rul'] = create_rul_labels(val_df, machine_id)
            test_df['rul'] = create_rul_labels(test_df, machine_id)
    
    return train_df, val_df, test_df

def create_failure_labels(df, machine_id, train_thresholds=None):
    """
    Create failure labels based on sensor thresholds
    
    Args:
        df: DataFrame with sensor data
        machine_id: Machine identifier
        train_thresholds: Dict of thresholds from training data (to prevent data leakage)
                         If None, calculate from current df (only for training data!)
    
    Returns:
        failure_status: Binary labels (0=normal, 1=failure)
    """
    
    # ⚠️ CRITICAL: To prevent data leakage, thresholds MUST come from training data only!
    if train_thresholds is None:
        # Only use this mode when creating thresholds from TRAINING data
        train_thresholds = {}
        
        # Temperature threshold
        temp_cols = [col for col in df.columns if 'temperature' in col.lower() or 'temp' in col.lower()]
        if temp_cols:
            train_thresholds['temp_threshold'] = df[temp_cols].quantile(0.90).max()
            train_thresholds['temp_cols'] = temp_cols
        
        # Vibration threshold
        vib_cols = [col for col in df.columns if 'vibration' in col.lower() or 'velocity' in col.lower()]
        if vib_cols:
            train_thresholds['vib_threshold'] = df[vib_cols].quantile(0.90).max()
            train_thresholds['vib_cols'] = vib_cols
        
        # Current threshold
        current_cols = [col for col in df.columns if 'current' in col.lower()]
        if current_cols:
            train_thresholds['current_threshold'] = df[current_cols].quantile(0.90).max()
            train_thresholds['current_cols'] = current_cols
    
    # Apply thresholds
    failure_score = 0
    
    if 'temp_threshold' in train_thresholds:
        temp_high = df[train_thresholds['temp_cols']].max(axis=1) > train_thresholds['temp_threshold']
        failure_score += temp_high.astype(int)
    
    if 'vib_threshold' in train_thresholds:
        vib_high = df[train_thresholds['vib_cols']].max(axis=1) > train_thresholds['vib_threshold']
        failure_score += vib_high.astype(int)
    
    if 'current_threshold' in train_thresholds:
        current_high = df[train_thresholds['current_cols']].max(axis=1) > train_thresholds['current_threshold']
        failure_score += current_high.astype(int)
    
    # Binary classification: failure if ANY threshold exceeded
    failure_status = (failure_score >= 1).astype(int)
    
    return failure_status, train_thresholds

def create_rul_labels(df, machine_id):
    """Create RUL (Remaining Useful Life) labels"""
    # Simple linear degradation model
    # In production, use domain expertise or historical data
    
    max_rul = 1000  # Maximum hours
    
    # Calculate degradation based on sensor values
    degradation_score = 0
    count = 0
    
    # Temperature degradation
    temp_cols = [col for col in df.columns if 'temperature' in col.lower() or 'temp' in col.lower()]
    if temp_cols:
        temp_min = df[temp_cols].min().min()
        temp_max = df[temp_cols].max().max()
        if temp_max > temp_min:
            temp_norm = (df[temp_cols].mean(axis=1) - temp_min) / (temp_max - temp_min)
            degradation_score += temp_norm
            count += 1
    
    # Vibration degradation
    vib_cols = [col for col in df.columns if 'vibration' in col.lower() or 'velocity' in col.lower()]
    if vib_cols:
        vib_min = df[vib_cols].min().min()
        vib_max = df[vib_cols].max().max()
        if vib_max > vib_min:
            vib_norm = (df[vib_cols].mean(axis=1) - vib_min) / (vib_max - vib_min)
            degradation_score += vib_norm
            count += 1
    
    # Current degradation
    current_cols = [col for col in df.columns if 'current' in col.lower()]
    if current_cols:
        current_min = df[current_cols].min().min()
        current_max = df[current_cols].max().max()
        if current_max > current_min:
            current_norm = (df[current_cols].mean(axis=1) - current_min) / (current_max - current_min)
            degradation_score += current_norm
            count += 1
    
    # Average degradation
    if count > 0:
        degradation_score = degradation_score / count
    else:
        degradation_score = 0
    
    # RUL decreases with degradation
    rul = max_rul * (1 - degradation_score)
    rul = rul.clip(0, max_rul)
    
    return rul

def prepare_pooled_data_for_task(task_type='classification'):
    """
    Prepare pooled data for specific ML task
    Loads pooled datasets and adds task-specific target labels
    """
    
    print(f"Preparing pooled data for {task_type}...")
    
    # Load pooled data
    pooled_train = pd.read_parquet('../data/processed/pooled_train.parquet')
    pooled_val = pd.read_parquet('../data/processed/pooled_val.parquet')
    pooled_test = pd.read_parquet('../data/processed/pooled_test.parquet')
    
    print(f"Loaded pooled train: {len(pooled_train):,} samples")
    print(f"Loaded pooled val: {len(pooled_val):,} samples")
    print(f"Loaded pooled test: {len(pooled_test):,} samples")
    
    # Add task-specific labels if not present
    if task_type == 'classification':
        if 'failure_status' not in pooled_train.columns:
            print("Creating failure labels...")
            pooled_train['failure_status'] = create_failure_labels(pooled_train, 'pooled')
            pooled_val['failure_status'] = create_failure_labels(pooled_val, 'pooled')
            pooled_test['failure_status'] = create_failure_labels(pooled_test, 'pooled')
            
            print(f"Failure distribution (train): {pooled_train['failure_status'].value_counts().to_dict()}")
    
    elif task_type == 'regression':
        if 'rul' not in pooled_train.columns:
            print("Creating RUL labels...")
            pooled_train['rul'] = create_rul_labels(pooled_train, 'pooled')
            pooled_val['rul'] = create_rul_labels(pooled_val, 'pooled')
            pooled_test['rul'] = create_rul_labels(pooled_test, 'pooled')
            
            print(f"RUL range (train): [{pooled_train['rul'].min():.2f}, {pooled_train['rul'].max():.2f}]")
    
    return pooled_train, pooled_val, pooled_test
