"""
Phase 2.5.1: Time-Series Forecasting - Machine-Specific Models
================================================================

Approach:
- Machine-specific LSTM models (one per machine)
- Adaptive target selection: <7 sensors → all; ≥7 sensors → 6 core
- Synthetic timestamps (hourly intervals assumed)
- 168-hour lookback → 24-hour forecast
- Metrics: RMSE, MAE, MAPE, SMAPE

Author: Phase 2.5 Implementation
Date: 2025-11-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
print(f"TensorFlow version: {tf.__version__}")
print("⚠️  Training optimized for CPU (reduced model size for speed)")
print("   Estimated time: ~2-3 minutes per machine")

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ML_MODELS_ROOT = PROJECT_ROOT / 'ml_models'
GAN_DATA_DIR = PROJECT_ROOT / 'GAN' / 'data' / 'synthetic'  # Machine-specific data
SAVE_DIR = ML_MODELS_ROOT / 'models' / 'timeseries'
REPORTS_DIR = ML_MODELS_ROOT / 'reports'

# Time-series parameters (optimized for CPU speed)
LOOKBACK_HOURS = 72       # 3 days of history (reduced from 7 for speed)
FORECAST_HOURS = 24       # Predict next 24 hours
BATCH_SIZE = 64           # Larger batch = faster training
EPOCHS = 30               # Reduced epochs for speed
LEARNING_RATE = 0.001

# Sensor categories for adaptive selection
CORE_SENSOR_KEYWORDS = [
    'temp', 'temperature',
    'pressure', 'psi', 'bar',
    'vibration', 'rms', 'mm_s',
    'current', '_a', '_amp',
    'voltage', '_v', '_volt',
    'power', '_kw', '_w'
]

# Metadata columns (not sensors)
METADATA_COLS = [
    'machine_id', 'machine_category', 'manufacturer',
    'power_rating_kw', 'rated_speed_rpm', 'operating_voltage',
    'equipment_age_years'
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def identify_sensors(df):
    """
    Identify active sensors from machine-specific dataframe
    
    Returns:
        list: Active sensor column names
    """
    all_cols = [c for c in df.columns if c not in METADATA_COLS]
    
    # Only include sensors with non-null values
    active_sensors = [
        col for col in all_cols 
        if df[col].notna().any()
    ]
    
    return active_sensors


def select_target_sensors(active_sensors, threshold=7):
    """
    Adaptive target selection based on sensor count
    
    Args:
        active_sensors: List of active sensor names
        threshold: Sensor count threshold for selection logic
        
    Returns:
        list: Selected target sensors
    """
    n_sensors = len(active_sensors)
    
    if n_sensors < threshold:
        # Predict all sensors
        return active_sensors
    
    else:
        # Select 6 core sensors based on keywords
        selected = []
        
        for keyword_group in [
            ['temp', 'temperature'],
            ['pressure', 'psi', 'bar'],
            ['vibration', 'rms', 'mm_s'],
            ['current', '_a'],
            ['voltage', '_v'],
            ['power', '_kw', '_w']
        ]:
            # Find first sensor matching this keyword group
            for sensor in active_sensors:
                if sensor in selected:
                    continue
                if any(kw in sensor.lower() for kw in keyword_group):
                    selected.append(sensor)
                    break
        
        # If we didn't get 6, fill with remaining sensors
        if len(selected) < 6:
            remaining = [s for s in active_sensors if s not in selected]
            selected.extend(remaining[:6 - len(selected)])
        
        return selected[:6]


def create_sequences(data, lookback, forecast):
    """
    Create sliding window sequences for time-series
    
    Args:
        data: numpy array (timesteps, features)
        lookback: number of past timesteps to use
        forecast: number of future timesteps to predict
        
    Returns:
        X, y: Input sequences and target sequences
    """
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[(i + lookback):(i + lookback + forecast)])
    
    return np.array(X), np.array(y)


def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true, y_pred, epsilon=1e-10):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator > epsilon
    return np.mean(numerator[mask] / denominator[mask]) * 100


def build_lstm_model(input_shape, output_shape, learning_rate):
    """
    Build LSTM model for time-series forecasting
    
    Args:
        input_shape: (lookback, n_features)
        output_shape: (forecast, n_targets)
        learning_rate: optimizer learning rate
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First LSTM layer (reduced size for speed)
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        
        # Second LSTM layer (reduced size for speed)
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layer for forecast
        layers.Dense(32, activation='relu'),
        
        # Output layer: reshape to (forecast, n_targets)
        layers.Dense(output_shape[0] * output_shape[1]),
        layers.Reshape(output_shape)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_timeseries_model(machine_id):
    """
    Train machine-specific time-series forecasting model
    
    Args:
        machine_id: Machine identifier
        
    Returns:
        dict: Training metrics and model info
    """
    print(f"\n{'='*80}")
    print(f"Training Time-Series Model: {machine_id}")
    print(f"{'='*80}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load Data (Machine-Specific)
    # -------------------------------------------------------------------------
    print("Step 1: Loading machine-specific data...")
    
    machine_dir = GAN_DATA_DIR / machine_id
    
    if not machine_dir.exists():
        raise FileNotFoundError(f"Machine directory not found: {machine_dir}")
    
    train_machine = pd.read_parquet(machine_dir / 'train.parquet')
    val_machine = pd.read_parquet(machine_dir / 'val.parquet')
    test_machine = pd.read_parquet(machine_dir / 'test.parquet')
    
    print(f"  Source: {machine_dir}")
    print(f"  Train: {len(train_machine)} samples")
    print(f"  Val:   {len(val_machine)} samples")
    print(f"  Test:  {len(test_machine)} samples")
    
    # -------------------------------------------------------------------------
    # 2. Identify and Select Sensors
    # -------------------------------------------------------------------------
    print("\nStep 2: Sensor selection...")
    
    active_sensors = identify_sensors(train_machine)
    target_sensors = select_target_sensors(active_sensors)
    
    print(f"  Active sensors: {len(active_sensors)}")
    print(f"  Selected targets: {len(target_sensors)}")
    print(f"  Targets: {target_sensors}")
    
    # -------------------------------------------------------------------------
    # 3. Prepare Data (Add synthetic timestamps)
    # -------------------------------------------------------------------------
    print("\nStep 3: Creating synthetic timestamps...")
    
    # Assume hourly data starting from arbitrary timestamp
    train_machine['timestamp'] = pd.date_range(
        start='2024-01-01', periods=len(train_machine), freq='h'
    )
    val_machine['timestamp'] = pd.date_range(
        start=train_machine['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
        periods=len(val_machine), freq='h'
    )
    test_machine['timestamp'] = pd.date_range(
        start=val_machine['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
        periods=len(test_machine), freq='h'
    )
    
    # Sort by timestamp
    train_machine = train_machine.sort_values('timestamp').reset_index(drop=True)
    val_machine = val_machine.sort_values('timestamp').reset_index(drop=True)
    test_machine = test_machine.sort_values('timestamp').reset_index(drop=True)
    
    print(f"  Timestamp range (train): {train_machine['timestamp'].min()} to {train_machine['timestamp'].max()}")
    
    # -------------------------------------------------------------------------
    # 4. Extract and Scale Features
    # -------------------------------------------------------------------------
    print("\nStep 4: Feature scaling...")
    
    # Extract only target sensors
    X_train = train_machine[target_sensors].values
    X_val = val_machine[target_sensors].values
    X_test = test_machine[target_sensors].values
    
    # Handle NaN values (fill with median)
    for i, col in enumerate(target_sensors):
        median_val = np.nanmedian(X_train[:, i])
        X_train[:, i] = np.nan_to_num(X_train[:, i], nan=median_val)
        X_val[:, i] = np.nan_to_num(X_val[:, i], nan=median_val)
        X_test[:, i] = np.nan_to_num(X_test[:, i], nan=median_val)
    
    # Normalize using training data statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Scaled data shape: {X_train_scaled.shape}")
    
    # -------------------------------------------------------------------------
    # 5. Create Sequences
    # -------------------------------------------------------------------------
    print(f"\nStep 5: Creating sequences (lookback={LOOKBACK_HOURS}h, forecast={FORECAST_HOURS}h)...")
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, LOOKBACK_HOURS, FORECAST_HOURS)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, LOOKBACK_HOURS, FORECAST_HOURS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, LOOKBACK_HOURS, FORECAST_HOURS)
    
    print(f"  Train sequences: {X_train_seq.shape} -> {y_train_seq.shape}")
    print(f"  Val sequences:   {X_val_seq.shape} -> {y_val_seq.shape}")
    print(f"  Test sequences:  {X_test_seq.shape} -> {y_test_seq.shape}")
    
    # Check if we have enough data
    if len(X_train_seq) < 100:
        print(f"  ⚠️  WARNING: Only {len(X_train_seq)} training sequences - may be insufficient!")
    
    # -------------------------------------------------------------------------
    # 6. Build Model
    # -------------------------------------------------------------------------
    print("\nStep 6: Building LSTM model...")
    
    input_shape = (LOOKBACK_HOURS, len(target_sensors))
    output_shape = (FORECAST_HOURS, len(target_sensors))
    
    model = build_lstm_model(input_shape, output_shape, LEARNING_RATE)
    
    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Total params: {model.count_params():,}")
    
    # -------------------------------------------------------------------------
    # 7. Train Model
    # -------------------------------------------------------------------------
    print("\nStep 7: Training model...")
    
    # Callbacks (optimized for speed)
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Reduced from 10 for faster completion
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Reduced from 5 for faster LR adjustment
        min_lr=1e-7
    )
    
    # Train with progress display
    print(f"  Training on {len(X_train_seq)} sequences...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1  # Show progress bar
    )
    
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"  Training completed - Best epoch: {best_epoch}/{EPOCHS}")
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss:   {history.history['val_loss'][-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # 8. Evaluate on Test Set
    # -------------------------------------------------------------------------
    print("\nStep 8: Evaluating on test set...")
    
    # Predict
    y_pred_scaled = model.predict(X_test_seq, verbose=0)
    
    # Inverse transform to original scale
    y_test_reshaped = y_test_seq.reshape(-1, len(target_sensors))
    y_pred_reshaped = y_pred_scaled.reshape(-1, len(target_sensors))
    
    y_test_original = scaler.inverse_transform(y_test_reshaped)
    y_pred_original = scaler.inverse_transform(y_pred_reshaped)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = calculate_mape(y_test_original, y_pred_original)
    smape = calculate_smape(y_test_original, y_pred_original)
    
    # Per-sensor metrics
    sensor_metrics = {}
    for i, sensor in enumerate(target_sensors):
        y_true_sensor = y_test_original[:, i]
        y_pred_sensor = y_pred_original[:, i]
        
        sensor_metrics[sensor] = {
            'rmse': float(np.sqrt(mean_squared_error(y_true_sensor, y_pred_sensor))),
            'mae': float(mean_absolute_error(y_true_sensor, y_pred_sensor)),
            'mape': float(calculate_mape(y_true_sensor, y_pred_sensor)),
            'smape': float(calculate_smape(y_true_sensor, y_pred_sensor))
        }
    
    print("\n  === OVERALL METRICS ===")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  SMAPE: {smape:.2f}%")
    
    print("\n  === PER-SENSOR METRICS ===")
    for sensor, metrics in sensor_metrics.items():
        print(f"  {sensor}:")
        print(f"    RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | "
              f"MAPE: {metrics['mape']:.2f}% | SMAPE: {metrics['smape']:.2f}%")
    
    # -------------------------------------------------------------------------
    # 9. Save Model and Artifacts
    # -------------------------------------------------------------------------
    print("\nStep 9: Saving model and artifacts...")
    
    save_path = SAVE_DIR / machine_id
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save Keras model
    model.save(save_path / 'lstm_model.keras')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, save_path / 'scaler.pkl')
    
    # Save configuration
    config = {
        'machine_id': machine_id,
        'target_sensors': target_sensors,
        'active_sensors': active_sensors,
        'lookback_hours': int(LOOKBACK_HOURS),
        'forecast_hours': int(FORECAST_HOURS),
        'n_features': int(len(target_sensors)),
        'model_params': int(model.count_params()),
        'training_samples': int(len(X_train_seq)),
        'best_epoch': int(best_epoch)
    }
    
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics report
    report = {
        'machine_id': machine_id,
        'n_sensors': len(active_sensors),
        'n_targets': len(target_sensors),
        'target_sensors': target_sensors,
        'overall_metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'smape': float(smape)
        },
        'per_sensor_metrics': sensor_metrics,
        'training_info': {
            'lookback_hours': int(LOOKBACK_HOURS),
            'forecast_hours': int(FORECAST_HOURS),
            'epochs_trained': int(best_epoch),
            'training_samples': int(len(X_train_seq)),
            'test_samples': int(len(X_test_seq))
        }
    }
    
    report_path = REPORTS_DIR / f'{machine_id}_timeseries_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✅ Model saved: {save_path}")
    print(f"  ✅ Report saved: {report_path}")
    
    return report


# =============================================================================
# BATCH TRAINING
# =============================================================================

def batch_train_all_machines():
    """Train time-series models for all machines"""
    
    print("\n" + "="*80)
    print("PHASE 2.5.1: TIME-SERIES FORECASTING - BATCH TRAINING")
    print("="*80 + "\n")
    
    # Get machine list from GAN synthetic data directory
    machine_ids = sorted([d.name for d in GAN_DATA_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(machine_ids)} machines to train\n")
    
    # Train each machine
    all_reports = []
    successful = 0
    failed = 0
    
    for i, machine_id in enumerate(machine_ids, 1):
        print(f"\n[{i}/{len(machine_ids)}] Processing: {machine_id}")
        
        try:
            report = train_timeseries_model(machine_id)
            all_reports.append(report)
            successful += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            failed += 1
            continue
    
    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("BATCH TRAINING COMPLETE")
    print("="*80 + "\n")
    
    print(f"Successful: {successful}/{len(machine_ids)}")
    print(f"Failed:     {failed}/{len(machine_ids)}")
    
    if all_reports:
        # Calculate summary statistics
        avg_rmse = np.mean([r['overall_metrics']['rmse'] for r in all_reports])
        avg_mae = np.mean([r['overall_metrics']['mae'] for r in all_reports])
        avg_mape = np.mean([r['overall_metrics']['mape'] for r in all_reports])
        avg_smape = np.mean([r['overall_metrics']['smape'] for r in all_reports])
        
        print("\n=== AVERAGE METRICS ACROSS ALL MACHINES ===")
        print(f"RMSE:  {avg_rmse:.4f}")
        print(f"MAE:   {avg_mae:.4f}")
        print(f"MAPE:  {avg_mape:.2f}%")
        print(f"SMAPE: {avg_smape:.2f}%")
        
        # Save batch summary
        summary = {
            'batch_date': pd.Timestamp.now().isoformat(),
            'total_machines': len(machine_ids),
            'successful': successful,
            'failed': failed,
            'average_metrics': {
                'rmse': float(avg_rmse),
                'mae': float(avg_mae),
                'mape': float(avg_mape),
                'smape': float(avg_smape)
            },
            'individual_reports': all_reports
        }
        
        summary_path = REPORTS_DIR / 'timeseries_batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Batch summary saved: {summary_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Train specific machine
        machine_id = sys.argv[1]
        train_timeseries_model(machine_id)
    else:
        # Batch train all machines
        batch_train_all_machines()
