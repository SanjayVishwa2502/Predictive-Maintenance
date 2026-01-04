"""
TIME-SERIES FORECASTING (PROPHET-BASED)
========================================
Lightweight time-series forecasting using Facebook Prophet.
No TensorFlow dependency - follows anomaly detection success pattern.

Predicts next 24 hours of sensor readings for predictive maintenance.
Uses per-machine approach for better accuracy.

Author: AI Assistant
Date: November 24, 2025
"""

import sys
import io
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Prophet for time-series forecasting
from prophet import Prophet

# UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent.parent))


class ProphetForecaster:
    """Prophet-based time-series forecasting for sensor data"""
    
    def __init__(self, machine_id, forecast_hours=24):
        """
        Args:
            machine_id: Machine identifier
            forecast_hours: Hours to forecast ahead (default 24)
        """
        self.machine_id = machine_id
        self.forecast_hours = forecast_hours
        self.models = {}  # One Prophet model per sensor
        self.sensor_cols = []
        # Default: hourly forecasts. Printers are higher-cadence; we downsample to 5-min
        # and forecast at 5-min steps to avoid huge arrays.
        self.freq = 'H'
        self.steps_per_hour = 1
        if str(machine_id).startswith('printer_'):
            self.freq = '5min'
            self.steps_per_hour = 12
        
    def load_data(self):
        """Load time-series data"""
        print(f"\n{'='*70}")
        print(f"Loading time-series data for {self.machine_id}...")
        print(f"{'='*70}\n")
        
        project_root = Path(__file__).parent.parent.parent.parent

        # For printers, train from real CSV telemetry instead of GAN synthetic parquet.
        if str(self.machine_id).startswith('printer_'):
            candidates = [
                project_root / '3Dprinterdata' / f'{self.machine_id}_temps_clean.csv',
                project_root / '3Dprinterdata' / 'ender3_temps_clean.csv',
                project_root / '3Dprinterdata' / 'printer_creality_ender3_001_temps_clean.csv',
            ]
            csv_path = next((p for p in candidates if p.exists()), None)
            if csv_path is None:
                raise FileNotFoundError(
                    f'Printer telemetry not found for {self.machine_id}. Expected CSV in 3Dprinterdata/'
                )
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            # Ensure required columns
            for c in ['bed_temp_c', 'nozzle_temp_c']:
                if c not in df.columns:
                    raise ValueError(f"Printer CSV missing column '{c}': {csv_path}")
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['bed_temp_c', 'nozzle_temp_c'])

            # Downsample to configured freq (5-min) so Prophet is stable and forecasting is tractable.
            df = df.set_index('timestamp')[['bed_temp_c', 'nozzle_temp_c']].resample(self.freq).mean().dropna().reset_index()

            # Prophet requires timezone-naive datetimes.
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            except Exception:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Simple split: 80/20 chronological
            n = len(df)
            split = max(10, int(n * 0.8))
            train_df = df.iloc[:split].copy()
            val_df = df.iloc[split:split].copy()
            test_df = df.iloc[split:].copy()
        else:
            base_path = project_root / 'GAN' / 'data' / 'synthetic' / self.machine_id
            train_df = pd.read_parquet(base_path / 'train.parquet')
            val_df = pd.read_parquet(base_path / 'val.parquet')
            test_df = pd.read_parquet(base_path / 'test.parquet')
        
        # Combine train + val for final training
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        
        print(f"  Train+Val: {len(train_val_df):,} samples")
        print(f"  Test:      {len(test_df):,} samples")
        
        # Sort by timestamp
        train_val_df = train_val_df.sort_values('timestamp').reset_index(drop=True)
        test_df = test_df.sort_values('timestamp').reset_index(drop=True)
        
        # Get sensor columns
        self.sensor_cols = [col for col in train_val_df.columns 
                   if col not in ['timestamp', 'rul']]
        
        print(f"  Sensors: {len(self.sensor_cols)} features")
        print(f"  Timespan: {train_val_df['timestamp'].min()} to {train_val_df['timestamp'].max()}")
        
        return train_val_df, test_df
    
    def train_sensor_model(self, df, sensor_col):
        """Train Prophet model for a single sensor"""
        # Prepare data in Prophet format (ds, y)
        ds = pd.to_datetime(df['timestamp'], errors='coerce')
        try:
            # If tz-aware, strip tz
            if hasattr(ds.dt, 'tz') and ds.dt.tz is not None:
                ds = ds.dt.tz_localize(None)
        except Exception:
            pass
        sensor_df = pd.DataFrame({
            'ds': ds,
            'y': df[sensor_col]
        })

        # Choose seasonality based on data span (prevents wild extrapolation on short logs).
        span_days = 0.0
        try:
            ds_min = pd.to_datetime(sensor_df['ds'], errors='coerce').min()
            ds_max = pd.to_datetime(sensor_df['ds'], errors='coerce').max()
            if pd.notna(ds_min) and pd.notna(ds_max):
                span_days = float((ds_max - ds_min).total_seconds() / 86400.0)
        except Exception:
            span_days = 0.0

        daily = span_days >= 2.0
        weekly = span_days >= 14.0

        # Printers: prioritize stability over flexibility.
        is_printer = str(self.machine_id).startswith('printer_')
        
        # Initialize Prophet with optimized parameters
        model = Prophet(
            changepoint_prior_scale=0.01 if is_printer else 0.05,  # trend flexibility
            seasonality_prior_scale=1.0 if is_printer else 10.0,   # seasonality flexibility
            daily_seasonality=daily,
            weekly_seasonality=weekly,
            yearly_seasonality=False,       # Not enough data for yearly
            seasonality_mode='additive',
            interval_width=0.95
        )
        
        # Fit model (suppress output)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(sensor_df)
        
        return model
    
    def train(self, train_df):
        """Train Prophet models for all sensors"""
        print(f"\n{'='*70}")
        print(f"Training Prophet models...")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for i, sensor in enumerate(self.sensor_cols, 1):
            print(f"  [{i}/{len(self.sensor_cols)}] Training {sensor}...", end='', flush=True)
            
            sensor_start = time.time()
            self.models[sensor] = self.train_sensor_model(train_df, sensor)
            sensor_time = time.time() - sensor_start
            
            print(f" ✅ ({sensor_time:.1f}s)")
        
        training_time = time.time() - start_time
        
        print(f"\n✅ All models trained: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return training_time
    
    def forecast_sensor(self, model, periods):
        """Generate forecast for a single sensor"""
        # Create future dataframe (only future periods, not historical)
        last_date = model.history['ds'].max()
        future = pd.DataFrame({
            'ds': pd.date_range(start=last_date, periods=periods+1, freq=self.freq)[1:]
        })
        
        # Generate forecast WITHOUT uncertainty intervals (faster)
        forecast = model.predict(future)
        
        # Return predictions
        return forecast[['ds', 'yhat']]
    
    def evaluate(self, test_df):
        """Evaluate models on test set"""
        print(f"\n{'='*70}")
        print(f"Evaluating models on test set...")
        print(f"{'='*70}\n")
        
        # Forecast the next horizon from training data. For sub-hourly printers,
        # this evaluates forecast_hours worth of steps at the configured frequency.
        all_predictions = []
        all_actuals = []
        sensor_metrics = {}

        steps = int(self.forecast_hours * (self.steps_per_hour or 1))
        steps_eval = int(min(max(1, steps), len(test_df)))
        
        for sensor in self.sensor_cols:
            # Get actual test values for first forecast_hours
            actual = test_df[sensor].iloc[:steps_eval].values
            
            # Generate forecast from trained model
            forecast = self.forecast_sensor(self.models[sensor], steps_eval)
            predicted = forecast['yhat'].values
            
            # Calculate metrics for this sensor
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            # MAPE calculation with proper handling of near-zero values
            actual_nonzero = np.where(np.abs(actual) < 0.01, np.nan, actual)
            mape_values = np.abs((actual - predicted) / actual_nonzero) * 100
            mape = np.nanmean(mape_values) if not np.all(np.isnan(mape_values)) else 100.0
            
            sensor_metrics[sensor] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(min(mape, 1000.0))  # Cap at 1000%
            }
            
            all_predictions.extend(predicted)
            all_actuals.extend(actual)
        
        # Overall metrics
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        
        # Overall MAPE with proper handling
        all_actuals_arr = np.array(all_actuals)
        all_predictions_arr = np.array(all_predictions)
        actuals_nonzero = np.where(np.abs(all_actuals_arr) < 0.01, np.nan, all_actuals_arr)
        mape_values = np.abs((all_actuals_arr - all_predictions_arr) / actuals_nonzero) * 100
        overall_mape = np.nanmean(mape_values) if not np.all(np.isnan(mape_values)) else 100.0
        overall_mape = min(overall_mape, 1000.0)  # Cap at 1000%
        
        metrics = {
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'mape': float(overall_mape),
            'n_test_samples': steps_eval,
            'n_sensors': len(self.sensor_cols),
            'sensor_metrics': sensor_metrics
        }
        
        print(f"Overall Metrics (across all sensors):")
        print(f"  MAE:  {overall_mae:.4f}")
        print(f"  RMSE: {overall_rmse:.4f}")
        print(f"  MAPE: {overall_mape:.2f}%")
        
        # Show best and worst sensors
        sorted_sensors = sorted(sensor_metrics.items(), key=lambda x: x[1]['mape'])
        
        print(f"\nBest 3 Sensors (lowest MAPE):")
        for sensor, m in sorted_sensors[:3]:
            print(f"  {sensor[:40]}: MAPE={m['mape']:.2f}%")
        
        if len(sorted_sensors) > 3:
            print(f"\nWorst 3 Sensors (highest MAPE):")
            for sensor, m in sorted_sensors[-3:]:
                print(f"  {sensor[:40]}: MAPE={m['mape']:.2f}%")
        
        return metrics
    
    def save_models(self, metrics, training_time):
        """Save Prophet models and metadata"""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        save_path = project_root / 'ml_models' / 'models' / 'timeseries' / self.machine_id
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save all Prophet models
        models_file = save_path / 'prophet_models.pkl'
        joblib.dump(self.models, str(models_file))
        
        # Save metadata
        metadata = {
            'machine_id': self.machine_id,
            'forecast_hours': self.forecast_hours,
            'freq': self.freq,
            'steps_per_hour': self.steps_per_hour,
            'sensor_cols': self.sensor_cols,
            'n_sensors': len(self.sensor_cols),
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'model_type': 'facebook_prophet',
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = save_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate model size
        model_size_mb = models_file.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"Models saved successfully!")
        print(f"{'='*70}")
        print(f"  Location: {save_path}")
        print(f"  Size: {model_size_mb:.2f} MB")
        print(f"  Models: {len(self.models)} Prophet models (one per sensor)")
        
        return metadata


def train_timeseries_model(machine_id, forecast_hours=24):
    """
    Train time-series forecasting model using Prophet
    
    Args:
        machine_id: Machine identifier
        forecast_hours: Hours to forecast ahead (default 24)
    """
    print(f"\n{'='*70}")
    print(f"TIME-SERIES FORECASTING (PROPHET)")
    print(f"Machine: {machine_id}")
    print(f"Forecast Horizon: {forecast_hours} hours")
    print(f"{'='*70}\n")
    
    # Initialize forecaster
    forecaster = ProphetForecaster(machine_id, forecast_hours)
    
    # Load data
    train_df, test_df = forecaster.load_data()
    
    # Train models
    training_time = forecaster.train(train_df)
    
    # Evaluate models
    metrics = forecaster.evaluate(test_df)
    
    # Save models
    metadata = forecaster.save_models(metrics, training_time)
    
    # Save performance report
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    report_path = project_root / 'ml_models' / 'reports' / 'performance_metrics'
    report_path.mkdir(parents=True, exist_ok=True)
    
    report = {
        'machine_id': machine_id,
        'model_type': 'timeseries_prophet',
        'forecast_hours': forecast_hours,
        'training_time_minutes': training_time / 60,
        'n_sensors': len(forecaster.sensor_cols),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    report_file = report_path / f'{machine_id}_timeseries_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved: {report_file}\n")
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train time-series forecasting with Prophet')
    parser.add_argument('--machine_id', required=True, help='Machine identifier')
    parser.add_argument('--forecast_hours', type=int, default=24,
                       help='Hours to forecast ahead (default: 24)')
    
    args = parser.parse_args()
    
    report = train_timeseries_model(
        machine_id=args.machine_id,
        forecast_hours=args.forecast_hours
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Machine:       {args.machine_id}")
    print(f"MAPE:          {report['metrics']['mape']:.2f}%")
    print(f"MAE:           {report['metrics']['mae']:.4f}")
    print(f"RMSE:          {report['metrics']['rmse']:.4f}")
    print(f"Training Time: {report['training_time_minutes']:.2f} minutes")
    print(f"Sensors:       {report['n_sensors']}")
    print(f"{'='*70}\n")
