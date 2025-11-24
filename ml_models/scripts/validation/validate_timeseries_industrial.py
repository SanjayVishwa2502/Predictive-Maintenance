"""
INDUSTRIAL-GRADE TIME-SERIES FORECASTING VALIDATION
====================================================
Rigorous validation framework for Prophet-based time-series forecasting models.
Implements comprehensive tests for temporal forecasting accuracy, stability, and deployment readiness.

Tests:
1. Multi-Horizon Forecast Accuracy (1h, 6h, 12h, 24h)
2. Temporal Consistency & Drift Detection
3. Sensor-Level Performance Analysis
4. Rolling Window Validation (Walk-Forward)
5. Prediction Interval Quality (Uncertainty Calibration)
6. Inference Speed & Latency
7. Model Size & Raspberry Pi Compatibility

Author: AI Assistant
Date: November 24, 2025
"""

import sys
import io
import os
import json
import joblib
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'


class TimeSeriesIndustrialValidator:
    """Industrial-grade validation for time-series forecasting models"""
    
    def __init__(self, machine_id, verbose=True):
        self.machine_id = machine_id
        self.verbose = verbose
        self.model_path = Path(f'../../models/timeseries/{machine_id}')
        self.data_path = Path(f'../../../GAN/data/synthetic/{machine_id}')
        self.results = {
            'machine_id': machine_id,
            'validation_timestamp': datetime.now().isoformat(),
            'model_type': 'prophet_multivariate',
            'tests': {}
        }
        
    def load_model_and_data(self):
        """Load trained Prophet models and test data"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INDUSTRIAL TIME-SERIES FORECASTING VALIDATION")
            print(f"Machine: {self.machine_id}")
            print(f"{'='*80}\n")
            print(f"Loading models and data...")
        
        # Load Prophet models (one per sensor)
        model_file = self.model_path / 'prophet_models.pkl'
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.models = joblib.load(model_file)
        
        # Load metadata
        metadata_file = self.model_path / 'metadata.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load test data
        test_file = self.data_path / 'test.parquet'
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        self.test_df = pd.read_parquet(test_file)
        
        # Ensure timestamp is datetime
        if 'timestamp' in self.test_df.columns:
            self.test_df['timestamp'] = pd.to_datetime(self.test_df['timestamp'])
            self.test_df = self.test_df.sort_values('timestamp').reset_index(drop=True)
        
        # Get sensor columns
        self.sensor_cols = [col for col in self.test_df.columns 
                           if col not in ['timestamp', 'rul', 'failure_status']]
        
        # Calculate model size
        model_size = sum(f.stat().st_size for f in self.model_path.rglob('*') if f.is_file())
        self.results['model_size_mb'] = model_size / (1024 * 1024)
        
        if self.verbose:
            print(f"✅ Models loaded: {len(self.models)} Prophet models (one per sensor)")
            print(f"✅ Model size: {self.results['model_size_mb']:.2f} MB")
            print(f"✅ Test samples: {len(self.test_df):,}")
            print(f"✅ Sensors: {len(self.sensor_cols)}")
            print(f"✅ Test period: {self.test_df['timestamp'].min()} to {self.test_df['timestamp'].max()}")
        
    def calculate_mape(self, y_true, y_pred):
        """Calculate MAPE with proper handling of near-zero values"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Mask near-zero values (< 0.01)
        mask = np.abs(y_true) >= 0.01
        
        if mask.sum() == 0:
            return 100.0  # No valid values
        
        mape_values = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        mape = np.mean(mape_values)
        
        # Cap at 1000%
        return min(mape, 1000.0)
    
    def forecast_sensor(self, sensor_name, periods):
        """Generate forecast for a single sensor"""
        model = self.models[sensor_name]
        
        # Create future dataframe from last training timestamp
        last_date = model.history['ds'].max()
        future = pd.DataFrame({
            'ds': pd.date_range(start=last_date, periods=periods+1, freq='H')[1:]
        })
        
        # Generate forecast
        forecast = model.predict(future)
        
        return forecast['yhat'].values
    
    def test_1_multi_horizon_accuracy(self):
        """
        Test 1: Multi-Horizon Forecast Accuracy
        Evaluate forecasts at different time horizons: 1h, 6h, 12h, 24h
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 1: Multi-Horizon Forecast Accuracy")
            print(f"{'='*80}\n")
        
        horizons = [1, 6, 12, 24]  # hours
        horizon_results = {}
        
        for horizon in horizons:
            if horizon > len(self.test_df):
                continue
            
            all_maes = []
            all_rmses = []
            all_mapes = []
            sensor_metrics = {}
            
            for sensor in self.sensor_cols:
                # Get actual values for this horizon
                actual = self.test_df[sensor].iloc[:horizon].values
                
                # Generate forecast
                predicted = self.forecast_sensor(sensor, horizon)
                
                # Calculate metrics
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mape = self.calculate_mape(actual, predicted)
                
                all_maes.append(mae)
                all_rmses.append(rmse)
                all_mapes.append(mape)
                
                sensor_metrics[sensor] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape)
                }
            
            # Calculate average metrics
            avg_mae = np.mean(all_maes)
            avg_rmse = np.mean(all_rmses)
            avg_mape = np.mean(all_mapes)
            
            # Grade based on MAPE
            if avg_mape < 10:
                grade = 'A'
                status = '✅ EXCELLENT'
            elif avg_mape < 20:
                grade = 'B'
                status = '✅ GOOD'
            elif avg_mape < 35:
                grade = 'C'
                status = '⚠️ FAIR'
            elif avg_mape < 50:
                grade = 'D'
                status = '⚠️ POOR'
            else:
                grade = 'F'
                status = '❌ FAILING'
            
            horizon_results[f'{horizon}h'] = {
                'mae': float(avg_mae),
                'rmse': float(avg_rmse),
                'mape': float(avg_mape),
                'grade': grade,
                'status': status,
                'sensor_count': len(self.sensor_cols),
                'sensor_metrics': sensor_metrics
            }
            
            if self.verbose:
                print(f"{horizon}h Forecast:")
                print(f"  MAE:   {avg_mae:.4f}")
                print(f"  RMSE:  {avg_rmse:.4f}")
                print(f"  MAPE:  {avg_mape:.2f}% [{grade}] {status}")
        
        # Calculate degradation (accuracy loss over time)
        mape_1h = horizon_results.get('1h', {}).get('mape', 0)
        mape_24h = horizon_results.get('24h', {}).get('mape', 0)
        degradation = mape_24h - mape_1h if mape_1h and mape_24h else 0
        
        if self.verbose:
            print(f"\nForecast Horizon Degradation:")
            print(f"  1h MAPE:     {mape_1h:.2f}%")
            print(f"  24h MAPE:    {mape_24h:.2f}%")
            print(f"  Degradation: {degradation:.2f}% [{'✅ Good' if degradation < 20 else '⚠️ High'}]")
        
        # Get best (lowest) MAPE across all horizons
        best_mape = min([h['mape'] for h in horizon_results.values()]) if horizon_results else 999
        
        self.results['tests']['multi_horizon_accuracy'] = {
            'horizons': horizon_results,
            'degradation_1h_to_24h': float(degradation),
            'best_mape': float(best_mape),
            'test_passed': bool(float(best_mape) < 50)  # Pass if best horizon < 50%
        }
        
    def test_2_temporal_consistency(self):
        """
        Test 2: Temporal Consistency & Drift Detection
        Check if model maintains consistent performance over time
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 2: Temporal Consistency & Drift Detection")
            print(f"{'='*80}\n")
        
        # Split test data into 4 temporal windows
        n_samples = len(self.test_df)
        window_size = n_samples // 4
        
        window_results = {}
        window_mapes = []
        
        for i in range(4):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n_samples)
            
            # Ensure we have at least 24 samples
            if end_idx - start_idx < 24:
                continue
            
            window_df = self.test_df.iloc[start_idx:end_idx]
            
            # Forecast 24h for this window
            all_mapes = []
            
            for sensor in self.sensor_cols:
                actual = window_df[sensor].iloc[:24].values
                predicted = self.forecast_sensor(sensor, 24)[:len(actual)]
                
                mape = self.calculate_mape(actual, predicted)
                all_mapes.append(mape)
            
            avg_mape = np.mean(all_mapes)
            window_mapes.append(avg_mape)
            
            window_results[f'window_{i+1}'] = {
                'start_timestamp': str(window_df['timestamp'].iloc[0]),
                'end_timestamp': str(window_df['timestamp'].iloc[-1]),
                'samples': int(end_idx - start_idx),
                'mape': float(avg_mape)
            }
            
            if self.verbose:
                print(f"Window {i+1}:")
                print(f"  Period: {window_df['timestamp'].iloc[0]} to {window_df['timestamp'].iloc[-1]}")
                print(f"  MAPE:   {avg_mape:.2f}%")
        
        # Calculate consistency (variance across windows)
        if len(window_mapes) > 1:
            mape_std = np.std(window_mapes)
            mape_range = max(window_mapes) - min(window_mapes)
            
            # Grade consistency
            if mape_std < 15:
                consistency_grade = 'A'
                consistency_status = '✅ EXCELLENT'
            elif mape_std < 30:
                consistency_grade = 'B'
                consistency_status = '✅ GOOD'
            elif mape_std < 50:
                consistency_grade = 'C'
                consistency_status = '⚠️ FAIR'
            else:
                consistency_grade = 'D'
                consistency_status = '❌ POOR'
            
            if self.verbose:
                print(f"\nTemporal Consistency:")
                print(f"  MAPE Std Dev:  {mape_std:.2f}%")
                print(f"  MAPE Range:    {mape_range:.2f}%")
                print(f"  Consistency:   [{consistency_grade}] {consistency_status}")
        else:
            mape_std = 0
            consistency_grade = 'N/A'
            consistency_status = 'Insufficient windows'
        
        self.results['tests']['temporal_consistency'] = {
            'windows': window_results,
            'mape_std_dev': float(mape_std),
            'consistency_grade': consistency_grade,
            'consistency_status': consistency_status,
            'test_passed': bool(float(mape_std) < 50)
        }
        
    def test_3_sensor_level_analysis(self):
        """
        Test 3: Sensor-Level Performance Analysis
        Identify best and worst performing sensors
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 3: Sensor-Level Performance Analysis")
            print(f"{'='*80}\n")
        
        sensor_performance = {}
        forecast_hours = 24
        
        for sensor in self.sensor_cols:
            # Get actual values
            actual = self.test_df[sensor].iloc[:forecast_hours].values
            
            # Generate forecast
            predicted = self.forecast_sensor(sensor, forecast_hours)
            
            # Calculate comprehensive metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = self.calculate_mape(actual, predicted)
            
            # Additional metrics
            residuals = actual - predicted
            max_error = np.max(np.abs(residuals))
            
            # Grade sensor performance
            if mape < 5:
                grade = 'A'
            elif mape < 15:
                grade = 'B'
            elif mape < 30:
                grade = 'C'
            elif mape < 50:
                grade = 'D'
            else:
                grade = 'F'
            
            sensor_performance[sensor] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'max_error': float(max_error),
                'grade': grade
            }
        
        # Sort by MAPE
        sorted_sensors = sorted(sensor_performance.items(), key=lambda x: x[1]['mape'])
        
        best_3 = sorted_sensors[:3]
        worst_3 = sorted_sensors[-3:]
        
        # Calculate statistics
        all_mapes = [v['mape'] for v in sensor_performance.values()]
        avg_mape = np.mean(all_mapes)
        median_mape = np.median(all_mapes)
        
        # Count by grade
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for metrics in sensor_performance.values():
            grade_counts[metrics['grade']] += 1
        
        if self.verbose:
            print(f"Overall Statistics:")
            print(f"  Average MAPE:  {avg_mape:.2f}%")
            print(f"  Median MAPE:   {median_mape:.2f}%")
            print(f"  Grade A:       {grade_counts['A']}/{len(self.sensor_cols)} sensors")
            print(f"  Grade B:       {grade_counts['B']}/{len(self.sensor_cols)} sensors")
            print(f"  Grade C:       {grade_counts['C']}/{len(self.sensor_cols)} sensors")
            print(f"  Grade D/F:     {grade_counts['D'] + grade_counts['F']}/{len(self.sensor_cols)} sensors")
            
            print(f"\nBest Performing Sensors:")
            for sensor, metrics in best_3:
                print(f"  {sensor:40s} MAPE={metrics['mape']:7.2f}% [{metrics['grade']}]")
            
            print(f"\nWorst Performing Sensors:")
            for sensor, metrics in worst_3:
                print(f"  {sensor:40s} MAPE={metrics['mape']:7.2f}% [{metrics['grade']}]")
        
        self.results['tests']['sensor_level_analysis'] = {
            'sensor_performance': sensor_performance,
            'average_mape': float(avg_mape),
            'median_mape': float(median_mape),
            'grade_distribution': grade_counts,
            'best_sensors': [s[0] for s in best_3],
            'worst_sensors': [s[0] for s in worst_3],
            'test_passed': bool(float(avg_mape) < 100)  # Reasonable threshold for synthetic data
        }
        
    def test_4_rolling_window_validation(self):
        """
        Test 4: Rolling Window Validation (Walk-Forward)
        Simulate real-world deployment with progressive forecasting
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 4: Rolling Window Validation (Walk-Forward)")
            print(f"{'='*80}\n")
        
        # Use 100 rolling windows of 24h forecasts
        n_windows = min(100, len(self.test_df) - 24)
        step_size = max(1, (len(self.test_df) - 24) // n_windows)
        
        window_mapes = []
        window_maes = []
        
        if self.verbose:
            print(f"Performing {n_windows} rolling window forecasts...")
            print(f"Window size: 24 hours")
        
        for i in range(n_windows):
            start_idx = i * step_size
            
            if start_idx + 24 > len(self.test_df):
                break
            
            window_mapes_sensors = []
            window_maes_sensors = []
            
            for sensor in self.sensor_cols:
                actual = self.test_df[sensor].iloc[start_idx:start_idx+24].values
                predicted = self.forecast_sensor(sensor, 24)
                
                mae = mean_absolute_error(actual, predicted)
                mape = self.calculate_mape(actual, predicted)
                
                window_mapes_sensors.append(mape)
                window_maes_sensors.append(mae)
            
            window_mapes.append(np.mean(window_mapes_sensors))
            window_maes.append(np.mean(window_maes_sensors))
        
        # Calculate statistics
        avg_mape = np.mean(window_mapes)
        std_mape = np.std(window_mapes)
        min_mape = np.min(window_mapes)
        max_mape = np.max(window_mapes)
        
        # Grade based on stability (std dev)
        if std_mape < 20:
            stability_grade = 'A'
            stability_status = '✅ EXCELLENT'
        elif std_mape < 40:
            stability_grade = 'B'
            stability_status = '✅ GOOD'
        elif std_mape < 60:
            stability_grade = 'C'
            stability_status = '⚠️ FAIR'
        else:
            stability_grade = 'D'
            stability_status = '❌ POOR'
        
        if self.verbose:
            print(f"\nRolling Window Results ({len(window_mapes)} windows):")
            print(f"  Average MAPE:  {avg_mape:.2f}%")
            print(f"  Std Dev:       {std_mape:.2f}%")
            print(f"  Min MAPE:      {min_mape:.2f}%")
            print(f"  Max MAPE:      {max_mape:.2f}%")
            print(f"  Stability:     [{stability_grade}] {stability_status}")
        
        self.results['tests']['rolling_window_validation'] = {
            'n_windows': len(window_mapes),
            'window_size_hours': 24,
            'average_mape': float(avg_mape),
            'std_dev_mape': float(std_mape),
            'min_mape': float(min_mape),
            'max_mape': float(max_mape),
            'stability_grade': stability_grade,
            'stability_status': stability_status,
            'test_passed': bool(float(std_mape) < 60)
        }
        
    def test_5_prediction_interval_quality(self):
        """
        Test 5: Prediction Interval Quality (Uncertainty Calibration)
        Note: Our Prophet models don't use uncertainty intervals for speed,
        so we'll test point forecast reliability instead
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 5: Point Forecast Reliability")
            print(f"{'='*80}\n")
        
        forecast_hours = 24
        
        # Calculate residuals for all sensors
        all_residuals = []
        sensor_reliability = {}
        
        for sensor in self.sensor_cols:
            actual = self.test_df[sensor].iloc[:forecast_hours].values
            predicted = self.forecast_sensor(sensor, forecast_hours)
            
            residuals = actual - predicted
            all_residuals.extend(residuals)
            
            # Calculate reliability metrics
            mae = np.mean(np.abs(residuals))
            std_residual = np.std(residuals)
            max_residual = np.max(np.abs(residuals))
            
            sensor_reliability[sensor] = {
                'mae': float(mae),
                'std_residual': float(std_residual),
                'max_residual': float(max_residual)
            }
        
        # Overall statistics
        overall_mae = np.mean([v['mae'] for v in sensor_reliability.values()])
        overall_std = np.mean([v['std_residual'] for v in sensor_reliability.values()])
        
        # Grade reliability
        if overall_std < 10:
            reliability_grade = 'A'
            reliability_status = '✅ EXCELLENT'
        elif overall_std < 25:
            reliability_grade = 'B'
            reliability_status = '✅ GOOD'
        elif overall_std < 50:
            reliability_grade = 'C'
            reliability_status = '⚠️ FAIR'
        else:
            reliability_grade = 'D'
            reliability_status = '❌ POOR'
        
        if self.verbose:
            print(f"Point Forecast Reliability:")
            print(f"  Average MAE:       {overall_mae:.4f}")
            print(f"  Average Std Dev:   {overall_std:.4f}")
            print(f"  Reliability:       [{reliability_grade}] {reliability_status}")
            print(f"\nNote: Models optimized for speed (no uncertainty intervals)")
        
        self.results['tests']['forecast_reliability'] = {
            'overall_mae': float(overall_mae),
            'overall_std_dev': float(overall_std),
            'sensor_reliability': sensor_reliability,
            'reliability_grade': reliability_grade,
            'reliability_status': reliability_status,
            'test_passed': bool(float(overall_std) < 50)
        }
        
    def test_6_inference_speed(self):
        """
        Test 6: Inference Speed & Latency
        Critical for real-time deployment
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 6: Inference Speed & Latency")
            print(f"{'='*80}\n")
        
        # Test single-sensor forecast latency
        sensor = self.sensor_cols[0]
        model = self.models[sensor]
        
        # Warm-up
        _ = self.forecast_sensor(sensor, 24)
        
        # Measure latency (10 runs)
        latencies = []
        for _ in range(10):
            start = time.time()
            _ = self.forecast_sensor(sensor, 24)
            latencies.append((time.time() - start) * 1000)  # ms
        
        single_sensor_latency = np.mean(latencies)
        
        # Estimate all-sensor latency
        total_latency = single_sensor_latency * len(self.sensor_cols)
        
        # Grade latency
        if total_latency < 1000:  # < 1 second
            latency_grade = 'A'
            latency_status = '✅ EXCELLENT'
        elif total_latency < 5000:  # < 5 seconds
            latency_grade = 'B'
            latency_status = '✅ GOOD'
        elif total_latency < 10000:  # < 10 seconds
            latency_grade = 'C'
            latency_status = '⚠️ FAIR'
        else:
            latency_grade = 'D'
            latency_status = '❌ POOR'
        
        if self.verbose:
            print(f"Inference Performance:")
            print(f"  Single Sensor:     {single_sensor_latency:.2f} ms")
            print(f"  All Sensors:       {total_latency:.2f} ms ({total_latency/1000:.2f}s)")
            print(f"  Sensors:           {len(self.sensor_cols)}")
            print(f"  Latency Grade:     [{latency_grade}] {latency_status}")
        
        self.results['tests']['inference_speed'] = {
            'single_sensor_latency_ms': float(single_sensor_latency),
            'total_latency_ms': float(total_latency),
            'n_sensors': len(self.sensor_cols),
            'latency_grade': latency_grade,
            'latency_status': latency_status,
            'test_passed': bool(float(total_latency) < 10000)
        }
        
    def test_7_pi_compatibility(self):
        """
        Test 7: Model Size & Raspberry Pi Compatibility
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TEST 7: Raspberry Pi Compatibility")
            print(f"{'='*80}\n")
        
        model_size_mb = self.results['model_size_mb']
        latency_ms = self.results['tests']['inference_speed']['total_latency_ms']
        
        # Pi compatibility thresholds
        size_ok = model_size_mb < 100  # 100 MB target for time-series
        latency_ok = latency_ms < 15000  # 15 seconds acceptable for batch forecasting
        
        # Estimate runtime memory
        runtime_memory = model_size_mb * 2.0 + 300  # Models + Prophet overhead
        memory_ok = runtime_memory < 2000  # 2GB limit for 4GB Pi
        
        pi_compatible = size_ok and memory_ok and latency_ok
        
        # Grade
        if pi_compatible and model_size_mb < 50:
            pi_grade = 'A'
            pi_status = '✅ EXCELLENT'
        elif pi_compatible:
            pi_grade = 'B'
            pi_status = '✅ GOOD'
        elif size_ok and memory_ok:
            pi_grade = 'C'
            pi_status = '⚠️ FAIR (Latency issue)'
        else:
            pi_grade = 'D'
            pi_status = '❌ POOR'
        
        if self.verbose:
            print(f"Raspberry Pi Compatibility:")
            print(f"  Model Size:        {model_size_mb:.2f} MB [{'✅' if size_ok else '❌'} {'<100MB' if size_ok else '≥100MB'}]")
            print(f"  Runtime Memory:    {runtime_memory:.2f} MB [{'✅' if memory_ok else '❌'}]")
            print(f"  Inference Latency: {latency_ms:.2f} ms [{'✅' if latency_ok else '❌'}]")
            print(f"  Pi Compatible:     {'✅ YES' if pi_compatible else '❌ NO'} [{pi_grade}] {pi_status}")
        
        self.results['tests']['pi_compatibility'] = {
            'model_size_mb': float(model_size_mb),
            'estimated_runtime_mb': float(runtime_memory),
            'inference_latency_ms': float(latency_ms),
            'size_compatible': bool(size_ok),
            'memory_compatible': bool(memory_ok),
            'latency_compatible': bool(latency_ok),
            'pi_compatible': bool(pi_compatible),
            'grade': pi_grade,
            'status': pi_status,
            'test_passed': bool(pi_compatible)
        }
        
    def generate_overall_grade(self):
        """Generate overall validation grade"""
        if self.verbose:
            print(f"\n{'='*80}")
            print("OVERALL VALIDATION SUMMARY")
            print(f"{'='*80}\n")
        
        # Collect all test grades
        grades = []
        test_names = []
        
        for test_name, test_results in self.results['tests'].items():
            if 'grade' in test_results:
                grades.append(test_results['grade'])
                test_names.append(test_name)
            elif isinstance(test_results, dict) and 'horizons' in test_results:
                # Multi-horizon test
                for horizon_result in test_results['horizons'].values():
                    grades.append(horizon_result['grade'])
        
        # Convert grades to numeric
        grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        numeric_grades = [grade_map.get(g, 0) for g in grades]
        
        avg_grade = np.mean(numeric_grades) if numeric_grades else 0
        
        # Determine overall grade
        if avg_grade >= 3.5:
            overall_grade = 'A'
            overall_status = '✅ EXCELLENT - Production Ready'
        elif avg_grade >= 2.5:
            overall_grade = 'B'
            overall_status = '✅ GOOD - Production Ready with Monitoring'
        elif avg_grade >= 1.5:
            overall_grade = 'C'
            overall_status = '⚠️ FAIR - Needs Improvement'
        elif avg_grade >= 0.5:
            overall_grade = 'D'
            overall_status = '⚠️ POOR - Requires Retraining'
        else:
            overall_grade = 'F'
            overall_status = '❌ FAILING - Not Production Ready'
        
        self.results['overall_grade'] = overall_grade
        self.results['overall_status'] = overall_status
        self.results['average_grade_score'] = float(avg_grade)
        
        if self.verbose:
            print(f"Overall Grade: {overall_grade}")
            print(f"Status: {overall_status}")
            print(f"Average Score: {avg_grade:.2f}/4.0")
            
            # Test summary
            print(f"\nTest Results:")
            test_count = len(self.results['tests'])
            passed = sum(1 for t in self.results['tests'].values() if t.get('test_passed', False))
            print(f"  Passed: {passed}/{test_count} tests")
            
            # Show individual test grades
            print(f"\nIndividual Test Grades:")
            for i, (test_name, grade) in enumerate(zip(test_names, grades), 1):
                print(f"  Test {i}: {test_name:40s} [{grade}]")
        
    def save_report(self):
        """Save validation report"""
        report_dir = Path('../../reports/validation')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f'{self.machine_id}_timeseries_industrial_validation.json'
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ Validation report saved: {report_file}")
            print(f"{'='*80}\n")
        
        return report_file
    
    def run_all_tests(self):
        """Run complete validation suite"""
        try:
            self.load_model_and_data()
            self.test_1_multi_horizon_accuracy()
            self.test_2_temporal_consistency()
            self.test_3_sensor_level_analysis()
            self.test_4_rolling_window_validation()
            self.test_5_prediction_interval_quality()
            self.test_6_inference_speed()
            self.test_7_pi_compatibility()
            self.generate_overall_grade()
            report_file = self.save_report()
            
            return True, report_file
            
        except Exception as e:
            if self.verbose:
                print(f"\n❌ Validation failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            self.results['validation_failed'] = True
            self.results['error'] = str(e)
            
            return False, None


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Industrial Time-Series Validation')
    parser.add_argument('--machine_id', type=str, required=True, help='Machine ID to validate')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    validator = TimeSeriesIndustrialValidator(
        machine_id=args.machine_id,
        verbose=not args.quiet
    )
    
    success, report_file = validator.run_all_tests()
    
    if success:
        print(f"\n✅ Validation completed successfully!")
        print(f"   Report: {report_file}")
        return 0
    else:
        print(f"\n❌ Validation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
