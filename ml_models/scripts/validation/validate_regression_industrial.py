"""
Industrial-Grade Validation Suite for Regression Models (RUL Prediction)
Implements rigorous validation techniques for time-series regression with temporal validation
"""
import sys
import os

# Fix Unicode encoding for Windows terminal
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


class RegressionIndustrialValidator:
    """
    Industrial-grade regression model validation with temporal integrity checks
    """
    
    def __init__(self, machine_id, model_path, verbose=True):
        self.machine_id = machine_id
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.predictor = None
        self.results = {
            'machine_id': machine_id,
            'validation_timestamp': datetime.now().isoformat(),
            'model_path': str(model_path)
        }
        
    def load_model(self):
        """Load trained model"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INDUSTRIAL REGRESSION VALIDATION: {self.machine_id}")
            print(f"{'='*80}")
        
        self.predictor = TabularPredictor.load(str(self.model_path))
        if self.verbose:
            print(f"✅ Model loaded: {self.predictor.model_best}")
        
        self.results['best_model'] = self.predictor.model_best
        
        # Get model size
        model_size_mb = sum(f.stat().st_size for f in self.model_path.rglob('*') if f.is_file()) / (1024**2)
        self.results['model_size_mb'] = round(model_size_mb, 2)
    
    def load_data(self):
        """Load all data splits"""
        data_path = Path(f'../../../GAN/data/synthetic/{self.machine_id}')
        
        if self.verbose:
            print(f"\nLoading data from: {data_path}")
        
        # Load all splits
        train_df = pd.read_parquet(data_path / 'train.parquet')
        val_df = pd.read_parquet(data_path / 'val.parquet')
        test_df = pd.read_parquet(data_path / 'test.parquet')
        
        # Verify RUL column exists
        if 'rul' not in train_df.columns:
            message = (
                f"⚠️  RUL column missing in {self.machine_id} synthetic data. "
                "Skipping regression validation."
            )
            self.results['status'] = 'skipped'
            self.results['skipped'] = True
            self.results['skip_reason'] = message
            if self.verbose:
                print(message)
            return None, None, None
        
        if self.verbose:
            print(f"✅ Data loaded:")
            print(f"   Train: {len(train_df):,} samples")
            print(f"   Val:   {len(val_df):,} samples")
            print(f"   Test:  {len(test_df):,} samples")
            print(f"   RUL Range: {train_df['rul'].min():.1f} - {train_df['rul'].max():.1f} hours")
        
        return train_df, val_df, test_df
    
    def basic_performance_metrics(self, y_true, y_pred, split_name):
        """Calculate basic regression metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE with safe division
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf
        
        # Additional metrics
        residuals = y_true - y_pred
        max_error = np.max(np.abs(residuals))
        std_error = np.std(residuals)
        
        metrics = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'max_error': float(max_error),
            'std_error': float(std_error),
            'mean_residual': float(np.mean(residuals))
        }
        
        if self.verbose:
            print(f"\n{split_name} Performance:")
            print(f"  R² Score:     {r2:.4f}")
            print(f"  RMSE:         {rmse:.2f} hours")
            print(f"  MAE:          {mae:.2f} hours")
            print(f"  Max Error:    {max_error:.2f} hours")
            print(f"  Std Error:    {std_error:.2f} hours")
        
        return metrics
    
    def temporal_robustness_test(self, df, target_col='rul'):
        """
        Test model performance across time windows
        Critical for time-series regression
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 1: TEMPORAL ROBUSTNESS (Time-Series Split Validation)")
            print("="*80)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 5-fold time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_scores = []
        fold_details = []
        
        # Get features (exclude only target, keep timestamp for datetime engineering)
        features = [col for col in df.columns if col != target_col]
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
            fold_test = df.iloc[test_idx]
            
            X_test = fold_test[features]
            y_test = fold_test[target_col]
            
            # Predict
            y_pred = self.predictor.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            fold_scores.append(r2)
            fold_details.append({
                'fold': fold_idx,
                'samples': len(test_idx),
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            })
            
            if self.verbose:
                print(f"Fold {fold_idx}: R²={r2:.4f}, RMSE={rmse:.2f}h, MAE={mae:.2f}h (n={len(test_idx)})")
        
        # Calculate stability metrics
        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        cv_r2 = (std_r2 / mean_r2) if mean_r2 != 0 else np.inf
        
        # Grade temporal stability
        if std_r2 < 0.02:
            stability_grade = 'A'
            stability_status = '✅ EXCELLENT'
        elif std_r2 < 0.05:
            stability_grade = 'B'
            stability_status = '⚠️ GOOD'
        elif std_r2 < 0.10:
            stability_grade = 'C'
            stability_status = '⚠️ FAIR'
        else:
            stability_grade = 'D'
            stability_status = '❌ POOR'
        
        if self.verbose:
            print(f"\nTemporal Stability Analysis:")
            print(f"  Mean R²:          {mean_r2:.4f}")
            print(f"  Std Dev:          {std_r2:.4f}")
            print(f"  Stability Grade:  {stability_grade} {stability_status}")
        
        results = {
            'fold_details': fold_details,
            'mean_r2': float(mean_r2),
            'std_r2': float(std_r2),
            'cv_r2': float(cv_r2),
            'stability_grade': stability_grade,
            'stability_status': stability_status,
            'test_passed': std_r2 < 0.10
        }
        
        return results
    
    def early_vs_late_rul_performance(self, df, target_col='rul'):
        """
        Test model performance on early life vs late life predictions
        Critical for RUL applications
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 2: EARLY vs LATE RUL PREDICTION ACCURACY")
            print("="*80)
        
        # Get features (exclude only target, keep timestamp for datetime engineering)
        features = [col for col in df.columns if col != target_col]
        
        # Split into RUL ranges
        max_rul = df[target_col].max()
        early_life = df[df[target_col] > max_rul * 0.66]  # >66% remaining life
        mid_life = df[(df[target_col] > max_rul * 0.33) & (df[target_col] <= max_rul * 0.66)]
        late_life = df[df[target_col] <= max_rul * 0.33]  # <33% remaining life
        
        results = {}
        
        for phase_name, phase_df in [('early_life', early_life), ('mid_life', mid_life), ('late_life', late_life)]:
            if len(phase_df) < 50:
                continue
            
            X = phase_df[features]
            y = phase_df[target_col]
            
            y_pred = self.predictor.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            results[phase_name] = {
                'samples': len(phase_df),
                'rul_range': f"{phase_df[target_col].min():.1f}-{phase_df[target_col].max():.1f}h",
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            }
            
            if self.verbose:
                print(f"\n{phase_name.replace('_', ' ').title()}:")
                print(f"  RUL Range:  {results[phase_name]['rul_range']}")
                print(f"  Samples:    {len(phase_df):,}")
                print(f"  R²:         {r2:.4f}")
                print(f"  RMSE:       {rmse:.2f}h")
                print(f"  MAE:        {mae:.2f}h")
        
        # Check consistency across life phases
        r2_values = [v['r2'] for v in results.values()]
        r2_spread = max(r2_values) - min(r2_values) if r2_values else 0
        
        if r2_spread < 0.05:
            consistency_grade = 'A'
            consistency_status = '✅ EXCELLENT'
        elif r2_spread < 0.10:
            consistency_grade = 'B'
            consistency_status = '⚠️ GOOD'
        elif r2_spread < 0.15:
            consistency_grade = 'C'
            consistency_status = '⚠️ FAIR'
        else:
            consistency_grade = 'D'
            consistency_status = '❌ POOR'
        
        if self.verbose:
            print(f"\nLife Phase Consistency:")
            print(f"  R² Spread:        {r2_spread:.4f}")
            print(f"  Consistency:      {consistency_grade} {consistency_status}")
        
        results['consistency_grade'] = consistency_grade
        results['consistency_status'] = consistency_status
        results['r2_spread'] = float(r2_spread)
        results['test_passed'] = r2_spread < 0.15
        
        return results
    
    def prediction_bias_analysis(self, df, target_col='rul'):
        """
        Check for systematic over/under-estimation
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 3: PREDICTION BIAS ANALYSIS")
            print("="*80)
        
        # Get features (exclude only target, keep timestamp for datetime engineering)
        features = [col for col in df.columns if col != target_col]
        
        X = df[features]
        y_true = df[target_col]
        y_pred = self.predictor.predict(X)
        
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        median_residual = np.median(residuals)
        
        # Check for bias
        over_predictions = (residuals > 0).sum()
        under_predictions = (residuals < 0).sum()
        
        bias_ratio = over_predictions / len(residuals) if len(residuals) > 0 else 0.5
        
        # Grade bias
        if abs(mean_residual) < 5:
            bias_grade = 'A'
            bias_status = '✅ EXCELLENT'
        elif abs(mean_residual) < 10:
            bias_grade = 'B'
            bias_status = '⚠️ GOOD'
        elif abs(mean_residual) < 20:
            bias_grade = 'C'
            bias_status = '⚠️ FAIR'
        else:
            bias_grade = 'D'
            bias_status = '❌ POOR'
        
        if self.verbose:
            print(f"Residual Statistics:")
            print(f"  Mean Residual:    {mean_residual:.2f} hours")
            print(f"  Median Residual:  {median_residual:.2f} hours")
            print(f"  Over-predictions: {over_predictions:,} ({over_predictions/len(residuals)*100:.1f}%)")
            print(f"  Under-predictions: {under_predictions:,} ({under_predictions/len(residuals)*100:.1f}%)")
            print(f"  Bias Grade:       {bias_grade} {bias_status}")
        
        results = {
            'mean_residual': float(mean_residual),
            'median_residual': float(median_residual),
            'over_predictions': int(over_predictions),
            'under_predictions': int(under_predictions),
            'bias_ratio': float(bias_ratio),
            'bias_grade': bias_grade,
            'bias_status': bias_status,
            'test_passed': abs(mean_residual) < 20
        }
        
        return results
    
    def critical_range_accuracy(self, df, target_col='rul', critical_threshold=100):
        """
        Test accuracy in critical RUL range (e.g., <100 hours)
        This is most important for maintenance planning
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"TEST 4: CRITICAL RANGE ACCURACY (RUL < {critical_threshold}h)")
            print("="*80)
        
        # Get features (exclude only target, keep timestamp for datetime engineering)
        features = [col for col in df.columns if col != target_col]
        
        # Filter critical range
        critical_df = df[df[target_col] < critical_threshold]
        
        if len(critical_df) < 50:
            if self.verbose:
                print(f"⚠️ Insufficient critical samples ({len(critical_df)}) - test skipped")
            return {
                'test_passed': False,
                'reason': 'insufficient_samples',
                'samples': len(critical_df)
            }
        
        X_critical = critical_df[features]
        y_critical = critical_df[target_col]
        
        y_pred = self.predictor.predict(X_critical)
        
        r2 = r2_score(y_critical, y_pred)
        rmse = np.sqrt(mean_squared_error(y_critical, y_pred))
        mae = mean_absolute_error(y_critical, y_pred)
        
        # Check for dangerous over-estimation (predicting more life than actual)
        over_estimate = y_pred - y_critical
        dangerous_over = (over_estimate > 50).sum()  # Over-estimating by >50 hours
        dangerous_ratio = dangerous_over / len(critical_df)
        
        # Grade critical accuracy
        if mae < 15 and dangerous_ratio < 0.05:
            critical_grade = 'A'
            critical_status = '✅ EXCELLENT'
        elif mae < 25 and dangerous_ratio < 0.10:
            critical_grade = 'B'
            critical_status = '⚠️ GOOD'
        elif mae < 40 and dangerous_ratio < 0.20:
            critical_grade = 'C'
            critical_status = '⚠️ FAIR'
        else:
            critical_grade = 'D'
            critical_status = '❌ POOR'
        
        if self.verbose:
            print(f"Critical Range Performance:")
            print(f"  Samples:              {len(critical_df):,}")
            print(f"  R²:                   {r2:.4f}")
            print(f"  RMSE:                 {rmse:.2f}h")
            print(f"  MAE:                  {mae:.2f}h")
            print(f"  Dangerous Over-Est:   {dangerous_over} ({dangerous_ratio*100:.1f}%)")
            print(f"  Critical Grade:       {critical_grade} {critical_status}")
        
        results = {
            'samples': len(critical_df),
            'threshold': critical_threshold,
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'dangerous_over_estimates': int(dangerous_over),
            'dangerous_ratio': float(dangerous_ratio),
            'critical_grade': critical_grade,
            'critical_status': critical_status,
            'test_passed': mae < 40 and dangerous_ratio < 0.20
        }
        
        return results
    
    def raspberry_pi_compatibility(self):
        """
        Check Raspberry Pi deployment readiness
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEST 5: RASPBERRY PI COMPATIBILITY")
            print("="*80)
        
        model_size_mb = self.results['model_size_mb']
        
        # Size check
        if model_size_mb < 50:
            size_grade = 'A'
            size_status = '✅ EXCELLENT'
        elif model_size_mb < 100:
            size_grade = 'B'
            size_status = '⚠️ ACCEPTABLE'
        elif model_size_mb < 500:
            size_grade = 'C'
            size_status = '⚠️ LARGE'
        else:
            size_grade = 'D'
            size_status = '❌ TOO LARGE'
        
        # Inference speed estimate (based on model type)
        model_name = self.predictor.model_best.lower()
        
        if 'lightgbm' in model_name and 'bag' not in model_name:
            inference_ms = 5
            speed_grade = 'A'
        elif 'lightgbm' in model_name or 'xgboost' in model_name:
            inference_ms = 20
            speed_grade = 'B'
        elif 'catboost' in model_name or 'randomforest' in model_name:
            inference_ms = 40
            speed_grade = 'C'
        else:
            inference_ms = 100
            speed_grade = 'D'
        
        pi_compatible = size_grade in ['A', 'B'] and speed_grade in ['A', 'B', 'C']
        
        if self.verbose:
            print(f"Model Size:           {model_size_mb:.2f} MB ({size_grade} {size_status})")
            print(f"Est. Inference:       ~{inference_ms}ms per sample (Grade {speed_grade})")
            print(f"Pi Compatible:        {'✅ YES' if pi_compatible else '❌ NO - NEEDS OPTIMIZATION'}")
        
        results = {
            'model_size_mb': model_size_mb,
            'size_grade': size_grade,
            'size_status': size_status,
            'estimated_inference_ms': inference_ms,
            'speed_grade': speed_grade,
            'pi_compatible': pi_compatible,
            'test_passed': pi_compatible
        }
        
        return results
    
    def overall_grade(self):
        """
        Calculate overall validation grade
        """
        # Collect all test results
        test_scores = []
        
        # Test data R² (40% weight)
        test_r2 = self.results.get('test_metrics', {}).get('r2', 0)
        if test_r2 > 0.90:
            test_scores.append(('A', 40))
        elif test_r2 > 0.80:
            test_scores.append(('B', 40))
        elif test_r2 > 0.70:
            test_scores.append(('C', 40))
        else:
            test_scores.append(('D', 40))
        
        # Temporal stability (20% weight)
        temporal = self.results.get('temporal_robustness', {})
        if temporal.get('stability_grade'):
            test_scores.append((temporal['stability_grade'], 20))
        
        # Life phase consistency (15% weight)
        life_phase = self.results.get('early_vs_late_rul', {})
        if life_phase.get('consistency_grade'):
            test_scores.append((life_phase['consistency_grade'], 15))
        
        # Prediction bias (10% weight)
        bias = self.results.get('prediction_bias', {})
        if bias.get('bias_grade'):
            test_scores.append((bias['bias_grade'], 10))
        
        # Critical range (15% weight)
        critical = self.results.get('critical_range', {})
        if critical.get('critical_grade'):
            test_scores.append((critical['critical_grade'], 15))
        
        # Calculate weighted score
        grade_points = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
        
        total_points = 0
        total_weight = 0
        
        for grade, weight in test_scores:
            total_points += grade_points.get(grade, 0) * weight
            total_weight += weight
        
        avg_score = total_points / total_weight if total_weight > 0 else 0
        
        # Assign overall grade
        if avg_score >= 3.5:
            overall_grade = 'A'
            deployment_ready = True
        elif avg_score >= 2.5:
            overall_grade = 'B'
            deployment_ready = True
        elif avg_score >= 1.5:
            overall_grade = 'C'
            deployment_ready = False
        else:
            overall_grade = 'D'
            deployment_ready = False
        
        return {
            'overall_grade': overall_grade,
            'weighted_score': round(avg_score, 2),
            'deployment_ready': deployment_ready,
            'component_scores': test_scores
        }
    
    def run_full_validation(self):
        """
        Run complete industrial validation suite
        """
        # Load model and data
        self.load_model()
        train_df, val_df, test_df = self.load_data()

        if train_df is None or val_df is None or test_df is None:
            # RUL not available; skip gracefully.
            return self.results
        
        # Test 1: Basic performance on all splits
        if self.verbose:
            print("\n" + "="*80)
            print("BASIC PERFORMANCE METRICS")
            print("="*80)
        
        # Get features (exclude only target, keep timestamp for datetime engineering)
        features = [col for col in test_df.columns if col != 'rul']
        
        # Test set
        X_test = test_df[features]
        y_test = test_df['rul']
        y_pred_test = self.predictor.predict(X_test)
        self.results['test_metrics'] = self.basic_performance_metrics(y_test, y_pred_test, "Test Set")
        
        # Validation set
        X_val = val_df[features]
        y_val = val_df['rul']
        y_pred_val = self.predictor.predict(X_val)
        self.results['val_metrics'] = self.basic_performance_metrics(y_val, y_pred_val, "Validation Set")
        
        # Test 2-5: Industrial validation tests
        self.results['temporal_robustness'] = self.temporal_robustness_test(test_df)
        self.results['early_vs_late_rul'] = self.early_vs_late_rul_performance(test_df)
        self.results['prediction_bias'] = self.prediction_bias_analysis(test_df)
        self.results['critical_range'] = self.critical_range_accuracy(test_df, critical_threshold=100)
        self.results['pi_compatibility'] = self.raspberry_pi_compatibility()
        
        # Overall grade
        self.results['overall_assessment'] = self.overall_grade()
        
        # Print summary
        if self.verbose:
            self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        overall = self.results['overall_assessment']
        
        print(f"\n🎯 Overall Grade: {overall['overall_grade']}")
        print(f"📊 Weighted Score: {overall['weighted_score']}/4.0")
        print(f"🚀 Deployment Ready: {'✅ YES' if overall['deployment_ready'] else '❌ NO - NEEDS IMPROVEMENT'}")
        
        print(f"\n📈 Test Set Performance:")
        test_metrics = self.results['test_metrics']
        print(f"   R²:    {test_metrics['r2']:.4f}")
        print(f"   RMSE:  {test_metrics['rmse']:.2f} hours")
        print(f"   MAE:   {test_metrics['mae']:.2f} hours")
        
        print(f"\n🔍 Validation Tests:")
        print(f"   Temporal Stability:  {self.results['temporal_robustness']['stability_grade']}")
        print(f"   Life Phase Consist:  {self.results['early_vs_late_rul']['consistency_grade']}")
        print(f"   Prediction Bias:     {self.results['prediction_bias']['bias_grade']}")
        print(f"   Critical Range:      {self.results['critical_range'].get('critical_grade', 'N/A')}")
        print(f"   Pi Compatibility:    {self.results['pi_compatibility']['size_grade']}")
        
        print(f"\n💾 Model Info:")
        print(f"   Best Model: {self.results['best_model']}")
        print(f"   Size: {self.results['model_size_mb']:.2f} MB")
        print(f"   Pi Compatible: {'✅' if self.results['pi_compatibility']['pi_compatible'] else '❌'}")
    
    def save_report(self, output_dir='../../reports/industrial_validation_regression'):
        """Save validation report to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f'{self.machine_id}_regression_validation.json'
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super().default(obj)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)
        
        if self.verbose:
            print(f"\n✅ Validation report saved: {output_file}")
        
        return output_file


def validate_all_regression_models(machines_file='../../config/priority_10_machines.txt'):
    """
    Validate all regression models and generate summary report
    """
    print("="*80)
    print("BATCH INDUSTRIAL VALIDATION: REGRESSION MODELS")
    print("="*80)
    
    # Load machine list (skip comments)
    with open(machines_file, 'r') as f:
        machines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    print(f"\nValidating {len(machines)} regression models...")
    
    all_results = []
    summary = {
        'validation_timestamp': datetime.now().isoformat(),
        'total_machines': len(machines),
        'results': []
    }
    
    for idx, machine_id in enumerate(machines, 1):
        print(f"\n{'#'*80}")
        print(f"# Machine {idx}/{len(machines)}: {machine_id}")
        print(f"{'#'*80}")
        
        model_path = Path(f'../../models/regression/{machine_id}')
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            summary['results'].append({
                'machine_id': machine_id,
                'status': 'model_not_found'
            })
            continue
        
        try:
            validator = RegressionIndustrialValidator(
                machine_id=machine_id,
                model_path=model_path,
                verbose=True
            )
            
            results = validator.run_full_validation()
            validator.save_report()

            if results.get('skipped'):
                summary['results'].append({
                    'machine_id': machine_id,
                    'status': 'skipped',
                    'reason': results.get('skip_reason', 'RUL not available')
                })
                continue
            
            # Add to summary
            summary['results'].append({
                'machine_id': machine_id,
                'status': 'success',
                'overall_grade': results['overall_assessment']['overall_grade'],
                'deployment_ready': results['overall_assessment']['deployment_ready'],
                'test_r2': results['test_metrics']['r2'],
                'test_rmse': results['test_metrics']['rmse'],
                'test_mae': results['test_metrics']['mae'],
                'model_size_mb': results['model_size_mb'],
                'pi_compatible': results['pi_compatibility']['pi_compatible'],
                'temporal_stability': results['temporal_robustness']['stability_grade'],
                'critical_range': results['critical_range'].get('critical_grade', 'N/A')
            })
            
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            summary['results'].append({
                'machine_id': machine_id,
                'status': 'failed',
                'error': str(e)
            })
    
    # Calculate summary statistics
    successful = [r for r in summary['results'] if r.get('status') == 'success']
    
    if successful:
        summary['statistics'] = {
            'successful_validations': len(successful),
            'avg_r2': round(np.mean([r['test_r2'] for r in successful]), 4),
            'avg_rmse': round(np.mean([r['test_rmse'] for r in successful]), 2),
            'avg_mae': round(np.mean([r['test_mae'] for r in successful]), 2),
            'avg_model_size_mb': round(np.mean([r['model_size_mb'] for r in successful]), 2),
            'deployment_ready_count': sum(1 for r in successful if r['deployment_ready']),
            'pi_compatible_count': sum(1 for r in successful if r['pi_compatible']),
            'grade_distribution': {
                'A': sum(1 for r in successful if r['overall_grade'] == 'A'),
                'B': sum(1 for r in successful if r['overall_grade'] == 'B'),
                'C': sum(1 for r in successful if r['overall_grade'] == 'C'),
                'D': sum(1 for r in successful if r['overall_grade'] == 'D')
            }
        }
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super().default(obj)
    
    # Save summary report
    summary_path = Path('../../reports/regression_industrial_validation_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # Print final summary
    print("\n" + "="*80)
    print("BATCH VALIDATION COMPLETED")
    print("="*80)
    
    if successful:
        stats = summary['statistics']
        print(f"\n✅ Successful: {stats['successful_validations']}/{len(machines)}")
        print(f"\n📊 Overall Statistics:")
        print(f"   Average R²:          {stats['avg_r2']:.4f}")
        print(f"   Average RMSE:        {stats['avg_rmse']:.2f} hours")
        print(f"   Average MAE:         {stats['avg_mae']:.2f} hours")
        print(f"   Average Size:        {stats['avg_model_size_mb']:.2f} MB")
        print(f"\n🚀 Deployment Status:")
        print(f"   Deployment Ready:    {stats['deployment_ready_count']}/{len(successful)}")
        print(f"   Pi Compatible:       {stats['pi_compatible_count']}/{len(successful)}")
        print(f"\n📈 Grade Distribution:")
        for grade in ['A', 'B', 'C', 'D']:
            count = stats['grade_distribution'][grade]
            print(f"   Grade {grade}: {count}")
    
    print(f"\n✅ Summary saved: {summary_path}")
    
    return summary


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Single machine validation
        machine_id = sys.argv[1]
        model_path = Path(f'../../models/regression/{machine_id}')
        
        validator = RegressionIndustrialValidator(
            machine_id=machine_id,
            model_path=model_path,
            verbose=True
        )
        
        results = validator.run_full_validation()
        validator.save_report()
    else:
        # Batch validation
        validate_all_regression_models()
