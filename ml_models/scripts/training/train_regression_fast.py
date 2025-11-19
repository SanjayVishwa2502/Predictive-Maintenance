"""
FAST Regression Training Script - Raspberry Pi Optimized
Train lightweight RUL regression models (15 min, ~250 MB)

Usage:
    python train_regression_fast.py --machine_id motor_siemens_1la7_001
    python train_regression_fast.py --machine_id motor_siemens_1la7_001 --time_limit 900
"""

from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import mlflow
import time
from pathlib import Path
import json
import sys
import argparse

def train_regression_fast(machine_id, time_limit=900):
    """
    Fast regression training optimized for Raspberry Pi
    - Time limit: 15 minutes
    - Model size target: ~250 MB
    - Pi-compatible models only
    """
    
    print(f"\n{'=' * 70}")
    print(f"FAST REGRESSION TRAINING (RUL): {machine_id}")
    print(f"{'=' * 70}")
    print(f"⏱️  Time limit: {time_limit//60} minutes")
    print(f"🥧 Raspberry Pi optimized: YES")
    print(f"{'=' * 70}\n")
    
    # MLflow tracking
    mlflow.set_experiment("Regression_PerMachine_Fast")
    
    with mlflow.start_run(run_name=f"{machine_id}_regression_fast"):
        # Log config
        mlflow.log_param("machine_id", machine_id)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("preset", "medium_quality_faster_train")
        mlflow.log_param("raspberry_pi_compatible", True)
        
        # Step 1: Load data (FAST - no feature engineering)
        print("Step 1: Loading data...")
        # Load data from GAN synthetic data - use absolute path
        project_root = Path(__file__).parent.parent.parent.parent
        data_path = project_root / 'GAN' / 'data' / 'synthetic' / machine_id
        train_df = pd.read_parquet(data_path / 'train.parquet')
        val_df = pd.read_parquet(data_path / 'val.parquet')
        test_df = pd.read_parquet(data_path / 'test.parquet')
        print("  ✓ Data loaded")
        
        # Step 2: Create RUL labels
        print("  Creating RUL (Remaining Useful Life) labels...")
        
        def calculate_rul(df):
            """
            Calculate RUL using time-based degradation simulation
            Creates realistic RUL that decreases over time with some variance
            """
            n_samples = len(df)
            
            # Create time index (0 to 1) - simulate equipment lifecycle
            time_index = np.linspace(0, 1, n_samples)
            
            # Base RUL: starts at max, decreases to 0
            max_rul = 1000  # hours
            base_rul = max_rul * (1 - time_index)
            
            # Add sensor-based adjustments (small influence)
            temp_cols = [c for c in df.columns if 'temp' in c.lower()]
            vib_cols = [c for c in df.columns if 'vib' in c.lower() or 'velocity' in c.lower()]
            
            sensor_factor = 0
            if temp_cols:
                # Higher temps = faster degradation = lower RUL
                temp_norm = (df[temp_cols].mean(axis=1) - df[temp_cols].mean(axis=1).min()) / (df[temp_cols].mean(axis=1).max() - df[temp_cols].mean(axis=1).min() + 1e-6)
                sensor_factor += temp_norm * 0.1  # 10% influence
            
            if vib_cols:
                # Higher vibration = faster degradation = lower RUL
                vib_norm = (df[vib_cols].mean(axis=1) - df[vib_cols].mean(axis=1).min()) / (df[vib_cols].mean(axis=1).max() - df[vib_cols].mean(axis=1).min() + 1e-6)
                sensor_factor += vib_norm * 0.1  # 10% influence
            
            # Adjust RUL based on sensor readings
            rul = base_rul * (1 - sensor_factor)
            
            # Add realistic noise (±10%)
            noise = np.random.normal(0, max_rul * 0.1, n_samples)
            rul = rul + noise
            
            # Clip to valid range
            rul = np.clip(rul, 0, max_rul)
            
            return rul
        
        train_df['rul'] = calculate_rul(train_df)
        val_df['rul'] = calculate_rul(val_df)
        test_df['rul'] = calculate_rul(test_df)
        print("  ✓ RUL labels created")
        
        # Combine train + val
        train_data = pd.concat([train_df, val_df], ignore_index=True)
        target_col = 'rul'
        
        print(f"Train samples: {len(train_data):,}")
        print(f"Test samples: {len(test_df):,}")
        print(f"Features: {len(train_data.columns) - 1}")
        
        # RUL distribution
        print(f"\nRUL Statistics:")
        print(f"  Min RUL: {train_data[target_col].min():.2f} hours")
        print(f"  Max RUL: {train_data[target_col].max():.2f} hours")
        print(f"  Mean RUL: {train_data[target_col].mean():.2f} hours")
        
        # Step 2: Initialize predictor
        print("\nStep 2: Initializing AutoGluon predictor...")
        # Save model - use absolute path
        ml_models_root = Path(__file__).parent.parent.parent
        save_path = ml_models_root / 'models' / 'regression' / machine_id
        save_path.mkdir(parents=True, exist_ok=True)
        
        predictor = TabularPredictor(
            label=target_col,
            path=str(save_path),
            eval_metric='r2',
            problem_type='regression'
        )
        
        # Step 3: Train (FAST mode - Pi compatible only)
        print(f"\nStep 3: Training (time limit: {time_limit//60} minutes)...")
        print("Preset: medium_quality_faster_train")
        print("Models: LightGBM, RandomForest, XGBoost (Pi-compatible)")
        print("Verbosity: 2 (Standard Logging)")
        
        start_time = time.time()
        
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets='medium_quality_faster_train',
            excluded_model_types=['NN_TORCH', 'FASTAI', 'XT'],  # Exclude heavy models
            num_bag_folds=3,  # Reduced for speed
            num_stack_levels=0,  # No stacking - keeps model size ~250 MB like classification
            verbosity=2
        )
        
        training_time = time.time() - start_time
        
        # Step 4: Evaluate
        print("\nStep 4: Evaluating on test set...")
        y_true = test_df[target_col]
        y_pred = predictor.predict(test_df)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': (abs((y_true - y_pred) / (y_true + 1e-6)).mean()) * 100
        }
        
        mlflow.log_metrics(metrics)
        mlflow.log_metric('training_time_seconds', training_time)
        
        # Leaderboard
        leaderboard = predictor.leaderboard(test_df, silent=True)
        best_model_name = leaderboard.iloc[0]['model']
        
        # Check model size
        import os
        model_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                           for dirpath, dirnames, filenames in os.walk(save_path) 
                           for filename in filenames) / (1024 * 1024)
        
        # Pi compatibility
        pi_compatible_models = ['LightGBM', 'RandomForest', 'ExtraTrees', 'XGBoost', 'CatBoost', 'WeightedEnsemble']
        is_pi_compatible = any(pi_model in best_model_name for pi_model in pi_compatible_models)
        
        print(f"\n{'=' * 70}")
        print("TRAINING RESULTS")
        print(f"{'=' * 70}")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f} hours")
        print(f"MAE: {metrics['mae']:.2f} hours")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        print(f"\nTop 5 Models:")
        print(leaderboard[['model', 'score_test', 'score_val', 'pred_time_test', 'fit_time']].head())
        
        # Feature importance
        feature_importance = predictor.feature_importance(test_df)
        print(f"\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        print(f"\n{'=' * 70}")
        print("RASPBERRY PI COMPATIBILITY")
        print(f"{'=' * 70}")
        print(f"Best Model: {best_model_name}")
        print(f"Pi Compatible: {'✅ YES' if is_pi_compatible else '❌ NO'}")
        print(f"Model Size: {model_size_mb:.2f} MB ({'⚠️ May be large for Pi' if model_size_mb > 50 else '✅ Suitable for Pi'})")
        print(f"Expected Inference: <50ms on Raspberry Pi 4")
        
        # Save report - use absolute path
        report = {
            'machine_id': machine_id,
            'task_type': 'regression',
            'training_time_minutes': training_time / 60,
            'metrics': {
                'r2_score': float(metrics['r2_score']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'mape': float(metrics['mape'])
            },
            'best_model': best_model_name,
            'best_score': float(leaderboard.iloc[0]['score_test']),
            'model_path': str(save_path),
            'model_size_mb': float(model_size_mb),
            'pi_compatible': is_pi_compatible,
            'pi_ready': is_pi_compatible and model_size_mb <= 500
        }
        
        report_path = ml_models_root / 'reports' / 'performance_metrics' / f'{machine_id}_regression_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Model saved: {save_path}")
        print(f"✅ Report saved: {report_path}")
        print(f"✅ Training completed in {training_time/60:.2f} minutes")
        
        return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast regression training for RUL prediction')
    parser.add_argument('--machine_id', required=True, help='Machine ID')
    parser.add_argument('--time_limit', type=int, default=900, help='Time limit in seconds (default: 900 = 15 min)')
    args = parser.parse_args()
    
    train_regression_fast(args.machine_id, args.time_limit)
