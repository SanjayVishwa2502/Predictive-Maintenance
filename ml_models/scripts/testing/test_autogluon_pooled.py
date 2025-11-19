"""
AutoGluon Baseline Testing Script - POOLED DATA VERSION
Phase 2.1.3: Test AutoGluon on sample machines using pooled data

This is MUCH FASTER than loading per-machine data!
"""

from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import time
from pathlib import Path
import json


def create_labels(df, task_type='classification'):
    """Create labels based on sensor thresholds"""
    
    if task_type == 'classification':
        # Create failure labels based on sensor anomalies
        failure_score = pd.Series(0, index=df.index)
        
        # Temperature anomaly (top 10% = failure)
        temp_cols = [col for col in df.columns if 'temp' in col.lower() and 'normalized' not in col.lower()]
        if temp_cols:
            for col in temp_cols:
                if df[col].notna().sum() > 100:
                    threshold = df[col].quantile(0.90)
                    failure_score += (df[col] > threshold).astype(int)
        
        # Vibration anomaly (top 10% = failure)
        vib_cols = [col for col in df.columns if 'vib' in col.lower() and col != 'vib_rms']
        if vib_cols:
            for col in vib_cols:
                if df[col].notna().sum() > 100:
                    threshold = df[col].quantile(0.90)
                    failure_score += (df[col] > threshold).astype(int)
        
        # Binary: failure if score >= 2
        df['failure_status'] = (failure_score >= 2).astype(int)
        return 'failure_status'
        
    elif task_type == 'regression':
        # Create RUL (Remaining Useful Life) labels
        max_rul = 1000.0
        
        # Use health_score if available, else calculate
        if 'health_score' in df.columns:
            # RUL inversely proportional to (100 - health_score)
            df['rul'] = max_rul * (df['health_score'] / 100.0)
        else:
            # Simple degradation model
            degradation = pd.Series(0.0, index=df.index)
            
            temp_cols = [col for col in df.columns if 'temp' in col.lower()]
            if temp_cols:
                for col in temp_cols[:3]:  # Use max 3 temp sensors
                    if df[col].notna().sum() > 100:
                        col_min, col_max = df[col].quantile(0.05), df[col].quantile(0.95)
                        if col_max > col_min:
                            degradation += ((df[col] - col_min) / (col_max - col_min)).fillna(0.5)
            
            # RUL decreases with degradation
            avg_degradation = degradation / max(1, len([c for c in temp_cols[:3] if df[c].notna().sum() > 100]))
            df['rul'] = max_rul * (1 - avg_degradation.clip(0, 1))
        
        df['rul'] = df['rul'].clip(50, max_rul)  # Minimum RUL of 50 hours
        return 'rul'


def test_autogluon_on_pooled_data(sample_machines, task_type='classification'):
    """Test AutoGluon using pooled data (FAST!)"""
    
    print(f"\n{'=' * 60}")
    print(f"Testing AutoGluon: POOLED DATA - {task_type.upper()}")
    print(f"Sample machines: {', '.join(sample_machines)}")
    print(f"{'=' * 60}\n")
    
    start_load = time.time()
    
    # Load pooled data (already combined from all machines!)
    print("Loading pooled data...")
    data_path = Path('../data/processed')
    
    train_df = pd.read_parquet(data_path / 'pooled_train.parquet')
    val_df = pd.read_parquet(data_path / 'pooled_val.parquet')
    test_df = pd.read_parquet(data_path / 'pooled_test.parquet')
    
    print(f"✓ Loaded in {time.time() - start_load:.1f}s: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")
    
    # Filter for sample machines (for quick test)
    print(f"\nFiltering to {len(sample_machines)} sample machines...")
    train_df = train_df[train_df['machine_id'].isin(sample_machines)].copy()
    val_df = val_df[val_df['machine_id'].isin(sample_machines)].copy()
    test_df = test_df[test_df['machine_id'].isin(sample_machines)].copy()
    
    print(f"✓ Filtered: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")
    
    # Create labels
    print(f"\nCreating {task_type} labels...")
    target_col = create_labels(train_df, task_type)
    create_labels(val_df, task_type)
    create_labels(test_df, task_type)
    
    # Check label distribution
    print(f"\nLabel distribution (train):")
    if task_type == 'classification':
        print(train_df[target_col].value_counts())
        print(f"Failure rate: {train_df[target_col].mean()*100:.1f}%")
    else:
        print(f"RUL range: [{train_df[target_col].min():.0f}, {train_df[target_col].max():.0f}]")
        print(f"RUL mean: {train_df[target_col].mean():.0f} hours")
    
    # Combine train + val for AutoGluon
    train_data = pd.concat([train_df, val_df], ignore_index=True)
    
    # Remove non-feature columns
    exclude_cols = ['machine_id', 'timestamp', target_col]
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Training samples: {len(train_data):,}")
    print(f"Test samples: {len(test_df):,}")
    
    # Initialize AutoGluon
    # Save model
    save_path = f'../models/{task_type}/pooled_test_{len(sample_machines)}_machines'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    predictor = TabularPredictor(
        label=target_col,
        path=save_path,
        eval_metric='f1' if task_type == 'classification' else 'r2',
        problem_type='binary' if task_type == 'classification' else 'regression'
    )
    
    # Train (quick 5-minute test)
    print(f"\n{'=' * 60}")
    print("Training AutoGluon (5-minute quick test)...")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    
    predictor.fit(
        train_data=train_data,
        time_limit=300,  # 5 minutes
        presets='medium_quality',  # Fast preset for testing
        verbosity=2
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    print(f"\n{'=' * 60}")
    print("Evaluating on test set...")
    print(f"{'=' * 60}\n")
    
    performance = predictor.evaluate(test_df)
    
    # Get leaderboard
    leaderboard = predictor.leaderboard(test_df, silent=True)
    
    # Detailed metrics
    y_true = test_df[target_col]
    y_pred = predictor.predict(test_df)
    
    if task_type == 'classification':
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'test_score': float(performance) if isinstance(performance, (int, float)) else float(leaderboard.iloc[0]['score_test'])
        }
    else:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': rmse,
            'mae': mean_absolute_error(y_true, y_pred),
            'test_score': float(performance) if isinstance(performance, (int, float)) else float(leaderboard.iloc[0]['score_test'])
        }
        
        print(f"\nR² Score: {metrics['r2_score']:.4f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {task_type.upper()}")
    print(f"{'=' * 60}")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"Performance: {performance}")
    print(f"\nTop 5 Models:")
    print(leaderboard.head())
    
    # Save report
    report = {
        'task_type': task_type,
        'sample_machines': sample_machines,
        'n_machines': len(sample_machines),
        'n_train_samples': len(train_data),
        'n_test_samples': len(test_df),
        'n_features': len(feature_cols),
        'training_time_minutes': training_time / 60,
        'metrics': metrics,
        'best_model': leaderboard.iloc[0]['model'],
        'best_score': float(leaderboard.iloc[0]['score_test']),
        'model_path': save_path
    }
    
    report_path = f'../reports/autogluon_test_{task_type}_{len(sample_machines)}_machines.json'
    Path('../reports').mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved: {report_path}")
    
    return report


if __name__ == "__main__":
    # Test on 3 sample machines
    sample_machines = [
        'motor_siemens_1la7_001',
        'pump_grundfos_cr3_004',
        'compressor_atlas_copco_ga30_001'
    ]
    
    print("=" * 70)
    print("AUTOGLUON BASELINE TESTING - POOLED DATA (FAST VERSION)")
    print("=" * 70)
    print(f"\nTesting on {len(sample_machines)} machines:")
    for machine in sample_machines:
        print(f"  - {machine}")
    
    results = []
    
    # Test classification
    print("\n\n" + "=" * 70)
    print("TASK 1: BINARY CLASSIFICATION (Failure Prediction)")
    print("=" * 70)
    result_cls = test_autogluon_on_pooled_data(sample_machines, 'classification')
    results.append(result_cls)
    
    # Test regression (RUL)
    print("\n\n" + "=" * 70)
    print("TASK 2: REGRESSION (RUL Prediction)")
    print("=" * 70)
    result_reg = test_autogluon_on_pooled_data(sample_machines, 'regression')
    results.append(result_reg)
    
    # Summary
    print("\n\n" + "=" * 70)
    print("AUTOGLUON TEST SUMMARY")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['task_type'].upper()}:")
        print(f"  Machines: {result['n_machines']}")
        print(f"  Training samples: {result['n_train_samples']:,}")
        print(f"  Training time: {result['training_time_minutes']:.2f} min")
        print(f"  Best model: {result['best_model']}")
        print(f"  Best score: {result['best_score']:.4f}")
        
        if result['task_type'] == 'classification':
            print(f"  F1 Score: {result['metrics']['f1_score']:.4f}")
            print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        else:
            print(f"  R² Score: {result['metrics']['r2_score']:.4f}")
            print(f"  RMSE: {result['metrics']['rmse']:.2f}")
            print(f"  MAE: {result['metrics']['mae']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ AutoGluon baseline testing complete!")
    print(f"✅ Total time: {sum(r['training_time_minutes'] for r in results):.1f} minutes")
    print("=" * 70)
