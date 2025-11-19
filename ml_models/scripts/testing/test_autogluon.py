"""
AutoGluon Baseline Testing Script
Phase 2.1.3: Test AutoGluon on 2-3 sample machines

This script tests AutoGluon with quick 5-minute training runs to:
1. Validate the training pipeline
2. Establish baseline performance metrics
3. Estimate full training times
"""

from autogluon.tabular import TabularPredictor
import pandas as pd
import time
from pathlib import Path
import json
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))
from feature_engineering import prepare_ml_data


def test_autogluon_on_machine(machine_id, task_type='classification'):
    """Test AutoGluon on single machine"""
    
    print(f"\n{'=' * 60}")
    print(f"Testing AutoGluon: {machine_id} - {task_type}")
    print(f"{'=' * 60}\n")
    
    try:
        # Load data
        print("Loading and preparing data...")
        train_df, val_df, test_df = prepare_ml_data(machine_id, task_type)
        
        # Combine train + val for AutoGluon
        train_data = pd.concat([train_df, val_df], ignore_index=True)
        
        # Define target
        target_col = 'failure_status' if task_type == 'classification' else 'rul'
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Target: {target_col}")
        
        # Check target distribution
        if task_type == 'classification':
            print(f"\nTarget distribution:")
            print(train_data[target_col].value_counts())
        else:
            print(f"\nTarget statistics:")
            print(train_data[target_col].describe())
        
        # Initialize AutoGluon
        # Save model
        save_path = f'../models/{task_type}/{machine_id}_autogluon'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        predictor = TabularPredictor(
            label=target_col,
            path=save_path,
            eval_metric='f1' if task_type == 'classification' else 'r2',
            problem_type='binary' if task_type == 'classification' else 'regression'
        )
        
        # Train (quick test: 5 minutes)
        print(f"\nStarting AutoGluon training (5-minute quick test)...")
        start_time = time.time()
        
        predictor.fit(
            train_data=train_data,
            time_limit=300,  # 5 minutes for quick test
            presets='medium_quality',  # Fast preset for testing
            verbosity=2
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        print("\nEvaluating on test set...")
        performance = predictor.evaluate(test_df)
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(test_df, silent=True)
        
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {machine_id}")
        print(f"{'=' * 60}")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"Performance: {performance}")
        print(f"\nTop 5 Models:")
        print(leaderboard.head())
        
        # Feature importance
        try:
            feature_importance = predictor.feature_importance(test_df)
            print(f"\nTop 10 Important Features:")
            print(feature_importance.head(10))
        except Exception as e:
            print(f"\nCould not compute feature importance: {e}")
            feature_importance = None
        
        # Save report
        report = {
            'machine_id': machine_id,
            'task_type': task_type,
            'training_time_minutes': round(training_time / 60, 2),
            'performance': float(performance) if isinstance(performance, (int, float)) else str(performance),
            'best_model': str(leaderboard.iloc[0]['model']),
            'best_score': float(leaderboard.iloc[0]['score_test']),
            'train_samples': len(train_data),
            'test_samples': len(test_df),
            'n_features': len(train_data.columns) - 1,
            'status': 'success'
        }
        
        if feature_importance is not None:
            report['top_features'] = feature_importance.head(10).to_dict()
        
        report_path = f'../reports/{machine_id}_{task_type}_autogluon_test.json'
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved: {report_path}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error report
        error_report = {
            'machine_id': machine_id,
            'task_type': task_type,
            'status': 'failed',
            'error': str(e)
        }
        
        report_path = f'../reports/{machine_id}_{task_type}_autogluon_test.json'
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        return error_report


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2.1.3: AutoML Baseline Testing")
    print("=" * 70)
    print("\nTesting AutoGluon on 3 sample machines")
    print("Quick 5-minute training runs to establish baseline performance\n")
    
    # Test on 3 sample machines
    test_machines = [
        'motor_siemens_1la7_001',
        'pump_grundfos_cr3_004',
        'compressor_atlas_copco_ga30_001'
    ]
    
    results = []
    
    for machine_id in test_machines:
        # Test classification
        print(f"\n{'#' * 70}")
        print(f"Testing Machine: {machine_id}")
        print(f"{'#' * 70}")
        
        result_cls = test_autogluon_on_machine(machine_id, 'classification')
        results.append(result_cls)
        
        # Test regression (RUL)
        result_reg = test_autogluon_on_machine(machine_id, 'regression')
        results.append(result_reg)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("AUTOGLUON TEST SUMMARY")
    print(f"{'=' * 70}")
    
    successful_tests = [r for r in results if r.get('status') == 'success']
    failed_tests = [r for r in results if r.get('status') == 'failed']
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n{'=' * 70}")
        print("SUCCESSFUL TEST RESULTS:")
        print(f"{'=' * 70}")
        
        for result in successful_tests:
            print(f"\n{result['machine_id']} - {result['task_type']}")
            print(f"  Time: {result['training_time_minutes']:.2f} min")
            print(f"  Best Model: {result['best_model']}")
            print(f"  Score: {result['best_score']:.4f}")
    
    if failed_tests:
        print(f"\n{'=' * 70}")
        print("FAILED TESTS:")
        print(f"{'=' * 70}")
        
        for result in failed_tests:
            print(f"\n{result['machine_id']} - {result['task_type']}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Save summary report
    summary_report = {
        'phase': '2.1.3',
        'total_tests': len(results),
        'successful_tests': len(successful_tests),
        'failed_tests': len(failed_tests),
        'test_machines': test_machines,
        'results': results
    }
    
    summary_path = '../reports/phase_2_1_3_autogluon_test_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\n✅ Summary report saved: {summary_path}")
    print(f"\n{'=' * 70}")
    print("PHASE 2.1.3: AutoML Baseline Testing Complete!")
    print(f"{'=' * 70}")
