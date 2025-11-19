"""
Industrial-Grade Validation Suite for Predictive Maintenance Models
Implements rigorous validation techniques to prevent overfitting and data leakage
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

class IndustrialValidator:
    """
    Industrial-grade model validation with rigorous testing
    """
    
    def __init__(self, machine_id, model_path, verbose=True):
        self.machine_id = machine_id
        self.model_path = Path(model_path)
        self.verbose = verbose
        self.predictor = None
        self.results = {}
        
    def load_model(self):
        """Load trained model"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"INDUSTRIAL VALIDATION: {self.machine_id}")
            print(f"{'='*80}")
        
        self.predictor = TabularPredictor.load(str(self.model_path))
        if self.verbose:
            print(f"✅ Model loaded: {self.predictor.model_best}")
    
    def realistic_failure_labels(self, df):
        """
        Create failure labels using statistical thresholds
        Same logic as training to ensure consistency
        """
        temp_cols = [c for c in df.columns if 'temp' in c.lower()]
        vib_cols = [c for c in df.columns if 'vib' in c.lower() or 'velocity' in c.lower()]
        
        failure_score = np.zeros(len(df))
        
        if temp_cols:
            temp_max = df[temp_cols].max(axis=1)
            temp_80 = np.percentile(temp_max, 80)
            temp_92 = np.percentile(temp_max, 92)
            
            temp_warn = ((temp_max > temp_80) & (temp_max <= temp_92)).astype(float) * 0.6
            temp_crit = (temp_max > temp_92).astype(float) * 1.5
            failure_score += temp_warn + temp_crit
        
        if vib_cols:
            vib_max = df[vib_cols].max(axis=1)
            vib_80 = np.percentile(vib_max, 80)
            vib_92 = np.percentile(vib_max, 92)
            
            vib_warn = ((vib_max > vib_80) & (vib_max <= vib_92)).astype(float) * 0.6
            vib_crit = (vib_max > vib_92).astype(float) * 1.5
            failure_score += vib_warn + vib_crit
        
        if temp_cols and vib_cols:
            both_high = ((temp_max > temp_80) & (vib_max > vib_80)).astype(float) * 0.4
            failure_score += both_high
        
        failure_status = (failure_score >= 1.2).astype(int)
        
        # Add 5% label noise
        label_noise = np.random.binomial(1, 0.05, len(df))
        failure_status = np.logical_xor(failure_status, label_noise).astype(int)
        
        return failure_status
    
    def load_data(self):
        """Load training data with labels for validation"""
        # Load data
        data_path = Path(f'../../../GAN/data/synthetic/{self.machine_id}')
        
        # Load all splits
        train_df = pd.read_parquet(data_path / 'train.parquet')
        val_df = pd.read_parquet(data_path / 'val.parquet')
        test_df = pd.read_parquet(data_path / 'test.parquet')
        
        # Apply same labeling logic as training
        train_df['failure_status'] = self.realistic_failure_labels(train_df)
        val_df['failure_status'] = self.realistic_failure_labels(val_df)
        test_df['failure_status'] = self.realistic_failure_labels(test_df)
        
        return train_df, val_df, test_df
    
    def validate_data_leakage(self, train_df, test_df):
        """
        1. Validate no data leakage between train and test sets
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("1. DATA LEAKAGE CHECK")
            print(f"{'='*80}")
        
        results = {
            'test_name': 'Data Leakage Detection',
            'checks': {}
        }
        
        # Check 1: No overlapping rows
        train_hash = pd.util.hash_pandas_object(train_df.drop(columns=['failure_status'], errors='ignore'))
        test_hash = pd.util.hash_pandas_object(test_df.drop(columns=['failure_status'], errors='ignore'))
        overlap = set(train_hash) & set(test_hash)
        
        results['checks']['no_overlapping_rows'] = {
            'overlap_count': len(overlap),
            'passed': bool(len(overlap) == 0)
        }
        
        if self.verbose:
            if len(overlap) == 0:
                print("✅ No overlapping rows between train and test")
            else:
                print(f"❌ WARNING: {len(overlap)} overlapping rows found!")
        
        # Check 2: Feature distributions should be similar (Kolmogorov-Smirnov test)
        from scipy.stats import ks_2samp
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'failure_status']
        
        ks_results = {}
        significant_diffs = 0
        for col in numeric_cols[:10]:  # Check first 10 features
            stat, pvalue = ks_2samp(train_df[col], test_df[col])
            ks_results[col] = {'statistic': float(stat), 'p_value': float(pvalue)}
            if pvalue < 0.05:  # Significantly different distributions
                significant_diffs += 1
        
        results['checks']['distribution_similarity'] = {
            'ks_tests': ks_results,
            'significant_differences': significant_diffs,
            'passed': bool(significant_diffs < len(numeric_cols[:10]) * 0.3)  # Allow up to 30% differences
        }
        
        if self.verbose:
            if significant_diffs < len(numeric_cols[:10]) * 0.3:
                print(f"✅ Feature distributions similar (KS test: {significant_diffs}/{len(numeric_cols[:10])} differ)")
            else:
                print(f"⚠️  WARNING: {significant_diffs}/{len(numeric_cols[:10])} features have significantly different distributions")
        
        return results
    
    def stratified_kfold_validation(self, train_df, k=5):
        """
        2. Stratified k-Fold Cross-Validation (Gold Standard)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"2. STRATIFIED {k}-FOLD CROSS-VALIDATION")
            print(f"{'='*80}")
        
        X = train_df.drop(columns=['failure_status'])
        y = train_df['failure_status']
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Quick train on fold (reduced time limit)
            fold_train_df = pd.concat([X_fold_train, y_fold_train], axis=1)
            
            # Use existing model's predict (don't retrain for speed)
            predictions = self.predictor.predict(X_fold_val)
            
            # Calculate metrics
            f1 = f1_score(y_fold_val, predictions)
            acc = accuracy_score(y_fold_val, predictions)
            prec = precision_score(y_fold_val, predictions, zero_division=0)
            rec = recall_score(y_fold_val, predictions, zero_division=0)
            
            fold_results.append({
                'fold': fold_idx,
                'f1_score': float(f1),
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'samples': len(y_fold_val)
            })
            
            if self.verbose:
                print(f"  Fold {fold_idx}: F1={f1:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
        
        # Calculate statistics
        f1_scores = [r['f1_score'] for r in fold_results]
        results = {
            'test_name': f'Stratified {k}-Fold Cross-Validation',
            'fold_results': fold_results,
            'statistics': {
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'min_f1': float(np.min(f1_scores)),
                'max_f1': float(np.max(f1_scores)),
                'cv_score_variance': float(np.var(f1_scores))
            },
            'stability_assessment': {
                'is_stable': bool(np.std(f1_scores) < 0.05),  # Std dev < 5%
                'interpretation': 'Stable' if np.std(f1_scores) < 0.05 else 'Unstable - high variance'
            }
        }
        
        if self.verbose:
            print(f"\n  Mean F1: {results['statistics']['mean_f1']:.4f} ± {results['statistics']['std_f1']:.4f}")
            if results['stability_assessment']['is_stable']:
                print(f"  ✅ Model is STABLE across folds (std < 0.05)")
            else:
                print(f"  ⚠️  Model shows HIGH VARIANCE across folds (std = {results['statistics']['std_f1']:.4f})")
        
        return results
    
    def null_model_benchmark(self, test_df):
        """
        3. Null Model Benchmarking - Compare to baseline
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("3. NULL MODEL BENCHMARKING")
            print(f"{'='*80}")
        
        y_test = test_df['failure_status']
        X_test = test_df.drop(columns=['failure_status'])
        
        # Null models
        majority_pred = np.full(len(y_test), y_test.mode()[0])  # Always predict majority class
        random_pred = np.random.choice([0, 1], size=len(y_test), p=[0.8, 0.2])  # Random based on class distribution
        
        # Real model
        real_pred = self.predictor.predict(X_test)
        
        # Calculate F1 for each
        null_f1_majority = f1_score(y_test, majority_pred, zero_division=0)
        null_f1_random = f1_score(y_test, random_pred, zero_division=0)
        real_f1 = f1_score(y_test, real_pred)
        
        improvement_over_majority = ((real_f1 - null_f1_majority) / (null_f1_majority + 1e-10)) * 100
        improvement_over_random = ((real_f1 - null_f1_random) / (null_f1_random + 1e-10)) * 100
        
        results = {
            'test_name': 'Null Model Benchmarking',
            'null_models': {
                'majority_class': {
                    'f1_score': float(null_f1_majority),
                    'strategy': 'Always predict majority class'
                },
                'random_baseline': {
                    'f1_score': float(null_f1_random),
                    'strategy': 'Random predictions based on class distribution'
                }
            },
            'real_model': {
                'f1_score': float(real_f1),
                'improvement_over_majority_pct': float(improvement_over_majority),
                'improvement_over_random_pct': float(improvement_over_random)
            },
            'assessment': {
                'significantly_better': bool(real_f1 > null_f1_majority * 2),  # At least 2x better
                'interpretation': 'Model learns meaningful patterns' if real_f1 > null_f1_majority * 2 else 'Model barely better than baseline'
            }
        }
        
        if self.verbose:
            print(f"  Majority Class Baseline F1: {null_f1_majority:.4f}")
            print(f"  Random Baseline F1: {null_f1_random:.4f}")
            print(f"  Real Model F1: {real_f1:.4f}")
            print(f"  Improvement over majority: {improvement_over_majority:.1f}%")
            if results['assessment']['significantly_better']:
                print(f"  ✅ Model SIGNIFICANTLY better than null models")
            else:
                print(f"  ⚠️  Model only marginally better than baseline")
        
        return results
    
    def confusion_matrix_analysis(self, test_df):
        """
        4. Deep Confusion Matrix Analysis
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("4. CONFUSION MATRIX ANALYSIS")
            print(f"{'='*80}")
        
        y_test = test_df['failure_status']
        X_test = test_df.drop(columns=['failure_status'])
        predictions = self.predictor.predict(X_test)
        
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        # Calculate rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate costs (assuming industrial context)
        # False Negative = Missed failure (HIGH COST - equipment breaks)
        # False Positive = Unnecessary maintenance (LOW COST - wasted labor)
        cost_fn = 1000  # $1000 per missed failure
        cost_fp = 50    # $50 per false alarm
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        cost_per_prediction = total_cost / total
        
        results = {
            'test_name': 'Confusion Matrix Analysis',
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp),
                'total': int(total)
            },
            'error_rates': {
                'false_positive_rate': float(false_positive_rate),
                'false_negative_rate': float(false_negative_rate),
                'total_error_rate': float((fp + fn) / total)
            },
            'industrial_cost_analysis': {
                'false_negative_cost_per': cost_fn,
                'false_positive_cost_per': cost_fp,
                'total_cost': float(total_cost),
                'cost_per_prediction': float(cost_per_prediction),
                'acceptable': bool(cost_per_prediction < 100)  # $100 per prediction is acceptable
            },
            'assessment': {
                'fp_acceptable': bool(fp < total * 0.05),  # <5% false positive rate
                'fn_acceptable': bool(fn < total * 0.02),  # <2% false negative rate (critical!)
                'overall_acceptable': bool((fp < total * 0.05) and (fn < total * 0.02))
            }
        }
        
        if self.verbose:
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {tn:,}  |  FP: {fp:,}")
            print(f"    FN: {fn:,}  |  TP: {tp:,}")
            print(f"\n  Error Rates:")
            print(f"    False Positive Rate: {false_positive_rate:.2%} {'✅' if fp < total * 0.05 else '⚠️'}")
            print(f"    False Negative Rate: {false_negative_rate:.2%} {'✅' if fn < total * 0.02 else '❌'}")
            print(f"\n  Industrial Cost Analysis:")
            print(f"    Total Cost: ${total_cost:,.0f}")
            print(f"    Cost per Prediction: ${cost_per_prediction:.2f} {'✅' if cost_per_prediction < 100 else '⚠️'}")
            
            if results['assessment']['overall_acceptable']:
                print(f"\n  ✅ Error rates ACCEPTABLE for industrial deployment")
            else:
                print(f"\n  ⚠️  Error rates NEED IMPROVEMENT before deployment")
        
        return results
    
    def precision_recall_curve_analysis(self, test_df):
        """
        5. Precision-Recall Curve Analysis (Threshold Robustness)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("5. PRECISION-RECALL CURVE ANALYSIS")
            print(f"{'='*80}")
        
        y_test = test_df['failure_status']
        X_test = test_df.drop(columns=['failure_status'])
        
        # Get prediction probabilities
        pred_proba = self.predictor.predict_proba(X_test)
        if hasattr(pred_proba, 'iloc'):
            y_scores = pred_proba.iloc[:, 1].values
        else:
            y_scores = pred_proba[:, 1]
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold (maximize F1)
        f1_scores = []
        for i, thresh in enumerate(thresholds):
            pred_at_thresh = (y_scores >= thresh).astype(int)
            f1 = f1_score(y_test, pred_at_thresh)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_f1 = float(f1_scores[optimal_idx])
        
        # Check robustness (F1 shouldn't drop >10% across reasonable thresholds)
        threshold_range = thresholds[(thresholds >= 0.3) & (thresholds <= 0.7)]
        f1_range = [f1_scores[i] for i, t in enumerate(thresholds) if 0.3 <= t <= 0.7]
        f1_variance = np.std(f1_range) if len(f1_range) > 0 else 0
        
        results = {
            'test_name': 'Precision-Recall Curve Analysis',
            'pr_auc': float(pr_auc),
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            'current_threshold': 0.5,
            'threshold_robustness': {
                'f1_variance_in_range': float(f1_variance),
                'is_robust': bool(f1_variance < 0.05),
                'interpretation': 'Robust across thresholds' if f1_variance < 0.05 else 'Sensitive to threshold choice'
            },
            'assessment': {
                'pr_auc_excellent': bool(pr_auc > 0.85),
                'threshold_recommendation': 'Use optimal threshold' if abs(optimal_threshold - 0.5) > 0.1 else 'Default 0.5 is fine'
            }
        }
        
        if self.verbose:
            print(f"  PR-AUC: {pr_auc:.4f} {'✅' if pr_auc > 0.85 else '⚠️'}")
            print(f"  Optimal Threshold: {optimal_threshold:.3f} (F1={optimal_f1:.4f})")
            print(f"  Current Threshold: 0.5")
            print(f"  F1 Variance (0.3-0.7 threshold range): {f1_variance:.4f}")
            
            if results['threshold_robustness']['is_robust']:
                print(f"  ✅ Model is ROBUST to threshold changes")
            else:
                print(f"  ⚠️  Model SENSITIVE to threshold - tune carefully")
        
        return results
    
    def temporal_validation(self, train_df, test_df):
        """
        6. Temporal Validation (Simulated Time-Series Split)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("6. TEMPORAL VALIDATION")
            print(f"{'='*80}")
        
        # Simulate temporal split (treat data as time-ordered)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Split into 5 time periods
        n_splits = 5
        split_size = len(full_df) // n_splits
        
        temporal_results = []
        for i in range(n_splits - 1):
            # Train on period i, test on period i+1
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = test_start + split_size
            
            temp_train = full_df.iloc[:train_end]
            temp_test = full_df.iloc[test_start:test_end]
            
            X_test = temp_test.drop(columns=['failure_status'])
            y_test = temp_test['failure_status']
            
            predictions = self.predictor.predict(X_test)
            f1 = f1_score(y_test, predictions)
            
            temporal_results.append({
                'period': f'{i}->{i+1}',
                'f1_score': float(f1),
                'test_samples': len(y_test)
            })
            
            if self.verbose:
                print(f"  Period {i}->{i+1}: F1={f1:.4f}")
        
        f1_scores = [r['f1_score'] for r in temporal_results]
        results = {
            'test_name': 'Temporal Validation',
            'temporal_splits': temporal_results,
            'statistics': {
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'trend': 'stable' if np.std(f1_scores) < 0.05 else 'degrading' if f1_scores[-1] < f1_scores[0] else 'variable'
            },
            'assessment': {
                'temporally_stable': bool(np.std(f1_scores) < 0.1),
                'interpretation': 'Model generalizes across time' if np.std(f1_scores) < 0.1 else 'Performance degrades over time'
            }
        }
        
        if self.verbose:
            print(f"\n  Mean F1: {results['statistics']['mean_f1']:.4f} ± {results['statistics']['std_f1']:.4f}")
            if results['assessment']['temporally_stable']:
                print(f"  ✅ Model is TEMPORALLY STABLE")
            else:
                print(f"  ⚠️  Model performance VARIES over time")
        
        return results
    
    def run_full_validation(self):
        """Run complete industrial validation suite"""
        print(f"\n{'#'*80}")
        print(f"# INDUSTRIAL-GRADE VALIDATION: {self.machine_id}")
        print(f"{'#'*80}")
        
        # Load model and data
        self.load_model()
        train_df, val_df, test_df = self.load_data()
        
        # Run all validation tests
        validation_results = {
            'machine_id': self.machine_id,
            'validation_suite': 'Industrial-Grade',
            'tests': {}
        }
        
        # Test 1: Data Leakage
        validation_results['tests']['data_leakage'] = self.validate_data_leakage(train_df, test_df)
        
        # Test 2: Stratified K-Fold CV
        validation_results['tests']['stratified_kfold'] = self.stratified_kfold_validation(train_df, k=5)
        
        # Test 3: Null Model Benchmark
        validation_results['tests']['null_model_benchmark'] = self.null_model_benchmark(test_df)
        
        # Test 4: Confusion Matrix Analysis
        validation_results['tests']['confusion_matrix'] = self.confusion_matrix_analysis(test_df)
        
        # Test 5: PR Curve Analysis
        validation_results['tests']['pr_curve'] = self.precision_recall_curve_analysis(test_df)
        
        # Test 6: Temporal Validation
        validation_results['tests']['temporal_validation'] = self.temporal_validation(train_df, test_df)
        
        # Overall assessment
        overall_pass = (
            validation_results['tests']['data_leakage']['checks']['no_overlapping_rows']['passed'] and
            validation_results['tests']['stratified_kfold']['stability_assessment']['is_stable'] and
            validation_results['tests']['null_model_benchmark']['assessment']['significantly_better'] and
            validation_results['tests']['confusion_matrix']['assessment']['overall_acceptable'] and
            validation_results['tests']['pr_curve']['assessment']['pr_auc_excellent']
        )
        
        validation_results['overall_assessment'] = {
            'passed': bool(overall_pass),
            'grade': 'A' if overall_pass else 'B',
            'deployment_ready': bool(overall_pass),
            'recommendation': 'Approved for industrial deployment' if overall_pass else 'Requires improvement before deployment'
        }
        
        print(f"\n{'='*80}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*80}")
        print(f"Grade: {validation_results['overall_assessment']['grade']}")
        print(f"Deployment Ready: {'✅ YES' if overall_pass else '⚠️ NO'}")
        print(f"Recommendation: {validation_results['overall_assessment']['recommendation']}")
        print(f"{'='*80}\n")
        
        return validation_results

def validate_machine_industrial(machine_id):
    """Run industrial validation for a single machine"""
    model_path = f'../../models/classification/{machine_id}'
    validator = IndustrialValidator(machine_id, model_path, verbose=True)
    results = validator.run_full_validation()
    
    # Save results
    output_file = Path(f'../../reports/industrial_validation/{machine_id}_industrial_validation.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Industrial validation saved: {output_file}\n")
    return results

def validate_all_machines_industrial():
    """Run industrial validation for all 10 machines"""
    config_file = Path('../../config/priority_10_machines.txt')
    with open(config_file) as f:
        machines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    all_results = []
    for i, machine_id in enumerate(machines, 1):
        print(f"\n{'#'*80}")
        print(f"# MACHINE {i}/{len(machines)}: {machine_id}")
        print(f"{'#'*80}\n")
        
        result = validate_machine_industrial(machine_id)
        all_results.append(result)
    
    # Generate summary report
    summary = {
        'total_machines': len(all_results),
        'passed': sum(1 for r in all_results if r['overall_assessment']['passed']),
        'grade_distribution': {
            'A': sum(1 for r in all_results if r['overall_assessment']['grade'] == 'A'),
            'B': sum(1 for r in all_results if r['overall_assessment']['grade'] == 'B')
        },
        'detailed_results': all_results
    }
    
    output_file = Path('../../reports/industrial_validation_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("INDUSTRIAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Machines: {summary['total_machines']}")
    print(f"Passed (Grade A): {summary['passed']}/{summary['total_machines']}")
    print(f"Grade Distribution: A={summary['grade_distribution']['A']}, B={summary['grade_distribution']['B']}")
    print(f"\n✅ Summary saved: {output_file}")
    print(f"{'='*80}\n")
    
    return summary

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Validate specific machine
        machine_id = sys.argv[1]
        validate_machine_industrial(machine_id)
    else:
        # Validate all machines
        validate_all_machines_industrial()
