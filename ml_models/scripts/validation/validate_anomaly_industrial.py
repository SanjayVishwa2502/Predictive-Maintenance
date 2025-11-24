"""
INDUSTRIAL-GRADE ANOMALY DETECTION VALIDATION
==============================================
Rigorous validation framework for anomaly detection models.
Similar to regression industrial validation with comprehensive tests.

Author: AI Assistant
Date: November 22, 2025
"""

import sys
import io
import json
import joblib
import time
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# UTF-8 encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent.parent))

# Import feature engineering functions
from scripts.data_preparation.feature_engineering import add_engineered_features


class AnomalyIndustrialValidator:
    """Industrial-grade validation for anomaly detection models"""
    
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.model_path = Path(f'../../models/anomaly/{machine_id}')
        self.results = {
            'machine_id': machine_id,
            'validation_timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
    def load_model_and_data(self):
        """Load trained model and test data"""
        print(f"\n{'='*70}")
        print(f"Loading model and data for {self.machine_id}...")
        print(f"{'='*70}\n")
        
        # Load model files
        model_files = {
            'all_detectors': self.model_path / 'all_detectors.pkl',
            'preprocessing': self.model_path / 'preprocessing.pkl',
            'features': self.model_path / 'features.json'
        }
        
        for name, path in model_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {name}: {path}")
        
        # Mock TensorFlow/Keras if not available to allow unpickling
        import sys
        if 'tensorflow' not in sys.modules:
            sys.modules['tensorflow'] = type(sys)('tensorflow')
            sys.modules['tensorflow'].compat = type(sys)('compat')
            sys.modules['tensorflow'].compat.v2 = type(sys)('v2')
        if 'keras' not in sys.modules:
            sys.modules['keras'] = type(sys)('keras')
        
        # Load models (saved with joblib)
        self.ensemble = joblib.load(model_files['all_detectors'])
        self.preprocessing = joblib.load(model_files['preprocessing'])
            
        with open(model_files['features'], 'r') as f:
            self.feature_info = json.load(f)
        
        # Load test data directly from GAN synthetic data
        print("Loading test data...")
        project_root = Path(__file__).parent.parent.parent.parent
        data_path = project_root / 'GAN' / 'data' / 'synthetic' / self.machine_id / 'test.parquet'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_path}")
        
        test_df = pd.read_parquet(data_path)
        
        # Create failure labels (simple threshold-based for validation)
        if 'failure_status' not in test_df.columns:
            # Create labels based on RUL threshold
            test_df['failure_status'] = (test_df['rul'] < 100).astype(int)
        
        # CRITICAL: Apply feature engineering pipeline (same as training)
        print("  Applying feature engineering pipeline...")
        test_df = add_engineered_features(test_df, self.machine_id)
        
        # Separate features and labels
        feature_names = self.feature_info.get('features', self.feature_info.get('feature_names', []))
        
        # Filter to only features used during training that exist in engineered data
        available_features = [f for f in feature_names if f in test_df.columns]
        missing_features = [f for f in feature_names if f not in test_df.columns]
        
        if missing_features:
            print(f"  Warning: {len(missing_features)} features missing after engineering: {missing_features[:5]}")
        
        self.X_test = test_df[available_features]
        self.y_test = test_df['failure_status']
        
        print(f"  Using {len(available_features)}/{len(feature_names)} features")
        
        # Calculate model size
        model_size = sum(f.stat().st_size for f in model_files.values() if f.exists())
        self.results['model_size_mb'] = model_size / (1024 * 1024)
        
        print(f"✅ Model loaded: {self.results['model_size_mb']:.2f} MB")
        print(f"✅ Test samples: {len(self.X_test):,}")
        feature_names = self.feature_info.get('features', self.feature_info.get('feature_names', []))
        print(f"✅ Features: {len(feature_names)}")
        print(f"✅ Anomaly rate: {self.y_test.mean()*100:.2f}%\n")
        
    def test_1_basic_performance(self):
        """Test 1: Basic Performance Metrics"""
        print(f"\n{'='*70}")
        print("TEST 1: Basic Performance Metrics")
        print(f"{'='*70}\n")
        
        # Get predictions from best model
        best_model_name = self.feature_info.get('best_model', 'ensemble_voting')
        
        # Handle preprocessing - use all_detectors structure
        X_scaled = self.ensemble['imputer'].transform(self.X_test)
        X_scaled = self.ensemble['scaler'].transform(X_scaled)
        
        # Get predictions from all models
        predictions = {}
        detectors = self.ensemble['detectors']
        
        for model_name, model in detectors.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_scaled)
                    # Convert to binary (1 = anomaly, 0 = normal)
                    # sklearn models use -1 for anomaly, 1 for normal
                    pred = np.where(pred == -1, 1, 0)
                    predictions[model_name] = pred
                except Exception as e:
                    print(f"  Warning: Could not get predictions from {model_name}: {str(e)}")
        
        # Use best model or ensemble
        if best_model_name in predictions:
            y_pred = predictions[best_model_name]
        else:
            # Ensemble voting - majority vote
            pred_matrix = np.array([predictions[k] for k in predictions.keys()])
            y_pred = (pred_matrix.sum(axis=0) > (len(predictions) / 2)).astype(int)
        
        # Calculate metrics
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        self.results['tests']['basic_performance'] = {
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'npv': float(npv),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'grade': self._grade_f1(f1)
        }
        
        print(f"F1 Score:              {f1:.4f} [{self._grade_f1(f1)}]")
        print(f"Precision:             {precision:.4f}")
        print(f"Recall:                {recall:.4f}")
        print(f"Accuracy:              {accuracy:.4f}")
        print(f"Specificity:           {specificity:.4f}")
        print(f"NPV:                   {npv:.4f}")
        print(f"False Positive Rate:   {false_positive_rate:.4f}")
        print(f"False Negative Rate:   {false_negative_rate:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:,}  FP: {fp:,}")
        print(f"  FN: {fn:,}  TP: {tp:,}\n")
        
        return y_pred
        
    def test_2_algorithm_consistency(self):
        """Test 2: Algorithm Consistency Analysis"""
        print(f"\n{'='*70}")
        print("TEST 2: Algorithm Consistency Analysis")
        print(f"{'='*70}\n")
        
        # Get predictions from all algorithms
        X_scaled = self.ensemble['imputer'].transform(self.X_test)
        X_scaled = self.ensemble['scaler'].transform(X_scaled)
        
        algorithm_metrics = {}
        detectors = self.ensemble['detectors']
        
        for model_name, model in detectors.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_scaled)
                    # Convert to binary
                    pred = np.where(pred == -1, 1, 0)
                except Exception as e:
                    print(f"  Warning: Could not get predictions from {model_name}: {str(e)}")
                    continue
                
                f1 = f1_score(self.y_test, pred, zero_division=0)
                precision = precision_score(self.y_test, pred, zero_division=0)
                recall = recall_score(self.y_test, pred, zero_division=0)
                
                algorithm_metrics[model_name] = {
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall)
                }
        
        # Calculate consistency metrics
        f1_scores = [m['f1'] for m in algorithm_metrics.values()]
        f1_std = np.std(f1_scores)
        f1_range = max(f1_scores) - min(f1_scores)
        
        consistency_grade = self._grade_consistency(f1_std)
        
        self.results['tests']['algorithm_consistency'] = {
            'algorithm_metrics': algorithm_metrics,
            'f1_std': float(f1_std),
            'f1_range': float(f1_range),
            'f1_mean': float(np.mean(f1_scores)),
            'n_algorithms': len(algorithm_metrics),
            'grade': consistency_grade
        }
        
        print(f"Number of Algorithms:  {len(algorithm_metrics)}")
        print(f"F1 Mean:               {np.mean(f1_scores):.4f}")
        print(f"F1 Std Dev:            {f1_std:.4f} [{consistency_grade}]")
        print(f"F1 Range:              {f1_range:.4f}")
        print(f"\nPer-Algorithm Performance:")
        for name, metrics in sorted(algorithm_metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"  {name:20s}: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        print()
        
    def test_3_false_positive_analysis(self):
        """Test 3: False Positive Analysis"""
        print(f"\n{'='*70}")
        print("TEST 3: False Positive Analysis")
        print(f"{'='*70}\n")
        
        # Get predictions
        X_scaled = self.ensemble['imputer'].transform(self.X_test)
        X_scaled = self.ensemble['scaler'].transform(X_scaled)
        
        best_model_name = self.feature_info.get('best_model', 'ensemble_voting')
        predictions = {}
        detectors = self.ensemble['detectors']
        
        for model_name, model in detectors.items():
            if hasattr(model, 'predict'):
                try:
                    pred = model.predict(X_scaled)
                    pred = np.where(pred == -1, 1, 0)
                    predictions[model_name] = pred
                except Exception as e:
                    print(f"  Warning: Could not get predictions from {model_name}: {str(e)}")
                    continue
        
        if best_model_name in predictions:
            y_pred = predictions[best_model_name]
        else:
            pred_matrix = np.array([predictions[k] for k in predictions.keys()])
            y_pred = (pred_matrix.sum(axis=0) > (len(predictions) / 2)).astype(int)
        
        # Calculate false positive metrics
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Analyze false positives
        normal_samples = (self.y_test == 0).sum()
        fp_rate = (fp / normal_samples * 100) if normal_samples > 0 else 0
        
        # Grade based on false positive rate
        fp_grade = self._grade_false_positive_rate(fpr)
        
        self.results['tests']['false_positive_analysis'] = {
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_positive_rate': float(fpr),
            'fp_percentage': float(fp_rate),
            'normal_samples': int(normal_samples),
            'grade': fp_grade
        }
        
        print(f"Normal Samples:        {normal_samples:,}")
        print(f"False Positives:       {fp:,}")
        print(f"True Negatives:        {tn:,}")
        print(f"False Positive Rate:   {fpr:.4f} ({fp_rate:.2f}%) [{fp_grade}]")
        print(f"\n✅ Lower FPR is better (less false alarms)\n")
        
    def test_4_detection_latency(self):
        """Test 4: Detection Latency (Inference Speed)"""
        print(f"\n{'='*70}")
        print("TEST 4: Detection Latency Analysis")
        print(f"{'='*70}\n")
        
        # Prepare data
        X_scaled = self.ensemble['imputer'].transform(self.X_test)
        X_scaled = self.ensemble['scaler'].transform(X_scaled)
        
        # Test best model
        best_model_name = self.feature_info.get('best_model', 'isolation_forest')
        detectors = self.ensemble['detectors']
        
        if best_model_name in detectors:
            model = detectors[best_model_name]
        else:
            model = list(detectors.values())[0]
        
        # Measure latency
        latencies = []
        for _ in range(10):  # 10 runs for average
            start = time.time()
            _ = model.predict(X_scaled)
            latencies.append((time.time() - start) * 1000)  # ms
        
        avg_latency = np.mean(latencies)
        latency_per_sample = avg_latency / len(X_scaled)
        
        # Grade latency
        latency_grade = self._grade_latency(latency_per_sample)
        
        self.results['tests']['detection_latency'] = {
            'avg_latency_ms': float(avg_latency),
            'latency_per_sample_ms': float(latency_per_sample),
            'samples_tested': len(X_scaled),
            'n_runs': 10,
            'grade': latency_grade
        }
        
        print(f"Total Latency (avg):   {avg_latency:.2f} ms")
        print(f"Per-Sample Latency:    {latency_per_sample:.4f} ms [{latency_grade}]")
        print(f"Samples Tested:        {len(X_scaled):,}")
        print(f"Throughput:            {1000/latency_per_sample:.0f} predictions/sec\n")
        
    def test_5_pi_compatibility(self):
        """Test 5: Raspberry Pi Compatibility"""
        print(f"\n{'='*70}")
        print("TEST 5: Raspberry Pi Compatibility")
        print(f"{'='*70}\n")
        
        model_size_mb = self.results['model_size_mb']
        latency_ms = self.results['tests']['detection_latency']['latency_per_sample_ms']
        
        # Pi compatibility thresholds
        size_ok = model_size_mb < 50  # 50 MB target
        latency_ok = latency_ms < 100  # 100ms target
        
        # Calculate Pi memory footprint estimate
        runtime_memory = model_size_mb * 1.5 + 200  # Model + 200MB overhead
        pi_compatible = size_ok and runtime_memory < 1800  # 1.8GB limit for 4GB Pi
        
        pi_grade = "A" if pi_compatible and size_ok else ("B" if size_ok else ("C" if model_size_mb < 100 else "D"))
        
        self.results['tests']['pi_compatibility'] = {
            'model_size_mb': float(model_size_mb),
            'estimated_runtime_mb': float(runtime_memory),
            'latency_ms': float(latency_ms),
            'size_compatible': size_ok,
            'latency_compatible': latency_ok,
            'pi_compatible': pi_compatible,
            'grade': pi_grade
        }
        
        print(f"Model Size:            {model_size_mb:.2f} MB [{'✅' if size_ok else '❌'} {'<50MB' if size_ok else '≥50MB'}]")
        print(f"Estimated Runtime:     {runtime_memory:.2f} MB [{'✅' if runtime_memory < 1800 else '❌'}]")
        print(f"Inference Latency:     {latency_ms:.4f} ms [{'✅' if latency_ok else '❌'}]")
        print(f"Pi Compatible:         {'✅ YES' if pi_compatible else '❌ NO'} [{pi_grade}]")
        print()
        
    def overall_grade(self):
        """Calculate overall grade"""
        print(f"\n{'='*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*70}\n")
        
        tests = self.results['tests']
        
        # Weight grades
        weights = {
            'basic_performance': 0.40,  # 40% - Most important
            'algorithm_consistency': 0.20,  # 20%
            'false_positive_analysis': 0.20,  # 20%
            'detection_latency': 0.10,  # 10%
            'pi_compatibility': 0.10  # 10%
        }
        
        grade_points = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
        
        weighted_score = 0
        for test_name, weight in weights.items():
            grade = tests[test_name]['grade']
            weighted_score += grade_points[grade] * weight
        
        # Convert to letter grade
        if weighted_score >= 3.5:
            overall = 'A'
        elif weighted_score >= 2.5:
            overall = 'B'
        elif weighted_score >= 1.5:
            overall = 'C'
        else:
            overall = 'D'
        
        self.results['overall_grade'] = overall
        self.results['weighted_score'] = float(weighted_score)
        
        # Deployment readiness
        f1 = tests['basic_performance']['f1_score']
        fpr = tests['false_positive_analysis']['false_positive_rate']
        deployment_ready = (f1 >= 0.70 and fpr < 0.15 and overall in ['A', 'B'])
        
        self.results['deployment_ready'] = deployment_ready
        
        print(f"Test Grades:")
        print(f"  Basic Performance:        {tests['basic_performance']['grade']} (F1={tests['basic_performance']['f1_score']:.4f})")
        print(f"  Algorithm Consistency:    {tests['algorithm_consistency']['grade']}")
        print(f"  False Positive Analysis:  {tests['false_positive_analysis']['grade']} (FPR={tests['false_positive_analysis']['false_positive_rate']:.4f})")
        print(f"  Detection Latency:        {tests['detection_latency']['grade']}")
        print(f"  Pi Compatibility:         {tests['pi_compatibility']['grade']}")
        print(f"\nWeighted Score:           {weighted_score:.2f}/4.0")
        print(f"Overall Grade:            {overall}")
        print(f"Deployment Ready:         {'✅ YES' if deployment_ready else '❌ NO'}")
        print()
        
    # Grading helper methods
    def _grade_f1(self, f1):
        if f1 >= 0.85: return 'A'
        elif f1 >= 0.75: return 'B'
        elif f1 >= 0.65: return 'C'
        else: return 'D'
    
    def _grade_consistency(self, std):
        if std < 0.05: return 'A'
        elif std < 0.10: return 'B'
        elif std < 0.15: return 'C'
        else: return 'D'
    
    def _grade_false_positive_rate(self, fpr):
        if fpr < 0.05: return 'A'
        elif fpr < 0.10: return 'B'
        elif fpr < 0.15: return 'C'
        else: return 'D'
    
    def _grade_latency(self, latency_ms):
        if latency_ms < 1.0: return 'A'
        elif latency_ms < 5.0: return 'B'
        elif latency_ms < 10.0: return 'C'
        else: return 'D'
    
    def save_results(self):
        """Save validation results"""
        output_dir = Path('../../reports/industrial_validation_anomaly')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{self.machine_id}_anomaly_validation.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Validation results saved: {output_file}\n")


def validate_all_anomaly_models():
    """Validate all 10 priority machines"""
    
    print("\n" + "=" * 100)
    print("=" * 100)
    print("  INDUSTRIAL-GRADE ANOMALY DETECTION VALIDATION")
    print("  Rigorous Testing Framework for All 10 Priority Machines")
    print("=" * 100)
    print("=" * 100 + "\n")
    
    machines = [
        'motor_siemens_1la7_001', 'motor_abb_m3bp_002', 'motor_weg_w22_003',
        'pump_grundfos_cr3_004', 'pump_flowserve_ansi_005',
        'compressor_atlas_copco_ga30_001', 'compressor_ingersoll_rand_2545_009',
        'cnc_dmg_mori_nlx_010', 'hydraulic_beckwood_press_011',
        'cooling_tower_bac_vti_018'
    ]
    
    all_results = []
    successful = 0
    failed = 0
    
    overall_start = time.time()
    
    for idx, machine_id in enumerate(machines, 1):
        print(f"\n{'#'*100}")
        print(f"MACHINE {idx}/{len(machines)}: {machine_id}")
        print(f"{'#'*100}\n")
        
        try:
            validator = AnomalyIndustrialValidator(machine_id)
            validator.load_model_and_data()
            validator.test_1_basic_performance()
            validator.test_2_algorithm_consistency()
            validator.test_3_false_positive_analysis()
            validator.test_4_detection_latency()
            validator.test_5_pi_compatibility()
            validator.overall_grade()
            validator.save_results()
            
            all_results.append(validator.results)
            successful += 1
            
            print(f"✅ {machine_id}: Grade {validator.results['overall_grade']}")
            
        except Exception as e:
            print(f"❌ {machine_id}: FAILED - {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print()
            failed += 1
    
    total_time = time.time() - overall_start
    
    # Generate summary
    print(f"\n{'='*100}")
    print("VALIDATION SUMMARY")
    print(f"{'='*100}\n")
    
    print(f"Total Machines:        {len(machines)}")
    print(f"Successful:            {successful}")
    print(f"Failed:                {failed}")
    print(f"Total Time:            {total_time/60:.2f} minutes\n")
    
    if all_results:
        # Calculate statistics
        avg_f1 = np.mean([r['tests']['basic_performance']['f1_score'] for r in all_results])
        avg_fpr = np.mean([r['tests']['false_positive_analysis']['false_positive_rate'] for r in all_results])
        avg_size = np.mean([r['model_size_mb'] for r in all_results])
        deployment_ready = sum([r['deployment_ready'] for r in all_results])
        pi_compatible = sum([r['tests']['pi_compatibility']['pi_compatible'] for r in all_results])
        
        # Grade distribution
        grades = [r['overall_grade'] for r in all_results]
        grade_dist = {g: grades.count(g) for g in ['A', 'B', 'C', 'D']}
        
        print(f"Average F1 Score:      {avg_f1:.4f}")
        print(f"Average FPR:           {avg_fpr:.4f}")
        print(f"Average Model Size:    {avg_size:.2f} MB")
        print(f"Deployment Ready:      {deployment_ready}/{len(all_results)}")
        print(f"Pi Compatible:         {pi_compatible}/{len(all_results)}")
        print(f"\nGrade Distribution:")
        for grade in ['A', 'B', 'C', 'D']:
            print(f"  Grade {grade}: {grade_dist.get(grade, 0)}")
        
        # Save summary
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_machines': len(machines),
            'successful': successful,
            'failed': failed,
            'total_time_minutes': total_time / 60,
            'results': all_results,
            'statistics': {
                'avg_f1': float(avg_f1),
                'avg_fpr': float(avg_fpr),
                'avg_model_size_mb': float(avg_size),
                'deployment_ready_count': int(deployment_ready),
                'pi_compatible_count': int(pi_compatible),
                'grade_distribution': grade_dist
            }
        }
        
        summary_file = Path('../../reports/anomaly_industrial_validation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Summary saved: {summary_file}")
    
    print(f"\n{'='*100}")
    print("INDUSTRIAL VALIDATION COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    validate_all_anomaly_models()
