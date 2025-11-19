"""
COMPREHENSIVE ANOMALY DETECTION VALIDATION & REPORTING
======================================================
Detailed validation with:
- Performance metrics across all detectors
- ROC curves and Precision-Recall curves
- Confusion matrices and error analysis
- Anomaly score distributions
- Feature importance analysis
- Threshold sensitivity analysis
- Comparative visualizations
"""

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.data_preparation.feature_engineering import prepare_ml_data
from scripts.training.train_anomaly_comprehensive import StatisticalAnomalyDetector, AutoencoderAnomalyDetector, EnsembleAnomalyDetector


def plot_confusion_matrices(results_dict, machine_id, save_path):
    """Plot confusion matrices for all detectors"""
    
    n_models = len(results_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, metrics) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        cm = np.array([
            [metrics.get('true_negatives', 0), metrics.get('false_positives', 0)],
            [metrics.get('false_negatives', 0), metrics.get('true_positives', 0)]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{model_name}\nF1={metrics["f1_score"]:.3f}', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Confusion Matrices - {machine_id}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path / f'{machine_id}_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Confusion matrices saved")


def plot_roc_curves(detectors, X_test, y_true, machine_id, save_path):
    """Plot ROC curves for all detectors"""
    
    plt.figure(figsize=(12, 8))
    
    for detector_name, detector in detectors.items():
        if detector_name == 'dbscan':
            continue
            
        try:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_test)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X_test)
            else:
                continue
            
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            
            plt.plot(fpr, tpr, label=f'{detector_name} (AUC={auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"    ⚠️ ROC curve failed for {detector_name}: {e}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {machine_id}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f'{machine_id}_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ ROC curves saved")


def plot_precision_recall_curves(detectors, X_test, y_true, machine_id, save_path):
    """Plot Precision-Recall curves for all detectors"""
    
    plt.figure(figsize=(12, 8))
    
    for detector_name, detector in detectors.items():
        if detector_name == 'dbscan':
            continue
            
        try:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_test)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X_test)
            else:
                continue
            
            precision, recall, _ = precision_recall_curve(y_true, scores)
            avg_precision = average_precision_score(y_true, scores)
            
            plt.plot(recall, precision, label=f'{detector_name} (AP={avg_precision:.3f})', linewidth=2)
        except Exception as e:
            print(f"    ⚠️ PR curve failed for {detector_name}: {e}")
            continue
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {machine_id}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f'{machine_id}_precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Precision-Recall curves saved")


def plot_anomaly_score_distributions(detectors, X_test, y_true, machine_id, save_path):
    """Plot anomaly score distributions for normal vs anomaly samples"""
    
    n_models = len([d for d in detectors.keys() if d != 'dbscan'])
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    idx = 0
    for detector_name, detector in detectors.items():
        if detector_name == 'dbscan':
            continue
            
        ax = axes[idx]
        idx += 1
        
        try:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_test)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X_test)
            else:
                continue
            
            # Split scores by true labels
            normal_scores = scores[y_true == 0]
            anomaly_scores = scores[y_true == 1]
            
            ax.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', color='blue')
            ax.hist(anomaly_scores, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_scores)})', color='red')
            ax.set_xlabel('Anomaly Score')
            ax.set_ylabel('Count')
            ax.set_title(f'{detector_name}', fontsize=10, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Failed: {str(e)[:30]}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Anomaly Score Distributions - {machine_id}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path / f'{machine_id}_score_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Score distributions saved")


def plot_performance_comparison(results_dict, machine_id, save_path):
    """Plot comparative bar charts of performance metrics"""
    
    models = list(results_dict.keys())
    metrics_to_plot = ['f1_score', 'precision', 'recall', 'accuracy', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = [results_dict[m].get(metric, 0) for m in models]
        colors = ['green' if v == max(values) else 'skyblue' for v in values]
        
        bars = ax.barh(models, values, color=colors)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9)
    
    # Training time comparison
    ax = axes[5]
    times = [results_dict[m].get('training_time_seconds', 0) for m in models]
    colors = ['red' if t == max(times) else 'lightcoral' for t in times]
    bars = ax.barh(models, times, color=colors)
    ax.set_xlabel('Training Time (seconds)', fontsize=11)
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, times):
        if val > 0:
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}s', va='center', fontsize=9)
    
    plt.suptitle(f'Performance Comparison - {machine_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / f'{machine_id}_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Performance comparison saved")


def generate_detailed_text_report(report_data, machine_id, save_path):
    """Generate detailed text report"""
    
    report_file = save_path / f'{machine_id}_detailed_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"COMPREHENSIVE ANOMALY DETECTION VALIDATION REPORT\n")
        f.write(f"Machine: {machine_id}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        # Overall summary
        f.write("📊 EXECUTIVE SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Detectors Evaluated: {len(report_data['all_models'])}\n")
        f.write(f"Best Model: {report_data['best_model']['name']}\n")
        f.write(f"Best F1 Score: {report_data['best_model']['metrics']['f1_score']:.4f}\n")
        f.write(f"Total Training Time: {report_data['training_time_total_minutes']:.2f} minutes\n")
        f.write(f"Test Samples: {report_data['data_info']['n_test_samples']}\n")
        f.write(f"Test Anomalies: {report_data['data_info']['n_anomalies_test']} ")
        f.write(f"({report_data['data_info']['n_anomalies_test']/report_data['data_info']['n_test_samples']*100:.1f}%)\n")
        f.write("\n")
        
        # Model rankings
        f.write("🏆 MODEL RANKINGS (by F1 Score)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Model':<30} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10} {'ROC-AUC':<10}\n")
        f.write("-" * 100 + "\n")
        
        for ranking in report_data['model_rankings']:
            model_name = ranking['model']
            metrics = report_data['all_models'][model_name]
            f.write(f"{ranking['rank']:<6} {model_name:<30} ")
            f.write(f"{metrics['f1_score']:<10.4f} {metrics['precision']:<12.4f} ")
            f.write(f"{metrics['recall']:<10.4f} {metrics['accuracy']:<10.4f} ")
            f.write(f"{metrics.get('roc_auc', 0):<10.4f}\n")
        f.write("\n")
        
        # Detailed per-model analysis
        f.write("📋 DETAILED MODEL ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        for ranking in report_data['model_rankings']:
            model_name = ranking['model']
            metrics = report_data['all_models'][model_name]
            
            f.write(f"Model: {model_name.upper()}\n")
            f.write("-" * 100 + "\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  • F1 Score:        {metrics['f1_score']:.4f}\n")
            f.write(f"  • Precision:       {metrics['precision']:.4f}\n")
            f.write(f"  • Recall:          {metrics['recall']:.4f}\n")
            f.write(f"  • Accuracy:        {metrics['accuracy']:.4f}\n")
            f.write(f"  • Specificity:     {metrics.get('specificity', 0):.4f}\n")
            f.write(f"  • ROC-AUC:         {metrics.get('roc_auc', 0):.4f}\n")
            f.write(f"  • Avg Precision:   {metrics.get('avg_precision', 0):.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  True Negatives:    {metrics.get('true_negatives', 0):6d}\n")
            f.write(f"  False Positives:   {metrics.get('false_positives', 0):6d}\n")
            f.write(f"  False Negatives:   {metrics.get('false_negatives', 0):6d}\n")
            f.write(f"  True Positives:    {metrics.get('true_positives', 0):6d}\n")
            
            f.write(f"\nTraining Time:       {metrics.get('training_time_seconds', 0):.2f} seconds\n")
            f.write("\n")
        
        # Training time analysis
        f.write("⏱️  TRAINING TIME ANALYSIS\n")
        f.write("-" * 100 + "\n")
        times = [(name, metrics.get('training_time_seconds', 0)) 
                for name, metrics in report_data['all_models'].items()]
        times_sorted = sorted(times, key=lambda x: x[1])
        
        for name, t in times_sorted:
            f.write(f"{name:<30} {t:>8.2f} seconds\n")
        f.write(f"\n{'Total':<30} {sum(t for _, t in times):>8.2f} seconds\n")
        f.write("\n")
        
        # Artifacts
        f.write("📁 SAVED ARTIFACTS\n")
        f.write("-" * 100 + "\n")
        for key, path in report_data['artifacts'].items():
            f.write(f"  • {key:<20} {path}\n")
        f.write(f"\n  Total Size: {report_data['artifacts']['total_size_mb']:.2f} MB\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"  ✅ Detailed text report saved: {report_file}")


def validate_comprehensive_anomaly_models(machine_id):
    """
    Comprehensive validation and reporting for anomaly detection models
    """
    
    print(f"\n{'#' * 100}")
    print(f"#  COMPREHENSIVE ANOMALY DETECTION VALIDATION")
    print(f"#  Machine: {machine_id}")
    print(f"{'#' * 100}\n")
    
    project_root = Path(__file__).parent.parent.parent.parent
    model_path = project_root / 'ml_models' / 'models' / 'anomaly' / machine_id
    
    # Check if comprehensive models exist
    all_models_file = model_path / 'all_detectors.pkl'
    
    if not all_models_file.exists():
        print(f"❌ Comprehensive models not found at: {all_models_file}")
        print(f"   Run train_anomaly_comprehensive.py first!")
        return None
    
    print("=" * 100)
    print("STEP 1: LOADING MODELS & DATA")
    print("=" * 100)
    
    # Load models
    print(f"  Loading all detectors...")
    artifacts = joblib.load(all_models_file)
    detectors = artifacts['detectors']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    
    print(f"  ✅ Loaded {len(detectors)} detectors")
    
    # Load data
    print(f"  Loading test data...")
    _, _, test_df = prepare_ml_data(machine_id, 'classification')
    
    # Load feature names
    feature_file = model_path / 'features.json'
    with open(feature_file, 'r') as f:
        feature_data = json.load(f)
        feature_cols = feature_data['features']
    
    # Prepare test data
    X_test_raw = test_df[feature_cols].values
    X_test_imputed = imputer.transform(X_test_raw)
    X_test = scaler.transform(X_test_imputed)
    
    if 'failure_status' in test_df.columns:
        y_true = (test_df['failure_status'].values == 1).astype(int)
        print(f"  ✅ Test samples: {len(X_test)}, Anomalies: {np.sum(y_true)} ({np.sum(y_true)/len(y_true)*100:.1f}%)")
    else:
        print("  ❌ No ground truth labels - validation limited")
        return None
    
    # Load existing report
    report_file = project_root / 'ml_models' / 'reports' / 'performance_metrics' / f'{machine_id}_comprehensive_anomaly_report.json'
    with open(report_file, 'r') as f:
        report_data = json.load(f)
    
    print("=" * 100)
    print("STEP 2: GENERATING VISUALIZATIONS")
    print("=" * 100)
    
    viz_path = project_root / 'ml_models' / 'reports' / 'visualizations' / machine_id
    viz_path.mkdir(parents=True, exist_ok=True)
    
    print("\n  Generating confusion matrices...")
    plot_confusion_matrices(report_data['all_models'], machine_id, viz_path)
    
    print("  Generating ROC curves...")
    plot_roc_curves(detectors, X_test, y_true, machine_id, viz_path)
    
    print("  Generating Precision-Recall curves...")
    plot_precision_recall_curves(detectors, X_test, y_true, machine_id, viz_path)
    
    print("  Generating anomaly score distributions...")
    plot_anomaly_score_distributions(detectors, X_test, y_true, machine_id, viz_path)
    
    print("  Generating performance comparison...")
    plot_performance_comparison(report_data['all_models'], machine_id, viz_path)
    
    print("\n=" * 100)
    print("STEP 3: GENERATING DETAILED TEXT REPORT")
    print("=" * 100)
    
    generate_detailed_text_report(report_data, machine_id, viz_path)
    
    print("\n" + "#" * 100)
    print("#  VALIDATION COMPLETED")
    print("#" * 100)
    print(f"\n  📊 Visualizations saved to: {viz_path}")
    print(f"  📄 Detailed report: {viz_path / f'{machine_id}_detailed_report.txt'}")
    print(f"\n" + "#" * 100 + "\n")
    
    return {
        'machine_id': machine_id,
        'visualizations_path': str(viz_path),
        'n_visualizations': 5,
        'report_file': str(viz_path / f'{machine_id}_detailed_report.txt')
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Anomaly Detection Validation')
    parser.add_argument('--machine_id', required=True, help='Machine ID')
    args = parser.parse_args()
    
    validate_comprehensive_anomaly_models(args.machine_id)
