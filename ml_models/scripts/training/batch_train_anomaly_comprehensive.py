"""
BATCH COMPREHENSIVE ANOMALY DETECTION TRAINING
==============================================
Train comprehensive anomaly detection models for all 10 priority machines.
"""

import sys
import time
from pathlib import Path
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.training.train_anomaly_comprehensive import train_comprehensive_anomaly_detection
from config.model_config import AUTOGLUON_CONFIG

# Priority machines
PRIORITY_MACHINES = [
    'motor_siemens_1la7_001',
    'motor_abb_m3bp_002',
    'motor_weg_w22_003',
    'pump_grundfos_cr3_004',
    'pump_flowserve_ansi_005',
    'compressor_atlas_copco_ga30_001',
    'compressor_ingersoll_rand_2545_009',
    'cnc_dmg_mori_nlx_010',
    'hydraulic_beckwood_press_011',
    'cooling_tower_bac_vti_018'
]


def batch_train_comprehensive_anomaly():
    """Train comprehensive anomaly detection for all 10 priority machines"""
    
    print("\n" + "=" * 100)
    print("=" * 100)
    print("  BATCH COMPREHENSIVE ANOMALY DETECTION TRAINING")
    print("  10 Priority Machines × Multiple Anomaly Detection Algorithms")
    print("=" * 100)
    print("=" * 100 + "\n")
    
    config = AUTOGLUON_CONFIG['anomaly'].copy()
    
    results = []
    successful = 0
    failed = 0
    
    overall_start = time.time()
    
    for idx, machine_id in enumerate(PRIORITY_MACHINES, 1):
        print(f"\n{'#' * 100}")
        print(f"#  MACHINE {idx}/{len(PRIORITY_MACHINES)}: {machine_id}")
        print(f"{'#' * 100}")
        
        machine_start = time.time()
        
        try:
            report = train_comprehensive_anomaly_detection(machine_id, config)
            
            machine_time = time.time() - machine_start
            elapsed_total = time.time() - overall_start
            
            # Extract key metrics
            result = {
                'machine_id': machine_id,
                'status': 'SUCCESS',
                'best_model': report['best_model']['name'],
                'f1_score': report['best_model']['metrics'].get('f1_score', 0),
                'precision': report['best_model']['metrics'].get('precision', 0),
                'recall': report['best_model']['metrics'].get('recall', 0),
                'accuracy': report['best_model']['metrics'].get('accuracy', 0),
                'model_size_mb': report['best_model']['model_size_mb'],
                'n_detectors_trained': len(report['all_models']),
                'training_time_minutes': machine_time / 60,
                'error': None
            }
            
            results.append(result)
            successful += 1
            
            print(f"\n✅ SUCCESS: {machine_id}")
            print(f"   Best Model: {result['best_model']}")
            print(f"   F1 Score: {result['f1_score']:.4f}")
            print(f"   Training Time: {result['training_time_minutes']:.2f} minutes")
            
            # Progress update
            remaining = len(PRIORITY_MACHINES) - idx
            avg_time = elapsed_total / idx
            estimated_remaining = avg_time * remaining
            
            print(f"\n📊 PROGRESS UPDATE:")
            print(f"   Completed: {idx}/{len(PRIORITY_MACHINES)} ({idx/len(PRIORITY_MACHINES)*100:.1f}%)")
            print(f"   Successful: {successful}, Failed: {failed}")
            print(f"   Elapsed: {elapsed_total/60:.2f} minutes")
            print(f"   Estimated Remaining: {estimated_remaining/60:.2f} minutes")
            print(f"   Estimated Total: {(elapsed_total + estimated_remaining)/60:.2f} minutes")
            
        except Exception as e:
            machine_time = time.time() - machine_start
            
            result = {
                'machine_id': machine_id,
                'status': 'FAILED',
                'error': str(e),
                'training_time_minutes': machine_time / 60
            }
            
            results.append(result)
            failed += 1
            
            print(f"\n❌ FAILED: {machine_id}")
            print(f"   Error: {str(e)[:100]}")
            print(f"   Continuing to next machine...")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - overall_start
    
    print("\n\n" + "=" * 100)
    print("=" * 100)
    print("  BATCH TRAINING COMPLETED")
    print("=" * 100)
    print("=" * 100 + "\n")
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"   Total Machines: {len(PRIORITY_MACHINES)}")
    print(f"   Successful: {successful} ({successful/len(PRIORITY_MACHINES)*100:.1f}%)")
    print(f"   Failed: {failed}")
    print(f"   Total Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Average Time per Machine: {total_time/len(PRIORITY_MACHINES)/60:.2f} minutes")
    
    # Performance statistics
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if successful_results:
        f1_scores = [r['f1_score'] for r in successful_results]
        model_sizes = [r['model_size_mb'] for r in successful_results]
        training_times = [r['training_time_minutes'] for r in successful_results]
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   F1 Score:")
        print(f"      Average:  {sum(f1_scores)/len(f1_scores):.4f}")
        print(f"      Range:    {min(f1_scores):.4f} - {max(f1_scores):.4f}")
        print(f"      Models ≥ 0.70: {sum(1 for f1 in f1_scores if f1 >= 0.70)}/{len(f1_scores)}")
        print(f"      Models ≥ 0.80: {sum(1 for f1 in f1_scores if f1 >= 0.80)}/{len(f1_scores)}")
        
        print(f"\n💾 STORAGE:")
        print(f"      Total Size:   {sum(model_sizes):.2f} MB")
        print(f"      Average Size: {sum(model_sizes)/len(model_sizes):.2f} MB")
        print(f"      Range:        {min(model_sizes):.2f} - {max(model_sizes):.2f} MB")
        
        print(f"\n⏱️  TRAINING TIME:")
        print(f"      Total:    {sum(training_times):.2f} minutes")
        print(f"      Average:  {sum(training_times)/len(training_times):.2f} minutes")
        print(f"      Range:    {min(training_times):.2f} - {max(training_times):.2f} minutes")
        
        # Top 3 performers
        top_3 = sorted(successful_results, key=lambda x: x['f1_score'], reverse=True)[:3]
        
        print(f"\n🏆 TOP 3 PERFORMERS:")
        for i, result in enumerate(top_3, 1):
            print(f"   {i}. {result['machine_id']}")
            print(f"      F1={result['f1_score']:.4f}, "
                  f"Model={result['best_model']}, "
                  f"Size={result['model_size_mb']:.2f}MB")
        
        # Best models distribution
        best_models = [r['best_model'] for r in successful_results]
        from collections import Counter
        model_counts = Counter(best_models)
        
        print(f"\n🔍 BEST MODEL DISTRIBUTION:")
        for model, count in model_counts.most_common():
            print(f"   {model}: {count}/{len(successful_results)} ({count/len(successful_results)*100:.1f}%)")
    
    # Failed machines
    if failed > 0:
        print(f"\n❌ FAILED MACHINES:")
        failed_results = [r for r in results if r['status'] == 'FAILED']
        for result in failed_results:
            print(f"   • {result['machine_id']}: {result['error'][:80]}")
    
    # Save batch report
    project_root = Path(__file__).parent.parent.parent.parent
    report_path = project_root / 'ml_models' / 'reports' / f'batch_comprehensive_anomaly_{len(PRIORITY_MACHINES)}_machines_report.json'
    
    batch_report = {
        'timestamp': datetime.now().isoformat(),
        'n_machines': len(PRIORITY_MACHINES),
        'successful': successful,
        'failed': failed,
        'total_time_minutes': total_time / 60,
        'results': results,
        'summary': {
            'avg_f1_score': sum(r['f1_score'] for r in successful_results) / len(successful_results) if successful_results else 0,
            'avg_model_size_mb': sum(r['model_size_mb'] for r in successful_results) / len(successful_results) if successful_results else 0,
            'total_storage_mb': sum(r['model_size_mb'] for r in successful_results) if successful_results else 0,
            'models_ge_70_f1': sum(1 for r in successful_results if r['f1_score'] >= 0.70) if successful_results else 0,
            'models_ge_80_f1': sum(1 for r in successful_results if r['f1_score'] >= 0.80) if successful_results else 0
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(batch_report, f, indent=2, default=str)
    
    print(f"\n📄 BATCH REPORT SAVED: {report_path}")
    
    print("\n" + "=" * 100)
    print("  ✅ BATCH TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 100 + "\n")
    
    return batch_report


if __name__ == "__main__":
    batch_train_comprehensive_anomaly()
