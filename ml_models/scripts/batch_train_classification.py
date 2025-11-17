# ml_models/scripts/batch_train_classification.py
# Batch training script for per-machine classification models
# Phase 2.2.2: Train all 10 priority machines sequentially

import subprocess
import time
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def load_machine_list(machines_file):
    """Load list of machines from file"""
    with open(machines_file, 'r') as f:
        machines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return machines

def train_all_machines(machines_file, time_limit=900, presets='medium_quality_faster_train'):
    """
    Train classification models for all machines in the list
    FAST MODE: Pi-compatible, lightweight models
    
    Args:
        machines_file: Path to file containing machine IDs (one per line)
        time_limit: Training time limit per machine in seconds (default: 900 = 15 min)
        presets: AutoGluon presets (default: medium_quality_faster_train)
    """
    
    print("\n" + "=" * 70)
    print("BATCH TRAINING: PER-MACHINE CLASSIFICATION MODELS (FAST MODE)")
    print("Phase 2.2.2: Pi-Compatible Models for 10 Priority Machines")
    print("=" * 70 + "\n")
    
    # Load machine list
    machines = load_machine_list(machines_file)
    total_machines = len(machines)
    
    print(f"Machines to train: {total_machines}")
    print(f"Time limit per machine: {time_limit}s ({time_limit/60:.0f} minutes)")
    print(f"Presets: {presets}")
    print(f"Estimated total time: {total_machines * time_limit / 3600:.1f} hours\n")
    
    print("Machine list:")
    for i, machine_id in enumerate(machines, 1):
        print(f"  {i}. {machine_id}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Training results
    results = []
    start_time_total = time.time()
    
    # Train each machine
    for i, machine_id in enumerate(machines, 1):
        print(f"\n{'#' * 70}")
        print(f"# Machine {i}/{total_machines}: {machine_id}")
        print(f"{'#' * 70}\n")
        
        machine_start_time = time.time()
        
        try:
            # Run FAST training script (Pi-compatible)
            cmd = [
                sys.executable,
                'scripts/train_classification_fast.py',
                '--machine_id', machine_id,
                '--time_limit', str(time_limit)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                capture_output=False,
                text=True,
                check=True
            )
            
            machine_training_time = time.time() - machine_start_time
            
            # Load report to get metrics
            report_path = Path(f'reports/performance_metrics/{machine_id}_classification_report.json')
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                results.append({
                    'machine_id': machine_id,
                    'status': 'success',
                    'f1_score': report['metrics']['f1_score'],
                    'accuracy': report['metrics']['accuracy'],
                    'training_time_minutes': machine_training_time / 60,
                    'best_model': report['best_model']
                })
                
                print(f"\n✅ SUCCESS: {machine_id}")
                print(f"   F1 Score: {report['metrics']['f1_score']:.4f}")
                print(f"   Time: {machine_training_time/60:.2f} minutes")
            else:
                results.append({
                    'machine_id': machine_id,
                    'status': 'success',
                    'training_time_minutes': machine_training_time / 60,
                    'note': 'Report file not found'
                })
        
        except subprocess.CalledProcessError as e:
            machine_training_time = time.time() - machine_start_time
            print(f"\n❌ FAILED: {machine_id}")
            print(f"   Error: {str(e)}")
            results.append({
                'machine_id': machine_id,
                'status': 'failed',
                'error': str(e),
                'training_time_minutes': machine_training_time / 60
            })
        
        except Exception as e:
            machine_training_time = time.time() - machine_start_time
            print(f"\n❌ ERROR: {machine_id}")
            print(f"   Error: {str(e)}")
            results.append({
                'machine_id': machine_id,
                'status': 'error',
                'error': str(e),
                'training_time_minutes': machine_training_time / 60
            })
        
        # Progress update
        elapsed_total = time.time() - start_time_total
        avg_time_per_machine = elapsed_total / i
        remaining_machines = total_machines - i
        estimated_remaining = avg_time_per_machine * remaining_machines
        
        print(f"\n{'=' * 70}")
        print(f"Progress: {i}/{total_machines} machines completed")
        print(f"Elapsed: {elapsed_total/3600:.2f} hours")
        print(f"Estimated remaining: {estimated_remaining/3600:.2f} hours")
        print(f"{'=' * 70}\n")
    
    total_time = time.time() - start_time_total
    
    # Final summary
    print("\n" + "=" * 70)
    print("BATCH TRAINING COMPLETED")
    print("=" * 70 + "\n")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    print(f"Total machines: {total_machines}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average time per machine: {total_time/total_machines/60:.1f} minutes\n")
    
    if successful:
        print("Successful models:")
        print(f"{'Machine ID':<40} {'F1 Score':>10} {'Time (min)':>12} {'Best Model':<20}")
        print("-" * 90)
        for r in successful:
            if 'f1_score' in r:
                print(f"{r['machine_id']:<40} {r['f1_score']:>10.4f} {r['training_time_minutes']:>12.1f} {r.get('best_model', 'N/A'):<20}")
            else:
                print(f"{r['machine_id']:<40} {'N/A':>10} {r['training_time_minutes']:>12.1f} {r.get('note', ''):<20}")
        
        if successful and all('f1_score' in r for r in successful):
            avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
            min_f1 = min(r['f1_score'] for r in successful)
            max_f1 = max(r['f1_score'] for r in successful)
            print("-" * 90)
            print(f"{'Average F1:':<40} {avg_f1:>10.4f}")
            print(f"{'Min F1:':<40} {min_f1:>10.4f}")
            print(f"{'Max F1:':<40} {max_f1:>10.4f}")
    
    if failed:
        print(f"\nFailed models:")
        for r in failed:
            print(f"  ❌ {r['machine_id']}: {r.get('error', 'Unknown error')}")
    
    # Save batch report
    batch_report = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2.2.2: Classification Models',
        'total_machines': total_machines,
        'successful': len(successful),
        'failed': len(failed),
        'total_time_hours': total_time / 3600,
        'avg_time_per_machine_minutes': total_time / total_machines / 60,
        'configuration': {
            'time_limit': time_limit,
            'presets': presets
        },
        'results': results
    }
    
    report_path = Path('reports/batch_training_classification_10_machines.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(batch_report, f, indent=2)
    
    print(f"\n✅ Batch report saved: {report_path}")
    print("=" * 70 + "\n")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch train classification models for multiple machines'
    )
    parser.add_argument(
        '--machines_file',
        type=str,
        default='config/priority_10_machines.txt',
        help='File containing machine IDs (one per line)'
    )
    parser.add_argument(
        '--time_limit',
        type=int,
        default=900,
        help='Training time limit per machine in seconds (default: 900 = 15 min for fast Pi-compatible training)'
    )
    
    args = parser.parse_args()
    
    # Run batch training (fast mode, Pi-compatible)
    results = train_all_machines(args.machines_file, args.time_limit)
    
    # Exit with error code if any failed
    if any(r['status'] != 'success' for r in results):
        sys.exit(1)
