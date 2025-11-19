# ml_models/scripts/training/batch_train_regression.py
# Batch training script for per-machine regression models (RUL prediction)
# Phase 2.3.1: Train all 10 priority machines sequentially

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
    # Use absolute path from script location
    if not Path(machines_file).is_absolute():
        script_dir = Path(__file__).parent
        machines_file = script_dir / machines_file
    
    with open(machines_file, 'r') as f:
        machines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return machines

def train_all_machines(machines_file='../../config/priority_10_machines.txt', time_limit=900):
    """
    Train regression models for all machines in batch
    
    Args:
        machines_file: Path to file with machine IDs (one per line)
        time_limit: Time limit per machine in seconds (default: 900 = 15 min)
    """
    
    print(f"\n{'#' * 80}")
    print("# BATCH REGRESSION TRAINING - FAST MODE (Pi-Optimized)")
    print(f"# Time limit per machine: {time_limit//60} minutes")
    print(f"{'#' * 80}\n")
    
    # Load machines
    machines = load_machine_list(machines_file)
    total_machines = len(machines)
    
    print(f"📋 Loaded {total_machines} machines from: {machines_file}")
    print(f"Machines: {', '.join(machines)}\n")
    
    # Training loop
    results = []
    start_time = time.time()
    
    # Train each machine
    for i, machine_id in enumerate(machines, 1):
        print(f"\n{'#' * 70}")
        print(f"# Machine {i}/{total_machines}: {machine_id}")
        print(f"{'#' * 70}\n")
        
        machine_start_time = time.time()
        
        try:
            # Run FAST training script (Pi-compatible)
            # Both scripts in same directory - use absolute path
            script_path = Path(__file__).parent / 'train_regression_fast.py'
            # Use venv Python to ensure autogluon is available
            venv_python = Path(__file__).parent.parent.parent.parent / 'venv' / 'Scripts' / 'python.exe'
            python_exe = str(venv_python) if venv_python.exists() else sys.executable
            cmd = [
                python_exe,
                str(script_path),
                '--machine_id', machine_id,
                '--time_limit', str(time_limit)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),  # Run in same directory as this script
                capture_output=False,
                text=True,
                check=True
            )
            
            machine_training_time = time.time() - machine_start_time
            
            # Load report to get metrics - use absolute path
            ml_models_root = Path(__file__).parent.parent.parent
            report_path = ml_models_root / 'reports' / 'performance_metrics' / f'{machine_id}_regression_report.json'
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                
                results.append({
                    'machine_id': machine_id,
                    'status': 'success',
                    'r2_score': report['metrics']['r2_score'],
                    'rmse': report['metrics']['rmse'],
                    'training_time_minutes': machine_training_time / 60,
                    'best_model': report['best_model']
                })
                
                print(f"\n✅ SUCCESS: {machine_id}")
                print(f"   R² Score: {report['metrics']['r2_score']:.4f}")
                print(f"   RMSE: {report['metrics']['rmse']:.2f} hours")
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
        elapsed = time.time() - start_time
        completed = i
        if completed > 0:
            avg_time_per_machine = elapsed / completed
            remaining = total_machines - completed
            estimated_remaining = (avg_time_per_machine * remaining) / 3600
        else:
            estimated_remaining = 0
        
        print(f"\n{'=' * 70}")
        print(f"Progress: {completed}/{total_machines} machines completed")
        print(f"Elapsed: {elapsed/3600:.2f} hours")
        print(f"Estimated remaining: {estimated_remaining:.2f} hours")
        print(f"{'=' * 70}\n")
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 70}")
    print("BATCH TRAINING COMPLETED")
    print(f"{'=' * 70}\n")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    print(f"Total machines: {total_machines}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Average time per machine: {total_time/total_machines/60:.1f} minutes\n")
    
    if successful:
        print("Successful models:")
        print(f"{'Machine ID':<40} {'R² Score':>10} {'RMSE':>10} {'Time (min)':>12} {'Best Model':<20}")
        print("-" * 100)
        for r in successful:
            if 'r2_score' in r:
                print(f"{r['machine_id']:<40} {r['r2_score']:>10.4f} {r['rmse']:>10.2f} {r['training_time_minutes']:>12.1f} {r.get('best_model', 'N/A'):<20}")
            else:
                print(f"{r['machine_id']:<40} {'N/A':>10} {'N/A':>10} {r['training_time_minutes']:>12.1f} {r.get('note', ''):<20}")
        
        if successful and all('r2_score' in r for r in successful):
            avg_r2 = sum(r['r2_score'] for r in successful) / len(successful)
            min_r2 = min(r['r2_score'] for r in successful)
            max_r2 = max(r['r2_score'] for r in successful)
            avg_rmse = sum(r['rmse'] for r in successful) / len(successful)
            print("-" * 100)
            print(f"{'Average R²:':<40} {avg_r2:>10.4f}")
            print(f"{'Min R²:':<40} {min_r2:>10.4f}")
            print(f"{'Max R²:':<40} {max_r2:>10.4f}")
            print(f"{'Average RMSE:':<40} {avg_rmse:>10.2f} hours")
    
    if failed:
        print(f"\nFailed models:")
        for r in failed:
            print(f"  ❌ {r['machine_id']}: {r.get('error', 'Unknown error')}")
    
    # Save batch report
    batch_report = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2.3.1: Regression Models (RUL)',
        'total_machines': total_machines,
        'successful': len(successful),
        'failed': len(failed),
        'total_time_hours': total_time / 3600,
        'average_time_minutes': total_time / total_machines / 60,
        'results': results
    }
    
    report_path = Path('../../reports/batch_training_regression_10_machines.json')
    with open(report_path, 'w') as f:
        json.dump(batch_report, f, indent=2, default=str)
    
    print(f"\n✅ Batch report saved: {report_path}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch train regression models for multiple machines')
    parser.add_argument('--machines_file', 
                       default='../../config/priority_10_machines.txt',
                       help='Path to file with machine IDs')
    parser.add_argument('--time_limit', type=int, default=900,
                       help='Time limit per machine in seconds (default: 900 = 15 min)')
    
    args = parser.parse_args()
    
    train_all_machines(args.machines_file, args.time_limit)
