"""
Batch training script for time-series forecasting models
Trains Prophet models for all 10 priority machines
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Priority machines for Phase 2
PRIORITY_MACHINES = [
    'motor_siemens_1la7_001',    # Already trained
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

def train_machine(machine_id, python_path, script_path):
    """Train time-series model for a single machine"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {machine_id}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        # Run training script
        cmd = [python_path, str(script_path), '--machine_id', machine_id]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {machine_id} trained in {duration:.2f} minutes")
            return {
                'machine_id': machine_id,
                'status': 'success',
                'training_time_minutes': duration,
                'timestamp': datetime.now().isoformat()
            }
        else:
            print(f"❌ FAILED: {machine_id}")
            print(f"Error: {result.stderr}")
            return {
                'machine_id': machine_id,
                'status': 'failed',
                'error': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {machine_id} (exceeded 10 minutes)")
        return {
            'machine_id': machine_id,
            'status': 'timeout',
            'error': 'Training exceeded 10 minute timeout',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ ERROR: {machine_id} - {str(e)}")
        return {
            'machine_id': machine_id,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    print("="*80)
    print("BATCH TIME-SERIES MODEL TRAINING")
    print("="*80)
    print(f"Machines to train: {len(PRIORITY_MACHINES)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    python_path = sys.executable
    script_path = Path(__file__).parent / 'train_timeseries.py'
    
    print(f"\nPython: {python_path}")
    print(f"Script: {script_path}")
    
    # Train all machines
    results = []
    successful = 0
    failed = 0
    
    batch_start = datetime.now()
    
    for i, machine_id in enumerate(PRIORITY_MACHINES, 1):
        print(f"\n[{i}/{len(PRIORITY_MACHINES)}] Processing: {machine_id}")
        
        result = train_machine(machine_id, python_path, script_path)
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    batch_end = datetime.now()
    total_time = (batch_end - batch_start).total_seconds() / 60
    
    # Save batch report
    report = {
        'batch_start': batch_start.isoformat(),
        'batch_end': batch_end.isoformat(),
        'total_time_minutes': total_time,
        'total_machines': len(PRIORITY_MACHINES),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    report_path = base_dir / 'reports' / 'performance_metrics' / 'batch_timeseries_training_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH TRAINING COMPLETE")
    print("="*80)
    print(f"Total Time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")
    print(f"Successful: {successful}/{len(PRIORITY_MACHINES)}")
    print(f"Failed: {failed}/{len(PRIORITY_MACHINES)}")
    print(f"Success Rate: {successful/len(PRIORITY_MACHINES)*100:.1f}%")
    print(f"\nReport saved: {report_path}")
    print("="*80)
    
    # Show successful machines
    if successful > 0:
        print("\n✅ Successfully trained:")
        for r in results:
            if r['status'] == 'success':
                print(f"  - {r['machine_id']} ({r['training_time_minutes']:.2f} min)")
    
    # Show failed machines
    if failed > 0:
        print("\n❌ Failed:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {r['machine_id']}: {r.get('error', 'Unknown error')}")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
