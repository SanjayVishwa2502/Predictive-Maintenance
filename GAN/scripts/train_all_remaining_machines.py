"""
Phase 1.3.3: Batch Training Script for Remaining 16 Machines
Train all remaining machines with TVAE (500 epochs, 10K samples)
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

# Remaining 16 machines (excluding 5 priority machines already trained)
REMAINING_MACHINES = [
    "motor_weg_w22_003",
    "pump_ksb_etanorm_006",
    "fan_ebm_papst_a3g710_007",
    "fan_howden_buffalo_008",
    "compressor_ingersoll_rand_2545_009",
    "cnc_dmg_mori_nlx_010",
    "hydraulic_beckwood_press_011",
    "hydraulic_parker_hpu_012",
    "conveyor_dorner_2200_013",
    "conveyor_hytrol_e24ez_014",
    "robot_fanuc_m20ia_015",
    "robot_abb_irb6700_016",
    "transformer_square_d_017",
    "cooling_tower_bac_vti_018",
    "cnc_haas_vf2_001",
    "turbofan_cfm56_7b_001"
]

def train_machine(machine_id, index, total):
    """Train a single machine and return results"""
    print(f"\n{'=' * 70}")
    print(f"[{index}/{total}] Training: {machine_id}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_tvae_machine.py', '--machine_id', machine_id],
            capture_output=True,
            text=True,
            check=True
        )
        
        training_time = time.time() - start_time
        
        # Parse quality score from output
        quality_score = None
        for line in result.stdout.split('\n'):
            if 'Quality Score:' in line:
                try:
                    quality_score = float(line.split(':')[1].strip())
                except:
                    pass
        
        print(f"\n✅ {machine_id} completed in {training_time/60:.2f} minutes")
        if quality_score:
            print(f"   Quality Score: {quality_score:.3f}")
        
        return {
            'machine_id': machine_id,
            'status': 'SUCCESS',
            'training_time_minutes': training_time / 60,
            'quality_score': quality_score
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {machine_id} FAILED")
        print(f"Error: {e.stderr}")
        
        return {
            'machine_id': machine_id,
            'status': 'FAILED',
            'training_time_minutes': 0,
            'quality_score': None,
            'error': str(e)
        }

def main():
    """Train all remaining machines sequentially"""
    
    print("\n" + "=" * 70)
    print("PHASE 1.3.3: BATCH TRAINING - REMAINING 16 MACHINES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: TVAE")
    print(f"  Epochs: 500")
    print(f"  Samples per machine: 10,000")
    print(f"  Total machines: {len(REMAINING_MACHINES)}")
    print(f"\nStarting batch training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    overall_start = time.time()
    results = []
    
    # Train each machine sequentially
    for idx, machine_id in enumerate(REMAINING_MACHINES, start=1):
        result = train_machine(machine_id, idx, len(REMAINING_MACHINES))
        results.append(result)
    
    overall_time = time.time() - overall_start
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH TRAINING COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"\nResults:")
    print(f"  Successful: {len(successful)}/{len(REMAINING_MACHINES)}")
    print(f"  Failed: {len(failed)}/{len(REMAINING_MACHINES)}")
    print(f"  Total Time: {overall_time/60:.2f} minutes ({overall_time/3600:.2f} hours)")
    
    if successful:
        avg_time = sum(r['training_time_minutes'] for r in successful) / len(successful)
        quality_scores = [r['quality_score'] for r in successful if r['quality_score'] is not None]
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            
            print(f"\nQuality Statistics:")
            print(f"  Average Quality: {avg_quality:.3f}")
            print(f"  Min Quality: {min_quality:.3f}")
            print(f"  Max Quality: {max_quality:.3f}")
        
        print(f"\nAverage Training Time: {avg_time:.2f} minutes per machine")
    
    if failed:
        print(f"\n⚠️  Failed Machines:")
        for r in failed:
            print(f"  - {r['machine_id']}")
    
    # Save results
    import json
    report_path = Path(__file__).parent.parent / 'reports' / 'batch_training_remaining_16_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'total_machines': len(REMAINING_MACHINES),
            'successful': len(successful),
            'failed': len(failed),
            'total_time_minutes': overall_time / 60,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Report saved: {report_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
