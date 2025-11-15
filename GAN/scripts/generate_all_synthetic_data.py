"""
Phase 1.4.1: Batch Synthetic Data Generation
Generate synthetic data for all 21 trained machines
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# All 21 trained machines
ALL_MACHINES = [
    "motor_siemens_1la7_001",
    "motor_abb_m3bp_002",
    "pump_grundfos_cr3_004",
    "pump_flowserve_ansi_005",
    "compressor_atlas_copco_ga30_001",
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

def generate_for_machine(machine_id, num_samples, index, total):
    """Generate synthetic data for one machine"""
    
    print(f"\n{'=' * 70}")
    print(f"[{index}/{total}] {machine_id}")
    print(f"{'=' * 70}")
    
    try:
        result = subprocess.run(
            [sys.executable, 'generate_synthetic_data.py', 
             '--machine_id', machine_id,
             '--num_samples', str(num_samples)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output for statistics
        output_lines = result.stdout.split('\n')
        train_samples = val_samples = test_samples = 0
        
        for line in output_lines:
            if 'Train:' in line:
                train_samples = int(line.split()[1])
            elif 'Val:' in line:
                val_samples = int(line.split()[1])
            elif 'Test:' in line:
                test_samples = int(line.split()[1])
        
        print(f"[OK] {machine_id}: {train_samples + val_samples + test_samples} samples generated")
        
        return {
            'machine_id': machine_id,
            'status': 'SUCCESS',
            'train': train_samples,
            'val': val_samples,
            'test': test_samples,
            'total': train_samples + val_samples + test_samples
        }
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {machine_id}: Generation failed")
        print(f"Error: {e.stderr}")
        
        return {
            'machine_id': machine_id,
            'status': 'FAILED',
            'error': str(e)
        }

def main():
    """Generate synthetic data for all machines"""
    
    print("\n" + "=" * 70)
    print("PHASE 1.4.1: SYNTHETIC DATA GENERATION - ALL 21 MACHINES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: TVAE")
    print(f"  Samples per machine: 5,000")
    print(f"  Total machines: {len(ALL_MACHINES)}")
    print(f"  Expected total samples: {len(ALL_MACHINES) * 5000:,}")
    print(f"  Splits: Train (70%) / Val (15%) / Test (15%)")
    print(f"\nStarting generation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = []
    
    # Generate for each machine
    for idx, machine_id in enumerate(ALL_MACHINES, start=1):
        result = generate_for_machine(machine_id, num_samples=50000, index=idx, total=len(ALL_MACHINES))
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 70)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    
    print(f"\nResults:")
    print(f"  [OK] Successful: {len(successful)}/{len(ALL_MACHINES)}")
    print(f"  [ERROR] Failed: {len(failed)}/{len(ALL_MACHINES)}")
    
    if successful:
        total_samples = sum(r['total'] for r in successful)
        total_train = sum(r['train'] for r in successful)
        total_val = sum(r['val'] for r in successful)
        total_test = sum(r['test'] for r in successful)
        
        print(f"\nGenerated Samples:")
        print(f"  Total: {total_samples:,} samples")
        print(f"  Train: {total_train:,} samples (70%)")
        print(f"  Val: {total_val:,} samples (15%)")
        print(f"  Test: {total_test:,} samples (15%)")
        print(f"\nData Location: GAN/data/synthetic/")
    
    if failed:
        print(f"\n[ERROR] Failed Machines:")
        for r in failed:
            print(f"  - {r['machine_id']}")
    
    # Save report
    report_path = Path(__file__).parent.parent / 'reports' / 'synthetic_data_generation_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'total_machines': len(ALL_MACHINES),
            'successful': len(successful),
            'failed': len(failed),
            'total_samples': sum(r.get('total', 0) for r in successful),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n[OK] Report saved: {report_path}")
    print("=" * 70 + "\n")
    
    if len(successful) == len(ALL_MACHINES):
        print("*** ALL SYNTHETIC DATA GENERATED SUCCESSFULLY! ***\n")
        return 0
    else:
        print(f"[WARNING] Generation incomplete: {len(successful)}/{len(ALL_MACHINES)} succeeded\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
