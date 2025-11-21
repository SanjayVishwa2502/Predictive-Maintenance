"""
Validate temporal structure of all 26 machines in data/synthetic_fixed/

Checks:
1. Timestamp column exists and is sorted chronologically
2. RUL column exists and is >90% decreasing
3. Proper start/end values (start > max*0.7, end < max*0.2)
"""

import pandas as pd
from pathlib import Path

# All 26 machines
machines = [
    "cnc_dmg_mori_nlx_010",
    "cnc_dmg_mori_ntx_001",
    "cnc_haas_vf2_001",
    "cnc_haas_vf3_001",
    "cnc_makino_a51nx_001",
    "cnc_mazak_variaxis_001",
    "cnc_okuma_lb3000_001",
    "compressor_atlas_copco_ga30_001",
    "compressor_ingersoll_rand_2545_009",
    "conveyor_dorner_2200_013",
    "conveyor_hytrol_e24ez_014",
    "cooling_tower_bac_vti_018",
    "fan_ebm_papst_a3g710_007",
    "fan_howden_buffalo_008",
    "hydraulic_beckwood_press_011",
    "hydraulic_parker_hpu_012",
    "motor_abb_m3bp_002",
    "motor_siemens_1la7_001",
    "motor_weg_w22_003",
    "pump_flowserve_ansi_005",
    "pump_grundfos_cr3_004",
    "pump_ksb_etanorm_006",
    "robot_abb_irb6700_016",
    "robot_fanuc_m20ia_015",
    "transformer_square_d_017",
    "turbofan_cfm56_7b_001"
]

results = []
passed = 0
failed = 0

print("="*100)
print("VALIDATING ALL 26 MACHINES FOR TEMPORAL STRUCTURE")
print("="*100)
print()

for machine in machines:
    try:
        # Load train data
        train_path = Path(f'data/synthetic_fixed/{machine}/train.parquet')
        
        if not train_path.exists():
            results.append({
                'machine': machine,
                'status': 'FAIL',
                'reason': 'Train file not found'
            })
            failed += 1
            continue
        
        df = pd.read_parquet(train_path)
        
        # Check 1: Timestamp exists and sorted
        if 'timestamp' not in df.columns:
            results.append({
                'machine': machine,
                'status': 'FAIL',
                'reason': 'No timestamp column'
            })
            failed += 1
            continue
        
        timestamp_sorted = df['timestamp'].is_monotonic_increasing
        
        # Check 2: RUL exists and >90% decreasing
        if 'rul' not in df.columns:
            results.append({
                'machine': machine,
                'status': 'FAIL',
                'reason': 'No RUL column'
            })
            failed += 1
            continue
        
        rul_dec_count = (df['rul'].diff()[1:] <= 0).sum()
        rul_dec_pct = rul_dec_count / (len(df) - 1) * 100
        rul_decreasing = rul_dec_pct > 90
        
        # Check 3: Start/end values
        rul_start = df['rul'].iloc[0]
        rul_end = df['rul'].iloc[-1]
        rul_max = df['rul'].max()
        rul_min = df['rul'].min()
        
        start_ok = rul_start > rul_max * 0.7
        end_ok = rul_end < rul_max * 0.2
        
        # Overall status
        all_pass = timestamp_sorted and rul_decreasing and start_ok and end_ok
        
        if all_pass:
            status = '✅ PASS'
            passed += 1
        else:
            status = '❌ FAIL'
            failed += 1
            
        results.append({
            'machine': machine,
            'status': status,
            'samples': len(df),
            'timestamp_sorted': '✅' if timestamp_sorted else '❌',
            'rul_dec_pct': f'{rul_dec_pct:.1f}%',
            'rul_dec_ok': '✅' if rul_decreasing else '❌',
            'rul_range': f'{rul_start:.2f} → {rul_end:.2f}',
            'start_ok': '✅' if start_ok else '❌',
            'end_ok': '✅' if end_ok else '❌'
        })
        
        # Print individual result
        print(f"{machine:<40} {status}")
        print(f"  Samples: {len(df):>6}  |  Timestamp sorted: {results[-1]['timestamp_sorted']}")
        print(f"  RUL decreasing: {rul_dec_pct:>5.1f}% {results[-1]['rul_dec_ok']}  |  Range: {rul_start:>7.2f} → {rul_end:>7.2f}")
        print(f"  Start value: {results[-1]['start_ok']} (>{rul_max*0.7:.2f})  |  End value: {results[-1]['end_ok']} (<{rul_max*0.2:.2f})")
        print()
        
    except Exception as e:
        results.append({
            'machine': machine,
            'status': '❌ ERROR',
            'reason': str(e)
        })
        failed += 1
        print(f"{machine:<40} ❌ ERROR: {str(e)}")
        print()

# Summary
print("="*100)
print("SUMMARY")
print("="*100)
print(f"Total machines: {len(machines)}")
print(f"✅ PASSED: {passed}")
print(f"❌ FAILED: {failed}")
print()

if passed == len(machines):
    print("🎉 ALL 26 MACHINES HAVE PROPER TEMPORAL STRUCTURE!")
    print("✅ READY FOR PHASE 2.5 TIME-SERIES TRAINING")
else:
    print("⚠️  Some machines need fixes before Phase 2.5 training")
    print()
    print("Failed machines:")
    for r in results:
        if 'FAIL' in r['status'] or 'ERROR' in r['status']:
            reason = r.get('reason', 'Validation checks failed')
            print(f"  - {r['machine']}: {reason}")
