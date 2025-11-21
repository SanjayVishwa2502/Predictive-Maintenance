"""
Verify machine-specific synthetic data from GAN
Checks temporal structure, RUL column, and data quality for all 27 machines
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def verify_machine_data(machine_id, gan_data_path='../../../GAN/data/synthetic'):
    """Verify single machine's data structure and quality"""
    
    machine_path = Path(gan_data_path) / machine_id
    
    if not machine_path.exists():
        return {
            'machine_id': machine_id,
            'status': 'MISSING',
            'error': f'Directory not found: {machine_path}'
        }
    
    try:
        # Load all splits
        train_df = pd.read_parquet(machine_path / 'train.parquet')
        val_df = pd.read_parquet(machine_path / 'val.parquet')
        test_df = pd.read_parquet(machine_path / 'test.parquet')
        
        # Basic checks
        results = {
            'machine_id': machine_id,
            'status': 'OK',
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'num_features': len(train_df.columns),
            'feature_names': list(train_df.columns),
        }
        
        # Check required columns
        results['has_timestamp'] = 'timestamp' in train_df.columns
        results['has_rul'] = 'rul' in train_df.columns
        
        # Check temporal ordering
        if 'timestamp' in train_df.columns:
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
            is_sorted = train_df['timestamp'].is_monotonic_increasing
            results['temporal_sorted'] = is_sorted
            results['time_span_days'] = (train_df['timestamp'].max() - train_df['timestamp'].min()).days
        
        # Check RUL characteristics
        if 'rul' in train_df.columns:
            results['rul_min'] = float(train_df['rul'].min())
            results['rul_max'] = float(train_df['rul'].max())
            results['rul_mean'] = float(train_df['rul'].mean())
            
            # Check RUL decreasing pattern
            rul_decreasing = (train_df['rul'].diff().dropna() <= 0).sum()
            rul_total = len(train_df) - 1
            results['rul_decreasing_pct'] = (rul_decreasing / rul_total) * 100 if rul_total > 0 else 0
        
        # Check for missing values
        results['missing_values'] = int(train_df.isnull().sum().sum())
        results['missing_pct'] = float((results['missing_values'] / (len(train_df) * len(train_df.columns))) * 100)
        
        # Sensor features (exclude timestamp and rul)
        sensor_cols = [col for col in train_df.columns if col not in ['timestamp', 'rul']]
        results['sensor_features'] = sensor_cols
        results['num_sensors'] = len(sensor_cols)
        
        return results
        
    except Exception as e:
        return {
            'machine_id': machine_id,
            'status': 'ERROR',
            'error': str(e)
        }

def verify_all_machines():
    """Verify all 27 machines in synthetic directory"""
    
    print(f"\n{'=' * 80}")
    print("PHASE 2.1.2: DATA VERIFICATION & LOADING")
    print(f"{'=' * 80}\n")
    
    # Get all machines
    gan_data_path = Path(__file__).parent.parent.parent.parent / 'GAN' / 'data' / 'synthetic'
    
    if not gan_data_path.exists():
        print(f"❌ ERROR: GAN data path not found: {gan_data_path}")
        return
    
    machine_dirs = sorted([d for d in gan_data_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(machine_dirs)} machines in {gan_data_path}\n")
    
    all_results = []
    
    # Verify each machine
    for i, machine_dir in enumerate(machine_dirs, 1):
        machine_id = machine_dir.name
        print(f"[{i}/{len(machine_dirs)}] Verifying: {machine_id}...", end=' ')
        
        result = verify_machine_data(machine_id, gan_data_path)
        all_results.append(result)
        
        if result['status'] == 'OK':
            print(f"✅ OK ({result['total_samples']:,} samples, {result['num_sensors']} sensors)")
        else:
            print(f"❌ {result['status']}: {result.get('error', 'Unknown error')}")
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 80}\n")
    
    ok_machines = [r for r in all_results if r['status'] == 'OK']
    error_machines = [r for r in all_results if r['status'] != 'OK']
    
    print(f"Total Machines: {len(all_results)}")
    print(f"✅ OK: {len(ok_machines)}")
    print(f"❌ Errors: {len(error_machines)}")
    
    if ok_machines:
        print(f"\n{'─' * 80}")
        print("DATA CHARACTERISTICS:")
        print(f"{'─' * 80}")
        
        total_samples = sum(r['total_samples'] for r in ok_machines)
        avg_sensors = sum(r['num_sensors'] for r in ok_machines) / len(ok_machines)
        
        print(f"Total Samples Across All Machines: {total_samples:,}")
        print(f"Average Sensors per Machine: {avg_sensors:.1f}")
        
        # Check RUL availability
        has_rul = sum(1 for r in ok_machines if r.get('has_rul', False))
        has_timestamp = sum(1 for r in ok_machines if r.get('has_timestamp', False))
        
        print(f"\n✅ Machines with RUL column: {has_rul}/{len(ok_machines)} ({(has_rul/len(ok_machines)*100):.0f}%)")
        print(f"✅ Machines with timestamp column: {has_timestamp}/{len(ok_machines)} ({(has_timestamp/len(ok_machines)*100):.0f}%)")
        
        # Temporal structure validation
        temporal_sorted = sum(1 for r in ok_machines if r.get('temporal_sorted', False))
        print(f"✅ Machines with sorted timestamps: {temporal_sorted}/{len(ok_machines)} ({(temporal_sorted/len(ok_machines)*100):.0f}%)")
        
        # RUL characteristics
        if has_rul > 0:
            rul_machines = [r for r in ok_machines if r.get('has_rul', False)]
            avg_rul_range = sum(r['rul_max'] - r['rul_min'] for r in rul_machines) / len(rul_machines)
            avg_rul_decreasing = sum(r.get('rul_decreasing_pct', 0) for r in rul_machines) / len(rul_machines)
            
            print(f"\nRUL CHARACTERISTICS:")
            print(f"  Average RUL Range: {avg_rul_range:.2f} hours")
            print(f"  Average RUL Decreasing %: {avg_rul_decreasing:.1f}%")
        
        # Sample distribution
        print(f"\n{'─' * 80}")
        print("SAMPLE DISTRIBUTION BY MACHINE:")
        print(f"{'─' * 80}")
        
        for result in sorted(ok_machines, key=lambda x: x['total_samples'], reverse=True)[:10]:
            print(f"  {result['machine_id']:45s} | {result['total_samples']:6,} samples | {result['num_sensors']:2d} sensors")
    
    if error_machines:
        print(f"\n{'─' * 80}")
        print("MACHINES WITH ERRORS:")
        print(f"{'─' * 80}")
        for result in error_machines:
            print(f"  ❌ {result['machine_id']}: {result.get('error', 'Unknown error')}")
    
    # Save detailed report
    report_path = Path(__file__).parent.parent.parent / 'reports' / 'data_verification_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'verification_date': datetime.now().isoformat(),
        'total_machines': len(all_results),
        'successful': len(ok_machines),
        'errors': len(error_machines),
        'total_samples': sum(r.get('total_samples', 0) for r in ok_machines),
        'machines': all_results
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Detailed report saved: {report_path}")
    
    print(f"\n{'=' * 80}")
    print("PHASE 2.1.2 STATUS")
    print(f"{'=' * 80}")
    
    if len(ok_machines) == len(all_results):
        print("✅ ALL MACHINES VERIFIED SUCCESSFULLY")
        print("✅ Ready for Phase 2.2 (Classification Training)")
    else:
        print(f"⚠️  {len(error_machines)} machines have issues - review before proceeding")
    
    return all_results

if __name__ == "__main__":
    verify_all_machines()
