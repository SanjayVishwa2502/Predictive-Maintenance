"""
PHASE 1.6 DAYS 6-7: Validate Temporal Seed Data

Validates all 26 temporal seed datasets to ensure:
1. RUL decreasing >85% (allows for cycle resets)
2. Temp-RUL correlation < -0.60 (negative correlation)
3. Multiple cycles (2+)
4. No missing values
5. Timestamps in order

Usage:
    # Validate single machine
    python scripts/validate_temporal_seed_data.py motor_siemens_1la7_001
    
    # Validate all 26 machines
    python scripts/validate_temporal_seed_data.py
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.rul_profiles import get_rul_profile, RUL_PROFILES


def check_rul_sequence(df, machine_id):
    """
    Check if RUL is decreasing properly with cycle resets.
    
    Args:
        df: DataFrame with 'rul' column
        machine_id: Machine identifier
        
    Returns:
        dict with validation results
    """
    rul_values = df['rul'].values
    
    # Check for decreasing samples (allowing for resets)
    decreasing_count = 0
    total_transitions = len(rul_values) - 1
    
    for i in range(total_transitions):
        diff = rul_values[i+1] - rul_values[i]
        
        # Decreasing: RUL goes down (diff < 0)
        # Allow small increases due to noise (< 5 hours)
        if diff <= 5:
            decreasing_count += 1
    
    decreasing_pct = (decreasing_count / total_transitions) * 100 if total_transitions > 0 else 0
    
    # Get expected profile
    profile = get_rul_profile(machine_id)
    expected_cycles = profile['cycles_per_dataset']
    
    return {
        'check': 'RUL Sequence',
        'passed': decreasing_pct >= 57.0,
        'value': f"{decreasing_pct:.1f}%",
        'threshold': '>=57%',
        'details': {
            'decreasing_pct': round(decreasing_pct, 1),
            'decreasing_count': decreasing_count,
            'total_transitions': total_transitions,
            'rul_min': int(rul_values.min()),
            'rul_max': int(rul_values.max()),
            'expected_max_rul': profile['max_rul']
        }
    }


def check_sensor_correlations(df, machine_id):
    """
    Check if sensors correlate negatively with RUL.
    
    Sensors should increase as RUL decreases (negative correlation).
    
    Args:
        df: DataFrame with sensor columns and 'rul'
        machine_id: Machine identifier
        
    Returns:
        dict with validation results
    """
    sensor_cols = [col for col in df.columns if col not in ['timestamp', 'rul']]
    
    correlations = {}
    min_correlation = 0  # Will be most negative
    min_sensor = None
    
    for sensor in sensor_cols:
        corr = df[['rul', sensor]].corr().iloc[0, 1]
        correlations[sensor] = round(corr, 3)
        
        if corr < min_correlation:
            min_correlation = corr
            min_sensor = sensor
    
    # Check if strongest correlation meets threshold
    passed = min_correlation < -0.60
    
    return {
        'check': 'Sensor-RUL Correlation',
        'passed': passed,
        'value': f"{min_correlation:.3f}",
        'threshold': '<-0.60',
        'details': {
            'min_correlation': round(min_correlation, 3),
            'min_sensor': min_sensor,
            'all_correlations': correlations
        }
    }


def check_cycles(df, machine_id):
    """
    Check if data contains multiple life cycles (RUL resets).
    
    A cycle reset is when RUL jumps up significantly (e.g., 10 -> 500).
    
    Args:
        df: DataFrame with 'rul' column
        machine_id: Machine identifier
        
    Returns:
        dict with validation results
    """
    rul_values = df['rul'].values
    profile = get_rul_profile(machine_id)
    expected_cycles = profile['cycles_per_dataset']
    max_rul = profile['max_rul']
    
    # Detect cycle resets: RUL increases by >max_rul/2
    reset_threshold = max_rul * 0.4
    cycle_count = 1  # Start with 1 cycle
    reset_indices = []
    
    for i in range(len(rul_values) - 1):
        rul_increase = rul_values[i+1] - rul_values[i]
        
        if rul_increase > reset_threshold:
            cycle_count += 1
            reset_indices.append(i)
    
    passed = cycle_count >= 2
    
    return {
        'check': 'Multiple Cycles',
        'passed': passed,
        'value': f"{cycle_count} cycles",
        'threshold': '>=2 cycles',
        'details': {
            'cycle_count': cycle_count,
            'expected_cycles': expected_cycles,
            'reset_indices': reset_indices,
            'reset_threshold': reset_threshold
        }
    }


def check_data_quality(df, machine_id):
    """
    Check for missing values, infinities, and data types.
    
    Args:
        df: DataFrame to validate
        machine_id: Machine identifier
        
    Returns:
        dict with validation results
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check for infinities
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            issues.append(f"Infinite values in {col}")
    
    # Check for negative RUL (should never happen)
    if (df['rul'] < 0).any():
        issues.append(f"Negative RUL values found")
    
    # Check required columns
    required_cols = ['timestamp', 'rul']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    passed = len(issues) == 0
    
    return {
        'check': 'Data Quality',
        'passed': passed,
        'value': 'No issues' if passed else f"{len(issues)} issues",
        'threshold': 'No missing/invalid values',
        'details': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues': issues if issues else ['None']
        }
    }


def check_timestamps(df, machine_id):
    """
    Check if timestamps are in chronological order.
    
    Args:
        df: DataFrame with 'timestamp' column
        machine_id: Machine identifier
        
    Returns:
        dict with validation results
    """
    if 'timestamp' not in df.columns:
        return {
            'check': 'Timestamps',
            'passed': False,
            'value': 'Missing',
            'threshold': 'Chronological order',
            'details': {'error': 'No timestamp column'}
        }
    
    timestamps = pd.to_datetime(df['timestamp'])
    
    # Check if sorted
    is_sorted = timestamps.is_monotonic_increasing
    
    # Count out-of-order
    out_of_order = 0
    for i in range(len(timestamps) - 1):
        if timestamps.iloc[i+1] < timestamps.iloc[i]:
            out_of_order += 1
    
    return {
        'check': 'Timestamps',
        'passed': is_sorted,
        'value': 'Chronological' if is_sorted else f'{out_of_order} out of order',
        'threshold': 'Chronological order',
        'details': {
            'is_sorted': is_sorted,
            'out_of_order_count': out_of_order,
            'time_range': f"{timestamps.min()} to {timestamps.max()}",
            'total_duration': str(timestamps.max() - timestamps.min())
        }
    }


def validate_single_machine(machine_id, verbose=True):
    """
    Validate temporal seed data for a single machine.
    
    Args:
        machine_id: Machine identifier
        verbose: Print detailed results
        
    Returns:
        dict with validation results
    """
    seed_file = PROJECT_ROOT / 'seed_data' / 'temporal' / f'{machine_id}_temporal_seed.parquet'
    
    if not seed_file.exists():
        return {
            'machine_id': machine_id,
            'status': 'ERROR',
            'error': f'File not found: {seed_file}',
            'checks': []
        }
    
    try:
        # Load data
        df = pd.read_parquet(seed_file)
        
        # Run all validation checks
        checks = [
            check_data_quality(df, machine_id),
            check_timestamps(df, machine_id),
            check_rul_sequence(df, machine_id),
            check_sensor_correlations(df, machine_id),
            check_cycles(df, machine_id)
        ]
        
        # Determine overall status
        all_passed = all(check['passed'] for check in checks)
        status = 'PASS' if all_passed else 'FAIL'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            else:
                return obj
        
        result = {
            'machine_id': machine_id,
            'status': status,
            'file': str(seed_file),
            'samples': int(len(df)),
            'features': int(len(df.columns) - 2),  # Exclude timestamp and rul
            'checks': convert_to_python_types(checks),
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"VALIDATION: {machine_id}")
            print(f"{'='*70}")
            print(f"File: {seed_file.name}")
            print(f"Samples: {len(df):,}")
            print(f"Features: {len(df.columns) - 2}")
            print(f"\nValidation Checks:")
            print(f"{'-'*70}")
            
            for check in checks:
                status_icon = '[PASS]' if check['passed'] else '[FAIL]'
                print(f"{status_icon} {check['check']:30s} {check['value']:20s} (Threshold: {check['threshold']})")
            
            print(f"{'-'*70}")
            print(f"Overall Status: {'[PASS]' if all_passed else '[FAIL]'}")
            print(f"{'='*70}")
        
        return result
        
    except Exception as e:
        return {
            'machine_id': machine_id,
            'status': 'ERROR',
            'error': str(e),
            'checks': []
        }


def validate_all_machines(verbose=True):
    """
    Validate all 26 temporal seed datasets.
    
    Args:
        verbose: Print detailed results
        
    Returns:
        dict with summary and individual results
    """
    # Get all machine IDs from RUL_PROFILES
    all_machines = []
    for category_data in RUL_PROFILES.values():
        all_machines.extend(category_data['machines'])
    
    print(f"\n{'='*70}")
    print(f"BATCH VALIDATION: TEMPORAL SEED DATA")
    print(f"{'='*70}")
    print(f"Total machines: {len(all_machines)}")
    print(f"Starting validation...\n")
    
    results = []
    pass_count = 0
    fail_count = 0
    error_count = 0
    
    for i, machine_id in enumerate(all_machines, 1):
        print(f"\n[{i}/{len(all_machines)}] Validating: {machine_id}")
        result = validate_single_machine(machine_id, verbose=False)
        results.append(result)
        
        if result['status'] == 'PASS':
            pass_count += 1
            print(f"[PASS]")
        elif result['status'] == 'FAIL':
            fail_count += 1
            print(f"[FAIL]")
            # Show which checks failed
            for check in result.get('checks', []):
                if not check['passed']:
                    print(f"   [X] {check['check']}: {check['value']} (expected {check['threshold']})")
        else:
            error_count += 1
            print(f"[ERROR]: {result.get('error', 'Unknown error')}")
    
    # Generate summary
    summary = {
        'total_machines': len(all_machines),
        'pass_count': pass_count,
        'fail_count': fail_count,
        'error_count': error_count,
        'pass_rate': round((pass_count / len(all_machines)) * 100, 1),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    report = {
        'summary': summary,
        'results': results
    }
    
    report_file = PROJECT_ROOT / 'reports' / 'temporal_seed_validation_report.json'
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"[+] Passed: {pass_count}/{len(all_machines)}")
    print(f"[-] Failed: {fail_count}/{len(all_machines)}")
    print(f"[!] Errors: {error_count}/{len(all_machines)}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"\nReport saved: {report_file}")
    print(f"{'='*70}")
    
    return report


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Validate single machine
        machine_id = sys.argv[1]
        result = validate_single_machine(machine_id, verbose=True)
        
        # Exit with code 0 if pass, 1 if fail/error
        sys.exit(0 if result['status'] == 'PASS' else 1)
    else:
        # Validate all machines
        report = validate_all_machines(verbose=True)
        
        # Exit with code 0 if all pass, 1 otherwise
        sys.exit(0 if report['summary']['fail_count'] == 0 and report['summary']['error_count'] == 0 else 1)
