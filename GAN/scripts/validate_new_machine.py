"""
Phase 1.5: Validate New Machine Dataset
Quality validation script for newly onboarded machines

This script performs comprehensive validation:
1. File structure checks
2. RUL label validation
3. Data quality assessment
4. Comparison with seed data
5. Final validation report
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def check_file_structure(machine_id, data_root):
    """Check if all required files exist"""
    
    print("[1/5] Checking file structure...")
    
    machine_path = data_root / machine_id
    metadata_path = Path(__file__).parent.parent / 'metadata' / f"{machine_id}_metadata.json"
    
    issues = []
    
    # Check data directory
    if not machine_path.exists():
        issues.append(f"Machine directory not found: {machine_path}")
        return False, issues
    
    # Check datasets
    for split in ['train', 'val', 'test']:
        file_path = machine_path / f"{split}.parquet"
        if not file_path.exists():
            issues.append(f"{split}.parquet not found")
        else:
            df = pd.read_parquet(file_path)
            expected_rows = {'train': 35000, 'val': 7500, 'test': 7500}
            if len(df) != expected_rows[split]:
                issues.append(f"{split}.parquet has {len(df)} rows, expected {expected_rows[split]}")
            else:
                print(f"✅ {split}.parquet exists ({len(df):,} rows, {len(df.columns)} columns)")
    
    # Check metadata
    if not metadata_path.exists():
        issues.append(f"Metadata file not found: {metadata_path}")
    else:
        print(f"✅ Metadata file exists")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        return False, issues
    
    return True, []


def validate_rul_labels(machine_id, data_root):
    """Validate RUL labels in datasets"""
    
    print("\n[2/5] Validating RUL labels...")
    
    machine_path = data_root / machine_id
    issues = []
    
    for split in ['train', 'val', 'test']:
        file_path = machine_path / f"{split}.parquet"
        df = pd.read_parquet(file_path)
        
        # Check RUL column exists
        if 'rul' not in df.columns:
            issues.append(f"{split}: RUL column missing")
            continue
        
        rul = df['rul'].values
        
        # Check for NaN or infinite values
        if np.isnan(rul).any():
            issues.append(f"{split}: RUL contains NaN values")
        if np.isinf(rul).any():
            issues.append(f"{split}: RUL contains infinite values")
        
        # Check for negative values
        if (rul < 0).any():
            issues.append(f"{split}: RUL contains negative values (min: {rul.min():.2f})")
        
        # Check range
        rul_min, rul_max = rul.min(), rul.max()
        
        # Check if RUL decreases over time (for train set)
        if split == 'train':
            first_quartile_mean = rul[:len(rul)//4].mean()
            last_quartile_mean = rul[-len(rul)//4:].mean()
            if first_quartile_mean <= last_quartile_mean:
                issues.append(f"{split}: RUL does not decrease over time (start: {first_quartile_mean:.1f}, end: {last_quartile_mean:.1f})")
            else:
                print(f"✅ {split}: RUL decreases properly ({first_quartile_mean:.1f} → {last_quartile_mean:.1f} hours)")
        
        print(f"✅ {split}: RUL range [{rul_min:.1f}, {rul_max:.1f}], mean: {rul.mean():.1f} hours")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        return False, issues
    
    print("✅ RUL labels valid in all datasets")
    return True, []


def check_data_quality(machine_id, data_root):
    """Check data quality metrics"""
    
    print("\n[3/5] Checking data quality...")
    
    machine_path = data_root / machine_id
    issues = []
    
    for split in ['train', 'val', 'test']:
        file_path = machine_path / f"{split}.parquet"
        df = pd.read_parquet(file_path)
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{split}: {duplicates} duplicate rows found")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            issues.append(f"{split}: {missing} missing values found")
        
        # Check variance (not all columns constant)
        low_variance_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if df[col].std() < 1e-10:
                    low_variance_cols.append(col)
        
        if low_variance_cols:
            issues.append(f"{split}: Columns with near-zero variance: {low_variance_cols}")
    
    if not issues:
        print("✅ No duplicates found")
        print("✅ No missing values")
        print("✅ All sensors have realistic variance")
        return True, []
    else:
        for issue in issues:
            print(f"⚠️  {issue}")
        return True, issues  # Warnings, not failures


def compare_with_seed(machine_id, data_root, seed_root):
    """Compare synthetic data with seed data"""
    
    print("\n[4/5] Comparing with seed data...")
    
    seed_path = seed_root / f"{machine_id}_seed.parquet"
    
    if not seed_path.exists():
        print(f"⚠️  Seed data not found at {seed_path}, skipping comparison")
        return True, []
    
    seed_data = pd.read_parquet(seed_path)
    test_path = data_root / machine_id / 'test.parquet'
    test_data = pd.read_parquet(test_path)
    
    # Remove RUL column for comparison
    test_data_no_rul = test_data.drop(columns=['rul']) if 'rul' in test_data.columns else test_data
    
    issues = []
    
    # Check column consistency
    seed_cols = set(seed_data.columns)
    test_cols = set(test_data_no_rul.columns)
    
    if seed_cols != test_cols:
        missing = seed_cols - test_cols
        extra = test_cols - seed_cols
        if missing:
            issues.append(f"Columns missing from synthetic data: {missing}")
        if extra:
            issues.append(f"Extra columns in synthetic data: {extra}")
    else:
        print(f"✅ Column consistency: {len(seed_cols)} columns match")
    
    # Compare distributions (simple check)
    common_cols = seed_cols & test_cols
    numeric_cols = [col for col in common_cols if seed_data[col].dtype in ['float64', 'int64']]
    
    correlations = []
    for col in numeric_cols:
        seed_mean = seed_data[col].mean()
        test_mean = test_data_no_rul[col].mean()
        seed_std = seed_data[col].std()
        test_std = test_data_no_rul[col].std()
        
        # Check if means are reasonably close (within 50%)
        if seed_mean != 0:
            mean_diff_pct = abs(test_mean - seed_mean) / abs(seed_mean) * 100
            if mean_diff_pct > 50:
                issues.append(f"Column '{col}': mean differs by {mean_diff_pct:.1f}%")
    
    if not issues:
        print(f"✅ Distributions match seed data (checked {len(numeric_cols)} numeric columns)")
        return True, []
    else:
        for issue in issues:
            print(f"⚠️  {issue}")
        return True, issues  # Warnings


def generate_validation_report(machine_id, results):
    """Generate validation report"""
    
    print("\n[5/5] Final validation...")
    
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    report = {
        'machine_id': machine_id,
        'validation_results': results,
        'overall_status': 'PASSED' if results['all_checks_passed'] else 'FAILED',
        'summary': {
            'file_structure': 'OK' if results['file_structure']['passed'] else 'FAILED',
            'rul_labels': 'OK' if results['rul_validation']['passed'] else 'FAILED',
            'data_quality': 'OK' if results['data_quality']['passed'] else 'WARNING',
            'seed_comparison': 'OK' if results['seed_comparison']['passed'] else 'WARNING'
        }
    }
    
    report_path = reports_dir / f"{machine_id}_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Validation report saved: {report_path.name}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Validate new machine dataset')
    parser.add_argument('--machine_id', type=str, required=True,
                        help='Machine identifier to validate')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"VALIDATING NEW MACHINE: {args.machine_id}")
    print("="*70 + "\n")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data' / 'synthetic'
    seed_root = project_root / 'seed_data'
    
    results = {
        'machine_id': args.machine_id,
        'file_structure': {},
        'rul_validation': {},
        'data_quality': {},
        'seed_comparison': {},
        'all_checks_passed': True
    }
    
    # Run validations
    try:
        # File structure
        passed, issues = check_file_structure(args.machine_id, data_root)
        results['file_structure'] = {'passed': passed, 'issues': issues}
        if not passed:
            results['all_checks_passed'] = False
        
        # RUL validation
        if passed:  # Only if files exist
            passed, issues = validate_rul_labels(args.machine_id, data_root)
            results['rul_validation'] = {'passed': passed, 'issues': issues}
            if not passed:
                results['all_checks_passed'] = False
        
        # Data quality
        if results['file_structure']['passed']:
            passed, issues = check_data_quality(args.machine_id, data_root)
            results['data_quality'] = {'passed': passed, 'issues': issues}
        
        # Seed comparison
        if results['file_structure']['passed']:
            passed, issues = compare_with_seed(args.machine_id, data_root, seed_root)
            results['seed_comparison'] = {'passed': passed, 'issues': issues}
        
        # Generate report
        report = generate_validation_report(args.machine_id, results)
        
        # Final summary
        print("\n" + "="*70)
        if results['all_checks_passed']:
            print("✅ VALIDATION SUCCESSFUL")
            print("="*70)
            print("Machine ready for ML training")
        else:
            print("❌ VALIDATION FAILED")
            print("="*70)
            print("Please review issues above and fix before ML training")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during validation: {e}")
        raise


if __name__ == '__main__':
    main()
