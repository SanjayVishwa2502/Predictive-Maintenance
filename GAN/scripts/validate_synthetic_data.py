"""
validate_synthetic_data.py
Phase 1.4.2: Validate Synthetic Data Quality

Comprehensive validation of all 21 machine synthetic datasets:
- Verify sample counts (train/val/test)
- Check for data quality issues (NaN, inf, duplicates)
- Validate feature ranges match original constraints
- Compare distributions to seed data
- Generate detailed validation report
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import KSComplement
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.parent
SYNTHETIC_DIR = BASE_DIR / 'data' / 'synthetic'
SEED_DIR = BASE_DIR / 'seed_data'
METADATA_DIR = BASE_DIR / 'metadata'
REPORT_DIR = BASE_DIR / 'reports'

# All machines
ALL_MACHINES = [
    "motor_siemens_1la7_001", "motor_abb_m3bp_002", "motor_weg_w22_003",
    "pump_grundfos_cr3_004", "pump_flowserve_ansi_005", "pump_ksb_etanorm_006",
    "fan_ebm_papst_a3g710_007", "fan_howden_buffalo_008",
    "compressor_ingersoll_rand_2545_009", "compressor_atlas_copco_ga30_001",
    "cnc_dmg_mori_nlx_010", "cnc_haas_vf2_001",
    "hydraulic_beckwood_press_011", "hydraulic_parker_hpu_012",
    "conveyor_dorner_2200_013", "conveyor_hytrol_e24ez_014",
    "robot_fanuc_m20ia_015", "robot_abb_irb6700_016",
    "transformer_square_d_017", "cooling_tower_bac_vti_018",
    "turbofan_cfm56_7b_001"
]

def validate_machine_data(machine_id):
    """
    Validate synthetic data for a single machine
    
    Returns:
        dict: Validation results
    """
    try:
        machine_dir = SYNTHETIC_DIR / machine_id
        
        # Step 1: Check if directory and files exist
        if not machine_dir.exists():
            return {
                'machine_id': machine_id,
                'status': 'FAILED',
                'error': 'Synthetic data directory not found'
            }
        
        train_file = machine_dir / 'train.parquet'
        val_file = machine_dir / 'val.parquet'
        test_file = machine_dir / 'test.parquet'
        
        if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
            return {
                'machine_id': machine_id,
                'status': 'FAILED',
                'error': 'Missing train/val/test files'
            }
        
        # Step 2: Load synthetic data
        train_data = pd.read_parquet(train_file)
        val_data = pd.read_parquet(val_file)
        test_data = pd.read_parquet(test_file)
        
        # Step 3: Verify sample counts
        expected_train = 35000
        expected_val = 7500
        expected_test = 7500
        
        count_issues = []
        if len(train_data) != expected_train:
            count_issues.append(f"Train: {len(train_data)} (expected {expected_train})")
        if len(val_data) != expected_val:
            count_issues.append(f"Val: {len(val_data)} (expected {expected_val})")
        if len(test_data) != expected_test:
            count_issues.append(f"Test: {len(test_data)} (expected {expected_test})")
        
        # Step 4: Check for data quality issues
        quality_issues = []
        
        # Combine all splits for quality checks
        all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
        
        # Check for NaN values
        nan_cols = all_data.columns[all_data.isna().any()].tolist()
        if nan_cols:
            quality_issues.append(f"NaN values in columns: {nan_cols}")
        
        # Check for infinite values (only numeric columns)
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(all_data[col]).any():
                inf_cols.append(col)
        if inf_cols:
            quality_issues.append(f"Infinite values in columns: {inf_cols}")
        
        # Check for duplicate rows
        duplicates = all_data.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Duplicate rows: {duplicates}")
        
        # Step 5: Verify machine_id column
        if 'machine_id' not in all_data.columns:
            quality_issues.append("Missing machine_id column")
        else:
            unique_ids = all_data['machine_id'].unique()
            if len(unique_ids) != 1 or unique_ids[0] != machine_id:
                quality_issues.append(f"Invalid machine_id values: {unique_ids}")
        
        # Step 6: Load seed data and metadata for comparison
        seed_file = SEED_DIR / f"{machine_id}_seed.parquet"
        metadata_file = METADATA_DIR / f"{machine_id}_metadata.json"
        
        if not seed_file.exists() or not metadata_file.exists():
            quality_issues.append("Cannot load seed data or metadata for comparison")
            quality_score = None
        else:
            # Load seed data and metadata
            seed_data = pd.read_parquet(seed_file)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Remove machine_id column for quality evaluation
            synthetic_eval = all_data.drop(columns=['machine_id'])
            
            # Step 7: Evaluate quality using SDMetrics
            try:
                quality_report = QualityReport()
                quality_report.generate(seed_data, synthetic_eval, metadata)
                quality_score = quality_report.get_score()
            except Exception as e:
                quality_issues.append(f"Quality evaluation failed: {str(e)}")
                quality_score = None
            
            # Step 8: Validate feature ranges
            range_issues = []
            for col in numeric_cols:
                if col in seed_data.columns:
                    seed_min = seed_data[col].min()
                    seed_max = seed_data[col].max()
                    synth_min = all_data[col].min()
                    synth_max = all_data[col].max()
                    
                    # Allow 10% margin outside seed data range
                    margin = (seed_max - seed_min) * 0.1
                    if synth_min < (seed_min - margin) or synth_max > (seed_max + margin):
                        range_issues.append(f"{col}: [{synth_min:.2f}, {synth_max:.2f}] vs seed [{seed_min:.2f}, {seed_max:.2f}]")
            
            if range_issues:
                quality_issues.append(f"Feature range violations: {range_issues}")
        
        # Step 9: Determine validation status
        if count_issues or quality_issues:
            status = 'WARNING' if not count_issues else 'FAILED'
        else:
            status = 'VALID'
        
        return {
            'machine_id': machine_id,
            'status': status,
            'sample_counts': {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data),
                'total': len(all_data)
            },
            'count_issues': count_issues,
            'quality_issues': quality_issues,
            'quality_score': quality_score,
            'num_features': len(all_data.columns) - 1,  # Exclude machine_id
            'duplicates': int(duplicates)
        }
        
    except Exception as e:
        return {
            'machine_id': machine_id,
            'status': 'FAILED',
            'error': str(e)
        }

def main():
    print("\n" + "="*70)
    print("PHASE 1.4.2: SYNTHETIC DATA VALIDATION - ALL 21 MACHINES")
    print("="*70)
    
    print("\nValidating synthetic data for all machines...")
    print(f"Data location: {SYNTHETIC_DIR}")
    
    start_time = datetime.now()
    
    validation_results = []
    valid_count = 0
    warning_count = 0
    failed_count = 0
    quality_scores = []
    
    print("\n" + "="*70)
    
    for i, machine_id in enumerate(ALL_MACHINES, 1):
        print(f"\n[{i}/{len(ALL_MACHINES)}] {machine_id}")
        print("-" * 70)
        
        result = validate_machine_data(machine_id)
        validation_results.append(result)
        
        # Print result
        status = result['status']
        print(f"Status: {status}")
        
        if status == 'VALID':
            valid_count += 1
            print(f"Samples: Train={result['sample_counts']['train']}, Val={result['sample_counts']['val']}, Test={result['sample_counts']['test']}")
            print(f"Quality Score: {result['quality_score']:.4f}")
            print(f"Features: {result['num_features']}")
            quality_scores.append(result['quality_score'])
        elif status == 'WARNING':
            warning_count += 1
            print(f"[WARNING] Issues found:")
            if result.get('count_issues'):
                for issue in result['count_issues']:
                    print(f"  - {issue}")
            if result.get('quality_issues'):
                for issue in result['quality_issues']:
                    print(f"  - {issue}")
        else:
            failed_count += 1
            print(f"[ERROR] {result.get('error', 'Unknown error')}")
    
    end_time = datetime.now()
    validation_time = (end_time - start_time).total_seconds()
    
    # Calculate statistics
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
    min_quality = min(quality_scores) if quality_scores else None
    max_quality = max(quality_scores) if quality_scores else None
    
    # Print summary
    print("\n" + "="*70)
    print("SYNTHETIC DATA VALIDATION COMPLETE")
    print("="*70)
    
    print("\nValidation Results:")
    print(f"  [OK] Valid: {valid_count}/{len(ALL_MACHINES)}")
    print(f"  [WARNING] Warnings: {warning_count}/{len(ALL_MACHINES)}")
    print(f"  [ERROR] Failed: {failed_count}/{len(ALL_MACHINES)}")
    
    if quality_scores:
        print("\nQuality Statistics:")
        print(f"  Average: {avg_quality:.4f}")
        print(f"  Min: {min_quality:.4f}")
        print(f"  Max: {max_quality:.4f}")
        print(f"  Range: {max_quality - min_quality:.4f}")
    
    print(f"\nValidation Time: {validation_time:.2f} seconds")
    
    # Save report
    report = {
        'phase': '1.4.2 - Synthetic Data Validation',
        'timestamp': datetime.now().isoformat(),
        'validation_time_seconds': validation_time,
        'summary': {
            'total_machines': len(ALL_MACHINES),
            'valid': valid_count,
            'warnings': warning_count,
            'failed': failed_count,
            'success_rate': f"{(valid_count / len(ALL_MACHINES) * 100):.1f}%"
        },
        'quality_statistics': {
            'average_quality': avg_quality,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'quality_range': max_quality - min_quality if (max_quality and min_quality) else None
        },
        'detailed_results': validation_results
    }
    
    report_file = REPORT_DIR / 'synthetic_data_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Report saved: {report_file}")
    print("="*70)
    
    if valid_count == len(ALL_MACHINES):
        print("\n*** ALL SYNTHETIC DATA VALIDATED SUCCESSFULLY! ***\n")
    elif warning_count > 0 and failed_count == 0:
        print("\n[WARNING] All datasets present but some have minor issues.\n")
    else:
        print(f"\n[ERROR] {failed_count} dataset(s) failed validation.\n")

if __name__ == '__main__':
    main()
