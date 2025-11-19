"""
Data Verification & Pooling Script for Phase 2.1.2
CRITICAL: Pool all machines into single dataset for generic training
"""

import pandas as pd
import json
from pathlib import Path
import sys

def load_machine_metadata(machine_id):
    """Load machine metadata from GAN metadata files"""
    metadata_path = Path(f'../../GAN/metadata/{machine_id}_metadata.json')
    
    if not metadata_path.exists():
        print(f"Warning: Metadata not found for {machine_id}")
        return {}
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def extract_machine_profile_features(machine_id):
    """
    Extract machine profile features from machine_id and metadata
    Returns: dict of features for the machine
    """
    # Parse machine_id to extract info
    parts = machine_id.split('_')
    
    # Extract category (first part)
    category = parts[0] if len(parts) > 0 else 'unknown'
    
    # Extract manufacturer (second part)
    manufacturer = parts[1] if len(parts) > 1 else 'unknown'
    
    # Load metadata if available
    metadata = load_machine_metadata(machine_id)
    
    # Build feature dict
    features = {
        'machine_id': machine_id,
        'machine_category': category,
        'manufacturer': manufacturer
    }
    
    # Add additional metadata if available
    # Note: The metadata JSON contains column definitions, not machine specs
    # For now, use placeholder values (can be enhanced with actual machine profiles)
    
    # Map categories to typical power ratings (placeholder)
    power_mapping = {
        'motor': 75.0,
        'pump': 50.0,
        'compressor': 100.0,
        'fan': 30.0,
        'cnc': 150.0,
        'hydraulic': 80.0,
        'conveyor': 25.0,
        'robot': 40.0,
        'transformer': 200.0,
        'cooling': 60.0,
        'turbofan': 500.0
    }
    
    features['power_rating_kw'] = power_mapping.get(category, 50.0)
    
    # Additional placeholder features
    features['rated_speed_rpm'] = 1500.0 if category in ['motor', 'pump', 'fan'] else 0.0
    features['operating_voltage'] = 480.0
    features['equipment_age_years'] = 5.0
    
    return features

def verify_and_pool_synthetic_data():
    """
    Verify Phase 1 data and create POOLED datasets
    This allows training generic models that work for ALL machines
    """
    
    print("=" * 70)
    print("PHASE 2.1.2: DATA VERIFICATION & POOLING")
    print("=" * 70)
    print()
    
    gan_data_path = Path('../../GAN/data/synthetic')
    
    if not gan_data_path.exists():
        print(f"ERROR: GAN data path not found: {gan_data_path}")
        sys.exit(1)
    
    # Containers for pooled data
    all_train_data = []
    all_val_data = []
    all_test_data = []
    
    # Verification report
    verification_report = []
    
    # Get all machine directories
    machine_dirs = sorted([d for d in gan_data_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(machine_dirs)} machines in {gan_data_path}")
    print()
    
    for machine_dir in machine_dirs:
        machine_id = machine_dir.name
        
        print(f"Processing: {machine_id}")
        
        # Check for required files
        train_file = machine_dir / 'train.parquet'
        val_file = machine_dir / 'val.parquet'
        test_file = machine_dir / 'test.parquet'
        
        missing_files = []
        if not train_file.exists():
            missing_files.append('train.parquet')
        if not val_file.exists():
            missing_files.append('val.parquet')
        if not test_file.exists():
            missing_files.append('test.parquet')
        
        if missing_files:
            print(f"  ERROR: Missing files: {missing_files}")
            verification_report.append({
                'machine_id': machine_id,
                'status': 'FAILED',
                'error': f"Missing files: {missing_files}"
            })
            continue
        
        try:
            # Load splits
            train_df = pd.read_parquet(train_file)
            val_df = pd.read_parquet(val_file)
            test_df = pd.read_parquet(test_file)
            
            # Get machine profile features
            profile_features = extract_machine_profile_features(machine_id)
            
            # Add machine metadata as features to each dataframe
            for key, value in profile_features.items():
                train_df[key] = value
                val_df[key] = value
                test_df[key] = value
            
            # Append to pooled containers
            all_train_data.append(train_df)
            all_val_data.append(val_df)
            all_test_data.append(test_df)
            
            # Add to verification report
            verification_report.append({
                'machine_id': machine_id,
                'status': 'SUCCESS',
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'total_samples': len(train_df) + len(val_df) + len(test_df),
                'n_features': len(train_df.columns),
                'feature_names': list(train_df.columns)
            })
            
            print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,} | Features: {len(train_df.columns)}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            verification_report.append({
                'machine_id': machine_id,
                'status': 'FAILED',
                'error': str(e)
            })
    
    print()
    print("=" * 70)
    print("POOLING DATA FROM ALL MACHINES")
    print("=" * 70)
    print()
    
    if not all_train_data:
        print("ERROR: No data to pool!")
        sys.exit(1)
    
    # Combine all machines
    pooled_train = pd.concat(all_train_data, ignore_index=True)
    pooled_val = pd.concat(all_val_data, ignore_index=True)
    pooled_test = pd.concat(all_test_data, ignore_index=True)
    
    print(f"Pooled Training Data: {len(pooled_train):,} samples from {len(all_train_data)} machines")
    print(f"Pooled Validation Data: {len(pooled_val):,} samples from {len(all_val_data)} machines")
    print(f"Pooled Test Data: {len(pooled_test):,} samples from {len(all_test_data)} machines")
    print(f"Total Features: {len(pooled_train.columns)}")
    print()
    
    # Show machine metadata features
    metadata_cols = ['machine_id', 'machine_category', 'manufacturer', 'power_rating_kw', 
                     'rated_speed_rpm', 'operating_voltage', 'equipment_age_years']
    
    print("Machine Metadata Features Added:")
    for col in metadata_cols:
        if col in pooled_train.columns:
            print(f"  - {col}")
    print()
    
    # Show sample distribution by category
    print("Distribution by Machine Category:")
    print(pooled_train['machine_category'].value_counts())
    print()
    
    # Save pooled datasets
    output_dir = Path('../data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output = output_dir / 'pooled_train.parquet'
    val_output = output_dir / 'pooled_val.parquet'
    test_output = output_dir / 'pooled_test.parquet'
    
    print("Saving pooled datasets...")
    pooled_train.to_parquet(train_output, index=False)
    pooled_val.to_parquet(val_output, index=False)
    pooled_test.to_parquet(test_output, index=False)
    
    print(f"  Saved: {train_output}")
    print(f"  Saved: {val_output}")
    print(f"  Saved: {test_output}")
    print()
    
    # Save verification report
    report_path = Path('../reports/data_verification_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_report = {
        'total_machines': len(machine_dirs),
        'successful_machines': len([r for r in verification_report if r['status'] == 'SUCCESS']),
        'failed_machines': len([r for r in verification_report if r['status'] == 'FAILED']),
        'pooled_train_samples': len(pooled_train),
        'pooled_val_samples': len(pooled_val),
        'pooled_test_samples': len(pooled_test),
        'total_features': len(pooled_train.columns),
        'machine_metadata_features': metadata_cols,
        'machine_details': verification_report
    }
    
    with open(report_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"Verification report saved: {report_path}")
    print()
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Machines: {summary_report['total_machines']}")
    print(f"Successful: {summary_report['successful_machines']}")
    print(f"Failed: {summary_report['failed_machines']}")
    print(f"Total Training Samples: {summary_report['pooled_train_samples']:,}")
    print(f"Total Validation Samples: {summary_report['pooled_val_samples']:,}")
    print(f"Total Test Samples: {summary_report['pooled_test_samples']:,}")
    print(f"Total Features: {summary_report['total_features']}")
    print()
    print("Phase 2.1.2 Data Verification & Pooling Complete!")
    print("=" * 70)
    
    return pooled_train, pooled_val, pooled_test, summary_report

if __name__ == "__main__":
    verify_and_pool_synthetic_data()
