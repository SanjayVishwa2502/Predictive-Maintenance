"""
Phase 1.4.1: Synthetic Data Generation Script
Generate synthetic data using trained TVAE models
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from sdv.single_table import TVAESynthesizer

def generate_machine_data(machine_id, num_samples=50000):
    """Generate synthetic data for specific machine"""
    
    print(f"\n{'=' * 70}")
    print(f"Generating synthetic data: {machine_id}")
    print(f"{'=' * 70}")
    print(f"Target samples: {num_samples}")
    
    base_path = Path(__file__).parent.parent
    model_path = base_path / "models" / "tvae" / f"{machine_id}_tvae_500epochs.pkl"
    output_dir = base_path / "data" / "synthetic" / machine_id
    
    try:
        # Load trained TVAE model
        print(f"Loading model: {model_path.name}")
        synthesizer = TVAESynthesizer.load(str(model_path))
        print(f"[OK] Model loaded successfully")
        
        # Generate synthetic samples
        print(f"Generating {num_samples} synthetic samples...")
        synthetic_data = synthesizer.sample(num_rows=num_samples)
        print(f"[OK] Generated {len(synthetic_data)} samples with {len(synthetic_data.columns)} features")
        
        # Add machine_id column
        synthetic_data['machine_id'] = machine_id
        
        # Split into train/val/test (70/15/15)
        n_total = len(synthetic_data)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        train_data = synthetic_data.iloc[:n_train]
        val_data = synthetic_data.iloc[n_train:n_train+n_val]
        test_data = synthetic_data.iloc[n_train+n_val:]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_data.to_parquet(output_dir / "train.parquet", index=False)
        val_data.to_parquet(output_dir / "val.parquet", index=False)
        test_data.to_parquet(output_dir / "test.parquet", index=False)
        
        print(f"\n[OK] Generated and saved:")
        print(f"   Train: {len(train_data)} samples (70%)")
        print(f"   Val: {len(val_data)} samples (15%)")
        print(f"   Test: {len(test_data)} samples (15%)")
        print(f"   Location: {output_dir}")
        
        return {
            'machine_id': machine_id,
            'status': 'SUCCESS',
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data),
            'total': len(synthetic_data),
            'features': len(synthetic_data.columns) - 1  # Exclude machine_id
        }
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate data for {machine_id}")
        print(f"Error: {str(e)}")
        return {
            'machine_id': machine_id,
            'status': 'FAILED',
            'error': str(e)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic data for a machine')
    parser.add_argument('--machine_id', required=True, help='Machine ID')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples to generate')
    args = parser.parse_args()
    
    result = generate_machine_data(args.machine_id, args.num_samples)
    
    if result['status'] == 'SUCCESS':
        print(f"\n[SUCCESS] Synthetic data generation complete for {args.machine_id}")
        sys.exit(0)
    else:
        print(f"\n[FAILED] Synthetic data generation failed for {args.machine_id}")
        sys.exit(1)
