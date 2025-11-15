"""
TVAE Training Script - Machine-Specific Training Pipeline
Phase 1.3.1: Training Pipeline Setup

Train TVAE (Tabular Variational AutoEncoder) for individual machines
Based on Phase 1.1.3 decision: TVAE chosen for higher quality (0.913) and faster training
"""

import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config.tvae_config import TVAE_CONFIG


def train_machine_tvae(machine_id, config, test_mode=False):
    """
    Train TVAE for specific machine
    
    Args:
        machine_id: Machine identifier (e.g., motor_siemens_1la7_001)
        config: Training configuration dictionary
        test_mode: If True, only train for 10 epochs for quick validation
    
    Returns:
        dict: Training results including quality score and timing
    """
    
    print(f"\n{'=' * 60}")
    print(f"PHASE 1.3.1: TVAE Training for {machine_id}")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    
    # Paths
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "metadata" / f"{machine_id}_metadata.json"
    seed_path = project_root / "seed_data" / f"{machine_id}_seed.parquet"
    models_dir = project_root / "models" / "tvae"
    reports_dir = project_root / "reports"
    
    # Create output directories
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load metadata
    print(f"[1/6] Loading metadata from: {metadata_path.name}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    metadata = SingleTableMetadata.load_from_json(str(metadata_path))
    print(f"   Metadata loaded: {len(metadata.columns)} columns")
    
    # Step 2: Load seed data
    print(f"\n[2/6] Loading seed data from: {seed_path.name}")
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed data not found: {seed_path}")
    
    seed_data = pd.read_parquet(seed_path)
    print(f"   Seed data shape: {seed_data.shape}")
    print(f"   Features: {list(seed_data.columns)}")
    
    # Step 3: Initialize TVAE
    print(f"\n[3/6] Initializing TVAE Synthesizer")
    
    # Override epochs if test mode
    epochs = 10 if test_mode else config['epochs']
    print(f"   Architecture: TVAE (Tabular Variational AutoEncoder)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   GPU Enabled: {config['cuda']}")
    print(f"   Compress/Decompress Dims: {config['compress_dims']}")
    print(f"   Embedding Dimension: {config.get('embedding_dim', 128)}")
    
    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=config['batch_size'],
        cuda=config['cuda'],
        verbose=config['verbose'],
        embedding_dim=config.get('embedding_dim', 128),
        compress_dims=config['compress_dims'],
        decompress_dims=config['decompress_dims'],
        l2scale=config['weight_decay'],
        loss_factor=config['loss_factor']
    )
    
    # Step 4: Train
    print(f"\n[4/6] Training TVAE model...")
    print(f"   Training started at: {time.strftime('%H:%M:%S')}")
    print(f"   Expected time: ~{0.2 if not test_mode else 0.05} minutes\n")
    
    train_start = time.time()
    synthesizer.fit(seed_data)
    train_time = time.time() - train_start
    
    print(f"\n   Training completed in: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    # Step 5: Save model
    print(f"\n[5/6] Saving trained model")
    model_filename = f"{machine_id}_tvae_{epochs}epochs.pkl"
    model_path = models_dir / model_filename
    synthesizer.save(str(model_path))
    
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   Model saved: {model_filename}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    
    # Step 6: Evaluate quality
    print(f"\n[6/6] Evaluating synthetic data quality")
    print(f"   Generating 1000 test samples...")
    
    test_samples = synthesizer.sample(num_rows=1000)
    
    print(f"   Evaluating quality metrics...")
    quality_report = evaluate_quality(seed_data, test_samples, metadata)
    quality_score = quality_report.get_score()
    
    print(f"\n   Quality Score: {quality_score:.3f}")
    print(f"   Target Score: >0.91 (based on Phase 1.1.3 results)")
    
    if quality_score >= 0.91:
        print(f"   Status: EXCELLENT (meets/exceeds expectations)")
    elif quality_score >= 0.80:
        print(f"   Status: GOOD (acceptable quality)")
    else:
        print(f"   Status: WARNING (below expected quality)")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Prepare results
    results = {
        'machine_id': machine_id,
        'architecture': 'TVAE',
        'epochs': epochs,
        'batch_size': config['batch_size'],
        'training_time_seconds': round(train_time, 2),
        'training_time_minutes': round(train_time / 60, 2),
        'total_time_seconds': round(total_time, 2),
        'total_time_minutes': round(total_time / 60, 2),
        'quality_score': round(quality_score, 3),
        'model_size_mb': round(model_size_mb, 2),
        'seed_samples': len(seed_data),
        'features': len(seed_data.columns),
        'model_path': str(model_path.relative_to(project_root)),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save detailed quality report
    report_path = reports_dir / f"{machine_id}_tvae_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE: {machine_id}")
    print(f"{'=' * 60}")
    print(f"\nResults Summary:")
    print(f"   Training Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"   Quality Score: {quality_score:.3f}")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    print(f"   Model: {model_filename}")
    print(f"   Report: {report_path.name}")
    print(f"\n{'=' * 60}\n")
    
    return results


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Train TVAE model for specific machine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (300 epochs)
  python train_tvae_machine.py --machine_id motor_siemens_1la7_001
  
  # Quick test (10 epochs)
  python train_tvae_machine.py --machine_id motor_siemens_1la7_001 --test
  
  # Custom configuration
  python train_tvae_machine.py --machine_id motor_siemens_1la7_001 --epochs 500 --batch_size 1000
        """
    )
    
    parser.add_argument(
        '--machine_id',
        required=True,
        help='Machine identifier (e.g., motor_siemens_1la7_001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {TVAE_CONFIG["epochs"]} from config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'Batch size for training (default: {TVAE_CONFIG["batch_size"]} from config)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test mode (10 epochs only)'
    )
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = TVAE_CONFIG.copy()
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Train
    try:
        results = train_machine_tvae(args.machine_id, config, test_mode=args.test)
        
        # Exit with success
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nPlease ensure Phase 1.2 is complete:")
        print(f"  - Metadata files in GAN/metadata/")
        print(f"  - Seed data files in GAN/seed_data/")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
