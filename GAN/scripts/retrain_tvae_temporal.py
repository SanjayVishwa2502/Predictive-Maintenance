"""
TVAE Temporal Retraining Script - Phase 1.6 Weeks 2-3
Retrain TVAE with temporal seed data including RUL as a feature

This script retrains TVAE models to learn the correlation between sensor values and RUL.
Unlike the original training (Phase 1.3) which used random seed data without RUL,
this retraining uses temporal seed data where:
- RUL column is included as a numerical feature
- Sensors are correlated with degradation (correlation < -0.60)
- Multiple life cycles are represented (2-7 cycles per machine)

The retrained TVAE will learn P(RUL, sensors) joint distribution, enabling generation
of synthetic data where sensor-RUL correlations are naturally preserved.
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


def create_temporal_metadata(seed_data, machine_id):
    """
    Create metadata for temporal seed data with RUL as numerical feature
    
    Args:
        seed_data: DataFrame with temporal seed data (timestamp, rul, sensors)
        machine_id: Machine identifier for context
    
    Returns:
        SingleTableMetadata: Configured metadata with proper data types
    """
    metadata = SingleTableMetadata()
    
    # Configure metadata based on column types
    # timestamp: datetime (SDV handles datetime columns)
    # rul: float (remaining useful life in hours)
    # sensors: float (various sensor measurements)
    
    for col in seed_data.columns:
        if col == 'timestamp':
            # Use datetime type for timestamp column
            metadata.add_column(
                column_name=col,
                sdtype='datetime',
                datetime_format='%Y-%m-%d %H:%M:%S'
            )
        else:
            # All other columns are numerical
            metadata.add_column(
                column_name=col,
                sdtype='numerical'
            )
    
    print(f"   Metadata created: {len(metadata.columns)} columns")
    print(f"   - timestamp: datetime")
    print(f"   - rul + sensors: {len(metadata.columns) - 1} numerical features")
    print(f"   RUL range: [{seed_data['rul'].min():.1f}, {seed_data['rul'].max():.1f}]")
    
    return metadata


def retrain_machine_tvae_temporal(machine_id, config, test_mode=False):
    """
    Retrain TVAE for specific machine using temporal seed data with RUL
    
    Args:
        machine_id: Machine identifier (e.g., motor_siemens_1la7_001)
        config: Training configuration dictionary (from tvae_config.py)
        test_mode: If True, only train for 10 epochs for quick validation
    
    Returns:
        dict: Training results including quality score, timing, and RUL correlation
    """
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 1.6 WEEKS 2-3: TVAE Temporal Retraining for {machine_id}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    # Paths
    project_root = Path(__file__).parent.parent
    seed_path = project_root / "seed_data" / "temporal" / f"{machine_id}_temporal_seed.parquet"
    models_dir = project_root / "models" / "tvae" / "temporal"
    reports_dir = project_root / "reports" / "tvae_temporal"
    
    # Create output directories
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load temporal seed data
    print(f"[1/7] Loading temporal seed data from: {seed_path.name}")
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Temporal seed data not found: {seed_path}\n"
            f"Please run Phase 1.6 Days 3-5 first to generate temporal seed data."
        )
    
    seed_data = pd.read_parquet(seed_path)
    print(f"   Seed data shape: {seed_data.shape}")
    print(f"   Features: {list(seed_data.columns)}")
    print(f"   RUL column present: {'rul' in seed_data.columns}")
    
    # Validate RUL column exists
    if 'rul' not in seed_data.columns:
        raise ValueError(
            f"RUL column not found in temporal seed data for {machine_id}\n"
            f"Expected columns: timestamp, rul, sensor_1, sensor_2, ...\n"
            f"Found columns: {list(seed_data.columns)}"
        )
    
    # Step 2: Create metadata with RUL as numerical feature
    print(f"\n[2/7] Creating metadata with RUL as numerical feature")
    metadata = create_temporal_metadata(seed_data, machine_id)
    
    # Step 3: Initialize TVAE
    print(f"\n[3/7] Initializing TVAE Synthesizer")
    
    # Override epochs if test mode
    epochs = 10 if test_mode else config['epochs']
    print(f"   Architecture: TVAE (Tabular Variational AutoEncoder)")
    print(f"   Training Mode: Temporal (with RUL correlation)")
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
    print(f"\n[4/7] Training TVAE model with temporal seed data...")
    print(f"   Training started at: {time.strftime('%H:%M:%S')}")
    
    if test_mode:
        print(f"   Expected time: ~1-2 minutes (test mode, 10 epochs)")
    else:
        print(f"   Expected time: ~2 hours (production mode, {epochs} epochs)")
        print(f"   Note: RTX 4070 GPU acceleration enabled")
    
    print(f"\n   Training in progress...")
    
    train_start = time.time()
    synthesizer.fit(seed_data)
    train_time = time.time() - train_start
    
    print(f"\n   Training completed in: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    # Step 5: Save model
    print(f"\n[5/7] Saving retrained temporal model")
    model_filename = f"{machine_id}_tvae_temporal_{epochs}epochs.pkl"
    model_path = models_dir / model_filename
    synthesizer.save(str(model_path))
    
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   Model saved: {model_filename}")
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Location: models/tvae/temporal/")
    
    # Step 6: Evaluate quality
    print(f"\n[6/7] Evaluating synthetic data quality")
    print(f"   Generating 1000 test samples...")
    
    test_samples = synthesizer.sample(num_rows=1000)
    # Save synthetic data to GAN/data/synthetic
    synthetic_dir = project_root / "data" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = synthetic_dir / f"{machine_id}_synthetic_temporal.parquet"
    test_samples.to_parquet(synthetic_path)
    print(f"   Synthetic data saved to: {synthetic_path}")
    
    # Verify RUL column in generated samples
    if 'rul' not in test_samples.columns:
        print(f"   WARNING: RUL column not found in generated samples!")
        rul_present = False
        rul_correlation = None
    else:
        rul_present = True
        print(f"   RUL column present in generated samples: YES")
        print(f"   Generated RUL range: [{test_samples['rul'].min():.1f}, {test_samples['rul'].max():.1f}]")
    
    print(f"\n   Evaluating quality metrics...")
    quality_report = evaluate_quality(seed_data, test_samples, metadata)
    quality_score = quality_report.get_score()
    
    print(f"\n   Quality Score: {quality_score:.3f}")
    print(f"   Target Score: >0.80 (acceptable for temporal data)")
    
    if quality_score >= 0.90:
        print(f"   Status: EXCELLENT (exceptional quality)")
    elif quality_score >= 0.80:
        print(f"   Status: GOOD (acceptable quality)")
    elif quality_score >= 0.70:
        print(f"   Status: ACCEPTABLE (meets minimum threshold)")
    else:
        print(f"   Status: WARNING (below expected quality)")
    
    # Step 7: Verify RUL-sensor correlations
    print(f"\n[7/7] Verifying RUL-sensor correlations in generated data")
    
    if rul_present:
        # Calculate correlations between RUL and all sensor columns
        sensor_cols = [col for col in test_samples.columns if col not in ['timestamp', 'rul']]
        
        if sensor_cols:
            correlations = []
            for sensor in sensor_cols:
                corr = test_samples['rul'].corr(test_samples[sensor])
                correlations.append({
                    'sensor': sensor,
                    'correlation': round(corr, 3)
                })
            
            # Find min/max correlations
            min_corr = min(correlations, key=lambda x: x['correlation'])
            max_corr = max(correlations, key=lambda x: x['correlation'])
            avg_corr = sum(c['correlation'] for c in correlations) / len(correlations)
            
            print(f"   Sensors analyzed: {len(sensor_cols)}")
            print(f"   Average RUL correlation: {avg_corr:.3f}")
            print(f"   Min correlation: {min_corr['correlation']:.3f} ({min_corr['sensor']})")
            print(f"   Max correlation: {max_corr['correlation']:.3f} ({max_corr['sensor']})")
            
            # Check if correlations are negative (expected for degradation)
            negative_count = sum(1 for c in correlations if c['correlation'] < 0)
            negative_pct = (negative_count / len(correlations)) * 100
            
            print(f"   Negative correlations: {negative_count}/{len(correlations)} ({negative_pct:.1f}%)")
            
            if negative_pct >= 80:
                print(f"   Correlation Status: EXCELLENT (degradation pattern learned)")
            elif negative_pct >= 60:
                print(f"   Correlation Status: GOOD (most sensors show degradation)")
            else:
                print(f"   Correlation Status: WARNING (weak degradation correlation)")
            
            rul_correlation = {
                'average': round(avg_corr, 3),
                'min': min_corr['correlation'],
                'max': max_corr['correlation'],
                'negative_percentage': round(negative_pct, 1),
                'all_correlations': correlations
            }
        else:
            print(f"   WARNING: No sensor columns found for correlation analysis")
            rul_correlation = None
    else:
        print(f"   SKIPPED: RUL column not present in generated samples")
        rul_correlation = None
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Prepare results
    results = {
        'machine_id': machine_id,
        'architecture': 'TVAE',
        'training_mode': 'temporal_with_rul',
        'epochs': epochs,
        'batch_size': config['batch_size'],
        'training_time_seconds': round(train_time, 2),
        'training_time_minutes': round(train_time / 60, 2),
        'training_time_hours': round(train_time / 3600, 2),
        'total_time_seconds': round(total_time, 2),
        'total_time_minutes': round(total_time / 60, 2),
        'quality_score': round(quality_score, 3),
        'model_size_mb': round(model_size_mb, 2),
        'seed_samples': len(seed_data),
        'features': len(seed_data.columns),
        'rul_present': rul_present,
        'rul_correlation': rul_correlation,
        'model_path': f"models/tvae/temporal/{model_filename}",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save detailed report
    report_path = reports_dir / f"{machine_id}_tvae_temporal_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"TEMPORAL RETRAINING COMPLETE: {machine_id}")
    print(f"{'=' * 70}")
    print(f"\nResults Summary:")
    print(f"   Training Time: {train_time:.2f}s ({train_time/60:.2f} min, {train_time/3600:.2f} hr)")
    print(f"   Quality Score: {quality_score:.3f}")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    print(f"   RUL Present: {rul_present}")
    if rul_correlation:
        print(f"   Avg RUL Correlation: {rul_correlation['average']:.3f}")
        print(f"   Negative Correlations: {rul_correlation['negative_percentage']:.1f}%")
    print(f"   Model: {model_filename}")
    print(f"   Report: reports/tvae_temporal/{report_path.name}")
    print(f"\n{'=' * 70}\n")
    
    return results


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Retrain TVAE model with temporal seed data (includes RUL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain with default config (500 epochs, ~2 hours)
  python retrain_tvae_temporal.py motor_siemens_1la7_001
  
  # Quick test (10 epochs, ~1-2 minutes)
  python retrain_tvae_temporal.py motor_siemens_1la7_001 --test
  
  # Custom epochs
  python retrain_tvae_temporal.py motor_siemens_1la7_001 --epochs 300

Phase Context:
  Phase 1.6 Weeks 2-3: TVAE Retraining with Temporal Seed Data
  
  This retraining teaches TVAE the correlation between sensor values and RUL.
  Unlike Phase 1.3 (random seed data), this uses temporal seed data where:
  - RUL is included as a feature
  - Sensors show degradation patterns (correlation < -0.60)
  - Multiple life cycles are represented
  
  Expected: ~2 hours per machine with 500 epochs on RTX 4070
  Total: ~52 hours for all 26 machines (run overnight/weekend)
        """
    )
    
    parser.add_argument(
        'machine_id',
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
        help='Quick test mode (10 epochs, ~1-2 minutes)'
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
        results = retrain_machine_tvae_temporal(args.machine_id, config, test_mode=args.test)
        
        # Print next steps
        print("\nNext Steps:")
        print("  1. Review correlation results in the report")
        print("  2. Test model: Load and generate samples to verify RUL is present")
        print("  3. Continue with batch retraining for all 26 machines")
        print("  4. After all retraining: Phase 1.6 Week 4 (generate 50K samples)\n")
        
        # Exit with success
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nPlease ensure Phase 1.6 Days 3-5 is complete:")
        print(f"  - Temporal seed data in seed_data/temporal/")
        print(f"  - Run: python scripts/create_temporal_seed_data.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR during retraining: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
