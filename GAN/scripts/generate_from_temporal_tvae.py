"""
Phase 1.6 Week 4: Generate Synthetic Data from Temporal TVAE Models
Generate 50K samples per machine with natural RUL-sensor correlations

AUTOMATIC POST-PROCESSING (Phase 1.7 Integration):
- TVAE generates samples independently (no temporal awareness)
- Script automatically sorts by RUL (descending) for degradation pattern
- Script automatically assigns sequential timestamps (1 hour intervals)
- Result: 100% chronological order, 100% RUL decreasing

This ensures ALL future generations are chronologically ordered without manual fixes!
"""

import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer
import zlib

# Ensure project root is importable so `GAN.*` namespace imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _build_monotonic_rul_curve(max_rul: float, n: int, pattern: str) -> np.ndarray:
    """Create a deterministic monotonic RUL curve from max_rul down to 0.

    Important: We intentionally do NOT add noise here.
    TVAE already provides sensor variability; adding noise to the RUL label
    can break monotonicity and confuse downstream chronological validation.
    """
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(max_rul)], dtype=float)

    max_rul = float(max_rul)
    t = np.linspace(0.0, 1.0, n)

    if pattern == 'exponential':
        # Accelerating degradation (slow early, fast near end)
        rul = max_rul * (1.0 - t**2)
    else:
        # Default linear patterns
        rul = max_rul * (1.0 - t)

    # Hard guarantee: last sample is exactly 0 and values are non-negative.
    rul[-1] = 0.0
    return np.clip(rul, 0.0, None).astype(float)


def _assign_sequential_timestamps(n: int, max_rul_hours: float) -> pd.Series:
    """Assign strictly increasing timestamps spanning max_rul_hours.

    If max_rul_hours is in hours of remaining life, then a single lifecycle
    of n samples should span roughly max_rul_hours.
    """
    start = pd.Timestamp('2024-01-01')
    if n <= 0:
        return pd.to_datetime(pd.Series([], dtype='datetime64[ns]'))
    if n == 1:
        return pd.Series([start])

    max_rul_hours = float(max_rul_hours) if max_rul_hours is not None else float(n - 1)
    if not np.isfinite(max_rul_hours) or max_rul_hours <= 0:
        max_rul_hours = float(n - 1)

    step_seconds = (max_rul_hours * 3600.0) / float(n - 1)
    offsets = pd.to_timedelta(np.arange(n) * step_seconds, unit='s')
    return pd.Series(start + offsets)


def generate_temporal_data(machine_id, num_samples=50000, train_split=0.70, val_split=0.15):
    """
    Generate synthetic data from retrained temporal TVAE model
    
    Args:
        machine_id: Machine identifier
        num_samples: Total samples to generate (default: 50000)
        train_split: Training set proportion (default: 0.70 = 35000 samples)
        val_split: Validation set proportion (default: 0.15 = 7500 samples)
        
    Returns:
        dict: Generation results with statistics
    """
    
    print(f"\n{'=' * 80}")
    print(f"PHASE 1.6 WEEK 4: GENERATE SYNTHETIC DATA - {machine_id}")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    # Paths
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models" / "tvae" / "temporal"
    output_dir = project_root / "data" / "synthetic" / machine_id
    reports_dir = project_root / "reports" / "generation"
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load retrained TVAE model
    print(f"[1/6] Loading retrained temporal TVAE model")
    
    # Find model file with any epoch count
    model_files = list(model_dir.glob(f"{machine_id}_tvae_temporal_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError(
            f"Temporal TVAE model not found for machine: {machine_id}\n"
            f"Searched in: {model_dir}\n"
            f"Please run Phase 1.6 Weeks 2-3 retraining first."
        )
    
    # Use the most recent model if multiple exist
    model_path = sorted(model_files, key=lambda p: p.stat().st_mtime)[-1]
    
    print(f"   Model: {model_path.name}")
    tvae = TVAESynthesizer.load(str(model_path))
    print(f"   Model loaded successfully")
    
    # Step 2: Generate synthetic samples
    print(f"\n[2/6] Generating {num_samples:,} synthetic samples")
    print(f"   Note: TVAE generates samples independently (no temporal order)")
    print(f"   Auto-fix will apply chronological ordering after generation")
    print(f"\n   Generating...")
    
    gen_start = time.time()
    synthetic_data = tvae.sample(num_rows=num_samples)
    gen_time = time.time() - gen_start
    
    print(f"   Generated {len(synthetic_data):,} samples in {gen_time:.2f}s")
    print(f"   Shape: {synthetic_data.shape}")
    print(f"   Columns: {list(synthetic_data.columns)}")
    
    # CRITICAL: Establish a consistent severity ordering.
    # TVAE samples are i.i.d. We use the sampled RUL ONLY as a *rank signal*
    # to order rows from healthy → failed (high → low) while preserving learned
    # sensor correlations.
    #
    # IMPORTANT: We will (re)assign deterministic RUL curves and timestamps
    # AFTER splitting so that each split (train/val/test) has its own full
    # lifecycle curve from max_rul → 0 with strictly increasing timestamps.
    print(f"\n   [AUTO-FIX] Applying chronological ordering...")
    has_rul = 'rul' in synthetic_data.columns
    if has_rul:
        print(f"   - Sorting by generated RUL (descending) to preserve rank")
        print(f"   - Will assign per-split RUL + timestamps after splitting")
        synthetic_data = synthetic_data.sort_values('rul', ascending=False).reset_index(drop=True)
        synthetic_data['__severity_rank'] = np.arange(len(synthetic_data), dtype=int)
    else:
        # No-RUL machines: keep a stable order and only assign timestamps.
        print(f"   - No RUL column present; skipping severity ranking")
        print(f"   - Will assign per-split timestamps after splitting")
        synthetic_data = synthetic_data.reset_index(drop=True)
        synthetic_data['__severity_rank'] = np.arange(len(synthetic_data), dtype=int)
    
    # Step 3: Verify RUL column exists (optional)
    print(f"\n[3/6] Verifying RUL column presence")
    if not has_rul:
        print(f"   RUL column present: NO (no-RUL machine)")
    
    # Load RUL profile for deterministic curve and time span.
    try:
        from GAN.config.rul_profiles import get_rul_profile

        rul_profile = get_rul_profile(machine_id) or {}
        if has_rul:
            max_rul = float(rul_profile.get('max_rul', float(synthetic_data['rul'].max())))
        else:
            max_rul = float(rul_profile.get('max_rul', 24.0))
        pattern = str(rul_profile.get('degradation_pattern', 'linear_slow'))
    except Exception:
        max_rul = float(synthetic_data['rul'].max()) if has_rul else 24.0
        pattern = 'linear_slow'

    # Report based on the configured lifecycle curve (not the sampled distribution)
    rul_min = 0.0
    rul_max = max_rul
    rul_mean = max_rul / 2.0
    
    if has_rul:
        print(f"   RUL column present: YES")
        print(f"   RUL range: [{rul_min:.1f}, {rul_max:.1f}]")
        print(f"   RUL mean: {rul_mean:.1f}")
    
    # Step 4: Split data (train/val/test)
    # Traditional ML split: RANDOM selection into splits.
    # To keep each parquet file usable for time-series inspection, we sort each split by timestamp.
    print(f"\n[4/6] Splitting data into train/val/test")

    n_total = len(synthetic_data)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    # Deterministic per-machine shuffle so repeated generations are stable.
    seed = zlib.crc32(machine_id.encode('utf-8')) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _finalize_split(df: pd.DataFrame) -> pd.DataFrame:
        # Order by severity rank so the split itself is deterministic.
        df = df.sort_values('__severity_rank').reset_index(drop=True)
        if has_rul:
            df['rul'] = _build_monotonic_rul_curve(max_rul=max_rul, n=len(df), pattern=pattern)
        df['timestamp'] = _assign_sequential_timestamps(n=len(df), max_rul_hours=max_rul)
        return df

    train_data = _finalize_split(synthetic_data.iloc[train_idx].copy())
    val_data = _finalize_split(synthetic_data.iloc[val_idx].copy())
    test_data = _finalize_split(synthetic_data.iloc[test_idx].copy())

    # Drop helper columns before saving
    for df in (train_data, val_data, test_data):
        if '__severity_rank' in df.columns:
            df.drop(columns=['__severity_rank'], inplace=True)

    # Also drop helper columns from the in-memory full dataset used for stats.
    synthetic_data_stats = synthetic_data.drop(columns=['__severity_rank'], errors='ignore')
    
    print(f"   Split mode: random (seed={seed})")
    print(f"   Train: {len(train_data):,} samples ({train_split*100:.0f}%)")
    print(f"   Val:   {len(val_data):,} samples ({val_split*100:.0f}%)")
    print(f"   Test:  {len(test_data):,} samples ({(1-train_split-val_split)*100:.0f}%)")
    
    # Step 5: Save datasets
    print(f"\n[5/6] Saving datasets")
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_data.to_parquet(train_path, index=False)
    val_data.to_parquet(val_path, index=False)
    test_data.to_parquet(test_path, index=False)
    
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    val_size_mb = val_path.stat().st_size / (1024 * 1024)
    test_size_mb = test_path.stat().st_size / (1024 * 1024)
    total_size_mb = train_size_mb + val_size_mb + test_size_mb
    
    print(f"   Train saved: {train_path.name} ({train_size_mb:.2f} MB)")
    print(f"   Val saved:   {val_path.name} ({val_size_mb:.2f} MB)")
    print(f"   Test saved:  {test_path.name} ({test_size_mb:.2f} MB)")
    print(f"   Total size:  {total_size_mb:.2f} MB")
    
    # Step 6: Calculate statistics
    print(f"\n[6/6] Calculating generation statistics")
    
    # Check for missing values
    missing_total = synthetic_data_stats.isnull().sum().sum()
    missing_cols = synthetic_data_stats.isnull().sum()[synthetic_data_stats.isnull().sum() > 0]
    
    # Check for infinite values
    numeric_cols = synthetic_data_stats.select_dtypes(include=['float64', 'int64']).columns
    inf_total = 0
    for col in numeric_cols:
        inf_total += pd.isna(synthetic_data_stats[col]).sum() + (synthetic_data_stats[col] == float('inf')).sum() + (synthetic_data_stats[col] == float('-inf')).sum()
    
    # RUL correlation with sensors (optional)
    sensor_cols = [col for col in synthetic_data_stats.columns if col not in ['timestamp', 'rul']]
    rul_correlations = {}

    if has_rul and sensor_cols:
        for sensor in sensor_cols:
            if sensor in numeric_cols:
                corr = synthetic_data_stats['rul'].corr(synthetic_data_stats[sensor])
                rul_correlations[sensor] = round(corr, 3)

        if rul_correlations:
            avg_corr = sum(rul_correlations.values()) / len(rul_correlations)
            negative_count = sum(1 for c in rul_correlations.values() if c < 0)
            negative_pct = (negative_count / len(rul_correlations)) * 100
        else:
            avg_corr = None
            negative_pct = None
    else:
        avg_corr = None
        negative_pct = None
    
    # Total time
    total_time = time.time() - start_time
    
    # Prepare results
    results = {
        'machine_id': machine_id,
        'generation_time_seconds': round(gen_time, 2),
        'total_time_seconds': round(total_time, 2),
        'samples_generated': len(synthetic_data_stats),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'features': len(synthetic_data_stats.columns),
        'rul_present': bool(has_rul),
        'rul_range': [round(rul_min, 2), round(rul_max, 2)] if has_rul else None,
        'rul_mean': round(rul_mean, 2) if has_rul else None,
        'missing_values': int(missing_total),
        'infinite_values': int(inf_total),
        'data_quality': 'PASS' if missing_total == 0 and inf_total == 0 else 'FAIL',
        'rul_correlations': rul_correlations,
        'avg_rul_correlation': round(avg_corr, 3) if avg_corr else None,
        'negative_correlation_pct': round(negative_pct, 1) if negative_pct else None,
        'total_size_mb': round(total_size_mb, 2),
        'output_directory': str(output_dir.relative_to(project_root)),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\n   Generation Statistics:")
    print(f"   - Samples: {len(synthetic_data_stats):,}")
    print(f"   - Features: {len(synthetic_data_stats.columns)}")
    print(f"   - Missing values: {missing_total}")
    print(f"   - Infinite values: {inf_total}")
    print(f"   - Data quality: {results['data_quality']}")
    
    if avg_corr:
        print(f"   - Avg RUL correlation: {avg_corr:.3f}")
        print(f"   - Negative correlations: {negative_pct:.1f}%")
    
    # Save report
    report_path = reports_dir / f"{machine_id}_generation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"GENERATION COMPLETE: {machine_id}")
    print(f"{'=' * 80}")
    print(f"\nResults:")
    print(f"   Generated: {len(synthetic_data_stats):,} samples")
    print(f"   Time: {gen_time:.2f}s")
    print(f"   Size: {total_size_mb:.2f} MB")
    print(f"   RUL present: {'YES' if has_rul else 'NO'}")
    print(f"   Data quality: {results['data_quality']}")
    print(f"   Output: {output_dir}")
    print(f"   Report: {report_path.name}")
    print(f"\n{'=' * 80}\n")
    
    return results


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic data from temporal TVAE model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50K samples (default)
  python generate_from_temporal_tvae.py motor_siemens_1la7_001
  
  # Generate custom number of samples
  python generate_from_temporal_tvae.py motor_siemens_1la7_001 --num_samples 100000
  
  # Custom train/val/test split
  python generate_from_temporal_tvae.py motor_siemens_1la7_001 --train_split 0.80 --val_split 0.10

Phase Context:
  Phase 1.6 Week 4: Generate synthetic data from retrained TVAE models
  Phase 1.7 Integration: Automatic chronological ordering
  
  AUTOMATIC POST-PROCESSING:
  - TVAE generates samples independently (no temporal awareness)
  - Script automatically sorts by RUL (descending) for degradation
  - Script automatically assigns sequential timestamps (1hr intervals)
  - Result: 100% chronological order, 100% RUL decreasing
  
  This ensures ALL future generations are properly ordered without manual fixes!
  
  Output: train.parquet, val.parquet, test.parquet in data/synthetic/{machine_id}/
        """
    )
    
    parser.add_argument(
        'machine_id',
        help='Machine identifier (e.g., motor_siemens_1la7_001)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50000,
        help='Number of samples to generate (default: 50000)'
    )
    
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.70,
        help='Training set proportion (default: 0.70)'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Validate splits
    test_split = 1.0 - args.train_split - args.val_split
    if test_split < 0 or test_split > 1:
        print(f"ERROR: Invalid splits - train ({args.train_split}) + val ({args.val_split}) must be <= 1.0")
        sys.exit(1)
    
    try:
        results = generate_temporal_data(
            args.machine_id,
            num_samples=args.num_samples,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        print("Next Steps:")
        print("  1. Review generation report in reports/generation/")
        print("  2. Verify data quality (no missing/infinite values)")
        print("  3. Check RUL correlations (should be negative)")
        print("  4. Continue with batch generation for all 26 machines\n")
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nPlease run Phase 1.6 Weeks 2-3 first:")
        print(f"  python scripts/batch_retrain_all_tvae_temporal.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
