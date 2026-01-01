"""
Phase 1.6 Days 3-5: Temporal Seed Data Generation
==================================================

Create physics-based seed data with temporal structure and RUL for TVAE retraining.

This seed data teaches TVAE:
- How sensors correlate with RUL degradation
- How degradation progresses over time
- Realistic failure patterns with noise

Author: GAN Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import time
import math

# Ensure project root is importable so `GAN.*` namespace imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from GAN.config.rul_profiles import RUL_PROFILES, get_machine_category, get_rul_profile


def _profile_path_for_machine(machine_id: str) -> Path:
    return PROJECT_ROOT / 'GAN' / 'data' / 'real_machines' / 'profiles' / f'{machine_id}.json'


def _load_machine_profile(machine_id: str) -> dict | None:
    try:
        p = _profile_path_for_machine(machine_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _is_missing_number(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in ('', 'none', 'null', 'nan'):
        return True
    if isinstance(v, (int, float)):
        try:
            return math.isnan(float(v))
        except Exception:
            return False
    return False


def _profile_has_rul(profile: dict | None) -> bool:
    if not isinstance(profile, dict):
        return True
    rul_min = profile.get('rul_min', None)
    rul_max = profile.get('rul_max', None)
    # If both are missing/NaN placeholders => explicitly no RUL.
    if _is_missing_number(rul_min) and _is_missing_number(rul_max):
        return False
    return True


def _sensors_from_profile(profile: dict | None) -> list[dict]:
    if not isinstance(profile, dict):
        return []

    sensors_config: list[dict] = []

    baseline = profile.get('baseline_normal_operation') or {}
    if isinstance(baseline, dict):
        # Common pattern: baseline_normal_operation.{temperature|thermal|vibration|...}.{sensor_name}.{min/max}
        for group_name, group in baseline.items():
            if not isinstance(group, dict):
                continue
            for sensor_name, spec in group.items():
                if not isinstance(spec, dict):
                    continue
                mn = spec.get('min', None)
                mx = spec.get('max', None)
                typ = spec.get('typical', None)

                # Establish a reasonable normal range
                if isinstance(mn, (int, float)) and isinstance(mx, (int, float)) and mx > mn:
                    normal_range = [float(mn), float(mx)]
                elif isinstance(typ, (int, float)):
                    normal_range = [float(typ) * 0.9, float(typ) * 1.1]
                else:
                    continue

                sensors_config.append({
                    'name': sensor_name,
                    'type': str(group_name),
                    'normal_range': normal_range,
                })

    # Also support explicit profile['sensors'] list if present.
    explicit = profile.get('sensors')
    if isinstance(explicit, list):
        for s in explicit:
            if not isinstance(s, dict):
                continue
            name = s.get('name')
            if not isinstance(name, str) or not name.strip():
                continue
            if any(x.get('name') == name for x in sensors_config):
                continue
            sensors_config.append({
                'name': name,
                'type': s.get('type', 'generic'),
                'normal_range': [0, 100],
            })

    return sensors_config


def create_temporal_seed_data(machine_id, rul_profile, n_samples=50000):
    """
    Create seed data with temporal structure and RUL
    
    TVAE will train on this data and learn:
    - How sensors correlate with RUL
    - How degradation progresses over time
    - Natural variations and noise patterns
    
    Args:
        machine_id: Machine identifier
        rul_profile: RUL configuration from rul_profiles.py
        n_samples: Total samples (default 50000, matching original production config)
        
    Returns:
        DataFrame with temporal structure, RUL, and degrading sensors
    """
    
    print(f"\n{'=' * 70}")
    print(f"Creating temporal seed data: {machine_id}")
    print(f"{'=' * 70}")
    
    profile = _load_machine_profile(machine_id)
    include_rul = _profile_has_rul(profile)

    max_rul = rul_profile['max_rul']
    cycles = rul_profile['cycles_per_dataset']
    pattern = rul_profile['degradation_pattern']
    noise_std = rul_profile['noise_std']
    correlations = rul_profile['sensor_correlation']
    
    print(f"Configuration:")
    print(f"  Max RUL: {max_rul} hours")
    print(f"  Include RUL column: {'YES' if include_rul else 'NO'}")
    print(f"  Cycles: {cycles}")
    print(f"  Pattern: {pattern}")
    print(f"  Total samples: {n_samples}")
    
    # Generate multiple run-to-failure cycles
    samples_per_cycle = n_samples // cycles
    all_cycles = []
    
    for cycle_idx in range(cycles):
        print(f"  Generating cycle {cycle_idx + 1}/{cycles}...", end=" ")
        
        # Adjust samples for last cycle
        cycle_samples = samples_per_cycle
        if cycle_idx == cycles - 1:
            cycle_samples = n_samples - len(all_cycles) * samples_per_cycle
        
        # Generate one degradation cycle
        cycle_data = generate_single_degradation_cycle(
            machine_id=machine_id,
            cycle_samples=cycle_samples,
            max_rul=max_rul,
            pattern=pattern,
            noise_std=noise_std,
            correlations=correlations,
            cycle_number=cycle_idx,
            include_rul=include_rul,
            profile=profile,
        )
        
        all_cycles.append(cycle_data)
        print(f"OK {len(cycle_data)} samples")
    
    # Combine all cycles
    full_data = pd.concat(all_cycles, ignore_index=True)
    
    # Add timestamps (hourly intervals)
    full_data['timestamp'] = pd.date_range(
        start='2024-01-01',
        periods=len(full_data),
        freq='1h'
    )
    
    # Reorder columns: timestamp, (optional) rul, sensors
    base_cols = ['timestamp'] + (['rul'] if include_rul and 'rul' in full_data.columns else [])
    cols = base_cols + [col for col in full_data.columns if col not in base_cols]
    full_data = full_data[cols]
    
    print(f"\n{'=' * 70}")
    print(f"Seed data creation complete!")
    print(f"  Total samples: {len(full_data)}")
    print(f"  Features: {len(full_data.columns)}")
    if include_rul and 'rul' in full_data.columns:
        print(f"  RUL range: {full_data['rul'].max():.0f} -> {full_data['rul'].min():.0f}")
    print(f"  Time range: {full_data['timestamp'].min()} to {full_data['timestamp'].max()}")
    print(f"{'=' * 70}\n")
    
    return full_data


def generate_single_degradation_cycle(machine_id, cycle_samples, max_rul, 
                                      pattern, noise_std, correlations, cycle_number,
                                      include_rul: bool = True,
                                      profile: dict | None = None):
    """
    Generate ONE run-to-failure cycle with physics-based degradation
    
    This creates realistic sensor behavior:
    - Temperature increases as bearing wears
    - Vibration increases exponentially near failure
    - Current changes with increased friction
    - All sensors have realistic noise
    """
    
    # ============================================
    # STEP 1: Generate degradation factor (+ optional RUL)
    # ============================================
    if include_rul:
        if pattern == 'exponential':
            t = np.linspace(0, 1, cycle_samples)
            rul_base = max_rul * (1 - t**2)  # Quadratic decay
        else:
            rul_base = np.linspace(max_rul, 0, cycle_samples)

        # Add realistic noise
        rul_noise = np.random.normal(0, noise_std, cycle_samples)
        rul = np.maximum(rul_base + rul_noise, 0)  # Ensure non-negative

        # degradation: 0 (healthy) -> 1 (failure)
        degradation = 1 - (rul / max_rul)
    else:
        # No-RUL machines: still generate a smooth degradation factor so sensors drift realistically.
        degradation = np.linspace(0.0, 1.0, cycle_samples)
        degradation = np.clip(degradation + np.random.normal(0.0, 0.01, cycle_samples), 0.0, 1.0)
    
    # ============================================
    # STEP 3: Generate physics-based sensor values
    # ============================================
    
    # Load machine metadata to get sensor names
    metadata_path = Path(__file__).parent.parent / "metadata" / f"{machine_id}_metadata.json"
    
    sensors_config = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Convert metadata columns to sensors_config format
        columns = metadata.get('columns', {})
        for col_name, col_info in columns.items():
            # Infer sensor type and normal range from column name
            if 'temp' in col_name.lower():
                sensor_type = 'temperature'
                normal_range = [20, 80]  # °C
            elif 'vib' in col_name.lower() or 'velocity' in col_name.lower():
                sensor_type = 'vibration'
                normal_range = [0.5, 8.0]  # mm/s
            elif 'current' in col_name.lower():
                sensor_type = 'current'
                normal_range = [5, 25]  # A
            elif 'voltage' in col_name.lower():
                sensor_type = 'voltage'
                normal_range = [380, 420]  # V
            elif 'pressure' in col_name.lower():
                sensor_type = 'pressure'
                normal_range = [1, 10]  # bar
            elif 'flow' in col_name.lower():
                sensor_type = 'flow'
                normal_range = [10, 100]  # L/min
            elif 'force' in col_name.lower():
                sensor_type = 'force'
                normal_range = [100, 500]  # N
            elif 'power' in col_name.lower() or 'efficiency' in col_name.lower():
                sensor_type = 'power'
                normal_range = [0.5, 1.0]  # factor/efficiency
            elif 'sound' in col_name.lower() or 'noise' in col_name.lower():
                sensor_type = 'sound'
                normal_range = [50, 90]  # dBA
            elif 'frequency' in col_name.lower():
                sensor_type = 'frequency'
                normal_range = [0, 1000]  # Hz
            else:
                # Generic numerical sensor
                sensor_type = 'generic'
                normal_range = [0, 100]
            
            sensors_config.append({
                'name': col_name,
                'type': sensor_type,
                'normal_range': normal_range
            })

    # If no metadata, attempt to derive sensors from the machine profile.
    if not sensors_config:
        sensors_config = _sensors_from_profile(profile)
    
    # Fallback if no metadata found
    if not sensors_config:
        sensors_config = [
            {'name': 'temperature_bearing_C', 'type': 'temperature', 'normal_range': [35, 65]},
            {'name': 'vibration_rms_mm_s', 'type': 'vibration', 'normal_range': [0.5, 4.5]},
            {'name': 'current_A', 'type': 'current', 'normal_range': [10, 20]}
        ]
    
    sensor_data = {}
    
    for sensor_config in sensors_config:
        sensor_name = sensor_config['name']
        sensor_type = sensor_config.get('type', 'unknown')
        normal_range = sensor_config.get('normal_range', [0, 100])
        
        baseline = normal_range[0]
        failure = normal_range[1]
        
        # Generate based on sensor type
        if 'temp' in sensor_type.lower() or 'temp' in sensor_name.lower():
            # Temperature: Linear increase with degradation
            correlation = correlations.get('temperature', 0.85)
            values = baseline + (failure - baseline) * degradation * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.03, cycle_samples)
            
        elif 'vib' in sensor_type.lower() or 'vib' in sensor_name.lower():
            # Vibration: Exponential increase near failure
            correlation = correlations.get('vibration', 0.80)
            # Ensure degradation is non-negative before power operation
            deg_safe = np.clip(degradation, 0, 1)
            values = baseline + (failure - baseline) * (deg_safe ** 1.5) * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.05, cycle_samples)
            
        elif 'current' in sensor_type.lower() or 'power' in sensor_type.lower() or 'current' in sensor_name.lower():
            # Current/Power: Moderate increase
            correlation = correlations.get('current', 0.70)
            values = baseline + (failure - baseline) * degradation * correlation * 0.7
            noise = np.random.normal(0, (failure - baseline) * 0.04, cycle_samples)
            
        elif 'pressure' in sensor_type.lower() or 'pressure' in sensor_name.lower():
            # Pressure: May decrease with wear (pumps, compressors)
            correlation = correlations.get('pressure', 0.75)
            values = failure - (failure - baseline) * degradation * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.03, cycle_samples)
            
        elif 'flow' in sensor_type.lower() or 'flow' in sensor_name.lower():
            # Flow: Decreases with clogging/wear
            correlation = correlations.get('flow', 0.85)
            values = failure - (failure - baseline) * degradation * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.04, cycle_samples)
            
        elif 'voltage' in sensor_type.lower() or 'voltage' in sensor_name.lower():
            # Voltage: Slight degradation
            correlation = correlations.get('voltage', 0.60)
            values = baseline + (failure - baseline) * degradation * correlation * 0.5
            noise = np.random.normal(0, (failure - baseline) * 0.02, cycle_samples)
            
        elif 'force' in sensor_type.lower() or 'force' in sensor_name.lower():
            # Cutting force: Increases with tool wear
            correlation = correlations.get('cutting_force', 0.75)
            values = baseline + (failure - baseline) * degradation * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.05, cycle_samples)
            
        elif 'spindle' in sensor_name.lower():
            # Spindle temperature
            correlation = correlations.get('spindle_temp', 0.85)
            values = baseline + (failure - baseline) * degradation * correlation
            noise = np.random.normal(0, (failure - baseline) * 0.03, cycle_samples)
            
        else:
            # Generic sensor: Linear increase
            values = baseline + (failure - baseline) * degradation * 0.8
            noise = np.random.normal(0, (failure - baseline) * 0.03, cycle_samples)
        
        # Add noise and clip to reasonable range
        values = values + noise
        values = np.clip(values, normal_range[0] * 0.8, normal_range[1] * 1.2)
        
        sensor_data[sensor_name] = values
    
    # ============================================
    # STEP 4: Create DataFrame
    # ============================================
    payload = {**sensor_data}
    if include_rul:
        payload = {'rul': rul, **payload}
    cycle_df = pd.DataFrame(payload)
    
    return cycle_df


def generate_all_machine_seed_data():
    """Generate temporal seed data for all 26 machines"""
    
    print(f"\n{'=' * 70}")
    print(f"BATCH TEMPORAL SEED DATA GENERATION")
    print(f"{'=' * 70}\n")
    
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "seed_data" / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    # Process each category
    for category, profile in RUL_PROFILES.items():
        machines = profile['machines']
        
        print(f"\n{'=' * 70}")
        print(f"CATEGORY: {category.upper()}")
        print(f"Machines: {len(machines)}")
        print(f"{'=' * 70}")
        
        for machine_id in machines:
            print(f"\n[{len(results) + 1}/26] Processing: {machine_id}")
            machine_start = time.time()
            
            try:
                # Create temporal seed data
                seed_data = create_temporal_seed_data(
                    machine_id=machine_id,
                    rul_profile=profile,
                    n_samples=10000  # 10000 samples for training TVAE
                )
                
                # Save
                output_path = output_dir / f"{machine_id}_temporal_seed.parquet"
                seed_data.to_parquet(output_path, index=False)
                
                machine_time = time.time() - machine_start
                
                print(f"[OK] Saved: {output_path.name} ({machine_time:.1f}s)")
                
                # Verify RUL decreasing
                rul_diff = seed_data['rul'].diff().dropna()
                decreasing_pct = (rul_diff <= 10).sum() / len(rul_diff) * 100
                
                results.append({
                    'machine_id': machine_id,
                    'status': 'SUCCESS',
                    'samples': len(seed_data),
                    'features': len(seed_data.columns),
                    'rul_decreasing_pct': round(decreasing_pct, 1),
                    'time_seconds': round(machine_time, 1),
                    'path': str(output_path.relative_to(base_path))
                })
                
            except Exception as e:
                print(f"[FAIL] FAILED: {machine_id}")
                print(f"   Error: {str(e)}")
                results.append({
                    'machine_id': machine_id,
                    'status': 'FAILED',
                    'error': str(e)
                })
    
    # Summary
    total_time = time.time() - total_start
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print(f"\n{'=' * 70}")
    print(f"BATCH COMPLETE")
    print(f"{'=' * 70}")
    print(f"[OK] Success: {success_count}/26")
    print(f"[FAIL] Failed: {26 - success_count}/26")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"[INFO] Average per machine: {total_time/26:.1f} seconds")
    
    # Quality check
    if success_count > 0:
        avg_decreasing = np.mean([r['rul_decreasing_pct'] for r in results if r['status'] == 'SUCCESS'])
        print(f"📈 Average RUL decreasing: {avg_decreasing:.1f}%")
    
    # Save report
    report_path = base_path / "reports" / "temporal_seed_generation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            'total_machines': 26,
            'success_count': success_count,
            'failed_count': 26 - success_count,
            'total_time_minutes': round(total_time/60, 2),
            'avg_time_seconds': round(total_time/26, 1),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }, f, indent=2)
    
    print(f"\nReport saved: {report_path.relative_to(base_path)}")
    print(f"{'=' * 70}\n")
    
    return success_count == 26


if __name__ == "__main__":
    import sys
    import importlib
    
    if len(sys.argv) > 1:
        # Single machine mode
        machine_id = sys.argv[1]
        
        # Reload rul_profiles to pick up any recent changes
        from GAN.config import rul_profiles as rul_profiles_module
        importlib.reload(rul_profiles_module)
        from GAN.config.rul_profiles import RUL_PROFILES as FRESH_RUL_PROFILES
        from GAN.config.rul_profiles import get_machine_category as fresh_get_machine_category
        from GAN.config.rul_profiles import get_rul_profile as fresh_get_rul_profile
        
        # Use fresh imports for single machine mode
        category = fresh_get_machine_category(machine_id)
        if category is None:
            print(f"ERROR: Machine '{machine_id}' not found in RUL profiles")
            print(f"\nAvailable machines:")
            for cat, prof in FRESH_RUL_PROFILES.items():
                print(f"  {cat}: {', '.join(prof['machines'])}")
            sys.exit(1)
        
        rul_profile = fresh_get_rul_profile(machine_id)
        
        print(f"Machine: {machine_id}")
        print(f"Category: {category}")
        print(f"Max RUL: {rul_profile['max_rul']} hours")
        
        seed_data = create_temporal_seed_data(machine_id, rul_profile)
        
        # Save
        base_path = Path(__file__).parent.parent
        output_path = base_path / "seed_data" / "temporal" / f"{machine_id}_temporal_seed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed_data.to_parquet(output_path, index=False)
        
        print(f"[OK] Saved: {output_path.relative_to(base_path)}")
        
        # Quick validation
        rul_diff = seed_data['rul'].diff().dropna()
        decreasing_pct = (rul_diff <= 10).sum() / len(rul_diff) * 100
        print(f"[OK] RUL decreasing: {decreasing_pct:.1f}%")
        
        sys.exit(0)
    else:
        # Batch mode - all machines
        success = generate_all_machine_seed_data()
        sys.exit(0 if success else 1)
