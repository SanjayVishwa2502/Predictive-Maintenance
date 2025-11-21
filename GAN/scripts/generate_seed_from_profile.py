"""
Generate seed data from machine profile JSON specifications
Creates realistic sensor readings from baseline_normal_operation specifications
"""
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

def load_profile(profile_path):
    """Load machine profile JSON"""
    with open(profile_path, 'r') as f:
        return json.load(f)

def generate_sensor_data(profile, num_rows=500):
    """Generate realistic sensor readings from profile specifications"""
    machine_id = profile['machine_id']
    baseline = profile['baseline_normal_operation']
    
    print(f"\n{'='*60}")
    print(f"Generating seed data for: {machine_id}")
    print(f"{'='*60}")
    
    data = {'timestamp': pd.date_range(start='2024-01-01', periods=num_rows, freq='1H')}
    
    # Thermal sensors
    if 'thermal' in baseline:
        thermal = baseline['thermal']
        
        if 'spindle_bearing_temp_C' in thermal:
            spec = thermal['spindle_bearing_temp_C']
            typical = spec['typical']
            max_val = spec['max']
            # Generate normal distribution around typical, clipped to realistic range
            data['spindle_bearing_temp'] = np.clip(
                np.random.normal(typical, (max_val - typical) * 0.25, num_rows),
                typical * 0.95, max_val * 0.95
            )
            print(f"  ✓ spindle_bearing_temp: {data['spindle_bearing_temp'].mean():.1f}°C (typical: {typical}°C)")
        
        if 'motor_temp_C' in thermal:
            spec = thermal['motor_temp_C']
            typical = spec['typical']
            max_val = spec['max']
            data['motor_temp'] = np.clip(
                np.random.normal(typical, (max_val - typical) * 0.25, num_rows),
                typical * 0.95, max_val * 0.95
            )
            print(f"  ✓ motor_temp: {data['motor_temp'].mean():.1f}°C (typical: {typical}°C)")
        
        if 'ambient_temp_K' in thermal:
            spec = thermal['ambient_temp_K']
            typical = spec['typical']
            temp_range = spec.get('range', [typical - 2, typical + 2])
            data['ambient_temp'] = np.random.uniform(temp_range[0], temp_range[1], num_rows)
            print(f"  ✓ ambient_temp: {data['ambient_temp'].mean():.1f}K (typical: {typical}K)")
        
        if 'process_temp_K' in thermal:
            spec = thermal['process_temp_K']
            typical = spec['typical']
            temp_range = spec.get('range', [typical - 2, typical + 2])
            data['process_temp'] = np.random.uniform(temp_range[0], temp_range[1], num_rows)
            print(f"  ✓ process_temp: {data['process_temp'].mean():.1f}K (typical: {typical}K)")
        
        if 'temp_difference_K' in thermal:
            spec = thermal['temp_difference_K']
            typical = spec['typical']
            temp_range = spec.get('range', [typical - 2, typical + 2])
            data['temp_difference'] = np.random.uniform(temp_range[0], temp_range[1], num_rows)
            print(f"  ✓ temp_difference: {data['temp_difference'].mean():.1f}K (typical: {typical}K)")
    
    # Mechanical sensors
    if 'mechanical' in baseline:
        mech = baseline['mechanical']
        
        if 'spindle_speed_rpm' in mech:
            spec = mech['spindle_speed_rpm']
            typical = spec['typical']
            max_val = spec['max']
            # Generate mixture: some idle time (0-20%), rest at working speeds
            idle_mask = np.random.random(num_rows) < 0.2
            data['spindle_speed'] = np.where(
                idle_mask,
                np.random.uniform(0, typical * 0.3, num_rows),
                np.random.uniform(typical * 0.7, typical * 1.3, num_rows)
            )
            data['spindle_speed'] = np.clip(data['spindle_speed'], 0, max_val)
            print(f"  ✓ spindle_speed: {data['spindle_speed'].mean():.0f} rpm (typical: {typical} rpm)")
        
        if 'cutting_force_N' in mech:
            spec = mech['cutting_force_N']
            typical = spec['typical']
            max_val = spec['max']
            # Correlated with spindle speed (higher speed = higher force)
            if 'spindle_speed' in data:
                normalized_speed = data['spindle_speed'] / data['spindle_speed'].max()
                data['cutting_force'] = typical * (0.5 + 0.5 * normalized_speed) + np.random.normal(0, typical * 0.15, num_rows)
                data['cutting_force'] = np.clip(data['cutting_force'], 0, max_val)
            else:
                data['cutting_force'] = np.random.uniform(typical * 0.6, typical * 1.4, num_rows)
            print(f"  ✓ cutting_force: {data['cutting_force'].mean():.0f}N (typical: {typical}N)")
        
        if 'spindle_torque_nm' in mech:
            spec = mech['spindle_torque_nm']
            typical = spec['typical']
            max_val = spec['max']
            data['spindle_torque'] = np.random.uniform(typical * 0.6, typical * 1.3, num_rows)
            data['spindle_torque'] = np.clip(data['spindle_torque'], 0, max_val * 0.9)
            print(f"  ✓ spindle_torque: {data['spindle_torque'].mean():.1f}Nm (typical: {typical}Nm)")
        
        if 'feed_rate_mm_min' in mech:
            spec = mech['feed_rate_mm_min']
            typical = spec['typical']
            max_val = spec['max']
            data['feed_rate'] = np.random.uniform(typical * 0.5, typical * 1.2, num_rows)
            data['feed_rate'] = np.clip(data['feed_rate'], 0, max_val * 0.8)
            print(f"  ✓ feed_rate: {data['feed_rate'].mean():.0f}mm/min (typical: {typical}mm/min)")
        
        if 'feed_rate_mm_rev' in mech:
            spec = mech['feed_rate_mm_rev']
            typical = spec['typical']
            max_val = spec['max']
            data['feed_rate'] = np.random.uniform(typical * 0.6, typical * 1.3, num_rows)
            data['feed_rate'] = np.clip(data['feed_rate'], 0, max_val * 0.8)
            print(f"  ✓ feed_rate: {data['feed_rate'].mean():.2f}mm/rev (typical: {typical}mm/rev)")
    
    # Vibration sensors
    if 'vibration' in baseline:
        vib = baseline['vibration']
        
        if 'spindle_vibration_mm_s' in vib:
            spec = vib['spindle_vibration_mm_s']
            typical = spec['typical']
            max_val = spec['max']
            data['vibration'] = np.clip(
                np.random.lognormal(np.log(typical), 0.3, num_rows),
                typical * 0.5, max_val * 0.85
            )
            print(f"  ✓ vibration: {data['vibration'].mean():.2f}mm/s (typical: {typical}mm/s)")
    
    # Power consumption
    if 'performance' in baseline and 'power_consumption_kW' in baseline['performance']:
        spec = baseline['performance']['power_consumption_kW']
        typical = spec['typical']
        max_val = spec['max']
        # Correlated with spindle speed
        if 'spindle_speed' in data:
            normalized_speed = data['spindle_speed'] / data['spindle_speed'].max()
            data['power_consumption'] = typical * (0.4 + 0.6 * normalized_speed) + np.random.normal(0, typical * 0.1, num_rows)
            data['power_consumption'] = np.clip(data['power_consumption'], typical * 0.3, max_val * 0.9)
        else:
            data['power_consumption'] = np.random.uniform(typical * 0.7, typical * 1.3, num_rows)
        print(f"  ✓ power_consumption: {data['power_consumption'].mean():.1f}kW (typical: {typical}kW)")
    
    df = pd.DataFrame(data)
    print(f"\nGenerated {len(df)} rows × {len(df.columns)} columns")
    return df

def main():
    """Process all 4 new machine profiles"""
    profiles_dir = Path('c:/GAN/data/real_machines/profiles')
    seed_dir = Path('c:/GAN/seed_data')
    seed_dir.mkdir(exist_ok=True)
    
    machine_files = [
        'cnc_makino_a51nx_001.json',
        'cnc_mazak_variaxis_001.json',
        'cnc_okuma_lb3000_001.json',
        'cnc_dmg_mori_ntx_001.json'
    ]
    
    print(f"\n{'#'*60}")
    print(f"# SEED DATA GENERATION FROM MACHINE PROFILES")
    print(f"{'#'*60}")
    
    for filename in machine_files:
        profile_path = profiles_dir / filename
        
        if not profile_path.exists():
            print(f"\n⚠ WARNING: {filename} not found at {profile_path}")
            continue
        
        # Load profile
        profile = load_profile(profile_path)
        machine_id = profile['machine_id']
        
        # Generate seed data
        seed_df = generate_sensor_data(profile, num_rows=500)
        
        # Save to parquet
        output_path = seed_dir / f'{machine_id}_seed.parquet'
        seed_df.to_parquet(output_path, index=False)
        
        print(f"  💾 Saved: {output_path}")
        print(f"  📊 Shape: {seed_df.shape}")
        print(f"  📋 Columns: {list(seed_df.columns)}")
    
    print(f"\n{'='*60}")
    print(f"✅ SEED DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext step: Run add_new_machine.py for each machine")

if __name__ == '__main__':
    main()
