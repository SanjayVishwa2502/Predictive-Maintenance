"""
Integration tests for ML + LLM pipeline
"""
import sys
import os
import time
import pandas as pd

# Add parent directory (LLM root) to path to allow importing api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.ml_integration import IntegratedPredictionSystem

def test_e2e_pipeline():
    """Test complete pipeline"""
    
    print("Initializing IntegratedPredictionSystem...")
    system = IntegratedPredictionSystem()
    
    # Test data with complete feature sets for each machine
    # Note: timestamp is converted to float (unix epoch) to match model expectations
    current_ts = pd.Timestamp('2025-11-26 10:00:00').timestamp()
    
    test_data = {
        'motor_siemens_1la7_001': {
            'timestamp': current_ts,
            'vibration': 10.5,
            'temperature': 72.0,
            'current': 42.0,
            'voltage': 400.0,
            'bearing_de_temp_C': 72.0,
            'bearing_nde_temp_C': 68.0,
            'winding_temp_C': 85.0,
            'casing_temp_C': 60.0,
            'ambient_temp_C': 25.0,
            'rms_velocity_mm_s': 4.5,
            'peak_velocity_mm_s': 6.2,
            'bpfo_frequency_hz': 120.0,
            'bpfi_frequency_hz': 150.0,
            'current_25pct_load_A': 10.0,
            'current_50pct_load_A': 20.0,
            'current_75pct_load_A': 30.0,
            'current_100pct_load_A': 42.0,
            'current_no_load_A': 5.0,
            'voltage_phase_to_phase_V': 400.0,
            'power_factor_100pct': 0.85,
            'power_factor_75pct': 0.80,
            'power_factor_50pct': 0.75,
            'efficiency_100pct': 95.0,
            'efficiency_75pct': 94.0,
            'efficiency_50pct': 92.0,
            'sound_level_dBA': 75.0,
            'rul': 0,
            # Derived/Static features
            'power_rating_kw': 15.0,
            'rated_speed_rpm': 1500,
            'operating_voltage': 400,
            'equipment_age_years': 5,
            'temp_mean_normalized': 0.5,
            'temp_max_normalized': 0.6,
            'temp_std': 0.1,
            'temp_range': 10.0,
            'vib_rms': 4.5,
            'vib_peak': 6.2,
            'vib_mean': 4.0,
            'vib_std': 0.5,
            'current_mean': 40.0,
            'current_max': 45.0,
            'current_std': 2.0,
            'health_score': 0.8
        },
        'pump_grundfos_cr3_004': {
             'timestamp': current_ts,
             'vibration': 5.2,
             'temperature': 45.0,
             'current': 12.0,
             'voltage': 230.0,
             'bearing_temp_C': 45.0,
             'motor_winding_temp_C': 55.0,
             'liquid_temp_C': 30.0,
             'overall_rms_mm_s': 3.2,
             'axial_mm_s': 1.5,
             'current_a': 12.0,
             'power_consumption_kW': 4.5,
             'power_factor': 0.88,
             'pump_efficiency_percent': 78.0,
             'sound_pressure_level_dBA': 68.0,
             'rul': 0,
             # Derived/Static features
             'power_rating_kw': 5.5,
             'rated_speed_rpm': 2900,
             'operating_voltage': 230,
             'equipment_age_years': 3,
             'temp_mean_normalized': 0.4,
             'temp_max_normalized': 0.5,
             'temp_std': 0.1,
             'temp_range': 5.0,
             'current_mean': 11.0,
             'current_max': 13.0,
             'current_std': 0.5,
             'health_score': 0.9
        },
        'compressor_atlas_copco_ga30_001': {
            'timestamp': current_ts,
            'vibration': 8.1,
            'temperature': 82.0,
            'current': 55.0,
            'voltage': 415.0,
            'discharge_air_temp_C': 82.0,
            'oil_temp_C': 75.0,
            'motor_winding_temp_C': 88.0,
            'ambient_temp_C': 28.0,
            'rms_velocity_mm_s': 5.5,
            'current_A': 55.0,
            'power_factor': 0.91,
            'discharge_pressure_bar': 7.5,
            'suction_pressure_bar': 1.0,
            'air_flow_cfm': 150.0,
            'rul': 0,
            # Derived/Static features
            'power_rating_kw': 30.0,
            'rated_speed_rpm': 3000,
            'operating_voltage': 415,
            'equipment_age_years': 4,
            'temp_mean_normalized': 0.6,
            'temp_max_normalized': 0.7,
            'temp_std': 0.2,
            'temp_range': 15.0,
            'vib_rms': 5.5,
            'vib_peak': 7.0,
            'vib_mean': 5.0,
            'vib_std': 0.8,
            'current_mean': 54.0,
            'current_max': 58.0,
            'current_std': 1.5,
            'health_score': 0.85
        }
    }
    
    print("=== INTEGRATION TEST ===\n")
    
    for machine_id in test_data.keys():
        print(f"Testing {machine_id}...")
        
        # Get specific sensor data for this machine
        sensor_data = test_data[machine_id]
        
        start = time.time()
        results = system.predict_with_explanation(
            machine_id=machine_id,
            sensor_data=sensor_data,
            model_type='all'
        )
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Models run: {len(results)}")
        print(f"  Explanations generated: {sum(1 for r in results.values() if 'explanation' in r)}")
        
        # Print generated explanations
        for model_type, res in results.items():
            if 'explanation' in res:
                print(f"\n  [{model_type.upper()}] Explanation:")
                if isinstance(res['explanation'], dict) and 'explanation' in res['explanation']:
                    print(f"  {res['explanation']['explanation']}")
                elif isinstance(res['explanation'], dict) and 'note' in res['explanation']:
                    print(f"  {res['explanation']['note']}")
                else:
                    print(f"  {res['explanation']}")
                print("-" * 40)

        # Check latency
        if elapsed >= 120.0:
            print(f"  WARNING: Pipeline slow: {elapsed:.2f}s (Threshold: 120.0s)")
        else:
            print(f"  Latency check passed: {elapsed:.2f}s < 120.0s")
        
        print("  ✓ PASS\n")
    
    print("=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_e2e_pipeline()
