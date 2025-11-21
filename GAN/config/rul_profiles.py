"""
Phase 1.6 Day 1-2: RUL Profiles Configuration
==============================================

Machine-specific RUL (Remaining Useful Life) characteristics grouped by category.
Each category defines:
- max_rul: Maximum operating hours before failure
- cycles_per_dataset: Number of run-to-failure cycles in 50K samples
- degradation_pattern: How the machine degrades over time
- noise_std: Standard deviation of RUL noise
- sensor_correlation: How strongly sensors correlate with RUL degradation

These profiles are used to:
1. Create temporal seed data (Phase 1.6.1)
2. Train TVAE models to learn degradation patterns (Phase 1.6.2)
3. Generate realistic synthetic data (Phase 1.6.3)

Author: GAN Team
Date: 2025-11-20
"""

RUL_PROFILES = {
    # =========================================================================
    # CATEGORY 1: MOTORS (3 machines)
    # Long operational life with slow, steady degradation
    # Primary failure: Bearing wear, winding insulation breakdown
    # =========================================================================
    'motor': {
        'machines': [
            'motor_siemens_1la7_001',
            'motor_abb_m3bp_002',
            'motor_weg_w22_003'
        ],
        'max_rul': 1000,                    # Hours to failure
        'cycles_per_dataset': 3,             # 3 complete life cycles in 50K samples
        'degradation_pattern': 'linear_slow',
        'noise_std': 10,                     # ±10 hour RUL variation
        'sensor_correlation': {
            'temperature': 0.85,             # Strong: Bearing temp rises with wear
            'vibration': 0.75,               # Medium-High: Imbalance increases
            'current': 0.65                  # Medium: Friction increases load
        }
    },
    
    # =========================================================================
    # CATEGORY 2: PUMPS (3 machines)
    # Medium life with steady degradation
    # Primary failure: Seal wear, impeller erosion, cavitation
    # =========================================================================
    'pump': {
        'machines': [
            'pump_grundfos_cr3_004',
            'pump_flowserve_ansi_005',
            'pump_ksb_etanorm_006'
        ],
        'max_rul': 800,
        'cycles_per_dataset': 4,
        'degradation_pattern': 'linear_medium',
        'noise_std': 12,
        'sensor_correlation': {
            'temperature': 0.80,             # High: Friction heat increases
            'vibration': 0.85,               # Very High: Cavitation/imbalance
            'pressure': 0.70                 # High: Output pressure drops
        }
    },
    
    # =========================================================================
    # CATEGORY 3: COMPRESSORS (2 machines)
    # Shorter life with faster degradation due to high stress
    # Primary failure: Valve wear, piston ring wear, bearing failure
    # =========================================================================
    'compressor': {
        'machines': [
            'compressor_ingersoll_rand_2545_009',
            'compressor_atlas_copco_ga30_001'
        ],
        'max_rul': 600,
        'cycles_per_dataset': 5,
        'degradation_pattern': 'linear_fast',
        'noise_std': 8,
        'sensor_correlation': {
            'temperature': 0.90,             # Very High: Compression heat increases rapidly
            'vibration': 0.80,               # High: Mechanical wear increases
            'pressure': 0.75                 # High: Efficiency drops
        }
    },
    
    # =========================================================================
    # CATEGORY 4: CNC MACHINES (7 machines)
    # Shorter cycles due to tool wear (not machine failure)
    # Primary failure: Cutting tool wear, spindle bearing degradation
    # =========================================================================
    'cnc': {
        'machines': [
            'cnc_dmg_mori_nlx_010',
            'cnc_dmg_mori_ntx_001',
            'cnc_haas_vf2_001',
            'cnc_haas_vf3_001',
            'cnc_makino_a51nx_001',
            'cnc_mazak_variaxis_001',
            'cnc_okuma_lb3000_001',
            'cnc_fanuc_robodrill_001'
        ],
        'max_rul': 500,                      # Cutting tool life (not machine life)
        'cycles_per_dataset': 7,             # Multiple tool changes
        'degradation_pattern': 'exponential', # Tool wear accelerates
        'noise_std': 15,
        'sensor_correlation': {
            'spindle_bearing_temp_C': {'base': 42.5, 'range': 12, 'noise': 0.8},
            'motor_temp_C': {'base': 48, 'range': 18, 'noise': 1.0},
            'ambient_temp_K': {'base': 299.2, 'range': 2.5, 'noise': 0.3},
            'process_temp_K': {'base': 309.1, 'range': 2.5, 'noise': 0.4},
            'temp_difference_K': {'base': 9.9, 'range': 4.0, 'noise': 0.5},
            'spindle_speed_rpm': {'base': 10000, 'range': 8000, 'noise': 200},
            'spindle_vibration_mm_s': {'base': 0.5, 'range': 1.2, 'noise': 0.08},
            'spindle_torque_nm': {'base': 8, 'range': 6, 'noise': 0.5},
            'power_consumption_kW': {'base': 5, 'range': 4, 'noise': 0.3}
        }
    },
    
    # =========================================================================
    # CATEGORY 5: FANS (2 machines)
    # Very long life with very slow degradation
    # Primary failure: Bearing wear, blade imbalance
    # =========================================================================
    'fan': {
        'machines': [
            'fan_ebm_papst_a3g710_007',
            'fan_howden_buffalo_008'
        ],
        'max_rul': 1200,                     # Very long operational life
        'cycles_per_dataset': 3,
        'degradation_pattern': 'linear_slow',
        'noise_std': 20,
        'sensor_correlation': {
            'temperature': 0.75,             # Medium-High: Bearing temp rises
            'vibration': 0.85,               # Very High: Imbalance primary indicator
            'current': 0.60                  # Medium: Slight increase with friction
        }
    },
    
    # =========================================================================
    # CATEGORY 6: CONVEYORS (2 machines)
    # Medium life with steady wear
    # Primary failure: Belt wear, roller bearing failure, motor overload
    # =========================================================================
    'conveyor': {
        'machines': [
            'conveyor_dorner_2200_013',
            'conveyor_hytrol_e24ez_014'
        ],
        'max_rul': 900,
        'cycles_per_dataset': 4,
        'degradation_pattern': 'linear_medium',
        'noise_std': 15,
        'sensor_correlation': {
            'temperature': 0.78,             # High: Motor and roller heating
            'vibration': 0.82,               # High: Belt misalignment, roller wear
            'current': 0.70                  # High: Increased friction
        }
    },
    
    # =========================================================================
    # CATEGORY 7: ROBOTS (2 machines)
    # Long life with steady degradation
    # Primary failure: Joint bearing wear, gear reducer degradation
    # =========================================================================
    'robot': {
        'machines': [
            'robot_fanuc_m20ia_015',
            'robot_abb_irb6700_016'
        ],
        'max_rul': 1100,
        'cycles_per_dataset': 3,
        'degradation_pattern': 'linear_slow',
        'noise_std': 18,
        'sensor_correlation': {
            'temperature': 0.80,             # High: Joint motor heating
            'vibration': 0.78,               # High: Backlash increases
            'current': 0.72                  # High: Resistance increases
        }
    },
    
    # =========================================================================
    # CATEGORY 8: HYDRAULIC SYSTEMS (2 machines)
    # Medium life with moderate degradation
    # Primary failure: Seal wear, fluid contamination, pump wear
    # =========================================================================
    'hydraulic': {
        'machines': [
            'hydraulic_beckwood_press_011',
            'hydraulic_parker_hpu_012'
        ],
        'max_rul': 850,
        'cycles_per_dataset': 4,
        'degradation_pattern': 'linear_medium',
        'noise_std': 14,
        'sensor_correlation': {
            'temperature': 0.83,             # High: Fluid heating with wear
            'pressure': 0.88,                # Very High: Primary failure indicator
            'vibration': 0.70                # Medium-High: Pump cavitation
        }
    },
    
    # =========================================================================
    # CATEGORY 9: TRANSFORMER (1 machine)
    # Very long life with slow degradation
    # Primary failure: Insulation breakdown, winding degradation
    # =========================================================================
    'transformer': {
        'machines': [
            'transformer_square_d_017'
        ],
        'max_rul': 1500,                     # Extremely long operational life
        'cycles_per_dataset': 2,
        'degradation_pattern': 'linear_slow',
        'noise_std': 25,
        'sensor_correlation': {
            'temperature': 0.88,             # Very High: Insulation breakdown indicator
            'current': 0.65,                 # Medium-High: Core losses increase
            'voltage': 0.60                  # Medium: Regulation degrades
        }
    },
    
    # =========================================================================
    # CATEGORY 10: COOLING TOWER (1 machine)
    # Long life with gradual degradation
    # Primary failure: Fill media clogging, fan bearing wear, pump failure
    # =========================================================================
    'cooling_tower': {
        'machines': [
            'cooling_tower_bac_vti_018'
        ],
        'max_rul': 1100,
        'cycles_per_dataset': 3,
        'degradation_pattern': 'linear_medium',
        'noise_std': 20,
        'sensor_correlation': {
            'temperature': 0.92,             # Very High: Cooling efficiency drops
            'vibration': 0.78,               # High: Fan bearing wear
            'flow': 0.85                     # Very High: Clogging reduces flow
        }
    },
    
    # =========================================================================
    # CATEGORY 11: TURBOFAN (1 machine)
    # Shorter life due to extreme operating conditions
    # Primary failure: Blade erosion, bearing wear, thermal fatigue
    # =========================================================================
    'turbofan': {
        'machines': [
            'turbofan_cfm56_7b_001'
        ],
        'max_rul': 700,
        'cycles_per_dataset': 5,
        'degradation_pattern': 'exponential', # Degradation accelerates
        'noise_std': 12,
        'sensor_correlation': {
            'temperature': 0.93,             # Extremely High: Thermal efficiency drops
            'vibration': 0.88,               # Very High: Blade imbalance
            'pressure': 0.85                 # Very High: Compression ratio drops
        }
    }
}

# =========================================================================
# VALIDATION: Ensure all 26 machines are covered
# =========================================================================

def get_all_machines():
    """Return list of all 26 machines across all categories"""
    all_machines = []
    for category, profile in RUL_PROFILES.items():
        all_machines.extend(profile['machines'])
    return all_machines

def get_machine_category(machine_id):
    """Get category for a specific machine"""
    for category, profile in RUL_PROFILES.items():
        if machine_id in profile['machines']:
            return category
    return None

def get_rul_profile(machine_id):
    """Get RUL profile for a specific machine"""
    category = get_machine_category(machine_id)
    if category:
        return RUL_PROFILES[category]
    return None

def validate_all_machines_covered():
    """Validate that all 26 machines have RUL profiles"""
    all_machines = get_all_machines()
    print(f"Total machines in RUL profiles: {len(all_machines)}")
    
    if len(all_machines) == 26:
        print("✅ All 26 machines covered!")
        return True
    else:
        print(f"❌ Expected 26 machines, found {len(all_machines)}")
        return False

# =========================================================================
# SUMMARY STATISTICS
# =========================================================================

def print_profile_summary():
    """Print summary of all RUL profiles"""
    print("\n" + "=" * 70)
    print("RUL PROFILES SUMMARY")
    print("=" * 70)
    
    total_machines = 0
    
    for category, profile in RUL_PROFILES.items():
        num_machines = len(profile['machines'])
        total_machines += num_machines
        
        print(f"\n{category.upper()}")
        print(f"  Machines: {num_machines}")
        print(f"  Max RUL: {profile['max_rul']} hours")
        print(f"  Cycles: {profile['cycles_per_dataset']}")
        print(f"  Pattern: {profile['degradation_pattern']}")
        print(f"  Samples per cycle: {50000 // profile['cycles_per_dataset']:,}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL MACHINES: {total_machines}/26")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Validate and print summary
    validate_all_machines_covered()
    print_profile_summary()
    
    # Test individual machine lookup
    test_machine = 'motor_siemens_1la7_001'
    profile = get_rul_profile(test_machine)
    if profile:
        print(f"\nTest lookup: {test_machine}")
        print(f"  Category: {get_machine_category(test_machine)}")
        print(f"  Max RUL: {profile['max_rul']} hours")
        print(f"  Cycles: {profile['cycles_per_dataset']}")
