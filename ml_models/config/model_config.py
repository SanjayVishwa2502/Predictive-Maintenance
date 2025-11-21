"""
Model Configuration for Phase 2 ML Training
Phase 2.1.4: Training Strategy & Configuration

This configuration defines training parameters, model types, and machine lists
for the predictive maintenance ML pipeline.
"""

# ==============================================================================
# MODEL TYPES TO TRAIN
# ==============================================================================

MODEL_TYPES = [
    'classification',  # Binary: normal vs failure prediction
    'regression',      # RUL (Remaining Useful Life) prediction
    'anomaly',         # Anomaly detection (unsupervised)
    'timeseries'       # Time-series forecasting (future sensor values)
]

# ==============================================================================
# AUTOGLUON TRAINING CONFIGURATIONS
# ==============================================================================

AUTOGLUON_CONFIG = {
    'classification': {
        'eval_metric': 'f1',                    # F1 score (balanced precision/recall)
        'problem_type': 'binary',               # Binary classification (normal/failure)
        'time_limit': 3600,                     # 1 hour per training run
        'presets': 'best_quality',              # Best quality (vs medium/high speed)
        'num_bag_folds': 5,                     # 5-fold bagging for robustness
        'num_stack_levels': 1,                  # 1-level stacking (base + ensemble)
        'verbosity': 2,                         # Standard logging
        'excluded_model_types': [],             # No exclusions (try all models)
        'hyperparameters': {                    # GPU only for neural networks
            'NN_TORCH': {'num_gpus': 1},       # Neural networks use GPU
            'FASTAI': {'num_gpus': 1},         # FastAI uses GPU
            'GBM': {'num_gpus': 0},            # LightGBM uses CPU only
            'CAT': {'num_gpus': 0},            # CatBoost uses CPU only
            'XGB': {'num_gpus': 0},            # XGBoost uses CPU only
            'RF': {},                           # RandomForest CPU only
            'XT': {}                            # ExtraTrees CPU only
        },
        'num_cpus': 6,                          # Low CPU usage (6 cores to prevent overheating)
        'num_gpus': 0,                          # Don't pass to fit() - set per model above
        'ag_args_fit': {'num_cpus': 6}         # Limit CPU cores globally
    },
    
    'regression': {
        'eval_metric': 'r2',                    # R² score (coefficient of determination)
        'problem_type': 'regression',           # Continuous value prediction
        'time_limit': 3600,                     # 1 hour per training run
        'presets': 'medium_quality_faster_train', # FASTER preset for Pi compatibility (not best_quality)
        'num_bag_folds': 3,                     # 3-fold bagging (reduced from 5 for speed/size)
        'num_stack_levels': 0,                  # NO STACKING (lighter models for Pi)
        'verbosity': 2,                         # Standard logging
        'excluded_model_types': ['NN_TORCH', 'FASTAI', 'XT', 'KNN'],  # Pi-incompatible models EXCLUDED
        'hyperparameters': {                    # Only Pi-compatible models
            'GBM': {'num_gpus': 0},            # LightGBM uses CPU only ✅ Pi-compatible
            'CAT': {'num_gpus': 0},            # CatBoost uses CPU only ✅ Pi-compatible  
            'XGB': {'num_gpus': 0},            # XGBoost uses CPU only ✅ Pi-compatible
            'RF': {},                           # RandomForest CPU only ✅ Pi-compatible
        },
        'num_cpus': 6,                          # Low CPU usage (6 cores)
        'num_gpus': 0,                          # Don't pass to fit() - set per model
        'ag_args_fit': {'num_cpus': 6}         # Limit CPU cores globally
    },
    
    'anomaly': {
        'algorithm': 'isolation_forest',        # Isolation Forest (best for high-dim data)
        'contamination': 0.1,                   # Expected anomaly rate (10%)
        'n_estimators': 100,                    # Number of trees
        'max_samples': 'auto',                  # Samples per tree (auto)
        'random_state': 42,                     # Reproducibility
        'n_jobs': -1,                           # Use all CPU cores
        'verbose': 1                            # Standard logging
    },
    
    'timeseries': {
        'prediction_length': 24,                # Forecast 24 hours ahead
        'time_limit': 3600,                     # 1 hour per training run
        'presets': 'best_quality',              # Best quality preset
        'freq': 'H',                            # Hourly frequency
        'target': 'sensor_value',               # Target column to forecast
        'eval_metric': 'MAPE',                  # Mean Absolute Percentage Error
        'hyperparameters': 'default',           # Default hyperparameters
        'num_cpus': 8,                          # Reduced CPU usage (8 cores)
        'num_gpus': 1                           # Enable GPU training (RTX 4070)
    }
}

# ==============================================================================
# QUICK TEST CONFIGURATION (Phase 2.1.3 - Already used)
# ==============================================================================

QUICK_TEST_CONFIG = {
    'classification': {
        'time_limit': 300,                      # 5 minutes (quick test)
        'presets': 'medium_quality',            # Medium quality (faster)
        'num_bag_folds': 0,                     # No bagging (faster)
        'num_stack_levels': 0                   # No stacking (faster)
    },
    'regression': {
        'time_limit': 300,                      # 5 minutes
        'presets': 'medium_quality',            # Medium quality
        'num_bag_folds': 0,                     # No bagging
        'num_stack_levels': 0                   # No stacking
    }
}

# ==============================================================================
# EDGE OPTIMIZATION CONFIGURATION
# ==============================================================================

EDGE_OPTIMIZATION_CONFIG = {
    'quantization': True,                       # Enable INT8 quantization
    'target_format': 'onnx',                    # ONNX format for edge deployment
    'max_model_size_mb': 10,                    # Target: <10 MB per model
    'optimization_level': 'O3',                 # Maximum optimization
    'opset_version': 13,                        # ONNX opset version
    'dynamic_axes': True,                       # Support variable batch sizes
    'test_accuracy_threshold': 0.95             # Retain >95% of original accuracy
}

# ==============================================================================
# MACHINE LIST (All 27 Machines - Updated November 21, 2025)
# ==============================================================================

MACHINES = [
    # Motors (3)
    'motor_siemens_1la7_001',
    'motor_abb_m3bp_002',
    'motor_weg_w22_003',
    
    # Pumps (3)
    'pump_grundfos_cr3_004',
    'pump_flowserve_ansi_005',
    'pump_ksb_etanorm_006',
    
    # Fans (2)
    'fan_ebm_papst_a3g710_007',
    'fan_howden_buffalo_008',
    
    # Compressors (2)
    'compressor_ingersoll_rand_2545_009',
    'compressor_atlas_copco_ga30_001',
    
    # CNC Machines (8) - EXPANDED with new temporal machines
    'cnc_dmg_mori_nlx_010',
    'cnc_dmg_mori_ntx_001',          # NEW
    'cnc_fanuc_robodrill_001',       # NEW (added Nov 21, 2025)
    'cnc_haas_vf2_001',
    'cnc_haas_vf3_001',              # NEW
    'cnc_makino_a51nx_001',          # NEW
    'cnc_mazak_variaxis_001',        # NEW
    'cnc_okuma_lb3000_001',          # NEW
    
    # Hydraulic Systems (2)
    'hydraulic_beckwood_press_011',
    'hydraulic_parker_hpu_012',
    
    # Conveyors (2)
    'conveyor_dorner_2200_013',
    'conveyor_hytrol_e24ez_014',
    
    # Robots (2)
    'robot_fanuc_m20ia_015',
    'robot_abb_irb6700_016',
    
    # Other Equipment (3)
    'transformer_square_d_017',
    'cooling_tower_bac_vti_018',
    'turbofan_cfm56_7b_001'
]

# Total: 27 machines (updated from 21)

# ==============================================================================
# MACHINE CATEGORIES (For grouping and analysis)
# ==============================================================================

MACHINE_CATEGORIES = {
    'motor': ['motor_siemens_1la7_001', 'motor_abb_m3bp_002', 'motor_weg_w22_003'],
    'pump': ['pump_grundfos_cr3_004', 'pump_flowserve_ansi_005', 'pump_ksb_etanorm_006'],
    'fan': ['fan_ebm_papst_a3g710_007', 'fan_howden_buffalo_008'],
    'compressor': ['compressor_ingersoll_rand_2545_009', 'compressor_atlas_copco_ga30_001'],
    'cnc': [
        'cnc_dmg_mori_nlx_010', 
        'cnc_dmg_mori_ntx_001',
        'cnc_fanuc_robodrill_001',      # NEW (added Nov 21, 2025)
        'cnc_haas_vf2_001',
        'cnc_haas_vf3_001',
        'cnc_makino_a51nx_001',
        'cnc_mazak_variaxis_001',
        'cnc_okuma_lb3000_001'
    ],
    'hydraulic': ['hydraulic_beckwood_press_011', 'hydraulic_parker_hpu_012'],
    'conveyor': ['conveyor_dorner_2200_013', 'conveyor_hytrol_e24ez_014'],
    'robot': ['robot_fanuc_m20ia_015', 'robot_abb_irb6700_016'],
    'other': ['transformer_square_d_017', 'cooling_tower_bac_vti_018', 'turbofan_cfm56_7b_001']
}

# ==============================================================================
# TRAINING PRIORITY ORDER
# ==============================================================================

PRIORITY_MACHINES = [
    # High priority (train first for quick validation)
    'motor_siemens_1la7_001',           # Motor with most features (29)
    'pump_grundfos_cr3_004',            # Tested in Phase 2.1.3
    'compressor_atlas_copco_ga30_001',  # Tested in Phase 2.1.3
    
    # Medium priority (diverse equipment types)
    'motor_abb_m3bp_002',               # Second motor variant
    'cnc_dmg_mori_nlx_010',             # CNC machine
    'robot_fanuc_m20ia_015',            # Industrial robot
    'fan_ebm_papst_a3g710_007',         # HVAC equipment
    
    # Remaining machines (standard priority)
    # ... (rest will be batch trained)
]

# ==============================================================================
# TRAINING RESOURCE ESTIMATES
# ==============================================================================

TRAINING_ESTIMATES = {
    'classification': {
        'time_per_machine': 3600,               # 1 hour per machine
        'total_time_sequential': 27 * 3600,     # 27 hours (all 27 machines sequential)
        'total_time_parallel': 3600,            # 1 hour (if parallelized across machines)
        'cpu_cores': 6,                         # 6 cores per training job (temperature controlled)
        'memory_gb': 8,                         # Memory per training job
        'disk_space_gb': 0.5,                   # ~500 MB per model
        'n_machines': 27,                       # Total machines
        'total_samples_per_machine': 50000,     # 50K temporal samples per machine
        'avg_features_per_machine': 7.3         # Average 7.3 sensors per machine
    },
    'regression': {
        'time_per_machine': 3600,               # 1 hour per machine  
        'total_time_sequential': 27 * 3600,     # 27 hours (updated from 21)
        'total_time_parallel': 3600,            # 1 hour (parallelized)
        'cpu_cores': 6,                         # 6 cores (reduced from 20)
        'memory_gb': 8,
        'disk_space_gb': 0.5,
        'n_machines': 27,
        'total_samples_per_machine': 50000,
        'avg_features_per_machine': 7.3
    },
    'anomaly': {
        'time_per_machine': 600,                # 10 minutes (faster, unsupervised)
        'total_time_sequential': 27 * 600,      # 4.5 hours (updated from 3.5)
        'total_time_parallel': 600,             # 10 minutes (parallelized)
        'cpu_cores': 6,                         # 6 cores
        'memory_gb': 4,
        'disk_space_gb': 0.1,
        'n_machines': 27,
        'total_samples_per_machine': 50000,
        'avg_features_per_machine': 7.3
    },
    'timeseries': {
        'time_per_machine': 3600,               # 1 hour per machine
        'total_time_sequential': 27 * 3600,     # 27 hours (updated from 21)
        'total_time_parallel': 3600,            # 1 hour (parallelized)
        'cpu_cores': 6,                         # 6 cores  
        'memory_gb': 10,
        'disk_space_gb': 1.0,
        'n_machines': 27,
        'total_samples_per_machine': 50000,
        'avg_features_per_machine': 7.3
    }
}

# TOTAL TRAINING TIME ESTIMATES:
# - Sequential (all 4 model types): 27 + 27 + 4.5 + 27 = 85.5 hours
# - Parallel (4 model types simultaneously): max(27, 27, 4.5, 27) = 27 hours
# - Per-machine (all 4 types): 1 + 1 + 0.17 + 1 = 3.17 hours per machine

# ==============================================================================
# MLFLOW EXPERIMENT TRACKING CONFIGURATION
# ==============================================================================

MLFLOW_CONFIG = {
    'tracking_uri': 'file:./mlruns',           # Local MLflow tracking
    'experiment_prefix': 'PredictiveMaintenance',  # Experiment name prefix
    'artifact_location': './mlartifacts',      # Artifact storage location
    'log_models': True,                        # Log models to MLflow
    'log_metrics': True,                       # Log performance metrics
    'log_params': True,                        # Log hyperparameters
    'log_artifacts': True,                     # Log reports and plots
    'autolog': True                            # Enable AutoGluon autologging
}

# ==============================================================================
# DATA PATHS
# ==============================================================================

DATA_PATHS = {
    'gan_synthetic': '../GAN/data/synthetic',      # Phase 1 GAN synthetic data
    'gan_metadata': '../GAN/metadata',             # Phase 1 machine metadata
    'pooled_train': 'data/processed/pooled_train.parquet',
    'pooled_val': 'data/processed/pooled_val.parquet',
    'pooled_test': 'data/processed/pooled_test.parquet',
    'models_base': 'models',                       # Model storage
    'reports_base': 'reports',                     # Reports storage
    'logs_base': 'logs'                            # Training logs
}

# ==============================================================================
# PERFORMANCE TARGETS (Based on Phase 2.1.3 Results)
# ==============================================================================

PERFORMANCE_TARGETS = {
    'classification': {
        'f1_score': 0.85,                       # Minimum F1 score target
        'accuracy': 0.85,                       # Minimum accuracy target
        'precision': 0.80,                      # Minimum precision target
        'recall': 0.80,                         # Minimum recall target
        'roc_auc': 0.90                         # Minimum ROC AUC target
    },
    'regression': {
        'r2_score': 0.75,                       # Minimum R² target
        'rmse': 50,                             # Maximum RMSE (hours)
        'mae': 30,                              # Maximum MAE (hours)
        'mape': 10                              # Maximum MAPE (%)
    },
    'anomaly': {
        'f1_score': 0.75,                       # Minimum F1 score
        'precision': 0.70,                      # Minimum precision
        'recall': 0.70                          # Minimum recall
    },
    'timeseries': {
        'mape': 15,                             # Maximum MAPE (%)
        'rmse': 5,                              # Maximum RMSE (sensor units)
        'mae': 3                                # Maximum MAE (sensor units)
    }
}

# Note: These targets are based on SYNTHETIC DATA. Real-world performance 
# expected to be 10-20% lower when deployed on actual sensor data.

# ==============================================================================
# SYNTHETIC DATA LIMITATION DOCUMENTATION
# ==============================================================================

KNOWN_LIMITATIONS = {
    'synthetic_data': {
        'issue': 'Training on GAN-generated synthetic data (not real sensors)',
        'impact': 'Performance metrics optimistically high (99.7% accuracy)',
        'expected_real_performance': {
            'classification_f1': '0.75-0.85 (vs 0.98 synthetic)',
            'regression_r2': '0.70-0.85 (vs 0.9998 synthetic)',
            'accuracy_drop': '10-20% when deployed on real data'
        },
        'mitigation': [
            'Document limitation in all reports',
            'Plan for real data integration (hybrid training)',
            'Implement continuous learning pipeline',
            'Deploy A/B testing framework',
            'Monitor data drift in production'
        ]
    },
    'label_quality': {
        'issue': 'Labels created from simple thresholds (data leakage)',
        'impact': 'Model learns overly simplified patterns',
        'mitigation': [
            'Use domain expertise for better label creation',
            'Incorporate real failure cases when available',
            'Add noise and complexity to synthetic labels'
        ]
    }
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_machine_category(machine_id: str) -> str:
    """Get category for a machine ID"""
    for category, machines in MACHINE_CATEGORIES.items():
        if machine_id in machines:
            return category
    return 'unknown'

def get_training_config(model_type: str, quick_test: bool = False):
    """Get training configuration for a model type"""
    if quick_test:
        return QUICK_TEST_CONFIG.get(model_type, {})
    return AUTOGLUON_CONFIG.get(model_type, {})

def estimate_training_time(model_types: list, parallel: bool = False):
    """Estimate total training time for given model types"""
    if parallel:
        return max([TRAINING_ESTIMATES[mt]['total_time_parallel'] for mt in model_types])
    else:
        return sum([TRAINING_ESTIMATES[mt]['total_time_sequential'] for mt in model_types])

# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_config():
    """Validate configuration consistency"""
    assert len(MACHINES) == 27, f"Expected 27 machines, found {len(MACHINES)}"
    assert len(MODEL_TYPES) == 4, f"Expected 4 model types, found {len(MODEL_TYPES)}"
    
    # Check all machines have categories
    all_categorized = []
    for machines in MACHINE_CATEGORIES.values():
        all_categorized.extend(machines)
    assert set(all_categorized) == set(MACHINES), "Machine category mismatch"
    
    # Check new machine is included
    assert 'cnc_fanuc_robodrill_001' in MACHINES, "New machine not found in list"
    assert 'cnc_fanuc_robodrill_001' in MACHINE_CATEGORIES['cnc'], "New machine not categorized"
    
    print("✅ Configuration validation passed")
    print(f"✅ {len(MACHINES)} machines configured (updated from 21 to 27)")
    print(f"✅ {len(MODEL_TYPES)} model types defined")
    print(f"✅ {len(MACHINE_CATEGORIES)} categories defined")
    print(f"✅ New machine 'cnc_fanuc_robodrill_001' verified")

if __name__ == "__main__":
    # Validate configuration when run directly
    validate_config()
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nMachines: {len(MACHINES)}")
    print(f"Model Types: {MODEL_TYPES}")
    print(f"Priority Machines: {len(PRIORITY_MACHINES)}")
    print(f"\nEstimated Training Time (Sequential):")
    for mt in MODEL_TYPES:
        time_hrs = TRAINING_ESTIMATES[mt]['total_time_sequential'] / 3600
        print(f"  {mt}: {time_hrs:.1f} hours")
    print(f"\nTotal Sequential Time: {sum([TRAINING_ESTIMATES[mt]['total_time_sequential'] for mt in MODEL_TYPES]) / 3600:.1f} hours")
    print(f"Total Parallel Time: {max([TRAINING_ESTIMATES[mt]['total_time_parallel'] for mt in MODEL_TYPES]) / 3600:.1f} hours")
    print("\n" + "="*70)
