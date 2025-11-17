# Fast Training Configuration for Raspberry Pi Deployment
# Created: 2025-11-17
# Optimized for: Speed + Raspberry Pi compatibility

# RASPBERRY PI CONSTRAINTS
# - CPU: ARM Cortex-A72 (Quad-core 1.5GHz)
# - RAM: 4-8 GB
# - Storage: SD Card (limited)
# - No GPU acceleration
# - Power consumption limits

# AutoGluon Configuration - FAST & LIGHTWEIGHT
AUTOGLUON_CONFIG_FAST = {
    'classification': {
        'eval_metric': 'f1',
        'problem_type': 'binary',
        'time_limit': 900,  # 15 minutes (vs 60 min before)
        'presets': 'medium_quality_faster_train',  # Faster preset
        'num_bag_folds': 3,  # Reduced from 5 (faster)
        'num_stack_levels': 0,  # No stacking (lighter models)
        
        # CRITICAL: Exclude heavy/incompatible models for Raspberry Pi
        'excluded_model_types': [
            'NN_TORCH',   # Neural networks (too heavy for Pi)
            'FASTAI',     # Deep learning (requires GPU)
            'XT',         # ExtraTrees (slower than RF)
            'KNN',        # Memory-intensive for large datasets
        ],
        
        # Keep only lightweight, Pi-compatible models
        # - LightGBM: Fast, small, CPU-optimized
        # - RandomForest: Efficient, parallelizable
        # - XGBoost: Good performance, reasonable size
        # - CatBoost: Good for categorical features
        
        'hyperparameters': {
            'GBM': {},  # LightGBM (primary model)
            'RF': [
                {'max_depth': 15, 'n_estimators': 100},  # Lighter than default
            ],
            'XGB': {},
            'CAT': {},
        },
        
        # Additional optimizations
        'ag_args_fit': {
            'num_cpus': 6,  # Match your i7-14700HX
            'num_gpus': 0,  # No GPU needed
        }
    },
    
    'regression': {
        'eval_metric': 'r2',
        'problem_type': 'regression',
        'time_limit': 900,  # 15 minutes
        'presets': 'medium_quality_faster_train',
        'num_bag_folds': 3,
        'num_stack_levels': 0,
        'excluded_model_types': ['NN_TORCH', 'FASTAI', 'XT', 'KNN'],
        'ag_args_fit': {
            'num_cpus': 6,
            'num_gpus': 0,
        }
    },
    
    'anomaly': {
        'algorithm': 'isolation_forest',
        'contamination': 0.1,
        'n_estimators': 50,  # Reduced from 100 (faster)
        'max_samples': 256,  # Limit training sample size
        'n_jobs': 4,  # Parallel processing
    },
    
    'timeseries': {
        'prediction_length': 24,
        'time_limit': 900,  # 15 minutes
        'presets': 'medium_quality_faster_train',
        'excluded_model_types': ['NN_TORCH', 'FASTAI'],
    }
}

# Expected Model Sizes (After Training)
MODEL_SIZE_ESTIMATES = {
    'classification': '5-10 MB',  # LightGBM + RF ensemble
    'regression': '5-10 MB',
    'anomaly': '2-5 MB',  # Isolation Forest
    'timeseries': '10-20 MB',
    'total_per_machine': '25-45 MB',
    'total_10_machines': '250-450 MB'
}

# Raspberry Pi Deployment Strategy
RASPBERRY_PI_DEPLOYMENT = {
    'model_format': 'pickle',  # Native Python (no ONNX conversion needed initially)
    'inference_optimization': {
        'batch_predictions': True,  # Batch multiple predictions
        'cache_results': True,  # Cache recent predictions
        'quantization': False,  # Not needed for tree models
    },
    'memory_management': {
        'lazy_loading': True,  # Load models on-demand
        'max_models_in_memory': 2,  # Limit concurrent loaded models
        'prediction_batching': 32,  # Batch size for efficiency
    },
    'performance_targets': {
        'inference_latency': '<200ms',  # Per prediction
        'memory_usage': '<1 GB',  # Total RAM
        'cpu_usage': '<60%',  # Average
        'power_consumption': '<10W',  # Total system
    }
}

# Training Time Estimates (Per Machine)
TRAINING_TIME_ESTIMATES = {
    'classification': '10-15 minutes',
    'regression': '10-15 minutes',
    'anomaly': '5 minutes',
    'timeseries': '10-15 minutes',
    'total_per_machine': '30-50 minutes',
    'total_10_machines': '5-8.5 hours'
}

# Performance Expectations (Realistic with Fast Training)
PERFORMANCE_TARGETS = {
    'classification': {
        'f1_score': '>0.90',  # Still achievable with medium_quality
        'accuracy': '>0.92',
        'inference_time': '<50ms',
    },
    'regression': {
        'r2_score': '>0.75',
        'mae': 'Low',
        'inference_time': '<50ms',
    },
    'anomaly': {
        'f1_score': '>0.80',  # Slightly lower for unsupervised
        'precision': '>0.75',
        'inference_time': '<30ms',
    }
}

# Validation
assert AUTOGLUON_CONFIG_FAST['classification']['time_limit'] == 900, "Should be 15 min"
assert 'NN_TORCH' in AUTOGLUON_CONFIG_FAST['classification']['excluded_model_types'], "Must exclude NN for Pi"
assert AUTOGLUON_CONFIG_FAST['classification']['num_stack_levels'] == 0, "No stacking for speed"

print("✅ Fast Training Config Loaded")
print(f"⏱️  Training time per machine: {TRAINING_TIME_ESTIMATES['total_per_machine']}")
print(f"📦 Model size per machine: {MODEL_SIZE_ESTIMATES['total_per_machine']}")
print(f"🥧 Raspberry Pi compatible: YES")
