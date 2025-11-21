"""
TVAE Production Configuration
Based on Phase 1.1.3 comparison results
"""

# TVAE was chosen over CTGAN based on:
# - Higher quality score (0.913 vs 0.788)
# - 2.5x faster training
# - Smaller model size (0.51 MB vs 0.95 MB)

TVAE_CONFIG = {
    # Training parameters
    'epochs': 500,  # High-confidence training (optimized for max quality)
    'batch_size': 100,  # Set to 100 as requested
    'cuda': True,  # GPU acceleration enabled
    'verbose': True,  # Show training progress
    
    # Model architecture (TVAE parameters)
    'embedding_dim': 128,  # Embedding dimension for categorical variables
    'compress_dims': (128, 128),  # Compressor dimensions
    'decompress_dims': (128, 128),  # Decompressor dimensions
    
    # Optimization
    'weight_decay': 1e-5,  # L2 regularization (l2scale parameter)
    
    # Loss function
    'loss_factor': 2,  # KL divergence weight
    
    # Training behavior
    'log_frequency': True,  # Log training metrics
}

# Production expectations - HIGH CONFIDENCE configuration
PRODUCTION_EXPECTATIONS = {
    'training_time_per_machine_minutes': 4.0,  # ~4 minutes with 10K samples + 500 epochs
    'total_training_time_21_machines_minutes': 84,  # ~84 minutes total (1.4 hours)
    'expected_quality_score': 0.935,  # High-confidence quality target
    'model_size_mb': 1.0,  # Per machine (larger due to more data)
    'total_storage_21_machines_mb': 21,  # All 21 models
    'samples_per_machine': 10000,  # High-confidence dataset size
    'total_synthetic_samples': 210000,  # 21 machines × 10K samples
}

# Batch processing configuration
BATCH_CONFIG = {
    'machines_per_batch': 5,  # Process 5 machines at a time
    'num_batches': 4,  # 20 machines / 5 per batch
    'parallel_training': False,  # Single GPU, sequential training
}

# MLflow tracking
MLFLOW_CONFIG = {
    'experiment_name_prefix': 'TVAE',
    'tracking_uri': './mlruns',
    'log_models': True,
    'log_metrics': True,
    'log_params': True,
}

# Data generation configuration
GENERATION_CONFIG = {
    'samples_per_machine': 5000,
    'train_split': 0.70,  # 3500 samples
    'val_split': 0.15,    # 750 samples
    'test_split': 0.15,   # 750 samples
    'add_machine_id': True,  # Add machine_id column
    'output_format': 'parquet',  # parquet or csv
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'minimum_acceptable': 0.75,  # Minimum quality score
    'good': 0.85,  # Good quality threshold
    'excellent': 0.90,  # Excellent quality threshold
    'action_on_failure': 'retrain',  # retrain or skip
}

# Production paths (relative to GAN folder)
PATHS = {
    'profiles': '../data/real_machines/profiles/',
    'seed_data': '../seed_data/',
    'metadata': '../metadata/',
    'models': '../models/tvae/',  # Changed from ctgan to tvae
    'synthetic_data': '../data/synthetic/',
    'reports': '../reports/',
    'mlflow_tracking': '../mlruns/',
}
