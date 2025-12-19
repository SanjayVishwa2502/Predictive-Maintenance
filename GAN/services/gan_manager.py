"""
GAN Manager Service
Industrial-Grade TVAE Model Management System

Centralized service for managing TVAE models, seed data generation,
and synthetic data generation with enterprise-level features:
- Singleton pattern for resource efficiency
- LRU caching for model instances (max 5 models)
- Comprehensive error handling and logging
- Metadata management
- Performance tracking

Author: GAN Engineering Team
Date: December 13, 2024
"""

import sys
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SeedGenerationResult:
    """Result object for seed data generation"""
    machine_id: str
    samples_generated: int
    file_path: str
    file_size_mb: float
    generation_time_seconds: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'machine_id': self.machine_id,
            'samples_generated': self.samples_generated,
            'file_path': self.file_path,
            'file_size_mb': self.file_size_mb,
            'generation_time_seconds': self.generation_time_seconds,
            'timestamp': self.timestamp
        }


@dataclass
class SyntheticGenerationResult:
    """Result object for synthetic data generation"""
    machine_id: str
    train_samples: int
    val_samples: int
    test_samples: int
    train_file: str
    val_file: str
    test_file: str
    generation_time_seconds: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'machine_id': self.machine_id,
            'samples': {
                'train': self.train_samples,
                'val': self.val_samples,
                'test': self.test_samples
            },
            'files': {
                'train': self.train_file,
                'val': self.val_file,
                'test': self.test_file
            },
            'generation_time_seconds': self.generation_time_seconds,
            'timestamp': self.timestamp
        }


@dataclass
class TVAEModelMetadata:
    """Metadata for TVAE model"""
    machine_id: str
    model_path: str
    is_trained: bool
    epochs: int
    loss: Optional[float]
    training_time_seconds: Optional[float]
    trained_at: Optional[str]
    num_features: int
    
    def to_dict(self) -> Dict:
        return {
            'machine_id': self.machine_id,
            'model_path': self.model_path,
            'is_trained': self.is_trained,
            'epochs': self.epochs,
            'loss': self.loss,
            'training_time_seconds': self.training_time_seconds,
            'trained_at': self.trained_at,
            'num_features': self.num_features
        }


class GANManager:
    """
    Singleton GAN Manager for centralized TVAE model and data management
    
    Features:
    - Singleton pattern for resource efficiency
    - LRU caching for TVAE models (max 5 models in memory)
    - Comprehensive error handling
    - Performance metrics tracking
    - Logging for all operations
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize GAN Manager (called only once due to singleton)"""
        if GANManager._initialized:
            return
        
        GANManager._initialized = True
        logger.info("Initializing GAN Manager Service...")
        
        # Base paths
        self.gan_root = PROJECT_ROOT / "GAN"
        self.models_path = self.gan_root / "models"
        self.seed_data_path = self.gan_root / "seed_data"
        self.synthetic_data_path = self.gan_root / "data"
        self.metadata_path = self.gan_root / "metadata"
        self.config_path = self.gan_root / "config"
        
        # Ensure directories exist
        self.models_path.mkdir(exist_ok=True, parents=True)
        self.seed_data_path.mkdir(exist_ok=True, parents=True)
        self.synthetic_data_path.mkdir(exist_ok=True, parents=True)
        self.metadata_path.mkdir(exist_ok=True, parents=True)
        
        # Model cache (LRU with max 5 models)
        self.model_cache: Dict[str, Any] = {}
        self.max_cache_size = 5
        
        # Performance tracking
        self.operation_count = 0
        self.seed_generations = 0
        self.synthetic_generations = 0
        self.model_trainings = 0
        
        logger.info(f"[OK] GAN Manager initialized successfully")
        logger.info(f"   Models path: {self.models_path}")
        logger.info(f"   Seed data path: {self.seed_data_path}")
        logger.info(f"   Synthetic data path: {self.synthetic_data_path}")
    
    @lru_cache(maxsize=5)
    def _load_tvae_model(self, machine_id: str):
        """
        Load TVAE model from disk with LRU caching
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Loaded TVAE model
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = self.models_path / f"{machine_id}_tvae_temporal.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"TVAE model not found at {model_path}")
        
        logger.info(f"Loading TVAE model for {machine_id}...")
        
        try:
            import joblib
            model = joblib.load(model_path)
            logger.info(f"[OK] TVAE model loaded for {machine_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load TVAE model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate_seed_data(self, machine_id: str, samples: int = 10000) -> SeedGenerationResult:
        """
        Generate temporal seed data for a machine
        
        Args:
            machine_id: Machine identifier
            samples: Number of samples to generate
            
        Returns:
            SeedGenerationResult with file info
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If metadata doesn't exist
        """
        if not machine_id:
            raise ValueError("machine_id cannot be empty")
        if samples <= 0:
            raise ValueError("samples must be positive")
        
        logger.info(f"Generating {samples} seed samples for {machine_id}...")
        self.operation_count += 1
        
        try:
            import time
            start_time = time.time()
            
            # Import dependencies
            from GAN.scripts.create_temporal_seed_data import create_temporal_seed_data
            from GAN.config.rul_profiles import get_rul_profile
            
            # Get RUL profile
            rul_profile = get_rul_profile(machine_id)
            if not rul_profile:
                raise ValueError(f"Machine {machine_id} not found in RUL profiles")
            
            # Generate seed dataframe
            # Note: The script function returns a DataFrame, doesn't save it automatically
            seed_df = create_temporal_seed_data(machine_id, rul_profile, n_samples=samples)
            
            # Save to parquet
            output_file = self.seed_data_path / f"{machine_id}_temporal_seed.parquet"
            # Ensure directory handles "temporal" subdirectory if needed, 
            # but initialized path is .../seed_data. 
            # Original script saves to .../seed_data/temporal/.
            # Let's match that structure.
            temporal_seed_path = self.seed_data_path / "temporal"
            temporal_seed_path.mkdir(exist_ok=True, parents=True)
            output_file = temporal_seed_path / f"{machine_id}_temporal_seed.parquet"
            
            seed_df.to_parquet(output_file, index=False)
            
            generation_time = time.time() - start_time
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            
            self.seed_generations += 1
            
            result = SeedGenerationResult(
                machine_id=machine_id,
                samples_generated=samples,
                file_path=str(output_file),
                file_size_mb=round(file_size_mb, 2),
                generation_time_seconds=round(generation_time, 2),
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
            logger.info(f"[OK] Seed data generated: {file_size_mb:.2f} MB in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Seed generation failed: {e}")
            raise RuntimeError(f"Seed generation failed: {e}")
    
    def train_tvae_model(self, machine_id: str, epochs: int = 300) -> TVAEModelMetadata:
        """
        Train TVAE model on seed data
        
        Args:
            machine_id: Machine identifier
            epochs: Training epochs
            
        Returns:
            TVAEModelMetadata with training info
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If seed data doesn't exist
        """
        if not machine_id:
            raise ValueError("machine_id cannot be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        
        logger.info(f"Training TVAE model for {machine_id} ({epochs} epochs)...")
        self.operation_count += 1
        
        try:
            import time
            start_time = time.time()
            
            # Import dependencies
            from GAN.scripts.retrain_tvae_temporal import retrain_machine_tvae_temporal
            from GAN.config.tvae_config import TVAE_CONFIG
            
            # Prepare config
            config = TVAE_CONFIG.copy()
            config['epochs'] = epochs
            # Ensure path consistency? The script handles paths relative to its location.
            # We trust the script's internal path logic if run from here?
            # Script uses Path(__file__).parent.parent.
            # If we import it, __file__ inside the script refers to the script file location.
            # So paths should resolve correctly.
            
            # Train model
            # Note: script function saves the model and returns results dict
            train_results = retrain_machine_tvae_temporal(machine_id, config, test_mode=False)
            
            # Extract info from results
            training_time = train_results.get('training_time_seconds', 0.0)
            final_loss = None # loss isn't returned explicitly in results dict shown, need to check if available
            # The results dict keys: machine_id, architecture, training_mode, epochs, batch_size,
            # training_time_seconds, total_time_seconds, quality_score, model_size_mb, 
            # seed_samples, features, rul_present, rul_correlation, model_path
            
            # Wait, retrain_machine_tvae_temporal implementation returns `results` dict.
            # It doesn't seem to include 'loss' in the keys I read earlier. 
            # But the metadata class has 'loss'.
            # I'll default to None or extract if added later.
            
            # Construct absolute model path
            # Result path is relative: "models/tvae/temporal/..."
            rel_path = train_results.get('model_path', '')
            model_path = self.gan_root / rel_path
            
            self.model_trainings += 1
            
            result = TVAEModelMetadata(
                machine_id=machine_id,
                model_path=str(model_path),
                is_trained=True,
                epochs=epochs,
                loss=None, # Not provided by script currently
                training_time_seconds=train_results.get('training_time_seconds'),
                trained_at=train_results.get('timestamp'),
                num_features=train_results.get('features', 0)
            )
            
            logger.info(f"[OK] TVAE trained in {training_time:.2f}s, Quality: {train_results.get('quality_score')}")
            return result
            
        except Exception as e:
            logger.error(f"TVAE training failed: {e}")
            raise RuntimeError(f"TVAE training failed: {e}")
    
    def generate_synthetic_data(
        self, 
        machine_id: str,
        train_samples: int = 35000,
        val_samples: int = 7500,
        test_samples: int = 7500
    ) -> SyntheticGenerationResult:
        """
        Generate synthetic datasets (train/val/test) using TVAE
        
        Args:
            machine_id: Machine identifier
            train_samples: Training samples (default: 35000)
            val_samples: Validation samples (default: 7500)
            test_samples: Test samples (default: 7500)
            
        Returns:
            SyntheticGenerationResult with file info
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If TVAE model doesn't exist
        """
        if not machine_id:
            raise ValueError("machine_id cannot be empty")
        # Allow 0 test samples if desired? But code enforces positive in validation.
        # Let's relax or check total > 0.
        if train_samples < 0 or val_samples < 0 or test_samples < 0:
             raise ValueError("Sample counts cannot be negative")
        
        total_samples = train_samples + val_samples + test_samples
        if total_samples == 0:
            raise ValueError("Total samples must be positive")
            
        logger.info(f"Generating synthetic data for {machine_id}...")
        self.operation_count += 1
        
        try:
            import time
            start_time = time.time()
            
            # Import dependencies
            from GAN.scripts.generate_from_temporal_tvae import generate_temporal_data
            
            # Calculate splits
            train_split = train_samples / total_samples
            val_split = val_samples / total_samples
            # test_split is implicitly the rest in the script
            
            # Generate data
            # Script function returns results dict
            gen_results = generate_temporal_data(
                machine_id, 
                num_samples=total_samples,
                train_split=train_split,
                val_split=val_split
            )
            
            generation_time = gen_results.get('generation_time_seconds', 0.0)
            
            # Construct absolute paths
            # script text: "output: train.parquet ... in data/synthetic/{machine_id}/"
            # results['output_directory'] gives relative path
            rel_out_dir = gen_results.get('output_directory', '')
            out_dir = self.gan_root / rel_out_dir
            
            self.synthetic_generations += 1
            
            result = SyntheticGenerationResult(
                machine_id=machine_id,
                train_samples=gen_results.get('train_samples'),
                val_samples=gen_results.get('val_samples'),
                test_samples=gen_results.get('test_samples'),
                train_file=str(out_dir / "train.parquet"),
                val_file=str(out_dir / "val.parquet"),
                test_file=str(out_dir / "test.parquet"),
                generation_time_seconds=generation_time,
                timestamp=gen_results.get('timestamp')
            )
            
            logger.info(f"[OK] Synthetic data generated: {total_samples} samples in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}")
            raise RuntimeError(f"Synthetic generation failed: {e}")
    
    def get_model_metadata(self, machine_id: str) -> TVAEModelMetadata:
        """
        Get metadata for a TVAE model
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            TVAEModelMetadata
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = self.models_path / f"{machine_id}_tvae_temporal.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {machine_id}")
        
        # Try to load training metadata if it exists
        metadata_file = self.models_path / f"{machine_id}_training_info.json"
        training_info = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                training_info = json.load(f)
        
        # Get feature count from seed data if available
        seed_file = self.seed_data_path / f"{machine_id}_temporal_seed.parquet"
        num_features = 0
        if seed_file.exists():
            import pandas as pd
            df = pd.read_parquet(seed_file)
            num_features = len(df.columns) - 2
        
        return TVAEModelMetadata(
            machine_id=machine_id,
            model_path=str(model_path),
            is_trained=True,
            epochs=training_info.get('epochs', 300),
            loss=training_info.get('final_loss'),
            training_time_seconds=training_info.get('training_time'),
            trained_at=training_info.get('trained_at'),
            num_features=num_features
        )
    
    def list_available_machines(self) -> List[str]:
        """
        List all machines with metadata
        
        Returns:
            List of machine IDs
        """
        metadata_files = list(self.metadata_path.glob("*_metadata.json"))
        return [f.stem.replace("_metadata", "") for f in metadata_files]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get GAN Manager statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_operations': self.operation_count,
            'seed_generations': self.seed_generations,
            'synthetic_generations': self.synthetic_generations,
            'model_trainings': self.model_trainings,
            'cached_models': len(self.model_cache),
            'available_machines': len(self.list_available_machines()),
            'models_path': str(self.models_path),
            'seed_data_path': str(self.seed_data_path),
            'synthetic_data_path': str(self.synthetic_data_path)
        }
    
    def clear_cache(self):
        """Clear model cache"""
        self._load_tvae_model.cache_clear()
        self.model_cache.clear()
        logger.info("Model cache cleared")


# Singleton instance
gan_manager = GANManager()
