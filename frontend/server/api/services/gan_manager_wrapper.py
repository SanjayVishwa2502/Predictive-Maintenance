"""
FastAPI Integration Wrapper for GAN Manager
Provides FastAPI-friendly interface to the GAN Manager singleton

This wrapper:
- Imports the existing GAN Manager from GAN/services/gan_manager.py
- Provides type-safe interfaces for FastAPI route handlers
- Adds async support where needed
- Handles path conversions between frontend and GAN directories
- Provides additional convenience methods for API operations
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from GAN.services.gan_manager import (
    gan_manager as _gan_manager,
    SeedGenerationResult,
    SyntheticGenerationResult,
    TVAEModelMetadata
)

logger = logging.getLogger(__name__)


class GANManagerWrapper:
    """
    FastAPI-friendly wrapper for GAN Manager
    
    Provides async methods and additional utilities for API integration
    """
    
    def __init__(self):
        """Initialize wrapper with GAN Manager singleton"""
        self.gan_manager = _gan_manager
        logger.info("GANManagerWrapper initialized with GAN Manager singleton")
    
    # =============================================================================
    # SEED DATA GENERATION
    # =============================================================================
    
    async def generate_seed_data_async(
        self, 
        machine_id: str, 
        samples: int = 10000
    ) -> SeedGenerationResult:
        """
        Async wrapper for seed data generation
        
        Args:
            machine_id: Machine identifier
            samples: Number of samples to generate
            
        Returns:
            SeedGenerationResult with file info
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If metadata doesn't exist
            RuntimeError: If generation fails
        """
        return self.gan_manager.generate_seed_data(machine_id, samples)
    
    def generate_seed_data(
        self, 
        machine_id: str, 
        samples: int = 10000
    ) -> SeedGenerationResult:
        """
        Synchronous seed data generation
        
        Args:
            machine_id: Machine identifier
            samples: Number of samples to generate
            
        Returns:
            SeedGenerationResult with file info
        """
        return self.gan_manager.generate_seed_data(machine_id, samples)
    
    # =============================================================================
    # TVAE MODEL TRAINING
    # =============================================================================
    
    async def train_tvae_model_async(
        self, 
        machine_id: str, 
        epochs: int = 300
    ) -> TVAEModelMetadata:
        """
        Async wrapper for TVAE training (for Celery task launching)
        
        Note: This is used to validate and prepare the training request.
        Actual training runs in Celery worker via gan_tasks.train_tvae_task
        
        Args:
            machine_id: Machine identifier
            epochs: Training epochs
            
        Returns:
            TVAEModelMetadata with training info
        """
        return self.gan_manager.train_tvae_model(machine_id, epochs)
    
    def train_tvae_model(
        self, 
        machine_id: str, 
        epochs: int = 300
    ) -> TVAEModelMetadata:
        """
        Synchronous TVAE model training
        
        Args:
            machine_id: Machine identifier
            epochs: Training epochs
            
        Returns:
            TVAEModelMetadata with training info
        """
        return self.gan_manager.train_tvae_model(machine_id, epochs)
    
    def validate_seed_data_exists(self, machine_id: str) -> bool:
        """
        Check if seed data exists for a machine (used before training)
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            True if seed data exists, False otherwise
        """
        seed_file = self.gan_manager.seed_data_path / "temporal" / f"{machine_id}_temporal_seed.parquet"
        return seed_file.exists()
    
    # =============================================================================
    # SYNTHETIC DATA GENERATION
    # =============================================================================
    
    async def generate_synthetic_data_async(
        self,
        machine_id: str,
        train_samples: int = 35000,
        val_samples: int = 7500,
        test_samples: int = 7500
    ) -> SyntheticGenerationResult:
        """
        Async wrapper for synthetic data generation
        
        Args:
            machine_id: Machine identifier
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            
        Returns:
            SyntheticGenerationResult with file info
        """
        return self.gan_manager.generate_synthetic_data(
            machine_id, train_samples, val_samples, test_samples
        )
    
    def generate_synthetic_data(
        self,
        machine_id: str,
        train_samples: int = 35000,
        val_samples: int = 7500,
        test_samples: int = 7500
    ) -> SyntheticGenerationResult:
        """
        Synchronous synthetic data generation
        
        Args:
            machine_id: Machine identifier
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            
        Returns:
            SyntheticGenerationResult with file info
        """
        return self.gan_manager.generate_synthetic_data(
            machine_id, train_samples, val_samples, test_samples
        )
    
    def validate_model_exists(self, machine_id: str) -> bool:
        """
        Check if TVAE model exists for a machine (used before generation)
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            True if model exists, False otherwise
        """
        # Check for model with epoch suffix pattern
        model_dir = self.gan_manager.models_path / "tvae" / "temporal"
        if not model_dir.exists():
            return False
        
        # Look for any model file matching the machine_id
        model_files = list(model_dir.glob(f"{machine_id}_tvae_temporal_*.pkl"))
        return len(model_files) > 0
    
    # =============================================================================
    # METADATA & INFORMATION
    # =============================================================================
    
    async def get_model_metadata_async(self, machine_id: str) -> TVAEModelMetadata:
        """
        Async wrapper for model metadata retrieval
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            TVAEModelMetadata
        """
        return self.gan_manager.get_model_metadata(machine_id)
    
    def get_model_metadata(self, machine_id: str) -> TVAEModelMetadata:
        """
        Get model metadata
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            TVAEModelMetadata
        """
        return self.gan_manager.get_model_metadata(machine_id)
    
    async def list_available_machines_async(self) -> List[str]:
        """
        Async wrapper for listing machines
        
        Returns:
            List of machine IDs
        """
        return self.gan_manager.list_available_machines()
    
    def list_available_machines(self) -> List[str]:
        """
        List all machines with metadata
        
        Returns:
            List of machine IDs
        """
        return self.gan_manager.list_available_machines()
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """
        Async wrapper for statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.gan_manager.get_statistics()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get GAN Manager statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.gan_manager.get_statistics()
    
    # =============================================================================
    # WORKFLOW STATUS HELPERS
    # =============================================================================
    
    def get_machine_workflow_status(self, machine_id: str) -> Dict[str, bool]:
        """
        Get workflow status for a machine (seed -> train -> generate)
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Dictionary with status flags:
            {
                'has_metadata': bool,
                'has_seed_data': bool,
                'has_trained_model': bool,
                'has_synthetic_data': bool,
                'can_generate_seed': bool,
                'can_train_model': bool,
                'can_generate_data': bool
            }
        """
        # Historical naming: the GAN manager writes a derived *metadata* file as
        # `{machine_id}_metadata.json`. The dashboard also stores the authored
        # profile as `{machine_id}.json`. Treat either as sufficient to consider
        # the machine "configured" for the next step.
        derived_metadata_exists = (self.gan_manager.metadata_path / f"{machine_id}_metadata.json").exists()
        profile_exists = (self.gan_manager.metadata_path / f"{machine_id}.json").exists()
        seed_exists = self.validate_seed_data_exists(machine_id)
        model_exists = self.validate_model_exists(machine_id)
        
        # Check synthetic data
        synthetic_dir = self.gan_manager.synthetic_data_path / "synthetic" / machine_id
        synthetic_exists = (
            synthetic_dir.exists() and
            (synthetic_dir / "train.parquet").exists()
        )

        has_profile_or_metadata = bool(derived_metadata_exists or profile_exists)
        
        return {
            'has_metadata': has_profile_or_metadata,
            'has_seed_data': seed_exists,
            'has_trained_model': model_exists,
            'has_synthetic_data': synthetic_exists,
            'can_generate_seed': has_profile_or_metadata,
            'can_train_model': has_profile_or_metadata and seed_exists,
            'can_generate_data': has_profile_or_metadata and model_exists
        }
    
    def get_machine_details(self, machine_id: str) -> Dict[str, Any]:
        """
        Get comprehensive machine details for API responses
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Dictionary with all machine info
        """
        workflow_status = self.get_machine_workflow_status(machine_id)
        
        details = {
            'machine_id': machine_id,
            'workflow_status': workflow_status
        }
        
        # Add metadata if model exists
        if workflow_status['has_trained_model']:
            try:
                metadata = self.get_model_metadata(machine_id)
                details['model_metadata'] = metadata.to_dict()
            except FileNotFoundError:
                pass
        
        return details
    
    # =============================================================================
    # CACHE MANAGEMENT
    # =============================================================================
    
    def clear_cache(self):
        """Clear model cache"""
        self.gan_manager.clear_cache()
        logger.info("Model cache cleared via wrapper")
    
    # =============================================================================
    # HEALTH CHECK
    # =============================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for GAN service
        
        Returns:
            Dictionary with health status
        """
        try:
            stats = self.get_statistics()
            
            return {
                'status': 'healthy',
                'service': 'GAN Manager',
                'total_operations': stats['total_operations'],
                'available_machines': stats['available_machines'],
                'paths_accessible': all([
                    Path(stats['models_path']).exists(),
                    Path(stats['seed_data_path']).exists(),
                    Path(stats['synthetic_data_path']).exists()
                ])
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'service': 'GAN Manager',
                'error': str(e)
            }


# Create singleton wrapper instance
gan_manager_wrapper = GANManagerWrapper()
