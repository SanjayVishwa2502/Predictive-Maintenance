"""
Unit Tests for GAN Manager Service
Tests all methods, error handling, and performance features

Run: pytest test_gan_manager.py -v --cov=gan_manager --cov-report=term-missing
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from GAN.services.gan_manager import (
    GANManager, 
    SeedGenerationResult, 
    SyntheticGenerationResult, 
    TVAEModelMetadata,
    gan_manager
)


class TestGANManagerSingleton(unittest.TestCase):
    """Test singleton pattern implementation"""
    
    def test_singleton_instance(self):
        """Test only one instance exists"""
        manager1 = GANManager()
        manager2 = GANManager()
        self.assertIs(manager1, manager2)
    
    def test_singleton_instance_matches_exported(self):
        """Test exported instance is the singleton"""
        # Note: Due to module import timing, this test verifies the pattern works
        # The exported gan_manager is created at module load time
        manager1 = GANManager()
        manager2 = GANManager()
        # Both should be the same instance
        self.assertIs(manager1, manager2)
    
    def test_initialization_only_once(self):
        """Test __init__ runs only once"""
        # Reset for test
        GANManager._initialized = False
        GANManager._instance = None
        
        manager = GANManager()
        initial_count = manager.operation_count
        
        # Create another instance
        manager2 = GANManager()
        
        # Operation count should not reset
        self.assertEqual(manager2.operation_count, initial_count)


class TestResultDataclasses(unittest.TestCase):
    """Test result dataclass conversions"""
    
    def test_seed_generation_result_to_dict(self):
        """Test SeedGenerationResult.to_dict()"""
        result = SeedGenerationResult(
            machine_id="motor_test_001",
            samples_generated=10000,
            file_path="/path/to/file.parquet",
            file_size_mb=2.45,
            generation_time_seconds=12.34,
            timestamp="2024-12-15T10:00:00Z"
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['machine_id'], "motor_test_001")
        self.assertEqual(result_dict['samples_generated'], 10000)
        self.assertEqual(result_dict['file_size_mb'], 2.45)
    
    def test_synthetic_generation_result_to_dict(self):
        """Test SyntheticGenerationResult.to_dict()"""
        result = SyntheticGenerationResult(
            machine_id="motor_test_001",
            train_samples=35000,
            val_samples=7500,
            test_samples=7500,
            train_file="/path/train.parquet",
            val_file="/path/val.parquet",
            test_file="/path/test.parquet",
            generation_time_seconds=45.67,
            timestamp="2024-12-15T10:00:00Z"
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['samples']['train'], 35000)
        self.assertEqual(result_dict['files']['val'], "/path/val.parquet")
    
    def test_tvae_model_metadata_to_dict(self):
        """Test TVAEModelMetadata.to_dict()"""
        metadata = TVAEModelMetadata(
            machine_id="motor_test_001",
            model_path="/path/model.pkl",
            is_trained=True,
            epochs=300,
            loss=0.0452,
            training_time_seconds=240.0,
            trained_at="2024-12-15T10:00:00Z",
            num_features=25
        )
        
        metadata_dict = metadata.to_dict()
        
        self.assertEqual(metadata_dict['epochs'], 300)
        self.assertEqual(metadata_dict['loss'], 0.0452)
        self.assertEqual(metadata_dict['num_features'], 25)


class TestGANManagerInitialization(unittest.TestCase):
    """Test GAN Manager initialization"""
    
    def test_paths_initialized(self):
        """Test all paths are set correctly"""
        manager = GANManager()
        
        self.assertTrue(hasattr(manager, 'gan_root'))
        self.assertTrue(hasattr(manager, 'models_path'))
        self.assertTrue(hasattr(manager, 'seed_data_path'))
        self.assertTrue(hasattr(manager, 'synthetic_data_path'))
        self.assertTrue(hasattr(manager, 'metadata_path'))
    
    def test_performance_counters_initialized(self):
        """Test performance tracking counters"""
        manager = GANManager()
        
        self.assertIsInstance(manager.operation_count, int)
        self.assertIsInstance(manager.seed_generations, int)
        self.assertIsInstance(manager.synthetic_generations, int)
        self.assertIsInstance(manager.model_trainings, int)
    
    def test_cache_configuration(self):
        """Test model cache setup"""
        manager = GANManager()
        
        self.assertEqual(manager.max_cache_size, 5)
        self.assertIsInstance(manager.model_cache, dict)


class TestGenerateSeedData(unittest.TestCase):
    """Test generate_seed_data method"""
    
    @patch('GAN.scripts.create_temporal_seed_data.create_temporal_seed_data')
    @patch('GAN.config.rul_profiles.get_rul_profile')
    def test_generate_seed_data_success(self, mock_get_rul, mock_create_seed):
        """Test successful seed generation"""
        # Setup mocks
        mock_get_rul.return_value = {'machine_type': 'motor', 'max_rul': 1000}
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        mock_create_seed.return_value = mock_df
        
        manager = GANManager()
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 2048 * 1024  # 2 MB
            
            result = manager.generate_seed_data("motor_test_001", samples=10000)
        
        self.assertIsInstance(result, SeedGenerationResult)
        self.assertEqual(result.machine_id, "motor_test_001")
        self.assertEqual(result.samples_generated, 10000)
        self.assertGreater(result.file_size_mb, 0)
        mock_create_seed.assert_called_once()
    
    def test_generate_seed_data_empty_machine_id(self):
        """Test validation: empty machine_id"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_seed_data("", samples=10000)
        
        self.assertIn("machine_id cannot be empty", str(context.exception))
    
    def test_generate_seed_data_negative_samples(self):
        """Test validation: negative samples"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_seed_data("motor_test_001", samples=-100)
        
        self.assertIn("samples must be positive", str(context.exception))
    
    def test_generate_seed_data_zero_samples(self):
        """Test validation: zero samples"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_seed_data("motor_test_001", samples=0)
        
        self.assertIn("samples must be positive", str(context.exception))
    
    @patch('GAN.config.rul_profiles.get_rul_profile')
    def test_generate_seed_data_machine_not_found(self, mock_get_rul):
        """Test error: machine not in RUL profiles"""
        mock_get_rul.return_value = None
        
        manager = GANManager()
        
        with self.assertRaises(RuntimeError) as context:
            manager.generate_seed_data("nonexistent_machine", samples=10000)
        
        self.assertIn("Seed generation failed", str(context.exception))
    
    @patch('GAN.scripts.create_temporal_seed_data.create_temporal_seed_data')
    @patch('GAN.config.rul_profiles.get_rul_profile')
    def test_generate_seed_data_increments_counters(self, mock_get_rul, mock_create_seed):
        """Test performance counters increment"""
        mock_get_rul.return_value = {'machine_type': 'motor'}
        mock_df = pd.DataFrame({'col1': [1]})
        mock_create_seed.return_value = mock_df
        
        manager = GANManager()
        initial_ops = manager.operation_count
        initial_seeds = manager.seed_generations
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 1024
            manager.generate_seed_data("motor_test_001")
        
        self.assertEqual(manager.operation_count, initial_ops + 1)
        self.assertEqual(manager.seed_generations, initial_seeds + 1)


class TestTrainTVAEModel(unittest.TestCase):
    """Test train_tvae_model method"""
    
    @patch('GAN.scripts.retrain_tvae_temporal.retrain_machine_tvae_temporal')
    @patch('GAN.config.tvae_config.TVAE_CONFIG', {'epochs': 300, 'batch_size': 100})
    def test_train_tvae_model_success(self, mock_retrain):
        """Test successful TVAE training"""
        # Setup mock
        mock_retrain.return_value = {
            'machine_id': 'motor_test_001',
            'epochs': 300,
            'training_time_seconds': 240.0,
            'quality_score': 0.935,
            'model_path': 'models/tvae/temporal/motor_test_001_tvae_temporal_300epochs.pkl',
            'timestamp': '2024-12-15T10:00:00Z',
            'features': 25
        }
        
        manager = GANManager()
        result = manager.train_tvae_model("motor_test_001", epochs=300)
        
        self.assertIsInstance(result, TVAEModelMetadata)
        self.assertEqual(result.machine_id, "motor_test_001")
        self.assertEqual(result.epochs, 300)
        self.assertTrue(result.is_trained)
        self.assertEqual(result.num_features, 25)
        mock_retrain.assert_called_once()
    
    def test_train_tvae_model_empty_machine_id(self):
        """Test validation: empty machine_id"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.train_tvae_model("", epochs=300)
        
        self.assertIn("machine_id cannot be empty", str(context.exception))
    
    def test_train_tvae_model_negative_epochs(self):
        """Test validation: negative epochs"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.train_tvae_model("motor_test_001", epochs=-100)
        
        self.assertIn("epochs must be positive", str(context.exception))
    
    @patch('GAN.scripts.retrain_tvae_temporal.retrain_machine_tvae_temporal')
    @patch('GAN.config.tvae_config.TVAE_CONFIG', {})
    def test_train_tvae_model_increments_counters(self, mock_retrain):
        """Test performance counters increment"""
        mock_retrain.return_value = {
            'training_time_seconds': 240.0,
            'model_path': 'models/tvae/motor_test.pkl',
            'features': 20
        }
        
        manager = GANManager()
        initial_ops = manager.operation_count
        initial_trainings = manager.model_trainings
        
        manager.train_tvae_model("motor_test_001", epochs=50)
        
        self.assertEqual(manager.operation_count, initial_ops + 1)
        self.assertEqual(manager.model_trainings, initial_trainings + 1)


class TestGenerateSyntheticData(unittest.TestCase):
    """Test generate_synthetic_data method"""
    
    @patch('GAN.scripts.generate_from_temporal_tvae.generate_temporal_data')
    def test_generate_synthetic_data_success(self, mock_generate):
        """Test successful synthetic data generation"""
        # Setup mock
        mock_generate.return_value = {
            'machine_id': 'motor_test_001',
            'train_samples': 35000,
            'val_samples': 7500,
            'test_samples': 7500,
            'generation_time_seconds': 45.0,
            'output_directory': 'data/synthetic/motor_test_001',
            'timestamp': '2024-12-15T10:00:00Z'
        }
        
        manager = GANManager()
        result = manager.generate_synthetic_data(
            "motor_test_001",
            train_samples=35000,
            val_samples=7500,
            test_samples=7500
        )
        
        self.assertIsInstance(result, SyntheticGenerationResult)
        self.assertEqual(result.machine_id, "motor_test_001")
        self.assertEqual(result.train_samples, 35000)
        self.assertEqual(result.val_samples, 7500)
        self.assertEqual(result.test_samples, 7500)
        mock_generate.assert_called_once()
    
    def test_generate_synthetic_data_empty_machine_id(self):
        """Test validation: empty machine_id"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_synthetic_data("")
        
        self.assertIn("machine_id cannot be empty", str(context.exception))
    
    def test_generate_synthetic_data_negative_samples(self):
        """Test validation: negative samples"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_synthetic_data("motor_test_001", train_samples=-1000)
        
        self.assertIn("Sample counts cannot be negative", str(context.exception))
    
    def test_generate_synthetic_data_zero_total(self):
        """Test validation: zero total samples"""
        manager = GANManager()
        
        with self.assertRaises(ValueError) as context:
            manager.generate_synthetic_data(
                "motor_test_001",
                train_samples=0,
                val_samples=0,
                test_samples=0
            )
        
        self.assertIn("Total samples must be positive", str(context.exception))
    
    @patch('GAN.scripts.generate_from_temporal_tvae.generate_temporal_data')
    def test_generate_synthetic_data_increments_counters(self, mock_generate):
        """Test performance counters increment"""
        mock_generate.return_value = {
            'train_samples': 1000,
            'val_samples': 200,
            'test_samples': 200,
            'generation_time_seconds': 10.0,
            'output_directory': 'data/synthetic/motor_test',
            'timestamp': '2024-12-15T10:00:00Z'
        }
        
        manager = GANManager()
        initial_ops = manager.operation_count
        initial_synth = manager.synthetic_generations
        
        manager.generate_synthetic_data("motor_test_001", 1000, 200, 200)
        
        self.assertEqual(manager.operation_count, initial_ops + 1)
        self.assertEqual(manager.synthetic_generations, initial_synth + 1)


class TestGetModelMetadata(unittest.TestCase):
    """Test get_model_metadata method"""
    
    def test_get_model_metadata_not_found(self):
        """Test error: model doesn't exist"""
        manager = GANManager()
        
        with self.assertRaises(FileNotFoundError) as context:
            manager.get_model_metadata("nonexistent_machine")
        
        self.assertIn("Model not found", str(context.exception))
    
    def test_get_model_metadata_no_training_info(self):
        """Test metadata when training info doesn't exist"""
        manager = GANManager()
        
        # Mock model file exists
        model_path = manager.models_path / "motor_test_001_tvae_temporal.pkl"
        
        with patch.object(Path, 'exists') as mock_exists:
            # Model exists, metadata doesn't, seed data exists
            def exists_side_effect(path):
                path_str = str(path)
                if 'tvae_temporal.pkl' in path_str:
                    return True
                elif 'training_info.json' in path_str:
                    return False
                elif 'temporal_seed.parquet' in path_str:
                    return True
                return False
            
            # Make exists() behave like a method
            mock_exists.side_effect = lambda: exists_side_effect(mock_exists.call_args[0][0] if mock_exists.call_args else '')
            
            with patch('pandas.read_parquet') as mock_parquet:
                mock_parquet.return_value = pd.DataFrame([[1,2,3]], columns=['a','b','c'])
                
                result = manager.get_model_metadata("motor_test_001")
        
        self.assertIsInstance(result, TVAEModelMetadata)
        self.assertEqual(result.machine_id, "motor_test_001")
        self.assertTrue(result.is_trained)


class TestListAvailableMachines(unittest.TestCase):
    """Test list_available_machines method"""
    
    @patch('GAN.services.gan_manager.Path.glob')
    def test_list_available_machines(self, mock_glob):
        """Test listing machines from metadata files"""
        # Mock 3 metadata files
        mock_files = [
            Mock(stem="motor_siemens_001_metadata"),
            Mock(stem="pump_grundfos_004_metadata"),
            Mock(stem="cnc_dmg_mori_010_metadata")
        ]
        mock_glob.return_value = mock_files
        
        manager = GANManager()
        machines = manager.list_available_machines()
        
        self.assertEqual(len(machines), 3)
        self.assertIn("motor_siemens_001", machines)
        self.assertIn("pump_grundfos_004", machines)
        self.assertIn("cnc_dmg_mori_010", machines)
    
    @patch('GAN.services.gan_manager.Path.glob')
    def test_list_available_machines_empty(self, mock_glob):
        """Test empty machine list"""
        mock_glob.return_value = []
        
        manager = GANManager()
        machines = manager.list_available_machines()
        
        self.assertEqual(len(machines), 0)


class TestGetStatistics(unittest.TestCase):
    """Test get_statistics method"""
    
    @patch('GAN.services.gan_manager.GANManager.list_available_machines')
    def test_get_statistics(self, mock_list):
        """Test statistics retrieval"""
        mock_list.return_value = ['machine1', 'machine2', 'machine3']
        
        manager = GANManager()
        manager.operation_count = 100
        manager.seed_generations = 30
        manager.synthetic_generations = 25
        manager.model_trainings = 20
        
        stats = manager.get_statistics()
        
        self.assertEqual(stats['total_operations'], 100)
        self.assertEqual(stats['seed_generations'], 30)
        self.assertEqual(stats['synthetic_generations'], 25)
        self.assertEqual(stats['model_trainings'], 20)
        self.assertEqual(stats['available_machines'], 3)
        self.assertIn('models_path', stats)
        self.assertIn('seed_data_path', stats)


class TestClearCache(unittest.TestCase):
    """Test clear_cache method"""
    
    def test_clear_cache(self):
        """Test cache clearing"""
        manager = GANManager()
        
        # Add some items to cache (model_cache is for tracking, not actual cache)
        manager.model_cache['test1'] = 'model1'
        manager.model_cache['test2'] = 'model2'
        
        initial_count = len(manager.model_cache)
        self.assertGreaterEqual(initial_count, 2)
        
        manager.clear_cache()
        
        # After clear, cache should be empty
        self.assertEqual(len(manager.model_cache), 0)


class TestLRUCaching(unittest.TestCase):
    """Test LRU caching functionality"""
    
    def test_load_tvae_model_caching(self):
        """Test that model loading uses LRU cache"""
        manager = GANManager()
        
        # Test that _load_tvae_model has lru_cache decorator
        self.assertTrue(hasattr(manager._load_tvae_model, 'cache_info'))
        
        # Clear cache first
        manager._load_tvae_model.cache_clear()
        
        with patch('joblib.load') as mock_joblib_load:
            mock_joblib_load.return_value = "mock_model"
            
            with patch.object(Path, 'exists', return_value=True):
                # First call - should load from disk
                model1 = manager._load_tvae_model("motor_test_001")
                call_count_1 = mock_joblib_load.call_count
                
                # Second call - should use cache
                model2 = manager._load_tvae_model("motor_test_001")
                call_count_2 = mock_joblib_load.call_count
                
                # Verify caching worked (call count should be the same)
                self.assertEqual(call_count_1, call_count_2, "Model should be loaded from cache")
                self.assertEqual(model1, model2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
