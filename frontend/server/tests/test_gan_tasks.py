"""
Unit Tests for GAN Celery Tasks
Tests all 3 tasks with progress broadcasting and error handling
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import json
import time
from datetime import datetime

# Import tasks
from tasks.gan_tasks import (
    train_tvae_task,
    generate_data_task,
    generate_seed_data_task,
    broadcast_progress,
    get_task_status,
    ProgressTask
)
from celery import Task


class TestBroadcastProgress(unittest.TestCase):
    """Test progress broadcasting utility"""
    
    @patch('tasks.gan_tasks.get_redis_pubsub')
    def test_broadcast_progress_success(self, mock_redis):
        """Test successful progress broadcast"""
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        broadcast_progress(
            task_id='test-task-123',
            machine_id='cnc_machine_001',
            current=50,
            total=100,
            status='RUNNING',
            message='Training progress',
            epoch=50,
            loss=0.05
        )
        
        # Verify publish was called
        mock_client.publish.assert_called_once()
        channel, message = mock_client.publish.call_args[0]
        
        self.assertEqual(channel, 'gan:training:test-task-123')
        
        message_data = json.loads(message)
        self.assertEqual(message_data['task_id'], 'test-task-123')
        self.assertEqual(message_data['machine_id'], 'cnc_machine_001')
        self.assertEqual(message_data['current'], 50)
        self.assertEqual(message_data['total'], 100)
        self.assertEqual(message_data['progress'], 50.0)
        self.assertEqual(message_data['epoch'], 50)
        self.assertEqual(message_data['loss'], 0.05)
    
    @patch('tasks.gan_tasks.get_redis_pubsub')
    def test_broadcast_progress_failure_handling(self, mock_redis):
        """Test progress broadcast failure is handled gracefully"""
        mock_redis.side_effect = Exception("Redis connection failed")
        
        # Should not raise exception
        try:
            broadcast_progress(
                task_id='test-task-456',
                machine_id='cnc_machine_002',
                current=10,
                total=100,
                status='RUNNING',
                message='Test'
            )
        except Exception:
            self.fail("broadcast_progress raised exception unexpectedly")


class TestProgressTask(unittest.TestCase):
    """Test ProgressTask base class"""
    
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_update_progress(self, mock_broadcast):
        """Test progress update method"""
        task = ProgressTask()

        task.update_state = Mock()

        task.update_progress(
            machine_id='cnc_machine_003',
            current=25,
            total=100,
            message='Processing',
            stage='training'
        )

        # Verify Celery state update
        task.update_state.assert_called_once_with(
            state='PROGRESS',
            meta={
                'machine_id': 'cnc_machine_003',
                'current': 25,
                'total': 100,
                'progress': 25.0,
                'message': 'Processing',
                'stage': 'training'
            }
        )

        # Verify Redis broadcast
        mock_broadcast.assert_called_once()


class TestTrainTvaeTask(unittest.TestCase):
    """Test train_tvae_task"""
    
    @patch('tasks.gan_tasks.gan_manager')
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_train_tvae_success(self, mock_broadcast, mock_manager):
        """Test successful TVAE training"""
        # Mock GANManager response
        mock_result = Mock(
            model_path='/models/cnc_machine_004_model.pkl',
            loss=0.045,
            trained_at='2024-01-15T10:30:00',
            num_features=25
        )
        mock_manager.train_tvae_model.return_value = mock_result
        
        with patch.object(train_tvae_task, 'update_state'):
            # Execute task - call the function directly
            result = train_tvae_task.run('cnc_machine_004', epochs=100)

            # Verify result
            self.assertTrue(result['success'])
            self.assertEqual(result['machine_id'], 'cnc_machine_004')
            self.assertEqual(result['epochs'], 100)
            self.assertEqual(result['model_path'], '/models/cnc_machine_004_model.pkl')
            self.assertEqual(result['final_loss'], 0.045)

            # Verify GANManager was called
            mock_manager.train_tvae_model.assert_called_once_with(
                machine_id='cnc_machine_004',
                epochs=100
            )

            # Verify progress was broadcast (at least twice: start and end)
            self.assertGreaterEqual(mock_broadcast.call_count, 2)
    
    @patch('tasks.gan_tasks.gan_manager')
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_train_tvae_validation_error(self, mock_broadcast, mock_manager):
        """Test training with validation error"""
        mock_manager.train_tvae_model.side_effect = ValueError("Invalid machine_id")
        
        with patch.object(train_tvae_task, 'update_state'):
            with self.assertRaises(ValueError) as context:
                train_tvae_task.run('invalid_machine', epochs=100)

            self.assertIn("Invalid machine_id", str(context.exception))

            # Verify failure broadcast
            failure_calls = [call for call in mock_broadcast.call_args_list 
                            if 'FAILURE' in str(call)]
            self.assertGreater(len(failure_calls), 0)
    
    @patch('tasks.gan_tasks.gan_manager')
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_train_tvae_file_not_found(self, mock_broadcast, mock_manager):
        """Test training with missing seed data"""
        mock_manager.train_tvae_model.side_effect = FileNotFoundError("Seed data not found")
        
        with patch.object(train_tvae_task, 'update_state'):
            with self.assertRaises(FileNotFoundError):
                train_tvae_task.run('cnc_machine_005', epochs=100)


class TestGenerateDataTask(unittest.TestCase):
    """Test generate_data_task"""
    
    @patch('tasks.gan_tasks.gan_manager')
    def test_generate_data_success(self, mock_manager):
        """Test successful synthetic data generation"""
        # Mock GANManager response
        mock_result = Mock(
            train_samples=35000,
            val_samples=7500,
            test_samples=7500,
            train_file='/data/train.csv',
            val_file='/data/val.csv',
            test_file='/data/test.csv',
            timestamp='2024-01-15T10:35:00'
        )
        mock_manager.generate_synthetic_data.return_value = mock_result
        
        with patch.object(generate_data_task, 'update_state'):
            # Execute task
            result = generate_data_task.run(
                'cnc_machine_006',
                train_samples=35000,
                val_samples=7500,
                test_samples=7500
            )

            # Verify result
            self.assertTrue(result['success'])
            self.assertEqual(result['machine_id'], 'cnc_machine_006')
            self.assertEqual(result['train_samples'], 35000)
            self.assertEqual(result['val_samples'], 7500)
            self.assertEqual(result['test_samples'], 7500)
            self.assertEqual(result['train_file'], '/data/train.csv')

            # Verify GANManager was called
            mock_manager.generate_synthetic_data.assert_called_once_with(
                machine_id='cnc_machine_006',
                train_samples=35000,
                val_samples=7500,
                test_samples=7500
            )
    
    @patch('tasks.gan_tasks.gan_manager')
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_generate_data_model_not_found(self, mock_broadcast, mock_manager):
        """Test generation with missing model"""
        mock_manager.generate_synthetic_data.side_effect = FileNotFoundError("Model not found")
        
        with patch.object(generate_data_task, 'update_state'):
            with self.assertRaises(FileNotFoundError):
                generate_data_task.run(
                    'cnc_machine_007',
                    train_samples=1000,
                    val_samples=200,
                    test_samples=200
                )

            # Verify failure broadcast
            failure_calls = [call for call in mock_broadcast.call_args_list 
                            if 'FAILURE' in str(call)]
            self.assertGreater(len(failure_calls), 0)


class TestGenerateSeedDataTask(unittest.TestCase):
    """Test generate_seed_data_task"""
    
    @patch('tasks.gan_tasks.gan_manager')
    def test_generate_seed_success(self, mock_manager):
        """Test successful seed data generation"""
        # Mock GANManager response
        mock_result = Mock(
            machine_id='cnc_machine_008',
            samples_generated=10000,
            file_path='/seed_data/cnc_machine_008_seed.csv',
            file_size_mb=2.5,
            timestamp='2024-01-15T10:40:00'
        )
        mock_manager.generate_seed_data.return_value = mock_result
        
        # Execute task
        result = generate_seed_data_task.run('cnc_machine_008', samples=10000)

        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['machine_id'], 'cnc_machine_008')
        self.assertEqual(result['samples_generated'], 10000)
        self.assertEqual(result['file_path'], '/seed_data/cnc_machine_008_seed.csv')
        self.assertEqual(result['file_size_mb'], 2.5)

        # Verify GANManager was called
        mock_manager.generate_seed_data.assert_called_once_with(
            machine_id='cnc_machine_008',
            samples=10000
        )
    
    @patch('tasks.gan_tasks.gan_manager')
    def test_generate_seed_machine_not_found(self, mock_manager):
        """Test seed generation with invalid machine"""
        mock_manager.generate_seed_data.side_effect = FileNotFoundError("Machine not found")
        
        with self.assertRaises(FileNotFoundError):
            generate_seed_data_task.run('invalid_machine', samples=5000)


class TestGetTaskStatus(unittest.TestCase):
    """Test get_task_status helper"""
    
    @patch('celery.result.AsyncResult')
    def test_get_task_status_success(self, mock_async_result):
        """Test getting status of successful task"""
        mock_result = Mock()
        mock_result.state = 'SUCCESS'
        mock_result.result = {'success': True, 'data': 'test'}
        mock_result.info = None
        mock_async_result.return_value = mock_result
        
        status = get_task_status('test-task-123')
        
        self.assertEqual(status['task_id'], 'test-task-123')
        self.assertEqual(status['status'], 'SUCCESS')
        self.assertEqual(status['result'], {'success': True, 'data': 'test'})
        self.assertIsNone(status['error'])
    
    @patch('celery.result.AsyncResult')
    def test_get_task_status_progress(self, mock_async_result):
        """Test getting status of in-progress task"""
        mock_result = Mock()
        mock_result.state = 'PROGRESS'
        mock_result.info = {'current': 50, 'total': 100, 'message': 'Processing'}
        mock_async_result.return_value = mock_result
        
        status = get_task_status('test-task-456')
        
        self.assertEqual(status['status'], 'PROGRESS')
        self.assertEqual(status['progress']['current'], 50)
        self.assertEqual(status['progress']['total'], 100)
    
    @patch('celery.result.AsyncResult')
    def test_get_task_status_failure(self, mock_async_result):
        """Test getting status of failed task"""
        mock_result = Mock()
        mock_result.state = 'FAILURE'
        mock_result.info = Exception("Task failed")
        mock_async_result.return_value = mock_result
        
        status = get_task_status('test-task-789')
        
        self.assertEqual(status['status'], 'FAILURE')
        self.assertIn('Task failed', status['error'])


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete GAN workflow"""
    
    @patch('tasks.gan_tasks.gan_manager')
    @patch('tasks.gan_tasks.broadcast_progress')
    def test_full_workflow(self, mock_broadcast, mock_manager):
        """Test complete workflow: seed → train → generate"""
        machine_id = 'cnc_machine_workflow_001'
        
        # Step 1: Generate seed data
        mock_manager.generate_seed_data.return_value = Mock(
            machine_id=machine_id,
            samples_generated=10000,
            file_path=f'/seed_data/{machine_id}_seed.csv',
            file_size_mb=2.0,
            timestamp='2024-01-15T10:00:00'
        )
        
        seed_result = generate_seed_data_task.run(machine_id, samples=10000)

        self.assertTrue(seed_result['success'])
        self.assertEqual(seed_result['samples_generated'], 10000)
        
        # Step 2: Train model
        mock_manager.train_tvae_model.return_value = Mock(
            model_path=f'/models/{machine_id}_model.pkl',
            loss=0.05,
            trained_at='2024-01-15T10:15:00',
            num_features=20
        )
        
        with patch.object(train_tvae_task, 'update_state'):
            train_result = train_tvae_task.run(machine_id, epochs=100)

            self.assertTrue(train_result['success'])
            self.assertEqual(train_result['final_loss'], 0.05)
        
        # Step 3: Generate synthetic data
        mock_manager.generate_synthetic_data.return_value = Mock(
            train_samples=35000,
            val_samples=7500,
            test_samples=7500,
            train_file='/data/train.csv',
            val_file='/data/val.csv',
            test_file='/data/test.csv',
            timestamp='2024-01-15T10:30:00'
        )
        
        with patch.object(generate_data_task, 'update_state'):
            gen_result = generate_data_task.run(machine_id, 35000, 7500, 7500)

            self.assertTrue(gen_result['success'])
            self.assertEqual(gen_result['train_samples'], 35000)
        
        # Verify all GANManager methods called
        mock_manager.generate_seed_data.assert_called_once()
        mock_manager.train_tvae_model.assert_called_once()
        mock_manager.generate_synthetic_data.assert_called_once()


if __name__ == '__main__':
    unittest.main()
