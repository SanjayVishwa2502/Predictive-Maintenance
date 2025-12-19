"""
GAN Celery Tasks - Asynchronous Operations
Implements 3 main tasks with progress broadcasting to Redis
"""

from celery import Task
from celery.utils.log import get_task_logger
import redis.asyncio as redis
import redis as sync_redis
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from collections import deque
import contextlib
import io
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from celery_app import celery_app
from config import settings
from GAN.services.gan_manager import gan_manager

logger = get_task_logger(__name__)


def _safe_task_id(task: Task) -> str:
    """Best-effort task id for both real Celery runs and direct `.run()` unit tests."""
    try:
        req = getattr(task, 'request', None)
        task_id = getattr(req, 'id', None)
        if task_id:
            return str(task_id)
    except Exception:
        pass
    return 'local-task'


# ============================================================================
# PROGRESS BROADCASTING UTILITIES
# ============================================================================

def get_redis_pubsub():
    """Get synchronous Redis client for pub/sub (DB 2)"""
    return sync_redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=2,  # DB 2 for pub/sub
        decode_responses=True
    )


def broadcast_progress(
    task_id: str,
    machine_id: str,
    current: int,
    total: int,
    status: str,
    message: str,
    **metadata
):
    """
    Broadcast progress update to Redis pub/sub channel
    
    Args:
        task_id: Celery task ID
        machine_id: Machine identifier
        current: Current progress value
        total: Total progress value
        status: Task status (RUNNING, SUCCESS, FAILURE)
        message: Human-readable message
        **metadata: Additional metadata (epoch, loss, stage, etc.)
    """
    try:
        redis_client = get_redis_pubsub()
        
        progress_data = {
            'task_id': task_id,
            'machine_id': machine_id,
            'timestamp': datetime.now().isoformat(),
            'current': current,
            'total': total,
            'progress': round((current / total) * 100, 2) if total > 0 else 0,
            'status': status,
            'message': message,
            **metadata
        }
        
        # Publish to channel
        channel = f"gan:training:{task_id}"
        redis_client.publish(channel, json.dumps(progress_data))
        
        redis_client.close()
        
        logger.info(f"Progress broadcast: {message} ({progress_data['progress']}%)")
        
    except Exception as e:
        logger.error(f"Failed to broadcast progress: {e}")


# ============================================================================
# BASE PROGRESS TASK
# ============================================================================

class ProgressTask(Task):
    """Base task class with progress tracking"""
    
    def update_progress(
        self,
        machine_id: str,
        current: int,
        total: int,
        message: str,
        **metadata
    ):
        """
        Update task progress and broadcast to Redis
        
        Args:
            machine_id: Machine identifier
            current: Current progress value
            total: Total progress value
            message: Progress message
            **metadata: Additional metadata
        """
        # Update Celery task state
        self.update_state(
            state='PROGRESS',
            meta={
                'machine_id': machine_id,
                'current': current,
                'total': total,
                'progress': round((current / total) * 100, 2) if total > 0 else 0,
                'message': message,
                **metadata
            }
        )
        
        # Broadcast to Redis pub/sub
        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=current,
            total=total,
            status='RUNNING',
            message=message,
            **metadata
        )


# ============================================================================
# TASK 1: TRAIN TVAE MODEL (ASYNC WITH PROGRESS)
# ============================================================================

@celery_app.task(bind=True, base=ProgressTask, name='tasks.gan_tasks.train_tvae_task')
def train_tvae_task(self, machine_id: str, epochs: int = 300) -> Dict[str, Any]:
    """
    Train TVAE model asynchronously with progress broadcasting
    
    Args:
        machine_id: Machine identifier
        epochs: Number of training epochs (default: 300)
    
    Returns:
        dict: Training result with model path, training time, final loss
    
    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: Seed data not found
        RuntimeError: Training failed
    """
    logger.info(f"Starting TVAE training for {machine_id} ({epochs} epochs)")
    start_time = time.time()
    started_at_iso = datetime.now().isoformat()

    # Rolling log buffer for UI (best-effort stdout-like view).
    # Keep last ~200 lines to avoid large Celery meta payloads.
    log_lines: deque[str] = deque(maxlen=200)

    def append_log(line: str) -> None:
        text = (line or '').strip()
        if not text:
            return
        # Timestamp each line for readability.
        log_lines.append(f"{datetime.now().isoformat()} {text}")

    # Capture last parsed metrics from training output (tqdm writes to stderr).
    last_epoch: int = 0
    last_loss: Optional[float] = None
    last_emit_ts: float = 0.0

    # Patterns seen in tqdm / training logs.
    # We keep these permissive because tqdm formatting differs by environment.
    epoch_total_re = re.compile(r"(?:Epoch\s*)?(?P<epoch>\d{1,6})\s*/\s*(?P<total>\d{1,6})")
    # Accept negative/positive losses and scientific notation.
    # Example tqdm fragment: "Loss: -3.599:  18%|..."
    loss_re = re.compile(r"(?:loss|Loss)\s*[:=]\s*(?P<loss>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)")
    percent_re = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)%")

    def maybe_emit_progress(epoch_val: Optional[int], loss_val: Optional[float], raw_text: str):
        nonlocal last_epoch, last_loss, last_emit_ts

        now = time.time()
        if loss_val is not None:
            last_loss = loss_val

        # Only emit when epoch increases, or at a low frequency if we only have percent updates.
        if epoch_val is None:
            if now - last_emit_ts < 1.5:
                return
            # No reliable epoch; emit an indeterminate-but-monotonic update if we can.
            epoch_val = last_epoch
        if epoch_val <= last_epoch and (now - last_emit_ts) < 1.5:
            return

        last_epoch = max(last_epoch, epoch_val)
        last_emit_ts = now

        msg_epoch = f"Epoch {last_epoch}/{epochs}"
        if last_loss is not None:
            message = f"{msg_epoch}, Loss: {last_loss:.6f}"
        else:
            message = msg_epoch

        append_log(message)

        self.update_progress(
            machine_id=machine_id,
            current=last_epoch,
            total=epochs,
            message=message,
            epoch=last_epoch,
            loss=last_loss,
            stage='training',
            started_at=started_at_iso,
            logs="\n".join(log_lines)
        )

    class _ProgressCapture(io.TextIOBase):
        def __init__(self, original: io.TextIOBase):
            super().__init__()
            self._original = original
            self._buf = ""

        def write(self, s: str) -> int:
            # Forward to original stream for normal logging/visibility.
            try:
                self._original.write(s)
                self._original.flush()
            except Exception:
                pass

            self._buf += s

            # tqdm updates frequently with carriage returns; treat both as delimiters.
            while True:
                split_at = None
                for delim in ("\r", "\n"):
                    idx = self._buf.find(delim)
                    if idx != -1 and (split_at is None or idx < split_at):
                        split_at = idx
                if split_at is None:
                    break

                line = self._buf[:split_at]
                self._buf = self._buf[split_at + 1:]
                if not line.strip():
                    continue

                # Parse epoch/total
                epoch_val: Optional[int] = None
                loss_val: Optional[float] = None

                m_epoch = epoch_total_re.search(line)
                if m_epoch:
                    try:
                        total = int(m_epoch.group('total'))
                        epoch_candidate = int(m_epoch.group('epoch'))
                        if total == epochs and 0 <= epoch_candidate <= epochs:
                            epoch_val = epoch_candidate
                    except Exception:
                        pass

                m_loss = loss_re.search(line)
                if m_loss:
                    try:
                        loss_val = float(m_loss.group('loss'))
                    except Exception:
                        pass

                # If no epoch but percent is present, approximate an epoch.
                if epoch_val is None:
                    m_pct = percent_re.search(line)
                    if m_pct:
                        try:
                            pct = float(m_pct.group('pct'))
                            if 0.0 <= pct <= 100.0:
                                epoch_val = int(round((pct / 100.0) * epochs))
                        except Exception:
                            pass

                if epoch_val is not None or loss_val is not None:
                    maybe_emit_progress(epoch_val=epoch_val, loss_val=loss_val, raw_text=line)

            return len(s)

    
    try:
        # Initial progress broadcast
        append_log(f"Initializing TVAE training for {machine_id}")
        self.update_progress(
            machine_id=machine_id,
            current=0,
            total=epochs,
            message=f"Initializing TVAE training for {machine_id}",
            epoch=0,
            loss=None,
            stage='initialization',
            started_at=started_at_iso,
            logs="\n".join(log_lines)
        )
        
        # Call GANManager to train model
        # Note: This will run the training script and we'll parse output
        logger.info(f"Calling GANManager.train_tvae_model({machine_id}, {epochs})")

        # Capture tqdm/stderr progress from SDV/CTGAN training and emit PROGRESS updates.
        with contextlib.redirect_stderr(_ProgressCapture(sys.stderr)), contextlib.redirect_stdout(_ProgressCapture(sys.stdout)):
            result = gan_manager.train_tvae_model(
                machine_id=machine_id,
                epochs=epochs
            )
        
        training_time = time.time() - start_time
        
        # Broadcast completion
        append_log(f"Training completed in {training_time:.1f}s")
        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=epochs,
            total=epochs,
            status='SUCCESS',
            message=f"Training completed in {training_time:.1f}s",
            epoch=epochs,
            loss=last_loss if last_loss is not None else result.loss,
            stage='completed',
            started_at=started_at_iso,
            logs="\n".join(log_lines)
        )
        
        logger.info(f"Training completed: {machine_id} in {training_time:.1f}s")
        
        return {
            'success': True,
            'machine_id': machine_id,
            'epochs': epochs,
            'model_path': result.model_path,
            'training_time_seconds': training_time,
            'final_loss': last_loss if last_loss is not None else result.loss,
            'trained_at': result.trained_at,
            'num_features': result.num_features,
            'logs': "\n".join(log_lines)
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=0,
            total=epochs,
            status='FAILURE',
            message=f"Validation error: {str(e)}",
            stage='failed'
        )
        raise
        
    except FileNotFoundError as e:
        logger.error(f"Seed data not found: {e}")
        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=0,
            total=epochs,
            status='FAILURE',
            message=f"Seed data not found: {str(e)}",
            stage='failed'
        )
        raise
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=0,
            total=epochs,
            status='FAILURE',
            message=f"Training failed: {str(e)}",
            stage='failed'
        )
        raise RuntimeError(f"Training failed: {str(e)}")


# ============================================================================
# TASK 2: GENERATE SYNTHETIC DATA (ASYNC WITH PROGRESS)
# ============================================================================

@celery_app.task(bind=True, base=ProgressTask, name='tasks.gan_tasks.generate_data_task')
def generate_data_task(
    self,
    machine_id: str,
    train_samples: int = 35000,
    val_samples: int = 7500,
    test_samples: int = 7500
) -> Dict[str, Any]:
    """
    Generate synthetic data asynchronously with progress tracking
    
    Args:
        machine_id: Machine identifier
        train_samples: Training set samples
        val_samples: Validation set samples
        test_samples: Test set samples
    
    Returns:
        dict: Generation result with file paths
    
    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: Model not found
        RuntimeError: Generation failed
    """
    logger.info(f"Starting synthetic data generation for {machine_id}")
    start_time = time.time()
    total_samples = train_samples + val_samples + test_samples
    
    try:
        # Stage 1: Initialization
        self.update_progress(
            machine_id=machine_id,
            current=0,
            total=total_samples,
            message=f"Initializing synthetic data generation",
            stage='initialization'
        )
        
        # Stage 2: Generating train set
        self.update_progress(
            machine_id=machine_id,
            current=0,
            total=total_samples,
            message=f"Generating training set ({train_samples} samples)",
            stage='train_generation'
        )
        
        # Stage 3: Call GANManager
        result = gan_manager.generate_synthetic_data(
            machine_id=machine_id,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples
        )
        
        generation_time = time.time() - start_time
        
        # Stage 4: Completion
        self.update_progress(
            machine_id=machine_id,
            current=total_samples,
            total=total_samples,
            message=f"Generation completed in {generation_time:.1f}s",
            stage='completed'
        )
        
        logger.info(f"Synthetic data generated: {machine_id} in {generation_time:.1f}s")
        
        return {
            'success': True,
            'machine_id': machine_id,
            'train_samples': result.train_samples,
            'val_samples': result.val_samples,
            'test_samples': result.test_samples,
            'train_file': result.train_file,
            'val_file': result.val_file,
            'test_file': result.test_file,
            'generation_time_seconds': generation_time,
            'timestamp': result.timestamp
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        broadcast_progress(
            task_id=self.request.id,
            machine_id=machine_id,
            current=0,
            total=total_samples,
            status='FAILURE',
            message=f"Validation error: {str(e)}",
            stage='failed'
        )
        raise
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        broadcast_progress(
            task_id=self.request.id,
            machine_id=machine_id,
            current=0,
            total=total_samples,
            status='FAILURE',
            message=f"Model not found: {str(e)}",
            stage='failed'
        )
        raise
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        broadcast_progress(
            task_id=self.request.id,
            machine_id=machine_id,
            current=0,
            total=total_samples,
            status='FAILURE',
            message=f"Generation failed: {str(e)}",
            stage='failed'
        )
        raise RuntimeError(f"Generation failed: {str(e)}")


# ============================================================================
# TASK 3: GENERATE SEED DATA (SIMPLE WRAPPER)
# ============================================================================

@celery_app.task(bind=True, name='tasks.gan_tasks.generate_seed_data_task')
def generate_seed_data_task(
    self,
    machine_id: str,
    samples: int = 10000
) -> Dict[str, Any]:
    """
    Generate seed data (simple wrapper, no streaming needed)
    
    Args:
        machine_id: Machine identifier
        samples: Number of samples (default: 10000)
    
    Returns:
        dict: Seed generation result
    
    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: Machine metadata not found
        RuntimeError: Generation failed
    """
    logger.info(f"Starting seed data generation for {machine_id} ({samples} samples)")
    start_time = time.time()
    
    try:
        # Call GANManager
        result = gan_manager.generate_seed_data(
            machine_id=machine_id,
            samples=samples
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"Seed data generated: {machine_id} in {generation_time:.1f}s")
        
        return {
            'success': True,
            'machine_id': result.machine_id,
            'samples_generated': result.samples_generated,
            'file_path': result.file_path,
            'file_size_mb': result.file_size_mb,
            'generation_time_seconds': generation_time,
            'timestamp': result.timestamp
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
        
    except FileNotFoundError as e:
        logger.error(f"Machine not found: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Seed generation failed: {e}")
        raise RuntimeError(f"Seed generation failed: {str(e)}")


# ============================================================================
# TASK STATUS HELPER
# ============================================================================

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of a Celery task
    
    Args:
        task_id: Celery task ID
    
    Returns:
        dict: Task status information
    """
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id, app=celery_app)
    
    status_info = {
        'task_id': task_id,
        'status': task_result.state,
        'result': None,
        'error': None,
        'progress': None
    }
    
    if task_result.state == 'PROGRESS':
        status_info['progress'] = task_result.info
    elif task_result.state == 'SUCCESS':
        status_info['result'] = task_result.result
    elif task_result.state == 'FAILURE':
        status_info['error'] = str(task_result.info)
    
    return status_info


# ============================================================================
# TASK REGISTRATION VERIFICATION
# ============================================================================

if __name__ == '__main__':
    """Verify task registration"""
    print("Registered Celery Tasks:")
    print(f"1. {train_tvae_task.name}")
    print(f"2. {generate_data_task.name}")
    print(f"3. {generate_seed_data_task.name}")
    print("\nAll GAN tasks registered successfully!")
