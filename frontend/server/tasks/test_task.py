"""
Test tasks for Celery worker verification.
Phase 3.7.1.4: Celery Worker Setup (Day 7)

Simple tasks to verify:
- Celery worker is running
- Tasks can be queued and executed
- Results are stored in Redis
"""
import time
import logging
from celery import shared_task

from tasks import LoggingTask

logger = logging.getLogger(__name__)


@shared_task(
    base=LoggingTask,
    name="tasks.test_task.add",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def add(self, x: int, y: int) -> int:
    """
    Simple addition task for testing Celery.
    
    Args:
        x: First number
        y: Second number
    
    Returns:
        Sum of x and y
    
    Example:
        # Synchronous execution
        result = add(4, 5)
        
        # Async execution
        task = add.delay(4, 5)
        result = task.get()  # Blocks until result is ready
    """
    logger.info(f"Executing add task: {x} + {y}")
    result = x + y
    logger.info(f"Add task result: {result}")
    return result


@shared_task(
    base=LoggingTask,
    name="tasks.test_task.multiply",
    bind=True,
)
def multiply(self, x: int, y: int) -> int:
    """
    Simple multiplication task for testing.
    
    Args:
        x: First number
        y: Second number
    
    Returns:
        Product of x and y
    """
    logger.info(f"Executing multiply task: {x} * {y}")
    result = x * y
    logger.info(f"Multiply task result: {result}")
    return result


@shared_task(
    base=LoggingTask,
    name="tasks.test_task.long_running",
    bind=True,
)
def long_running(self, duration: int = 5) -> dict:
    """
    Long-running task to test async execution and status updates.
    
    Args:
        duration: Number of seconds to run (default: 5)
    
    Returns:
        Dictionary with task status and duration
    """
    logger.info(f"Starting long-running task for {duration} seconds")
    
    for i in range(duration):
        # Update task state with progress
        self.update_state(
            state="PROGRESS",
            meta={
                "current": i + 1,
                "total": duration,
                "status": f"Processing step {i + 1} of {duration}",
            }
        )
        logger.info(f"Long-running task progress: {i + 1}/{duration}")
        time.sleep(1)
    
    result = {
        "status": "completed",
        "duration": duration,
        "message": f"Task ran for {duration} seconds"
    }
    
    logger.info(f"Long-running task completed: {result}")
    return result


@shared_task(
    base=LoggingTask,
    name="tasks.test_task.failing_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
)
def failing_task(self, should_fail: bool = True) -> dict:
    """
    Task that fails intentionally to test error handling and retries.
    
    Args:
        should_fail: Whether to raise an exception (default: True)
    
    Returns:
        Success message if should_fail is False
    
    Raises:
        ValueError: If should_fail is True
    """
    if should_fail:
        logger.error("Failing task raising exception")
        raise ValueError("This task is configured to fail for testing")
    
    logger.info("Failing task succeeded (should_fail=False)")
    return {"status": "success", "message": "Task completed without error"}
