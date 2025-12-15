"""
Celery tasks package.
Phase 3.7.1.4: Celery Worker Setup

Tasks are organized by domain:
- test_task: Simple test tasks for verification
- gan: GAN training and synthetic data generation
- ml: ML model training and predictions
- llm: LLM explanation generation
"""
from celery import Task

# Custom task base class for logging and error handling
class LoggingTask(Task):
    """Base task with logging and error handling."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Log successful task execution."""
        from celery.utils.log import get_task_logger
        logger = get_task_logger(__name__)
        logger.info(f"Task {self.name}[{task_id}] succeeded with result: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log failed task execution."""
        from celery.utils.log import get_task_logger
        logger = get_task_logger(__name__)
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}")
        logger.error(f"Traceback: {einfo}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log task retry."""
        from celery.utils.log import get_task_logger
        logger = get_task_logger(__name__)
        logger.warning(f"Task {self.name}[{task_id}] retrying due to: {exc}")
