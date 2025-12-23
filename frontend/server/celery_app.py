"""
Celery application configuration for async task processing.
Phase 3.7.1.4: Celery Worker Setup (Day 7)

Celery is used for:
- Long-running GAN training tasks
- ML model predictions (batch processing)
- LLM explanation generation
- Data preprocessing pipelines
"""
import logging
import sys
from pathlib import Path
from celery import Celery

# Add project root to path for imports (keeps GAN namespace imports reliable)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create Celery application instance
celery_app = Celery(
    "predictive_maintenance",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Celery Configuration
celery_app.conf.update(
    # Task Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task Execution
    task_acks_late=settings.CELERY_TASK_ACKS_LATE,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks to prevent memory leaks
    
    # Task Time Limits
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 minutes soft limit (sends exception)
    
    # Result Backend Settings
    result_expires=86400,  # Results expire after 24 hours
    result_persistent=True,  # Persist results to disk
    
    # Task Result Extended
    result_extended=True,  # Include task args, kwargs, and traceback in result
    
    # Task Routes (organize tasks by queue) - DISABLED FOR TESTING
    # Issue: Tasks weren't being queued when routes were enabled
    # Will re-enable after verifying basic functionality works
    # task_routes={
    #     "tasks.gan.*": {"queue": "gan"},
    #     "tasks.ml.*": {"queue": "ml"},
    #     "tasks.llm.*": {"queue": "llm"},
    #     "tasks.test_task.*": {"queue": "default"},
    # },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Manually import tasks to ensure they're registered
# Auto-discovery doesn't work reliably on Windows
import tasks.test_task  # noqa
import tasks.gan_tasks  # noqa
import tasks.ml_training_tasks  # noqa

logger.info(f"Celery app configured with broker: {settings.CELERY_BROKER_URL}")
logger.info(f"Result backend: {settings.CELERY_RESULT_BACKEND}")
logger.info(f"Worker concurrency: {settings.CELERY_WORKER_CONCURRENCY}")


# Celery Beat Schedule (for periodic tasks - Phase 3.7.5)
# celery_app.conf.beat_schedule = {
#     "health-check-every-hour": {
#         "task": "tasks.health.check_system_health",
#         "schedule": 3600.0,  # Every hour
#     },
# }


if __name__ == "__main__":
    celery_app.start()
