"""
ML Training API Request/Response Models
Phase 3.7.8.1: Backend API Routes

Keeps ML training start endpoints lightweight:
- Start endpoints return a Celery task_id.
- Task polling uses the existing GAN TaskStatusResponse shape.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TrainingRequest(BaseModel):
    machine_id: str = Field(..., description="Machine identifier")
    time_limit: Optional[int] = Field(900, description="Training time limit in seconds")


class BatchTrainingRequest(BaseModel):
    machine_id: str
    model_types: List[str] = Field(
        default=["classification", "regression", "anomaly", "timeseries"],
        description="Models to train",
    )
    time_limit_per_model: Optional[int] = Field(900, description="Per-model time limit in seconds")


class StartTrainingResponse(BaseModel):
    success: bool
    machine_id: str
    task_id: str
    message: str
