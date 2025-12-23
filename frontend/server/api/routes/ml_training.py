"""\
ML Training API Routes
Phase 3.7.8.1: Backend API Routes

Implements REST endpoints to start ML training jobs using existing training scripts
(via Celery tasks that will be implemented in Phase 3.7.8.2).

Key design:
- Start endpoints return Celery task_id (StartTrainingResponse)
- Task polling endpoints return the same TaskStatusResponse shape used by GAN
"""

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging

from celery_app import celery_app

from ..models.ml_training import TrainingRequest, BatchTrainingRequest, StartTrainingResponse
from api.models.gan import TaskStatusResponse, TaskProgress, TaskStatus

logger = logging.getLogger(__name__)

# Single router that covers:
# - /api/ml/train/*
# - /api/ml/tasks/*
router = APIRouter(prefix="/api/ml", tags=["ML Training"])


def _get_project_root() -> Path:
    """Resolve repository root regardless of current working directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "GAN").exists() and (parent / "frontend").exists():
            return parent
    return Path.cwd().resolve()


def _synthetic_train_path(machine_id: str) -> Path:
    return _get_project_root() / "GAN" / "data" / "synthetic" / machine_id / "train.parquet"


def _validate_has_synthetic_data(machine_id: str) -> None:
    data_path = _synthetic_train_path(machine_id)
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic data not found for {machine_id} (expected {data_path.as_posix()})",
        )


def _send_training_task(task_name: str, kwargs: Dict[str, Any]) -> str:
    """Send a Celery task by name.

    We avoid importing task functions here so Phase 3.7.8.1 stays independent from
    Phase 3.7.8.2 implementation details.
    """
    async_result = celery_app.send_task(task_name, kwargs=kwargs)
    return str(async_result.id)


# ============================================================================
# TRAINING START ENDPOINTS (5)
# ============================================================================


@router.post(
    "/train/classification",
    response_model=StartTrainingResponse,
    summary="Start classification training",
)
async def start_classification_training(request: TrainingRequest) -> StartTrainingResponse:
    _validate_has_synthetic_data(request.machine_id)
    task_id = _send_training_task(
        "ml.train_classification",
        {"machine_id": request.machine_id, "time_limit": request.time_limit or 900},
    )
    return StartTrainingResponse(
        success=True,
        task_id=task_id,
        machine_id=request.machine_id,
        message="Classification training task started",
    )


@router.post(
    "/train/regression",
    response_model=StartTrainingResponse,
    summary="Start regression (RUL) training",
)
async def start_regression_training(request: TrainingRequest) -> StartTrainingResponse:
    _validate_has_synthetic_data(request.machine_id)
    task_id = _send_training_task(
        "ml.train_regression",
        {"machine_id": request.machine_id, "time_limit": request.time_limit or 900},
    )
    return StartTrainingResponse(
        success=True,
        task_id=task_id,
        machine_id=request.machine_id,
        message="Regression training task started",
    )


@router.post(
    "/train/anomaly",
    response_model=StartTrainingResponse,
    summary="Start anomaly training",
)
async def start_anomaly_training(request: TrainingRequest) -> StartTrainingResponse:
    _validate_has_synthetic_data(request.machine_id)
    task_id = _send_training_task(
        "ml.train_anomaly",
        {"machine_id": request.machine_id, "time_limit": request.time_limit or 900},
    )
    return StartTrainingResponse(
        success=True,
        task_id=task_id,
        machine_id=request.machine_id,
        message="Anomaly training task started",
    )


@router.post(
    "/train/timeseries",
    response_model=StartTrainingResponse,
    summary="Start timeseries training",
)
async def start_timeseries_training(request: TrainingRequest) -> StartTrainingResponse:
    _validate_has_synthetic_data(request.machine_id)
    task_id = _send_training_task(
        "ml.train_timeseries",
        {"machine_id": request.machine_id, "time_limit": request.time_limit or 900},
    )
    return StartTrainingResponse(
        success=True,
        task_id=task_id,
        machine_id=request.machine_id,
        message="Timeseries training task started",
    )


@router.post(
    "/train/batch",
    response_model=StartTrainingResponse,
    summary="Start batch training (all 4 models)",
)
async def start_batch_training(request: BatchTrainingRequest) -> StartTrainingResponse:
    _validate_has_synthetic_data(request.machine_id)
    task_id = _send_training_task(
        "ml.train_batch",
        {
            "machine_id": request.machine_id,
            "model_types": request.model_types,
            "time_limit_per_model": request.time_limit_per_model or 900,
        },
    )
    return StartTrainingResponse(
        success=True,
        task_id=task_id,
        machine_id=request.machine_id,
        message="Batch training task started",
    )


@router.get(
    "/train/machines/available",
    summary="List machines with synthetic training data",
)
async def list_available_training_machines() -> Dict[str, Any]:
    base = _get_project_root() / "GAN" / "data" / "synthetic"
    machines: List[str] = []
    if base.exists():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if (child / "train.parquet").exists():
                machines.append(child.name)

    machines.sort()
    return {"total": len(machines), "machines": machines}


# ============================================================================
# TASK MONITORING ENDPOINTS (2)
# Same response shape as GAN TaskStatusResponse
# ============================================================================


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get ML training task status",
)
async def get_ml_task_status(task_id: str) -> TaskStatusResponse:
    try:
        task_result = celery_app.AsyncResult(task_id)

        response = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus(task_result.status),
            machine_id=None,
            progress=None,
            result=None,
            error=None,
            logs=None,
            started_at=None,
            completed_at=None,
        )

        if task_result.state == "PROGRESS":
            info = task_result.info
            if isinstance(info, dict):
                response.machine_id = info.get("machine_id")
                response.progress = TaskProgress(
                    current=info.get("current", 0),
                    total=info.get("total", 100),
                    progress_percent=info.get("progress", 0),
                    epoch=info.get("epoch"),
                    loss=info.get("loss"),
                    stage=info.get("stage"),
                    message=info.get("message"),
                )
                response.logs = info.get("logs")
                started_at = info.get("started_at")
                if isinstance(started_at, str):
                    try:
                        response.started_at = datetime.fromisoformat(started_at)
                    except Exception:
                        pass

        elif task_result.state == "SUCCESS":
            response.result = task_result.result
            if isinstance(task_result.result, dict) and task_result.result.get("logs"):
                response.logs = task_result.result.get("logs")
            response.completed_at = datetime.now()

        elif task_result.state == "FAILURE":
            response.error = str(task_result.info)
            response.completed_at = datetime.now()
            info = task_result.info
            if isinstance(info, dict) and info.get("logs"):
                response.logs = info.get("logs")

        return response

    except ValueError as e:
        # TaskStatus enum coercion failed (unexpected Celery state)
        logger.error(f"Unknown Celery status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unknown Celery task status: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to get ML task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.post(
    "/tasks/{task_id}/cancel",
    summary="Cancel ML training task (best-effort)",
)
async def cancel_ml_task(task_id: str) -> Dict[str, Any]:
    try:
        celery_app.control.revoke(task_id, terminate=False)
        return {
            "success": True,
            "task_id": task_id,
            "message": "Cancellation requested. If the task was pending, it will not run; if already running, it may continue until completion.",
        }
    except Exception as e:
        logger.error(f"Failed to cancel ML task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")
