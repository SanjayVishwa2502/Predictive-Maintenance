"""\
ML Training Celery Tasks - Asynchronous Model Training
PHASE 3.7.8.2: Celery Training Tasks (Days 3-4)

Implements 5 tasks:
- ml.train_classification
- ml.train_regression
- ml.train_anomaly
- ml.train_timeseries
- ml.train_batch

Key design goals (mirrors tasks.gan_tasks):
- Progress updates via Celery meta (state=PROGRESS)
- Progress broadcasting via Redis pub/sub (DB 2)
- Structured logging includes task_id + machine_id
- Training scripts remain the single source of truth
- Best-effort MLflow run tagging + artifact validation
"""

from __future__ import annotations

from celery import Task
from celery.utils.log import get_task_logger
import json
import os
import subprocess
import sys
import time
import threading
from collections import deque
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import redis as sync_redis

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from celery_app import celery_app
from config import settings

logger = get_task_logger(__name__)


def _safe_task_id(task: Task) -> str:
    """Best-effort task id for both real Celery runs and direct `.run()` calls."""
    try:
        req = getattr(task, "request", None)
        task_id = getattr(req, "id", None)
        if task_id:
            return str(task_id)
    except Exception:
        pass
    return "local-task"


# ============================================================================
# REDIS PUB/SUB PROGRESS BROADCASTING (SYNC CLIENT, DB 2)
# ============================================================================

def get_redis_pubsub():
    """Get synchronous Redis client for pub/sub (DB 2)."""
    return sync_redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=2,
        decode_responses=True,
    )


def broadcast_progress(
    task_id: str,
    machine_id: str,
    current: int,
    total: int,
    status: str,
    message: str,
    **metadata: Any,
) -> None:
    """Broadcast progress update to Redis pub/sub channel `ml:training:{task_id}`."""
    try:
        redis_client = get_redis_pubsub()

        progress_data: Dict[str, Any] = {
            "task_id": task_id,
            "machine_id": machine_id,
            "timestamp": datetime.now().isoformat(),
            "current": current,
            "total": total,
            "progress": round((current / total) * 100, 2) if total > 0 else 0,
            "status": status,
            "message": message,
            **metadata,
        }

        channel = f"ml:training:{task_id}"
        redis_client.publish(channel, json.dumps(progress_data))
        redis_client.close()

        logger.info(f"Progress broadcast: {message} ({progress_data['progress']}%)")
    except Exception as exc:
        logger.error(f"Failed to broadcast ML progress: {exc}")


# ============================================================================
# BASE PROGRESS TASK
# ============================================================================


def _tail_lines(text: str, max_lines: int = 200, max_chars: int = 20000) -> str:
    if not text:
        return ""
    lines = text.splitlines()[-max_lines:]
    tailed = "\n".join(lines)
    if len(tailed) > max_chars:
        return tailed[-max_chars:]
    return tailed


def _validate_has_synthetic_data(machine_id: str) -> Path:
    synthetic_dir = PROJECT_ROOT / "GAN" / "data" / "synthetic" / machine_id
    train_path = synthetic_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Synthetic data not found: {train_path.as_posix()}")
    return synthetic_dir


def _synthetic_has_column(machine_id: str, column: str) -> bool:
    """Best-effort check for a column in synthetic train.parquet.

    We prefer reading Parquet schema (fast) and fall back to reading the file
    via pandas when necessary.
    """
    train_path = PROJECT_ROOT / "GAN" / "data" / "synthetic" / machine_id / "train.parquet"
    if not train_path.exists():
        return False

    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.read_schema(train_path)
        return column in schema.names
    except Exception:
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(train_path)
            return column in df.columns
        except Exception:
            return False


def _model_dir(model_type: str, machine_id: str) -> Path:
    return PROJECT_ROOT / "ml_models" / "models" / model_type / machine_id


def _report_path(model_type: str, machine_id: str) -> Path:
    perf_dir = PROJECT_ROOT / "ml_models" / "reports" / "performance_metrics"
    if model_type == "classification":
        return perf_dir / f"{machine_id}_classification_report.json"
    if model_type == "regression":
        return perf_dir / f"{machine_id}_regression_report.json"
    if model_type == "anomaly":
        return perf_dir / f"{machine_id}_comprehensive_anomaly_report.json"
    if model_type == "timeseries":
        return perf_dir / f"{machine_id}_timeseries_report.json"
    raise ValueError(f"Unknown model_type: {model_type}")


def _validate_artifacts(model_type: str, machine_id: str) -> Tuple[Path, Path]:
    model_path = _model_dir(model_type, machine_id)
    report_path = _report_path(model_type, machine_id)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory missing: {model_path.as_posix()}")

    # Ensure non-empty model directory (best-effort)
    has_any_files = any(p.is_file() for p in model_path.rglob("*"))
    if not has_any_files:
        raise FileNotFoundError(f"Model directory is empty: {model_path.as_posix()}")

    if not report_path.exists():
        raise FileNotFoundError(f"Training report missing: {report_path.as_posix()}")

    return model_path, report_path


def _maybe_log_mlflow(
    *,
    task_id: str,
    machine_id: str,
    model_type: str,
    report_path: Optional[Path] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Best-effort MLflow tracking at the task level.

    The training scripts already log their own MLflow runs; this creates a small
    high-level run tagged with the Celery task_id, primarily for traceability.
    """
    try:
        import mlflow  # type: ignore

        mlflow.set_experiment("ML_Training_Tasks")
        with mlflow.start_run(run_name=f"{machine_id}_{model_type}_{task_id}"):
            mlflow.set_tag("celery_task_id", task_id)
            mlflow.set_tag("machine_id", machine_id)
            mlflow.set_tag("model_type", model_type)
            if extra_metrics:
                for key, val in extra_metrics.items():
                    mlflow.log_metric(key, float(val))
            if report_path and report_path.exists():
                # Log the JSON report as an artifact (safe + small)
                mlflow.log_artifact(str(report_path))
    except Exception:
        # Never fail the task due to MLflow tracking issues.
        return


class MLProgressTask(Task):
    """Base task class with progress tracking (Celery meta + Redis pub/sub)."""

    def update_progress(
        self,
        *,
        machine_id: str,
        current: int,
        total: int,
        status: str,
        message: str,
        logs: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        progress_pct = round((current / total) * 100, 2) if total > 0 else 0

        meta: Dict[str, Any] = {
            "machine_id": machine_id,
            "current": current,
            "total": total,
            "progress": progress_pct,
            "message": message,
            **metadata,
        }
        if logs is not None:
            meta["logs"] = logs

        # Celery polling endpoint uses this meta for PROGRESS states
        self.update_state(state="PROGRESS", meta=meta)

        broadcast_progress(
            task_id=_safe_task_id(self),
            machine_id=machine_id,
            current=current,
            total=total,
            status=status,
            message=message,
            **metadata,
            **({"logs": logs} if logs is not None else {}),
        )


def _run_script_with_progress(
    *,
    task: MLProgressTask,
    machine_id: str,
    model_type: str,
    script_path: Path,
    args: List[str],
    started_at: str,
    est_seconds: int,
    progress_start: int = 20,
    progress_end: int = 99,
    update_every_seconds: int = 5,
) -> Tuple[int, str, str]:
    """Run a training script while emitting time-based progress updates."""

    # Force UTF-8 for child scripts.
    # Some training scripts print emoji/status glyphs; under Windows services/legacy
    # consoles the default encoding can be cp1252, which can crash with UnicodeEncodeError.
    cmd = [sys.executable, "-X", "utf8", str(script_path), *args]

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    logger.info(
        f"[task_id={_safe_task_id(task)} machine_id={machine_id}] "
        f"Launching {model_type} script: {' '.join(cmd)}"
    )

    # NOTE: Use a single pipe (stdout) and merge stderr into it.
    # This prevents deadlocks where a child process blocks because stderr fills up.
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    output_lock = threading.Lock()
    output_lines: deque[str] = deque(maxlen=4000)

    def _drain_stdout() -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                with output_lock:
                    output_lines.append(line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    drain_thread = threading.Thread(
        target=_drain_stdout,
        name=f"ml-drain-{machine_id}-{model_type}",
        daemon=True,
    )
    drain_thread.start()

    start_ts = time.time()
    last_emit = 0.0
    last_current = progress_start

    while proc.poll() is None:
        now = time.time()
        if now - last_emit >= update_every_seconds:
            elapsed = max(0.0, now - start_ts)

            denom = max(30.0, float(est_seconds) if est_seconds else 30.0)

            # Progress model:
            # - linear ramp up to ~90% by the estimated duration
            # - then keep creeping toward progress_end for up to ~5x est_seconds
            # This avoids the UI getting stuck at 90% if a script runs longer than expected.
            mid = min(90, max(progress_start, progress_end - 1))
            if elapsed <= denom:
                frac = elapsed / denom
                current_f = float(progress_start) + frac * float(mid - progress_start)
            else:
                extra = min(1.0, (elapsed - denom) / (denom * 5.0))
                current_f = float(mid) + extra * float(progress_end - mid)

            current = int(round(current_f))
            current = max(progress_start, min(progress_end, current))
            current = max(last_current, current)

            with output_lock:
                combined = "".join(output_lines)
            live_logs = _tail_lines(combined)

            task.update_progress(
                machine_id=machine_id,
                current=current,
                total=100,
                status="RUNNING",
                message=f"Training {model_type} model...",
                stage="training",
                model_type=model_type,
                started_at=started_at,
                elapsed_seconds=int(elapsed),
                logs=live_logs,
            )
            last_current = current
            last_emit = now

        time.sleep(0.5)

    # Ensure output thread has a chance to drain any remaining buffered lines.
    try:
        drain_thread.join(timeout=2.0)
    except Exception:
        pass

    with output_lock:
        stdout = "".join(output_lines)

    return int(proc.returncode or 0), stdout or "", ""


# ============================================================================
# TRAINING TASKS (4 single + 1 batch)
# ============================================================================


def _train_classification_impl(task: MLProgressTask, machine_id: str, time_limit: int = 900) -> Dict[str, Any]:
    model_type = "classification"
    task_id = _safe_task_id(task)
    started_at = datetime.now().isoformat()

    try:
        task.update_progress(
            machine_id=machine_id,
            current=10,
            total=100,
            status="RUNNING",
            message="Validating synthetic data...",
            stage="validating",
            model_type=model_type,
            started_at=started_at,
        )
        _validate_has_synthetic_data(machine_id)

        script = PROJECT_ROOT / "ml_models" / "scripts" / "training" / "train_classification_fast.py"
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script.as_posix()}")

        task.update_progress(
            machine_id=machine_id,
            current=20,
            total=100,
            status="RUNNING",
            message="Launching training script...",
            stage="launching",
            model_type=model_type,
            started_at=started_at,
        )

        code, out, err = _run_script_with_progress(
            task=task,
            machine_id=machine_id,
            model_type=model_type,
            script_path=script,
            args=["--machine_id", machine_id, "--time_limit", str(int(time_limit))],
            started_at=started_at,
            est_seconds=int(time_limit) if time_limit else 900,
        )

        logs = _tail_lines(out + ("\n" + err if err else ""))

        if code != 0:
            task.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="FAILURE",
                message="Training failed",
                stage="failed",
                model_type=model_type,
                started_at=started_at,
                error=_tail_lines(err) or f"Exit code {code}",
                logs=logs,
            )
            raise RuntimeError(_tail_lines(err) or f"Training failed with exit code {code}")

        task.update_progress(
            machine_id=machine_id,
            current=95,
            total=100,
            status="RUNNING",
            message="Validating model artifacts...",
            stage="validating_artifacts",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        model_path, report_path = _validate_artifacts(model_type, machine_id)

        report_data: Optional[Dict[str, Any]] = None
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_data = None

        _maybe_log_mlflow(task_id=task_id, machine_id=machine_id, model_type=model_type, report_path=report_path)

        task.update_progress(
            machine_id=machine_id,
            current=100,
            total=100,
            status="SUCCESS",
            message="Training completed successfully",
            stage="completed",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        return {
            "status": "completed",
            "machine_id": machine_id,
            "model_type": model_type,
            "model_dir": model_path.as_posix(),
            "report_path": report_path.as_posix(),
            "report": report_data,
            "logs": logs,
        }

    except Exception as exc:
        logger.exception(f"[task_id={task_id} machine_id={machine_id}] Classification training failed: {exc}")
        raise


@celery_app.task(bind=True, base=MLProgressTask, name="ml.train_classification")
def train_classification(self: MLProgressTask, machine_id: str, time_limit: int = 900) -> Dict[str, Any]:
    return _train_classification_impl(self, machine_id=machine_id, time_limit=time_limit)


def _train_regression_impl(task: MLProgressTask, machine_id: str, time_limit: int = 900) -> Dict[str, Any]:
    model_type = "regression"
    task_id = _safe_task_id(task)
    started_at = datetime.now().isoformat()

    try:
        task.update_progress(
            machine_id=machine_id,
            current=10,
            total=100,
            status="RUNNING",
            message="Validating synthetic data...",
            stage="validating",
            model_type=model_type,
            started_at=started_at,
        )
        _validate_has_synthetic_data(machine_id)

        if not _synthetic_has_column(machine_id, "rul"):
            message = (
                "⚠️ RUL column not found in synthetic data; skipping regression (RUL) training. "
                "Proceed with classification/anomaly/timeseries instead."
            )
            logger.warning(f"[task_id={task_id} machine_id={machine_id}] {message}")
            task.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="SUCCESS",
                message=message,
                stage="skipped",
                model_type=model_type,
                started_at=started_at,
                logs=message,
            )
            return {
                "status": "skipped",
                "machine_id": machine_id,
                "model_type": model_type,
                "report": {"skipped": True, "reason": message},
                "logs": message,
            }

        script = PROJECT_ROOT / "ml_models" / "scripts" / "training" / "train_regression_fast.py"
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script.as_posix()}")

        task.update_progress(
            machine_id=machine_id,
            current=20,
            total=100,
            status="RUNNING",
            message="Launching training script...",
            stage="launching",
            model_type=model_type,
            started_at=started_at,
        )

        code, out, err = _run_script_with_progress(
            task=task,
            machine_id=machine_id,
            model_type=model_type,
            script_path=script,
            args=["--machine_id", machine_id, "--time_limit", str(int(time_limit))],
            started_at=started_at,
            est_seconds=int(time_limit) if time_limit else 900,
        )

        logs = _tail_lines(out + ("\n" + err if err else ""))

        if code != 0:
            task.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="FAILURE",
                message="Training failed",
                stage="failed",
                model_type=model_type,
                started_at=started_at,
                error=_tail_lines(err) or f"Exit code {code}",
                logs=logs,
            )
            raise RuntimeError(_tail_lines(err) or f"Training failed with exit code {code}")

        task.update_progress(
            machine_id=machine_id,
            current=95,
            total=100,
            status="RUNNING",
            message="Validating model artifacts...",
            stage="validating_artifacts",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        model_path, report_path = _validate_artifacts(model_type, machine_id)

        report_data: Optional[Dict[str, Any]] = None
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_data = None

        _maybe_log_mlflow(task_id=task_id, machine_id=machine_id, model_type=model_type, report_path=report_path)

        task.update_progress(
            machine_id=machine_id,
            current=100,
            total=100,
            status="SUCCESS",
            message="Training completed successfully",
            stage="completed",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        return {
            "status": "completed",
            "machine_id": machine_id,
            "model_type": model_type,
            "model_dir": model_path.as_posix(),
            "report_path": report_path.as_posix(),
            "report": report_data,
            "logs": logs,
        }

    except Exception as exc:
        logger.exception(f"[task_id={task_id} machine_id={machine_id}] Regression training failed: {exc}")
        raise


@celery_app.task(bind=True, base=MLProgressTask, name="ml.train_regression")
def train_regression(self: MLProgressTask, machine_id: str, time_limit: int = 900) -> Dict[str, Any]:
    return _train_regression_impl(self, machine_id=machine_id, time_limit=time_limit)


def _train_anomaly_impl(
    task: MLProgressTask,
    machine_id: str,
    contamination: float = 0.1,
    time_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Train anomaly detection models.

    Note: API Phase 3.7.8.1 currently sends `time_limit`; we accept it here for
    compatibility, but the script uses `contamination`.
    """
    model_type = "anomaly"
    task_id = _safe_task_id(task)
    started_at = datetime.now().isoformat()

    # Provide a reasonable progress duration even though anomaly training is fast.
    est_seconds = int(time_limit) if isinstance(time_limit, int) and time_limit > 0 else 60
    est_seconds = max(30, min(300, est_seconds))

    try:
        task.update_progress(
            machine_id=machine_id,
            current=10,
            total=100,
            status="RUNNING",
            message="Validating synthetic data...",
            stage="validating",
            model_type=model_type,
            started_at=started_at,
        )
        _validate_has_synthetic_data(machine_id)

        script = PROJECT_ROOT / "ml_models" / "scripts" / "training" / "train_anomaly_comprehensive.py"
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script.as_posix()}")

        task.update_progress(
            machine_id=machine_id,
            current=20,
            total=100,
            status="RUNNING",
            message="Launching training script...",
            stage="launching",
            model_type=model_type,
            started_at=started_at,
        )

        code, out, err = _run_script_with_progress(
            task=task,
            machine_id=machine_id,
            model_type=model_type,
            script_path=script,
            args=["--machine_id", machine_id, "--contamination", str(float(contamination))],
            started_at=started_at,
            est_seconds=est_seconds,
        )

        logs = _tail_lines(out + ("\n" + err if err else ""))

        if code != 0:
            task.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="FAILURE",
                message="Training failed",
                stage="failed",
                model_type=model_type,
                started_at=started_at,
                error=_tail_lines(err) or f"Exit code {code}",
                logs=logs,
            )
            raise RuntimeError(_tail_lines(err) or f"Training failed with exit code {code}")

        task.update_progress(
            machine_id=machine_id,
            current=95,
            total=100,
            status="RUNNING",
            message="Validating model artifacts...",
            stage="validating_artifacts",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        model_path, report_path = _validate_artifacts(model_type, machine_id)

        report_data: Optional[Dict[str, Any]] = None
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_data = None

        _maybe_log_mlflow(task_id=task_id, machine_id=machine_id, model_type=model_type, report_path=report_path)

        task.update_progress(
            machine_id=machine_id,
            current=100,
            total=100,
            status="SUCCESS",
            message="Training completed successfully",
            stage="completed",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        return {
            "status": "completed",
            "machine_id": machine_id,
            "model_type": model_type,
            "model_dir": model_path.as_posix(),
            "report_path": report_path.as_posix(),
            "report": report_data,
            "logs": logs,
        }

    except Exception as exc:
        logger.exception(f"[task_id={task_id} machine_id={machine_id}] Anomaly training failed: {exc}")
        raise


@celery_app.task(bind=True, base=MLProgressTask, name="ml.train_anomaly")
def train_anomaly(
    self: MLProgressTask,
    machine_id: str,
    contamination: float = 0.1,
    time_limit: Optional[int] = None,
) -> Dict[str, Any]:
    return _train_anomaly_impl(self, machine_id=machine_id, contamination=contamination, time_limit=time_limit)


def _train_timeseries_impl(
    task: MLProgressTask,
    machine_id: str,
    forecast_hours: int = 24,
    time_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Train time-series forecasting models.

    Note: API Phase 3.7.8.1 currently sends `time_limit`; we accept it here for
    compatibility, but the script uses `forecast_hours`.
    """
    model_type = "timeseries"
    task_id = _safe_task_id(task)
    started_at = datetime.now().isoformat()

    est_seconds = int(time_limit) if isinstance(time_limit, int) and time_limit > 0 else 60
    est_seconds = max(30, min(300, est_seconds))

    try:
        task.update_progress(
            machine_id=machine_id,
            current=10,
            total=100,
            status="RUNNING",
            message="Validating synthetic data...",
            stage="validating",
            model_type=model_type,
            started_at=started_at,
        )
        _validate_has_synthetic_data(machine_id)

        script = PROJECT_ROOT / "ml_models" / "scripts" / "training" / "train_timeseries.py"
        if not script.exists():
            raise FileNotFoundError(f"Training script not found: {script.as_posix()}")

        task.update_progress(
            machine_id=machine_id,
            current=20,
            total=100,
            status="RUNNING",
            message="Launching training script...",
            stage="launching",
            model_type=model_type,
            started_at=started_at,
        )

        code, out, err = _run_script_with_progress(
            task=task,
            machine_id=machine_id,
            model_type=model_type,
            script_path=script,
            args=["--machine_id", machine_id, "--forecast_hours", str(int(forecast_hours))],
            started_at=started_at,
            est_seconds=est_seconds,
        )

        logs = _tail_lines(out + ("\n" + err if err else ""))

        if code != 0:
            task.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="FAILURE",
                message="Training failed",
                stage="failed",
                model_type=model_type,
                started_at=started_at,
                error=_tail_lines(err) or f"Exit code {code}",
                logs=logs,
            )
            raise RuntimeError(_tail_lines(err) or f"Training failed with exit code {code}")

        task.update_progress(
            machine_id=machine_id,
            current=95,
            total=100,
            status="RUNNING",
            message="Validating model artifacts...",
            stage="validating_artifacts",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        model_path, report_path = _validate_artifacts(model_type, machine_id)

        report_data: Optional[Dict[str, Any]] = None
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report_data = None

        _maybe_log_mlflow(task_id=task_id, machine_id=machine_id, model_type=model_type, report_path=report_path)

        task.update_progress(
            machine_id=machine_id,
            current=100,
            total=100,
            status="SUCCESS",
            message="Training completed successfully",
            stage="completed",
            model_type=model_type,
            started_at=started_at,
            logs=logs,
        )

        return {
            "status": "completed",
            "machine_id": machine_id,
            "model_type": model_type,
            "model_dir": model_path.as_posix(),
            "report_path": report_path.as_posix(),
            "report": report_data,
            "logs": logs,
        }

    except Exception as exc:
        logger.exception(f"[task_id={task_id} machine_id={machine_id}] Timeseries training failed: {exc}")
        raise


@celery_app.task(bind=True, base=MLProgressTask, name="ml.train_timeseries")
def train_timeseries(
    self: MLProgressTask,
    machine_id: str,
    forecast_hours: int = 24,
    time_limit: Optional[int] = None,
) -> Dict[str, Any]:
    return _train_timeseries_impl(self, machine_id=machine_id, forecast_hours=forecast_hours, time_limit=time_limit)


@celery_app.task(bind=True, base=MLProgressTask, name="ml.train_batch")
def train_batch(
    self: MLProgressTask,
    machine_id: str,
    model_types: Optional[List[str]] = None,
    time_limit_per_model: int = 900,
) -> Dict[str, Any]:
    """Train multiple model types sequentially for a complete ML system."""
    task_id = _safe_task_id(self)
    started_at = datetime.now().isoformat()

    if model_types is None:
        model_types = ["classification", "regression", "anomaly", "timeseries"]

    allowed = {"classification", "regression", "anomaly", "timeseries"}
    requested = [m for m in model_types if m in allowed]
    if not requested:
        raise ValueError("No valid model_types specified")

    results: Dict[str, Any] = {}

    try:
        # Shared validation once
        self.update_progress(
            machine_id=machine_id,
            current=5,
            total=100,
            status="RUNNING",
            message="Validating synthetic data...",
            stage="validating",
            model_type="batch",
            started_at=started_at,
        )
        _validate_has_synthetic_data(machine_id)

        if "regression" in requested and not _synthetic_has_column(machine_id, "rul"):
            message = "⚠️ RUL column not found in synthetic data; skipping regression (RUL) in batch training."
            logger.warning(f"[task_id={task_id} machine_id={machine_id}] {message}")
            self.update_progress(
                machine_id=machine_id,
                current=8,
                total=100,
                status="RUNNING",
                message=message,
                stage="warning",
                model_type="batch",
                started_at=started_at,
            )
            requested = [m for m in requested if m != "regression"]

        if not requested:
            message = "⚠️ No trainable model types remain after applying RUL checks."
            logger.warning(f"[task_id={task_id} machine_id={machine_id}] {message}")
            self.update_progress(
                machine_id=machine_id,
                current=100,
                total=100,
                status="SUCCESS",
                message=message,
                stage="skipped",
                model_type="batch",
                started_at=started_at,
            )
            return {
                "machine_id": machine_id,
                "models_trained": 0,
                "complete_system": False,
                "results": {},
                "skipped": True,
                "reason": message,
            }

        per_model_span = 100.0 / float(len(requested))

        for idx, model_type in enumerate(requested):
            slice_start = int(idx * per_model_span)
            slice_end = int((idx + 1) * per_model_span)

            self.update_progress(
                machine_id=machine_id,
                current=max(slice_start, 10),
                total=100,
                status="RUNNING",
                message=f"Starting {model_type} training...",
                stage="starting_model",
                model_type=model_type,
                started_at=started_at,
                batch_index=idx,
                batch_total=len(requested),
            )

            if model_type == "classification":
                results[model_type] = _train_classification_impl(
                    self,
                    machine_id=machine_id,
                    time_limit=time_limit_per_model,
                )
            elif model_type == "regression":
                results[model_type] = _train_regression_impl(
                    self,
                    machine_id=machine_id,
                    time_limit=time_limit_per_model,
                )
            elif model_type == "anomaly":
                results[model_type] = _train_anomaly_impl(
                    self,
                    machine_id=machine_id,
                    contamination=0.1,
                    time_limit=min(300, max(30, int(time_limit_per_model / 15))),
                )
            elif model_type == "timeseries":
                results[model_type] = _train_timeseries_impl(
                    self,
                    machine_id=machine_id,
                    forecast_hours=24,
                    time_limit=min(300, max(30, int(time_limit_per_model / 15))),
                )
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Emit a batch milestone
            self.update_progress(
                machine_id=machine_id,
                current=min(99, slice_end),
                total=100,
                status="RUNNING",
                message=f"{model_type} training completed",
                stage="model_completed",
                model_type=model_type,
                started_at=started_at,
                batch_index=idx,
                batch_total=len(requested),
            )

        complete_system = set(requested) == {"classification", "regression", "anomaly", "timeseries"}

        self.update_progress(
            machine_id=machine_id,
            current=100,
            total=100,
            status="SUCCESS",
            message="Batch training completed successfully",
            stage="completed",
            model_type="batch",
            started_at=started_at,
            models_trained=len(requested),
            complete_system=complete_system,
        )

        _maybe_log_mlflow(
            task_id=task_id,
            machine_id=machine_id,
            model_type="batch",
            extra_metrics={"models_trained": float(len(requested))},
        )

        return {
            "machine_id": machine_id,
            "models_trained": len(requested),
            "complete_system": complete_system,
            "results": results,
        }

    except Exception as exc:
        logger.exception(f"[task_id={task_id} machine_id={machine_id}] Batch training failed: {exc}")
        raise
