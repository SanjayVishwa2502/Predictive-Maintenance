"""
ML API Routes
Phase 3.7.3 Day 15.2: ML Dashboard Backend

REST API endpoints for ML predictions and machine monitoring
Integrates with MLManager service for model inference
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import FileResponse
from typing import Optional
from datetime import datetime, timedelta, timezone
import logging

from ..models.ml import (
    PredictionRequest,
    MachineListResponse,
    MachineInfo,
    MachineStatusResponse,
    ClassificationResponse,
    ClassificationPrediction,
    ModelInfo,
    ExplanationInfo,
    RULResponse,
    RULPrediction,
    CriticalSensor,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    HealthCheckResponse,
    ModelsLoaded,
    GPUInfo,
    ErrorResponse,
    ModelArtifactStatus,
    MachineModelInventory,
    ModelInventoryResponse,
    DeleteModelResponse,
    DeleteAllModelsResponse,
    ModelType,
    AnomalyResponse,
    AnomalyPrediction,
    TimeSeriesResponse,
    TimeSeriesPrediction,
)
import shutil
import json as jsonlib

# Import MLManager singleton
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).resolve().parents[2]))
from services.ml_manager import ml_manager
from services.sensor_simulator import get_simulator
from services.printer_log_reader import read_latest_printer_reading, reading_to_status_fields
from services.history_store import add_snapshot, list_snapshots, list_runs, get_run
from services.auto_prediction_runner import auto_prediction_runner
from services.audit_csv import list_datasets, latest_dataset_path
from services.wide_dataset_csv import append_snapshot_row, latest_wide_dataset_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ml", tags=["ML Predictions"])


def _model_dir(model_type: str, machine_id: str) -> PathLib:
    return ml_manager.project_root / "ml_models" / "models" / model_type / machine_id


def _report_path(model_type: str, machine_id: str) -> PathLib:
    perf_dir = ml_manager.project_root / "ml_models" / "reports" / "performance_metrics"
    if model_type == "classification":
        return perf_dir / f"{machine_id}_classification_report.json"
    if model_type == "regression":
        return perf_dir / f"{machine_id}_regression_report.json"
    if model_type == "anomaly":
        return perf_dir / f"{machine_id}_comprehensive_anomaly_report.json"
    if model_type == "timeseries":
        return perf_dir / f"{machine_id}_timeseries_report.json"
    raise ValueError(f"Unknown model_type: {model_type}")


def _dir_non_empty(path: PathLib) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except Exception:
        return path.is_dir()


def _compute_model_status(model_type: str, machine_id: str) -> ModelArtifactStatus:
    model_path = _model_dir(model_type, machine_id)
    report_path = _report_path(model_type, machine_id)
    issues = []

    model_exists = model_path.exists()
    model_non_empty = _dir_non_empty(model_path)
    report_exists = report_path.exists()
    report_valid = True

    if model_exists and not model_non_empty:
        issues.append("model_dir_empty")

    if report_exists:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                jsonlib.load(f)
        except Exception:
            report_valid = False
            issues.append("report_invalid_json")
    else:
        issues.append("report_missing")
        report_valid = False

    if not model_exists:
        # If there's no model directory, treat as missing regardless of report.
        return ModelArtifactStatus(
            status="missing",
            model_dir=str(model_path),
            report_path=str(report_path),
            issues=["model_dir_missing"],
        )

    if model_non_empty and report_exists and report_valid:
        return ModelArtifactStatus(
            status="available",
            model_dir=str(model_path),
            report_path=str(report_path),
        )

    return ModelArtifactStatus(
        status="corrupted",
        model_dir=str(model_path),
        report_path=str(report_path),
        issues=issues,
    )


def _discover_machine_ids_from_models() -> set[str]:
    machine_ids: set[str] = set()
    models_root = ml_manager.project_root / "ml_models" / "models"
    for model_type in ("classification", "regression", "anomaly", "timeseries"):
        type_dir = models_root / model_type
        if not type_dir.exists():
            continue
        try:
            for entry in type_dir.iterdir():
                if entry.is_dir():
                    machine_ids.add(entry.name)
        except Exception:
            continue


    return machine_ids


def _discover_machine_ids_from_synthetic_data() -> set[str]:
    """Discover machines that have synthetic training data available.

    This is important for model management UX: if a machine has synthetic data but
    no model artifacts (e.g., after deletion), it should still appear in the
    inventory so the UI can show it as missing and allow retraining.
    """
    base = ml_manager.project_root / "GAN" / "data" / "synthetic"
    machine_ids: set[str] = set()
    if not base.exists():
        return machine_ids
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            if (child / "train.parquet").exists():
                machine_ids.add(child.name)
    except Exception:
        return machine_ids
    return machine_ids


@router.get(
    "/models/inventory",
    response_model=ModelInventoryResponse,
    summary="Model Inventory",
    description="Return per-machine, per-model-type status (missing/available/corrupted).",
)
async def model_inventory():
    try:
        machines_from_manager = list(ml_manager.get_machines())
        meta_by_id = {getattr(m, 'machine_id', None): m for m in machines_from_manager}

        machine_ids = set(m.machine_id for m in machines_from_manager)
        machine_ids.update(_discover_machine_ids_from_models())


        machine_ids.update(_discover_machine_ids_from_synthetic_data())

        machines: list[MachineModelInventory] = []
        for machine_id in sorted(machine_ids):
            meta = meta_by_id.get(machine_id)
            manufacturer = getattr(meta, 'manufacturer', None) if meta else None
            model = getattr(meta, 'model', None) if meta else None
            display_name = getattr(meta, 'display_name', None) if meta else None
            category = getattr(meta, 'category', None) if meta else None

            if not display_name and manufacturer and model:
                display_name = f"{manufacturer} {model}"

            statuses = {
                "classification": _compute_model_status("classification", machine_id),
                "regression": _compute_model_status("regression", machine_id),
                "anomaly": _compute_model_status("anomaly", machine_id),
                "timeseries": _compute_model_status("timeseries", machine_id),
            }
            machines.append(
                MachineModelInventory(
                    machine_id=machine_id,
                    display_name=display_name,
                    category=category,
                    manufacturer=manufacturer,
                    model=model,
                    models=statuses,
                )
            )

        return ModelInventoryResponse(machines=machines, total=len(machines))
    except Exception as e:
        logger.error(f"Failed to build model inventory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build model inventory: {str(e)}")


@router.delete(
    "/models/{machine_id}/{model_type}",
    response_model=DeleteModelResponse,
    summary="Delete Model Artifacts",
    description="Delete a machine's model directory and its performance report file.",
)
async def delete_model(
    machine_id: str = Path(..., description="Machine identifier"),
    model_type: ModelType = Path(..., description="Model type"),
):
    try:
        model_type_str = model_type.value
        model_path = _model_dir(model_type_str, machine_id)
        report_path = _report_path(model_type_str, machine_id)

        deleted_model_dir = False
        deleted_report_file = False

        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path)
            deleted_model_dir = True

        if report_path.exists() and report_path.is_file():
            report_path.unlink()
            deleted_report_file = True

        return DeleteModelResponse(
            machine_id=machine_id,
            model_type=model_type_str,
            deleted_model_dir=deleted_model_dir,
            deleted_report_file=deleted_report_file,
        )
    except Exception as e:
        logger.error(f"Failed to delete model artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model artifacts: {str(e)}")


@router.delete(
    "/models/{machine_id}",
    response_model=DeleteAllModelsResponse,
    summary="Delete All Model Artifacts",
    description="Delete all model directories and performance report files for a machine.",
)
async def delete_all_models(
    machine_id: str = Path(..., description="Machine identifier"),
):
    results: dict[str, DeleteModelResponse] = {}
    errors: dict[str, str] = {}

    for model_type_str in ("classification", "regression", "anomaly", "timeseries"):
        try:
            model_path = _model_dir(model_type_str, machine_id)
            report_path = _report_path(model_type_str, machine_id)

            deleted_model_dir = False
            deleted_report_file = False

            if model_path.exists() and model_path.is_dir():
                shutil.rmtree(model_path)
                deleted_model_dir = True

            if report_path.exists() and report_path.is_file():
                report_path.unlink()
                deleted_report_file = True

            results[model_type_str] = DeleteModelResponse(
                machine_id=machine_id,
                model_type=model_type_str,
                deleted_model_dir=deleted_model_dir,
                deleted_report_file=deleted_report_file,
            )
        except Exception as e:
            logger.error(f"Failed deleting {model_type_str} for {machine_id}: {e}")
            errors[model_type_str] = str(e)

    return DeleteAllModelsResponse(machine_id=machine_id, results=results, errors=errors)


# ============================================================
# Endpoint 1: List All Machines
# ============================================================

@router.get(
    "/machines",
    response_model=MachineListResponse,
    summary="List All Machines",
    description="""
    Retrieve list of all 26 machines with model availability flags.
    
    **Returns:**
    - Machine metadata (ID, name, category, manufacturer, model)
    - Sensor count
    - Model availability (classification, regression, anomaly, timeseries)
    
    **Use Case:** Populate machine selector dropdown in dashboard
    """,
    responses={
        200: {"description": "List of machines retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def list_machines():
    """
    List all available machines with model information
    """
    try:
        logger.info("Listing all machines...")
        
        # Get machines from MLManager
        machines_data = ml_manager.get_machines()

        # Default behavior: only expose machines that have at least one trained model.
        # This prevents the dashboard selector from showing machines that cannot run predictions.
        machines_data = [
            m for m in machines_data
            if (
                getattr(m, 'has_classification_model', False)
                or getattr(m, 'has_regression_model', False)
                or getattr(m, 'has_anomaly_model', False)
                or getattr(m, 'has_timeseries_model', False)
            )
        ]

        # Single-printer streaming (Ender 3): expose it when a clean temp log exists.
        # This allows monitoring even before any ML models are trained.
        # NOTE: We intentionally do NOT inject an extra non-metadata printer id here.
        # The printer should be represented by its canonical metadata id
        # (e.g. "printer_creality_ender3_001") to match the rest of the system.
        
        # Convert to Pydantic models (machines_data is already list of MachineInfo dataclasses)
        from dataclasses import asdict
        machines = [MachineInfo(**asdict(machine)) for machine in machines_data]
        
        logger.info(f"[OK] Listed {len(machines)} machines")
        
        return MachineListResponse(
            machines=machines,
            total=len(machines)
        )
        
    except Exception as e:
        logger.error(f"Failed to list machines: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve machine list: {str(e)}"
        )


# ============================================================
# Endpoint 2: Get Machine Status
# ============================================================

@router.get(
    "/machines/{machine_id}/status",
    response_model=MachineStatusResponse,
    summary="Get Machine Status",
    description="""
    Retrieve current status and latest sensor readings for a specific machine.
    
    **Parameters:**
    - machine_id: Machine identifier (e.g., motor_siemens_1la7_001)
    
    **Returns:**
    - Running status
    - Latest sensor readings (all sensors)
    - Last update timestamp
    - Sensor count
    
    **Use Case:** Display real-time sensor dashboard
    """,
    responses={
        200: {"description": "Machine status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Machine not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_machine_status(
    machine_id: str = Path(..., description="Machine identifier"),
    session_id: str = Query("", description="Per-browser-session id for dataset capture"),
    client_id: str = Query("", description="Dashboard client id (for correlation only)"),
):
    """
    Get current status and sensor readings for a machine
    """
    try:
        logger.info(f"Getting status for machine: {machine_id}")

        # Prefer real printer streaming data (clean CSV) when available.
        # IMPORTANT: allow monitoring-only mode even if MLManager has no metadata for this machine_id.
        printer_reading = read_latest_printer_reading(machine_id)
        if printer_reading is not None:
            from services.auto_prediction_service import (
                should_run_prediction,
                run_prediction_only,
                preload_models_for_machine,
            )
            
            # Preload models on first status call for this machine
            preload_models_for_machine(machine_id)
            
            status = {
                "machine_id": machine_id,
                **reading_to_status_fields(printer_reading),
            }
            
            # Store snapshot and check for auto-prediction
            sensor_data = status.get("latest_sensors", {})
            if sensor_data:
                try:
                    stamp = status["last_update"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if status.get("last_update") else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                    await add_snapshot(
                        machine_id=machine_id,
                        sensor_data=sensor_data,
                        data_stamp=stamp,
                    )
                    status["data_stamp"] = stamp
                    
                    # Auto-prediction every 5th data point (ML only, no LLM)
                    if should_run_prediction(machine_id):
                        logger.info(f"[AUTO] Triggering prediction-only run for {machine_id}")
                        pred_result = await run_prediction_only(
                            machine_id=machine_id,
                            sensor_data=sensor_data,
                            data_stamp=stamp,
                        )
                        if pred_result:
                            status["auto_prediction"] = {
                                "run_id": pred_result.get("run_id"),
                                "run_type": "prediction",
                            }
                            logger.info(f"[OK] Auto prediction completed for {machine_id}: {pred_result.get('run_id')}")
                    
                    # Wide CSV dataset capture
                    if session_id:
                        append_snapshot_row(
                            machine_id=machine_id,
                            session_id=session_id,
                            data_stamp=stamp,
                            sensor_data=sensor_data,
                            client_id=client_id,
                            run_id=str(status.get("auto_prediction", {}).get("run_id") or ""),
                        )
                except Exception as e:
                    logger.debug(f"Snapshot persistence skipped for {machine_id}: {e}")
            
            return MachineStatusResponse(**status)
        
        # Get machine info from MLManager
        machine_info = ml_manager.get_machine_info(machine_id)
        
        if machine_info is None:
            raise HTTPException(status_code=404, detail=f"Machine not found: {machine_id}")
        
        # Fall back to the existing validation-data simulator.
        simulator = get_simulator()

        # Auto-start simulation for machines that have validation data.
        # This keeps the UI simple for testing: selecting a machine immediately shows live readings.
        is_running = simulator.is_running(machine_id)
        if not is_running:
            try:
                # Start only if dataset exists; start_simulation returns False when missing.
                if simulator.start_simulation(machine_id):
                    is_running = True
                    logger.info(f"[OK] Auto-started sensor simulation for {machine_id}")
            except Exception as e:
                # Never fail the status endpoint due to simulation issues.
                logger.warning(f"Auto-start simulation failed for {machine_id}: {e}")

        latest_sensors: dict = {}
        if is_running:
            sensor_data = simulator.get_current_readings(machine_id)
            if sensor_data:
                latest_sensors = sensor_data
                logger.debug(
                    f"Returning simulated sensor data for {machine_id}: {len(latest_sensors)} sensors"
                )

        status = {
            "machine_id": machine_id,
            "is_running": is_running,
            "latest_sensors": latest_sensors,
            "last_update": datetime.now(),
            "sensor_count": len(latest_sensors) if latest_sensors else machine_info.sensor_count,
        }

        # Store the 5-second datapoints for history. Best-effort only.
        if latest_sensors:
            try:
                stamp = status["last_update"].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                await add_snapshot(
                    machine_id=machine_id,
                    sensor_data=latest_sensors,
                    data_stamp=stamp,
                )
                status["data_stamp"] = stamp

                # Wide CSV dataset capture (Excel-friendly, per session). Best-effort only.
                if session_id:
                    append_snapshot_row(
                        machine_id=machine_id,
                        session_id=session_id,
                        data_stamp=stamp,
                        sensor_data=latest_sensors,
                        client_id=client_id,
                        run_id=str(status.get("run_id") or ""),
                    )
            except Exception as e:
                logger.debug(f"Snapshot persistence skipped for {machine_id}: {e}")
        
        logger.info(f"[OK] Status retrieved for {machine_id} (running: {is_running}, {status['sensor_count']} sensors)")
        
        return MachineStatusResponse(**status)
        
    except FileNotFoundError as e:
        logger.error(f"Machine not found: {machine_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to get machine status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve machine status: {str(e)}"
        )


# ============================================================
# Snapshot + Run History (Persisted)
# ============================================================


@router.get(
    "/machines/{machine_id}/audit/datasets",
    summary="List prediction-history datasets (CSV)",
    description="Return a list of CSV audit datasets captured for this machine (session-scoped).",
)
async def get_audit_datasets(
    machine_id: str = Path(..., description="Machine identifier"),
):
    try:
        return {"machine_id": machine_id, "datasets": list_datasets(machine_id)}
    except Exception as e:
        logger.error(f"Failed to list audit datasets for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get(
    "/machines/{machine_id}/audit/datasets/latest",
    summary="Download latest prediction-history dataset (CSV)",
    description="Download the most recent CSV audit dataset for this machine.",
)
async def download_latest_audit_dataset(
    machine_id: str = Path(..., description="Machine identifier"),
):
    try:
        p = latest_dataset_path(machine_id)
        if not p or not p.exists():
            raise HTTPException(status_code=404, detail="No audit dataset found")
        return FileResponse(
            path=str(p),
            filename=p.name,
            media_type="text/csv",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download audit dataset for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")


@router.get(
    "/machines/{machine_id}/audit/wide/latest",
    summary="Download latest wide dataset (CSV)",
    description="Download an Excel-friendly wide CSV for this machine and session_id (one column per sensor).",
)
async def download_latest_wide_dataset(
    machine_id: str = Path(..., description="Machine identifier"),
    session_id: str = Query("", description="Per-browser-session id"),
):
    try:
        if not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        p = latest_wide_dataset_path(machine_id, session_id)
        if not p or not p.exists():
            raise HTTPException(status_code=404, detail="No wide dataset found for this session")
        return FileResponse(
            path=str(p),
            filename=p.name,
            media_type="text/csv",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download wide dataset for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download wide dataset: {str(e)}")


@router.get(
    "/machines/{machine_id}/snapshots",
    summary="List stored sensor snapshots",
    description="Return stored 5-second sensor snapshots for a machine (newest first).",
)
async def get_machine_snapshots(
    machine_id: str = Path(..., description="Machine identifier"),
    limit: int = Query(500, ge=1, le=5000, description="Max snapshots to return"),
):
    try:
        return {
            "machine_id": machine_id,
            "snapshots": await list_snapshots(machine_id, limit=limit),
        }
    except Exception as e:
        logger.error(f"Failed to list snapshots for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {str(e)}")


@router.get(
    "/machines/{machine_id}/runs",
    summary="List prediction runs",
    description="Return stored prediction runs for a machine (newest first).",
)
async def get_machine_runs(
    machine_id: str = Path(..., description="Machine identifier"),
    limit: int = Query(200, ge=1, le=2000, description="Max runs to return"),
):
    try:
        return {
            "machine_id": machine_id,
            "runs": await list_runs(machine_id, limit=limit),
        }
    except Exception as e:
        logger.error(f"Failed to list runs for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")


@router.get(
    "/runs/{run_id}",
    summary="Get prediction run details",
    description="Return full stored details (snapshot, predictions, LLM outputs) for a run_id.",
)
async def get_run_details(
    run_id: str = Path(..., description="Run ID"),
):
    try:
        run = await get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch run details for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch run details: {str(e)}")


# ============================================================
# Backend Auto-Run (150s Trigger)
# ============================================================


@router.post(
    "/machines/{machine_id}/auto/start",
    summary="Start backend auto prediction+LLM",
    description="Register a machine for backend-side auto runs (default 150 seconds).",
)
async def start_auto_runs(
    machine_id: str = Path(..., description="Machine identifier"),
    interval_seconds: int = Query(150, ge=30, le=3600, description="Interval in seconds"),
    client_id: str = Query("", description="Dashboard client id for WS push updates"),
):
    try:
        return await auto_prediction_runner.start(machine_id, interval_seconds=interval_seconds, client_id=client_id)
    except Exception as e:
        logger.error(f"Failed to start auto runs for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start auto runs: {str(e)}")


@router.post(
    "/machines/{machine_id}/auto/stop",
    summary="Stop backend auto prediction+LLM",
)
async def stop_auto_runs(
    machine_id: str = Path(..., description="Machine identifier"),
):
    try:
        return await auto_prediction_runner.stop(machine_id)
    except Exception as e:
        logger.error(f"Failed to stop auto runs for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop auto runs: {str(e)}")


@router.get(
    "/machines/{machine_id}/auto/status",
    summary="Get backend auto-run status",
)
async def auto_runs_status(
    machine_id: str = Path(..., description="Machine identifier"),
):
    try:
        return auto_prediction_runner.status(machine_id)
    except Exception as e:
        logger.error(f"Failed to get auto status for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get auto status: {str(e)}")


@router.post(
    "/machines/{machine_id}/auto/run_once",
    summary="Run one prediction+LLM cycle now",
    description="Immediately create a run (uses latest snapshot) and enqueue LLM tasks.",
)
async def auto_run_once(
    machine_id: str = Path(..., description="Machine identifier"),
    client_id: str = Query("", description="Dashboard client id for WS push updates"),
    session_id: str = Query("", description="Per-browser-session id for dataset capture"),
):
    try:
        return await auto_prediction_runner.run_once(machine_id, client_id=client_id, session_id=session_id)
    except Exception as e:
        logger.error(f"Failed to run once for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run once: {str(e)}")


# ============================================================
# Endpoint 3: Run Classification Prediction
# ============================================================

@router.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    summary="Run Classification Prediction",
    description="""
    Run health state classification prediction for a single machine.
    
    **Request Body:**
    - machine_id: Machine identifier
    - sensor_data: Dictionary of sensor readings (all required sensors)
    
    **Returns:**
    - Failure type (normal, bearing_wear, overheating, electrical_fault)
    - Confidence score (0.0 - 1.0)
    - Failure probability
    - All class probabilities
    - LLM-generated explanation
    - Maintenance recommendations
    
    **Model:** AutoGluon classification (WeightedEnsemble_L2)
    
    **Typical Response Time:** < 3 seconds (ML: 200ms, LLM: 2s)
    """,
    responses={
        200: {"description": "Prediction completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request (missing sensors, invalid values)"},
        404: {"model": ErrorResponse, "description": "Machine or model not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict_classification(request: PredictionRequest):
    """
    Run classification prediction with LLM explanation
    """
    try:
        logger.info(f"Running classification prediction for {request.machine_id}")
        
        # Run prediction via MLManager (includes LLM explanation)
        result = ml_manager.predict_classification(
            machine_id=request.machine_id,
            sensor_data=request.sensor_data
        )
        
        logger.info(
            f"[OK] Prediction complete: {result.failure_type} "
            f"(confidence: {result.confidence:.2%})"
        )
        
        # Convert dataclass to response model
        from dataclasses import asdict
        response = ClassificationResponse(
            machine_id=result.machine_id,
            prediction=ClassificationPrediction(
                failure_type=result.failure_type,
                confidence=result.confidence,
                failure_probability=result.failure_probability,
                all_probabilities=result.all_probabilities,
                model_info=ModelInfo(
                    path="ml_models/models/classification/" + result.machine_id,
                    best_model="AutoGluon",
                    num_features=len(request.sensor_data)
                )
            ),
            explanation=ExplanationInfo(
                summary=result.explanation or "Prediction completed successfully.",
                risk_factors=[],
                recommendations=["Continue monitoring"]
            ),
            timestamp=datetime.fromisoformat(result.timestamp) if result.timestamp else datetime.now()
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Machine or model not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{request.machine_id}' or model not found"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# Endpoint 4: Run RUL Prediction
# ============================================================

@router.post(
    "/predict/rul",
    response_model=RULResponse,
    summary="Run RUL Prediction",
    description="""
    Run Remaining Useful Life (RUL) prediction for a single machine.
    
    **Request Body:**
    - machine_id: Machine identifier
    - sensor_data: Dictionary of sensor readings
    
    **Returns:**
    - RUL in hours and days
    - Urgency level (low, medium, high, critical)
    - Maintenance window recommendation
    - Critical sensors requiring attention
    - Estimated failure date
    - LLM-generated explanation
    
    **Model:** AutoGluon regression
    
    **Typical Response Time:** < 3 seconds
    """,
    responses={
        200: {"description": "RUL prediction completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Machine or model not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict_rul(request: PredictionRequest):
    """
    Run RUL prediction with maintenance recommendations
    """
    try:
        logger.info(f"Running RUL prediction for {request.machine_id}")
        
        # Run prediction via MLManager
        result = ml_manager.predict_rul(
            machine_id=request.machine_id,
            sensor_data=request.sensor_data
        )
        
        logger.info(
            f"[OK] RUL prediction complete: {result.rul_hours:.1f} hours "
            f"({result.rul_days:.1f} days)"
        )
        
        # Convert to response model
        response = RULResponse(
            machine_id=result.machine_id,
            prediction=RULPrediction(
                rul_hours=result.rul_hours,
                rul_days=result.rul_days,
                urgency=result.urgency,
                maintenance_window=f"within {int(result.rul_days)} days",
                critical_sensors=[],  # To be implemented
                estimated_failure_date=datetime.now() + timedelta(hours=result.rul_hours),
                confidence=result.confidence
            ),
            explanation=ExplanationInfo(
                summary=result.explanation or f"Machine has {result.rul_days:.1f} days remaining useful life.",
                risk_factors=[],
                recommendations=["Schedule maintenance inspection"]
            ),
            timestamp=datetime.fromisoformat(result.timestamp.replace('Z', '+00:00')) if result.timestamp else datetime.now()
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Machine or model not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{request.machine_id}' or model not found"
        )
    except Exception as e:
        logger.error(f"RUL prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RUL prediction failed: {str(e)}"
        )


# ============================================================
# Endpoint 4b: Run Anomaly Detection
# ============================================================


@router.post(
    "/predict/anomaly",
    response_model=AnomalyResponse,
    summary="Run Anomaly Detection",
    description="""
    Run anomaly detection for a single machine.

    Notes:
    - This endpoint returns prediction-only results quickly.
    - LLM explanations should be generated asynchronously via /api/llm/explain.
    """,
    responses={
        200: {"description": "Anomaly detection completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Machine or model not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict_anomaly(request: PredictionRequest):
    try:
        logger.info(f"Running anomaly detection for {request.machine_id}")

        result = ml_manager.predict_anomaly(machine_id=request.machine_id, sensor_data=request.sensor_data)

        response = AnomalyResponse(
            machine_id=result.machine_id,
            prediction=AnomalyPrediction(
                is_anomaly=result.is_anomaly,
                anomaly_score=result.anomaly_score,
                detection_method=result.detection_method,
                abnormal_sensors=result.abnormal_sensors,
            ),
            explanation=ExplanationInfo(
                summary=(
                    "Anomaly detected." if result.is_anomaly else "No anomaly detected."
                ),
                risk_factors=[],
                recommendations=["Continue monitoring"],
            ),
            timestamp=datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
            if getattr(result, "timestamp", None)
            else datetime.now(),
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"Machine or model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Machine '{request.machine_id}' or model not found")
    except Exception as e:
        logger.error(f"Anomaly prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# Endpoint 4c: Run Time-series Forecast
# ============================================================


@router.post(
    "/predict/timeseries",
    response_model=TimeSeriesResponse,
    summary="Run Time-series Forecast",
    description="""
    Run time-series forecasting for a single machine.

    Notes:
    - This endpoint returns prediction-only results quickly.
    - LLM explanations should be generated asynchronously via /api/llm/explain.
    """,
    responses={
        200: {"description": "Forecast generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Machine or model not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict_timeseries(request: PredictionRequest):
    try:
        logger.info(f"Running timeseries forecast for {request.machine_id}")

        result = ml_manager.predict_timeseries(machine_id=request.machine_id, sensor_data=request.sensor_data)

        response = TimeSeriesResponse(
            machine_id=result.machine_id,
            prediction=TimeSeriesPrediction(
                forecast_summary=result.forecast_summary,
                confidence=result.confidence,
                forecast_horizon=result.forecast_horizon,
                forecasts=result.forecasts,
            ),
            explanation=ExplanationInfo(
                summary="Forecast generated.",
                risk_factors=[],
                recommendations=["Continue monitoring"],
            ),
            timestamp=datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
            if getattr(result, "timestamp", None)
            else datetime.now(),
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"Machine or model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Machine '{request.machine_id}' or model not found")
    except Exception as e:
        logger.error(f"Timeseries prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# Endpoint 5: Get Prediction History
# ============================================================

@router.get(
    "/machines/{machine_id}/history",
    response_model=PredictionHistoryResponse,
    summary="Get Prediction History",
    description="""
    Retrieve prediction history for a specific machine.
    
    **Parameters:**
    - machine_id: Machine identifier
    - limit: Maximum number of predictions to return (default: 100, max: 1000)
    - model_type: Filter by model type (optional)
    
    **Returns:**
    - List of past predictions (timestamp, failure type, confidence, RUL)
    - Total count
    - Pagination info
    
    **Use Case:** Display prediction trends and history chart
    """,
    responses={
        200: {"description": "Prediction history retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Machine not found"},
        500: {"model": ErrorResponse, "description": "Failed to retrieve history"}
    }
)
async def get_prediction_history(
    machine_id: str = Path(..., description="Machine identifier"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of predictions"),
    model_type: Optional[str] = Query(None, description="Filter by model type")
):
    """
    Get prediction history for a machine
    """
    try:
        logger.info(f"Getting prediction history for {machine_id} (limit: {limit})")
        
        # Get history from MLManager (to be implemented - returning empty for now)
        history = {
            "predictions": [],
            "total": 0
        }
        
        # Convert to response items
        predictions = [
            PredictionHistoryItem(**item)
            for item in history['predictions']
        ]
        
        logger.info(f"[OK] Retrieved {len(predictions)} predictions for {machine_id}")
        
        return PredictionHistoryResponse(
            machine_id=machine_id,
            predictions=predictions,
            total=history['total'],
            page=1,
            per_page=limit
        )
        
    except FileNotFoundError as e:
        logger.error(f"Machine not found: {machine_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_id}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )


# ============================================================
# Endpoint 6: Service Health Check
# ============================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="ML Service Health Check",
    description="""
    Check ML service health and readiness.
    
    **Returns:**
    - Service status (healthy, degraded, unhealthy)
    - Count of loaded models by type
    - LLM service status
    - GPU availability and info
    - IntegratedPredictionSystem readiness
    
    **Use Case:** Monitoring and dashboard status indicators
    """,
    responses={
        200: {"description": "Health check completed"}
    }
)
async def health_check():
    """
    Check ML service health
    """
    try:
        logger.info("Running ML service health check...")
        
        # Get health status from MLManager
        health = ml_manager.get_health()
        
        # Parse GPU info if available
        gpu_info = None
        if health.get('gpu_available'):
            gpu_info = GPUInfo(
                name=health['gpu_info']['name'],
                cuda_version=health['gpu_info']['cuda_version']
            )
        
        # Build response
        response = HealthCheckResponse(
            status=health['status'],
            models_loaded=ModelsLoaded(**health['models_loaded']),
            llm_status=health['llm_status'],
            gpu_available=health['gpu_available'],
            gpu_info=gpu_info,
            integrated_system_ready=health['integrated_system_ready'],
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"[OK] Health check complete: {health['status']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return unhealthy status instead of raising exception
        return HealthCheckResponse(
            status="unhealthy",
            models_loaded=ModelsLoaded(
                classification=0,
                regression=0,
                anomaly=0,
                timeseries=0
            ),
            llm_status="unavailable",
            gpu_available=False,
            gpu_info=None,
            integrated_system_ready=False,
            timestamp=datetime.utcnow()
        )


# ============================================================
# Sensor Simulation Control Endpoints
# ============================================================

@router.post(
    "/simulation/start/{machine_id}",
    summary="Start Sensor Simulation",
    description="""
    Start simulated sensor data streaming for a machine.
    
    Reads from validation dataset (val.parquet) and iterates through rows
    to simulate real-time sensor readings. Each call to /machines/{machine_id}/status
    will return the next row of data.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Success status and simulation details
    """,
    responses={
        200: {"description": "Simulation started successfully"},
        404: {"description": "Machine or validation dataset not found"},
        500: {"description": "Failed to start simulation"}
    }
)
async def start_simulation(
    machine_id: str = Path(..., description="Machine identifier")
):
    """Start sensor data simulation for a machine"""
    try:
        simulator = get_simulator()
        
        # Check if machine exists
        machine_info = ml_manager.get_machine_info(machine_id)
        if machine_info is None:
            raise HTTPException(status_code=404, detail=f"Machine not found: {machine_id}")
        
        # Start simulation
        success = simulator.start_simulation(machine_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Validation dataset not found for {machine_id}"
            )
        
        # Get initial status
        status = simulator.get_simulation_status(machine_id)
        
        logger.info(f"[OK] Started simulation for {machine_id}")
        
        return {
            "success": True,
            "message": f"Simulation started for {machine_id}",
            "simulation": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start simulation: {str(e)}"
        )


@router.post(
    "/simulation/stop/{machine_id}",
    summary="Stop Sensor Simulation",
    description="""
    Stop simulated sensor data streaming for a machine.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Success status
    """,
    responses={
        200: {"description": "Simulation stopped successfully"},
        404: {"description": "No active simulation for machine"}
    }
)
async def stop_simulation(
    machine_id: str = Path(..., description="Machine identifier")
):
    """Stop sensor data simulation for a machine"""
    try:
        simulator = get_simulator()
        
        success = simulator.stop_simulation(machine_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No active simulation for {machine_id}"
            )
        
        logger.info(f"[OK] Stopped simulation for {machine_id}")
        
        return {
            "success": True,
            "message": f"Simulation stopped for {machine_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop simulation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop simulation: {str(e)}"
        )


@router.get(
    "/simulation/status/{machine_id}",
    summary="Get Simulation Status",
    description="""
    Get current simulation status for a machine.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Simulation status including current row, progress, and sensor count
    """,
    responses={
        200: {"description": "Status retrieved successfully"},
        404: {"description": "No active simulation for machine"}
    }
)
async def get_simulation_status(
    machine_id: str = Path(..., description="Machine identifier")
):
    """Get simulation status for a machine"""
    try:
        simulator = get_simulator()
        
        status = simulator.get_simulation_status(machine_id)
        
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"No active simulation for {machine_id}"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get simulation status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get simulation status: {str(e)}"
        )


@router.get(
    "/simulation/active",
    summary="List Active Simulations",
    description="""
    Get list of all machines with active simulations.
    
    **Returns:**
    - List of machine IDs with running simulations
    """,
    responses={
        200: {"description": "Active simulations retrieved successfully"}
    }
)
async def list_active_simulations():
    """List all active simulations"""
    try:
        simulator = get_simulator()
        active = simulator.get_active_simulations()
        
        return {
            "active_simulations": active,
            "count": len(active)
        }
        
    except Exception as e:
        logger.error(f"Failed to list simulations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list simulations: {str(e)}"
        )


@router.post(
    "/simulation/stop-all",
    summary="Stop All Simulations",
    description="""
    Stop all active sensor simulations.
    
    **Returns:**
    - Number of simulations stopped
    """,
    responses={
        200: {"description": "All simulations stopped successfully"}
    }
)
async def stop_all_simulations():
    """Stop all active simulations"""
    try:
        simulator = get_simulator()
        count = simulator.stop_all_simulations()
        
        logger.info(f"[OK] Stopped all simulations ({count} total)")
        
        return {
            "success": True,
            "message": f"Stopped {count} simulation(s)",
            "count": count
        }
        
    except Exception as e:
        logger.error(f"Failed to stop all simulations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop all simulations: {str(e)}"
        )


# ============================================================
# Export Router
# ============================================================

__all__ = ['router']
