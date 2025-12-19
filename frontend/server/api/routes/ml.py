"""
ML API Routes
Phase 3.7.3 Day 15.2: ML Dashboard Backend

REST API endpoints for ML predictions and machine monitoring
Integrates with MLManager service for model inference
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import Optional
from datetime import datetime, timedelta
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
    ErrorResponse
)

# Import MLManager singleton
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).resolve().parents[2]))
from services.ml_manager import ml_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ml", tags=["ML Predictions"])


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
            if getattr(m, 'has_classification_model', False) or getattr(m, 'has_regression_model', False)
        ]
        
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
    machine_id: str = Path(..., description="Machine identifier")
):
    """
    Get current status and sensor readings for a machine
    """
    try:
        logger.info(f"Getting status for machine: {machine_id}")
        
        # Get machine info from MLManager
        machine_info = ml_manager.get_machine_info(machine_id)
        
        if machine_info is None:
            raise HTTPException(status_code=404, detail=f"Machine not found: {machine_id}")
        
        # For now, return basic status (real-time sensor data to be implemented)
        from dataclasses import asdict
        status = {
            "machine_id": machine_id,
            "is_running": False,  # To be implemented with real sensor data
            "latest_sensors": {},
            "last_update": datetime.now(),  # Use current time instead of None
            "sensor_count": max(machine_info.sensor_count, 1)  # Ensure at least 1
        }
        
        logger.info(f"[OK] Status retrieved for {machine_id} ({status['sensor_count']} sensors)")
        
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
            timestamp=datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
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
# Export Router
# ============================================================

__all__ = ['router']
