"""
Pydantic Models for ML API
Phase 3.7.3 Day 15.2: ML API Endpoints

Request/response schemas for ML prediction endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================================
# Enums
# ============================================================

class FailureType(str, Enum):
    """Failure type classification"""
    NORMAL = "normal"
    BEARING_WEAR = "bearing_wear"
    OVERHEATING = "overheating"
    ELECTRICAL_FAULT = "electrical_fault"


class UrgencyLevel(str, Enum):
    """Urgency level for maintenance"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelType(str, Enum):
    """Model type for prediction"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    TIMESERIES = "timeseries"


# ============================================================
# Response Models - Model Inventory / Management
# ============================================================


class ModelArtifactStatus(BaseModel):
    """Status for a single model type's artifacts."""

    status: str = Field(..., description="One of: missing, available, corrupted")
    model_dir: Optional[str] = Field(None, description="Filesystem path to model directory")
    report_path: Optional[str] = Field(None, description="Filesystem path to report JSON")
    issues: List[str] = Field(default_factory=list, description="Any detected issues")


class MachineModelInventory(BaseModel):
    """Inventory status for all model types for a machine."""

    machine_id: str = Field(..., description="Machine unique identifier")
    display_name: Optional[str] = Field(None, description="Human-friendly display name")
    category: Optional[str] = Field(None, description="Machine category/type")
    manufacturer: Optional[str] = Field(None, description="Machine manufacturer")
    model: Optional[str] = Field(None, description="Machine model")
    models: Dict[str, ModelArtifactStatus] = Field(
        ..., description="Per-model-type status keyed by model type string"
    )


class ModelInventoryResponse(BaseModel):
    """Response model for model inventory endpoint."""

    machines: List[MachineModelInventory] = Field(..., description="Inventory per machine")
    total: int = Field(..., description="Total count of machines", ge=0)


class DeleteModelResponse(BaseModel):
    """Response model for deleting a model's artifacts."""

    machine_id: str = Field(..., description="Machine identifier")
    model_type: str = Field(..., description="Model type deleted")
    deleted_model_dir: bool = Field(..., description="Whether the model directory was deleted")
    deleted_report_file: bool = Field(..., description="Whether the report file was deleted")


class DeleteAllModelsResponse(BaseModel):
    """Response model for deleting all model artifacts for a machine."""

    machine_id: str = Field(..., description="Machine identifier")
    results: Dict[str, DeleteModelResponse] = Field(
        default_factory=dict,
        description="Per-model-type delete results keyed by model type",
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-model-type error messages keyed by model type",
    )


# ============================================================
# Request Models
# ============================================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    machine_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Machine identifier (e.g., motor_siemens_1la7_001)",
        example="motor_siemens_1la7_001"
    )
    sensor_data: Optional[Dict[str, float]] = Field(
        None,
        description="Dictionary of sensor readings",
        example={
            "bearing_de_temp_C": 65.2,
            "bearing_nde_temp_C": 62.1,
            "winding_temp_C": 55.3,
            "rms_velocity_mm_s": 3.4,
            "current_100pct_load_A": 12.5,
            "voltage_phase_to_phase_V": 410.0
        }
    )

    @validator('sensor_data', pre=True)
    def normalize_sensor_data(cls, v):
        """Treat empty payloads as 'no sensor data provided'."""
        if v is None:
            return None
        if isinstance(v, dict) and len(v) == 0:
            return None
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "sensor_data": {
                    "bearing_de_temp_C": 65.2,
                    "bearing_nde_temp_C": 62.1,
                    "winding_temp_C": 55.3,
                    "rms_velocity_mm_s": 3.4,
                    "current_100pct_load_A": 12.5
                }
            }
        }


# ============================================================
# Response Models - Machine Management
# ============================================================

class MachineInfo(BaseModel):
    """Machine information with model availability"""
    machine_id: str = Field(..., description="Machine unique identifier")
    display_name: str = Field(..., description="Human-readable machine name")
    category: str = Field(..., description="Machine category (motor, pump, cnc, etc.)")
    manufacturer: str = Field(..., description="Manufacturer name")
    model: str = Field(..., description="Model name")
    sensor_count: int = Field(..., description="Number of sensors", ge=1)
    has_classification_model: bool = Field(..., description="Classification model available")
    has_regression_model: bool = Field(..., description="RUL regression model available")
    has_anomaly_model: bool = Field(..., description="Anomaly detection model available")
    has_timeseries_model: bool = Field(..., description="Time-series forecast model available")
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "display_name": "Motor Siemens 1LA7 001",
                "category": "motor",
                "manufacturer": "SIEMENS",
                "model": "1LA7",
                "sensor_count": 22,
                "has_classification_model": True,
                "has_regression_model": True,
                "has_anomaly_model": False,
                "has_timeseries_model": False
            }
        }


class MachineListResponse(BaseModel):
    """Response model for list machines endpoint"""
    machines: List[MachineInfo] = Field(..., description="List of machines")
    total: int = Field(..., description="Total count of machines", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "machines": [
                    {
                        "machine_id": "motor_siemens_1la7_001",
                        "display_name": "Motor Siemens 1LA7 001",
                        "category": "motor",
                        "manufacturer": "SIEMENS",
                        "model": "1LA7",
                        "sensor_count": 22,
                        "has_classification_model": True,
                        "has_regression_model": True,
                        "has_anomaly_model": False,
                        "has_timeseries_model": False
                    }
                ],
                "total": 26
            }
        }


class MachineStatusResponse(BaseModel):
    """Response model for machine status endpoint"""
    machine_id: str = Field(..., description="Machine identifier")
    is_running: bool = Field(..., description="Whether machine is currently running")
    latest_sensors: Dict[str, float] = Field(..., description="Latest sensor readings")
    last_update: datetime = Field(..., description="Timestamp of last sensor update")
    data_stamp: Optional[str] = Field(None, description="Stable UTC data stamp for this snapshot (ISO8601)")
    run_id: Optional[str] = Field(None, description="Prediction run id associated with this snapshot (if any)")
    sensor_count: int = Field(..., description="Number of sensors", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "is_running": True,
                "latest_sensors": {
                    "bearing_de_temp_C": 65.2,
                    "bearing_nde_temp_C": 62.1,
                    "winding_temp_C": 55.3,
                    "rms_velocity_mm_s": 3.4,
                    "current_100pct_load_A": 12.5,
                    "voltage_phase_to_phase_V": 410.0
                },
                "last_update": "2025-12-15T10:45:23Z",
                "data_stamp": "2025-12-15T10:45:23Z",
                "run_id": "5dd0d7f5-6a41-4ba0-8f6a-83b886d58c77",
                "sensor_count": 22
            }
        }


# ============================================================
# Response Models - Classification Prediction
# ============================================================

class ModelInfo(BaseModel):
    """Model metadata information"""
    path: str = Field(..., description="Model file path")
    best_model: str = Field(..., description="Best model name from ensemble")
    num_features: int = Field(..., description="Number of features", ge=1)


class ClassificationPrediction(BaseModel):
    """Classification prediction results"""
    failure_type: str = Field(..., description="Predicted failure type")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    failure_probability: float = Field(..., description="Overall failure probability", ge=0.0, le=1.0)
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    model_info: ModelInfo = Field(..., description="Model information")


class ExplanationInfo(BaseModel):
    """LLM-generated explanation"""
    summary: str = Field(..., description="Brief explanation summary")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Maintenance recommendations")


class ClassificationResponse(BaseModel):
    """Response model for classification prediction"""
    machine_id: str = Field(..., description="Machine identifier")
    prediction: ClassificationPrediction = Field(..., description="Prediction results")
    explanation: ExplanationInfo = Field(..., description="LLM explanation")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "prediction": {
                    "failure_type": "normal",
                    "confidence": 0.95,
                    "failure_probability": 0.05,
                    "all_probabilities": {
                        "normal": 0.95,
                        "bearing_wear": 0.03,
                        "overheating": 0.01,
                        "electrical_fault": 0.01
                    },
                    "model_info": {
                        "path": "ml_models/models/classification/motor_siemens_1la7_001",
                        "best_model": "WeightedEnsemble_L2",
                        "num_features": 22
                    }
                },
                "explanation": {
                    "summary": "Machine is operating normally with 95% confidence. All sensor readings are within normal ranges.",
                    "risk_factors": [],
                    "recommendations": [
                        "Continue normal operation",
                        "Monitor bearing temperature trends",
                        "Next inspection in 7 days"
                    ]
                },
                "timestamp": "2025-12-15T10:45:23Z"
            }
        }


# ============================================================
# Response Models - RUL Prediction
# ============================================================

class CriticalSensor(BaseModel):
    """Critical sensor information"""
    name: str = Field(..., description="Sensor name")
    value: float = Field(..., description="Current sensor value")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    threshold: float = Field(..., description="Threshold value")


class RULPrediction(BaseModel):
    """RUL prediction results"""
    rul_hours: float = Field(..., description="Remaining useful life in hours", ge=0)
    rul_days: float = Field(..., description="Remaining useful life in days", ge=0)
    urgency: str = Field(..., description="Urgency level (low, medium, high, critical)")
    maintenance_window: str = Field(..., description="Recommended maintenance window")
    critical_sensors: List[CriticalSensor] = Field(..., description="Critical sensors requiring attention")
    estimated_failure_date: datetime = Field(..., description="Estimated failure date")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)


class RULResponse(BaseModel):
    """Response model for RUL prediction"""
    machine_id: str = Field(..., description="Machine identifier")
    prediction: RULPrediction = Field(..., description="RUL prediction results")
    explanation: ExplanationInfo = Field(..., description="LLM explanation")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "prediction": {
                    "rul_hours": 156.3,
                    "rul_days": 6.51,
                    "urgency": "medium",
                    "maintenance_window": "within 3 days",
                    "critical_sensors": [
                        {
                            "name": "bearing_de_temp_C",
                            "value": 65.2,
                            "severity": "medium",
                            "threshold": 70.0
                        }
                    ],
                    "estimated_failure_date": "2025-12-21T10:45:00Z",
                    "confidence": 0.85
                },
                "explanation": {
                    "summary": "Machine has approximately 6.5 days of remaining useful life. Schedule maintenance within 3 days.",
                    "risk_factors": [
                        "Bearing temperature trending upward",
                        "Vibration levels increasing gradually"
                    ],
                    "recommendations": [
                        "Schedule maintenance inspection",
                        "Check bearing lubrication",
                        "Monitor temperature every 4 hours"
                    ]
                },
                "timestamp": "2025-12-15T10:45:23Z"
            }
        }


# ============================================================
# Response Models - Anomaly Detection
# ============================================================


class AnomalyPrediction(BaseModel):
    """Anomaly detection results"""
    is_anomaly: bool = Field(..., description="Whether the reading is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (0-1)")
    detection_method: str = Field(..., description="Detection method used")
    abnormal_sensors: Dict[str, float] = Field(..., description="Sensors contributing to anomaly")


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection"""
    machine_id: str = Field(..., description="Machine identifier")
    prediction: AnomalyPrediction = Field(..., description="Anomaly detection results")
    explanation: ExplanationInfo = Field(..., description="LLM explanation")
    timestamp: datetime = Field(..., description="Prediction timestamp")


# ============================================================
# Response Models - Time-series Forecast
# ============================================================


class TimeSeriesPrediction(BaseModel):
    """Time-series forecast results"""
    forecast_summary: str = Field(..., description="Human-readable forecast summary")
    confidence: float = Field(..., description="Forecast confidence", ge=0.0, le=1.0)
    forecast_horizon: str = Field(..., description="Forecast horizon")
    forecasts: Dict[str, Any] = Field(..., description="Optional structured forecast outputs")


class TimeSeriesResponse(BaseModel):
    """Response model for time-series forecast"""
    machine_id: str = Field(..., description="Machine identifier")
    prediction: TimeSeriesPrediction = Field(..., description="Forecast results")
    explanation: ExplanationInfo = Field(..., description="LLM explanation")
    timestamp: datetime = Field(..., description="Prediction timestamp")


# ============================================================
# Response Models - Prediction History
# ============================================================

class PredictionHistoryItem(BaseModel):
    """Single prediction history item"""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    failure_type: str = Field(..., description="Predicted failure type")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    rul_hours: Optional[float] = Field(None, description="RUL in hours (if available)")
    urgency: Optional[str] = Field(None, description="Urgency level (if available)")


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history"""
    machine_id: str = Field(..., description="Machine identifier")
    predictions: List[PredictionHistoryItem] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total count of predictions", ge=0)
    page: int = Field(..., description="Current page number", ge=1)
    per_page: int = Field(..., description="Items per page", ge=1)
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "predictions": [
                    {
                        "timestamp": "2025-12-15T10:45:00Z",
                        "failure_type": "normal",
                        "confidence": 0.95,
                        "rul_hours": 156.3,
                        "urgency": "medium"
                    },
                    {
                        "timestamp": "2025-12-15T10:30:00Z",
                        "failure_type": "normal",
                        "confidence": 0.94,
                        "rul_hours": 158.1,
                        "urgency": "medium"
                    }
                ],
                "total": 100,
                "page": 1,
                "per_page": 100
            }
        }


# ============================================================
# Response Models - Health Check
# ============================================================

class ModelsLoaded(BaseModel):
    """Count of loaded models by type"""
    classification: int = Field(..., description="Classification models loaded", ge=0)
    regression: int = Field(..., description="Regression models loaded", ge=0)
    anomaly: int = Field(..., description="Anomaly models loaded", ge=0)
    timeseries: int = Field(..., description="Time-series models loaded", ge=0)


class GPUInfo(BaseModel):
    """GPU information"""
    name: str = Field(..., description="GPU name")
    cuda_version: str = Field(..., description="CUDA version")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    models_loaded: ModelsLoaded = Field(..., description="Count of loaded models")
    llm_status: str = Field(..., description="LLM service status")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_info: Optional[GPUInfo] = Field(None, description="GPU information (if available)")
    integrated_system_ready: bool = Field(..., description="IntegratedPredictionSystem status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "classification": 10,
                    "regression": 8,
                    "anomaly": 5,
                    "timeseries": 3
                },
                "llm_status": "operational",
                "gpu_available": True,
                "gpu_info": {
                    "name": "NVIDIA GeForce RTX 4070",
                    "cuda_version": "12.1"
                },
                "integrated_system_ready": True,
                "timestamp": "2025-12-15T10:45:23Z"
            }
        }


# ============================================================
# Error Response Model
# ============================================================

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "MachineNotFound",
                "detail": "Machine 'invalid_machine_id' not found",
                "timestamp": "2025-12-15T10:45:23Z"
            }
        }
