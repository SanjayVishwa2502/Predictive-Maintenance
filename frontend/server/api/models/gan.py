"""
GAN API Request/Response Models
Professional Pydantic models with comprehensive validation
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class MachineStatus(str, Enum):
    """Machine workflow status"""
    NOT_STARTED = "not_started"
    SEED_GENERATED = "seed_generated"
    TRAINING = "training"
    TRAINED = "trained"
    SYNTHETIC_GENERATED = "synthetic_generated"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Celery task status"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class FileFormat(str, Enum):
    """Supported file formats for profile upload"""
    JSON = "json"
    YAML = "yaml"
    EXCEL = "excel"
    CSV = "csv"


# ============================================================================
# SENSOR CONFIGURATION
# ============================================================================

class SensorConfig(BaseModel):
    """Individual sensor configuration"""
    name: str = Field(..., min_length=2, max_length=100,
                     description="Sensor name (e.g., 'winding_temp_C')")
    display_name: str = Field(..., min_length=2, max_length=100,
                              description="Human-readable name")
    unit: str = Field(..., min_length=1, max_length=20,
                     description="Unit of measurement (e.g., '°C', 'A', 'mm/s')")
    min_value: float = Field(..., description="Minimum sensor value")
    max_value: float = Field(..., description="Maximum sensor value")
    sensor_type: str = Field(..., description="Type: temperature, vibration, current, etc.")
    is_critical: bool = Field(default=False, description="Critical sensor flag")
    
    @field_validator('max_value')
    @classmethod
    def validate_max_greater_than_min(cls, v, info):
        min_val = info.data.get('min_value')
        if min_val is not None and v <= min_val:
            raise ValueError('max_value must be greater than min_value')
        return v
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "name": "winding_temp_C",
                "display_name": "Winding Temperature",
                "unit": "°C",
                "min_value": 20.0,
                "max_value": 120.0,
                "sensor_type": "temperature",
                "is_critical": True
            }
        }


# ============================================================================
# PROFILE MANAGEMENT - REQUEST MODELS
# ============================================================================

class ProfileUploadRequest(BaseModel):
    """Upload machine profile"""
    machine_id: str = Field(
        ..., 
        min_length=3, 
        max_length=100,
        pattern="^[a-z0-9_]+$",
        description="Lowercase machine ID with underscores"
    )
    machine_type: str = Field(
        ..., 
        min_length=2,
        max_length=50,
        description="Machine category (motor, pump, cnc, etc.)"
    )
    manufacturer: str = Field(..., min_length=2, max_length=100)
    model: str = Field(..., min_length=2, max_length=100)
    sensors: List[SensorConfig] = Field(
        ..., 
        min_items=1, 
        max_items=50,
        description="List of sensor configurations"
    )
    degradation_states: int = Field(
        4, 
        ge=2, 
        le=10,
        description="Number of health states (default: 4)"
    )
    rul_min: int = Field(
        0,
        ge=0,
        description="Minimum RUL in hours"
    )
    rul_max: int = Field(
        1000,
        ge=1,
        le=100000,
        description="Maximum RUL in hours"
    )
    
    @field_validator('machine_id')
    @classmethod
    def validate_machine_id_format(cls, v):
        if not v.islower():
            raise ValueError('machine_id must be lowercase')
        if '__' in v:
            raise ValueError('machine_id cannot contain consecutive underscores')
        return v
    
    @model_validator(mode='after')
    def validate_rul_range(self):
        if self.rul_max <= self.rul_min:
            raise ValueError('rul_max must be greater than rul_min')
        return self
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "machine_type": "motor",
                "manufacturer": "Siemens",
                "model": "1LA7 090-4AA60",
                "sensors": [
                    {
                        "name": "winding_temp_C",
                        "display_name": "Winding Temperature",
                        "unit": "°C",
                        "min_value": 20.0,
                        "max_value": 120.0,
                        "sensor_type": "temperature",
                        "is_critical": True
                    }
                ],
                "degradation_states": 4,
                "rul_min": 0,
                "rul_max": 1000
            }
        }


class ProfileValidationRequest(BaseModel):
    """Validate uploaded profile"""
    profile_id: str = Field(..., description="Profile UUID")
    strict: bool = Field(
        default=True,
        description="Strict validation (check for duplicates, etc.)"
    )


class ProfileEditRequest(BaseModel):
    """Edit profile after validation"""
    profile_id: str = Field(..., description="Profile UUID")
    updates: Dict[str, Any] = Field(
        ...,
        description="Dictionary of fields to update"
    )


# ============================================================================
# PROFILE MANAGEMENT - RESPONSE MODELS
# ============================================================================

class TemplateInfo(BaseModel):
    """Machine profile template info"""
    machine_type: str
    display_name: str
    manufacturer: str
    model: str
    num_sensors: int
    degradation_states: int
    file_path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_type": "motor",
                "display_name": "AC Induction Motor",
                "manufacturer": "Siemens",
                "model": "1LA7 Series",
                "num_sensors": 8,
                "degradation_states": 4,
                "file_path": "GAN/metadata/templates/motor_template.json"
            }
        }


class ProfileUploadResponse(BaseModel):
    """Profile upload response"""
    success: bool
    profile_id: str
    machine_id: str
    message: str
    validation_required: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "profile_id": "a1b2c3d4-e5f6-7890",
                "machine_id": "motor_siemens_1la7_001",
                "message": "Profile uploaded successfully. Validation required.",
                "validation_required": True
            }
        }


class ValidationIssue(BaseModel):
    """Single validation issue"""
    severity: str = Field(..., description="error | warning | info")
    field: str
    message: str


class InlineProfileValidationRequest(BaseModel):
    """Validate a profile JSON payload without staging it on disk."""
    profile_data: Dict[str, Any] = Field(..., description="Machine profile JSON payload")
    strict: bool = Field(default=True, description="Strict validation (duplicates, TVAE rules)")


class InlineProfileValidationResponse(BaseModel):
    """Inline validation result."""
    valid: bool
    machine_id: str
    issues: List[ValidationIssue] = []
    can_proceed: bool
    message: str


class ProfileValidationResponse(BaseModel):
    """Profile validation response"""
    valid: bool
    profile_id: str
    machine_id: str
    issues: List[ValidationIssue] = []
    can_proceed: bool
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "valid": True,
                "profile_id": "a1b2c3d4-e5f6-7890",
                "machine_id": "motor_siemens_1la7_001",
                "issues": [],
                "can_proceed": True,
                "message": "Profile validation passed"
            }
        }


# ============================================================================
# WORKFLOW OPERATIONS - REQUEST MODELS
# ============================================================================

class SeedGenerationRequest(BaseModel):
    """Request to generate seed data"""
    samples: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Number of temporal samples (1K-100K)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "samples": 10000
            }
        }


class TrainingRequest(BaseModel):
    """Request to train TVAE model"""
    epochs: int = Field(
        300,
        ge=50,
        le=1000,
        description="Training epochs (50-1000)"
    )
    batch_size: int = Field(
        500,
        ge=100,
        le=2000,
        description="Batch size for training"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "epochs": 300,
                "batch_size": 500
            }
        }


class GenerationRequest(BaseModel):
    """Request to generate synthetic data"""
    train_samples: int = Field(
        35000,
        ge=1000,
        le=100000,
        description="Training set samples"
    )
    val_samples: int = Field(
        7500,
        ge=100,
        le=20000,
        description="Validation set samples"
    )
    test_samples: int = Field(
        7500,
        ge=100,
        le=20000,
        description="Test set samples"
    )
    
    @field_validator('val_samples', 'test_samples')
    @classmethod
    def validate_split_ratio(cls, v, info):
        train = info.data.get('train_samples', 0)
        if v > train * 0.5:
            raise ValueError(
                'Validation/Test samples too large relative to training samples. '
                'Should not exceed 50% of training size.'
            )
        return v
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "train_samples": 35000,
                "val_samples": 7500,
                "test_samples": 7500
            }
        }


# ============================================================================
# WORKFLOW OPERATIONS - RESPONSE MODELS
# ============================================================================

class SeedGenerationResponse(BaseModel):
    """Seed generation response"""
    machine_id: str
    samples_generated: int
    file_path: str
    file_size_mb: float
    generation_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "samples_generated": 10000,
                "file_path": "GAN/seed_data/motor_siemens_1la7_001_temporal_seed.parquet",
                "file_size_mb": 2.45,
                "generation_time_seconds": 12.34,
                "timestamp": "2024-12-15T16:30:00Z"
            }
        }


class TrainingResponse(BaseModel):
    """Training initiation response"""
    success: bool
    machine_id: str
    task_id: str
    epochs: int
    estimated_time_minutes: float
    websocket_url: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "machine_id": "motor_siemens_1la7_001",
                "task_id": "a1b2c3d4-e5f6-7890-abcd-ef0123456789",
                "epochs": 300,
                "estimated_time_minutes": 4.0,
                "websocket_url": "/ws/gan/training/a1b2c3d4-e5f6-7890-abcd-ef0123456789",
                "message": "Training started successfully"
            }
        }


class GenerationResponse(BaseModel):
    """Synthetic data generation response"""
    machine_id: str
    train_samples: int
    val_samples: int
    test_samples: int
    train_file: str
    val_file: str
    test_file: str
    generation_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "train_samples": 35000,
                "val_samples": 7500,
                "test_samples": 7500,
                "train_file": "GAN/data/motor_siemens_1la7_001_train.parquet",
                "val_file": "GAN/data/motor_siemens_1la7_001_val.parquet",
                "test_file": "GAN/data/motor_siemens_1la7_001_test.parquet",
                "generation_time_seconds": 45.67,
                "timestamp": "2024-12-15T16:35:00Z"
            }
        }


# ============================================================================
# MACHINE MANAGEMENT - RESPONSE MODELS
# ============================================================================

class MachineWorkflowStatus(BaseModel):
    """Machine workflow status"""
    machine_id: str
    status: MachineStatus
    has_metadata: bool
    has_seed_data: bool
    has_trained_model: bool
    has_synthetic_data: bool
    can_generate_seed: bool
    can_train_model: bool
    can_generate_synthetic: bool
    last_updated: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "status": "trained",
                "has_metadata": True,
                "has_seed_data": True,
                "has_trained_model": True,
                "has_synthetic_data": False,
                "can_generate_seed": True,
                "can_train_model": True,
                "can_generate_synthetic": True,
                "last_updated": "2024-12-15T16:00:00Z"
            }
        }


class MachineDetails(BaseModel):
    """Detailed machine information"""
    machine_id: str
    machine_type: str
    manufacturer: str
    model: str
    num_sensors: int
    degradation_states: int
    status: MachineWorkflowStatus
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "machine_type": "motor",
                "manufacturer": "Siemens",
                "model": "1LA7 090-4AA60",
                "num_sensors": 8,
                "degradation_states": 4,
                "status": {
                    "machine_id": "motor_siemens_1la7_001",
                    "status": "trained",
                    "has_metadata": True,
                    "has_seed_data": True,
                    "has_trained_model": True,
                    "has_synthetic_data": False,
                    "can_generate_seed": True,
                    "can_train_model": True,
                    "can_generate_synthetic": True,
                    "last_updated": "2024-12-15T16:00:00Z"
                },
                "created_at": "2024-12-10T10:00:00Z",
                "updated_at": "2024-12-15T16:00:00Z"
            }
        }


class MachineListResponse(BaseModel):
    """List of machines"""
    total: int
    machines: List[str]
    machine_details: Optional[List[MachineDetails]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 29,
                "machines": [
                    "motor_siemens_1la7_001",
                    "pump_grundfos_cr3_004",
                    "cnc_dmg_nlx_010"
                ],
                "machine_details": [
                    {
                        "machine_id": "motor_siemens_1la7_001",
                        "machine_type": "motor",
                        "manufacturer": "Siemens",
                        "model": "1LA7 090-4AA60",
                        "num_sensors": 8,
                        "degradation_states": 4,
                        "status": {
                            "machine_id": "motor_siemens_1la7_001",
                            "status": "trained",
                            "has_metadata": True,
                            "has_seed_data": True,
                            "has_trained_model": True,
                            "has_synthetic_data": False,
                            "can_generate_seed": True,
                            "can_train_model": True,
                            "can_generate_synthetic": True,
                            "last_updated": "2024-12-15T16:00:00Z"
                        },
                        "created_at": "2024-12-10T10:00:00Z",
                        "updated_at": "2024-12-15T16:00:00Z"
                    }
                ]
            }
        }


# ============================================================================
# TASK MONITORING - RESPONSE MODELS
# ============================================================================

class TaskProgress(BaseModel):
    """Task progress details"""
    current: int
    total: int
    progress_percent: float
    epoch: Optional[int] = None
    loss: Optional[float] = None
    stage: Optional[str] = None
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Celery task status"""
    task_id: str
    status: TaskStatus
    machine_id: Optional[str] = None
    progress: Optional[TaskProgress] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "a1b2c3d4-e5f6-7890",
                "status": "PROGRESS",
                "machine_id": "motor_siemens_1la7_001",
                "progress": {
                    "current": 150,
                    "total": 300,
                    "progress_percent": 50.0,
                    "epoch": 150,
                    "loss": 0.0452,
                    "stage": "training",
                    "message": "Epoch 150/300, Loss: 0.0452"
                },
                "result": None,
                "error": None,
                "started_at": "2024-12-15T16:20:00Z",
                "completed_at": None
            }
        }


# ============================================================================
# HEALTH CHECK - RESPONSE MODEL
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Service health check"""
    status: str = Field(..., description="healthy | degraded | unhealthy")
    service: str = "GAN Manager"
    total_operations: int
    available_machines: int
    paths_accessible: bool
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "GAN Manager",
                "total_operations": 145,
                "available_machines": 29,
                "paths_accessible": True,
                "timestamp": "2024-12-15T16:30:00Z"
            }
        }


# ============================================================================
# ERROR RESPONSE MODEL
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    status_code: int
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "machine_id must be lowercase",
                "status_code": 400,
                "timestamp": "2024-12-15T16:30:00Z"
            }
        }

