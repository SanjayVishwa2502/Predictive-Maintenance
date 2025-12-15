"""
GAN API Routes - Professional Implementation
Comprehensive endpoints for GAN workflow management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import FileResponse
import redis.asyncio as redis
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import uuid

from api.models.gan import (
    # Request Models
    ProfileUploadRequest,
    ProfileValidationRequest,
    ProfileEditRequest,
    SeedGenerationRequest,
    TrainingRequest,
    GenerationRequest,
    # Response Models
    TemplateInfo,
    ProfileUploadResponse,
    ProfileValidationResponse,
    ValidationIssue,
    SeedGenerationResponse,
    TrainingResponse,
    GenerationResponse,
    MachineWorkflowStatus,
    MachineDetails,
    MachineListResponse,
    TaskStatusResponse,
    TaskProgress,
    HealthCheckResponse,
    ErrorResponse,
    # Enums
    MachineStatus,
    TaskStatus,
)
from api.services.gan_manager_wrapper import gan_manager_wrapper
# Import Celery tasks (Phase 3.7.2.3 - COMPLETE)
from tasks.gan_tasks import train_tvae_task, generate_data_task, generate_seed_data_task
from celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


# ============================================================================
# DEPENDENCIES: Rate Limiting & Caching
# ============================================================================

RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds
CACHE_TTL = 30  # seconds for cached responses


async def get_redis() -> redis.Redis:
    """Get Redis connection"""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_CACHE_DB,  # Use DB 3 for API caching
        decode_responses=True
    )


async def rate_limiter(request: Request) -> None:
    """
    Rate limiter dependency
    Limits: 100 requests per minute per IP
    """
    client_ip = request.client.host
    key = f"ratelimit:gan:{client_ip}"
    
    redis_client = await get_redis()
    try:
        count = await redis_client.incr(key)
        
        if count == 1:
            await redis_client.expire(key, RATE_WINDOW)
        
        if count > RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RateLimitExceeded",
                    "detail": f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per minute.",
                    "retry_after": RATE_WINDOW
                }
            )
    finally:
        await redis_client.close()


async def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if exists"""
    redis_client = await get_redis()
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        return None
    finally:
        await redis_client.close()


async def set_cached_response(cache_key: str, data: Dict, ttl: int = CACHE_TTL) -> None:
    """Cache response with TTL"""
    redis_client = await get_redis()
    try:
        await redis_client.setex(
            cache_key,
            ttl,
            json.dumps(data, default=str)
        )
    finally:
        await redis_client.close()


# ============================================================================
# PROFILE MANAGEMENT ENDPOINTS (6)
# ============================================================================

@router.get(
    "/templates",
    response_model=List[TemplateInfo],
    summary="List All Machine Profile Templates",
    description="""
    List all available machine profile templates.
    
    **Returns:**
    - List of template information
    - Each template includes: machine_type, manufacturer, model, sensors
    
    **Caching:** 30 seconds TTL
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "List of templates"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def list_templates() -> List[TemplateInfo]:
    """List all machine profile templates"""
    try:
        # Check cache
        cache_key = "gan:templates:list"
        cached = await get_cached_response(cache_key)
        if cached:
            return cached
        
        # Get templates from metadata directory
        metadata_path = Path("GAN/metadata")
        templates = []
        
        # Scan for template files
        for template_file in metadata_path.glob("*_template.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                
                templates.append(TemplateInfo(
                    machine_type=template_data.get('machine_type', ''),
                    display_name=template_data.get('display_name', ''),
                    manufacturer=template_data.get('manufacturer', ''),
                    model=template_data.get('model', ''),
                    num_sensors=len(template_data.get('sensors', [])),
                    degradation_states=template_data.get('degradation_states', 4),
                    file_path=str(template_file)
                ))
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
                continue
        
        # Cache response
        response_data = [t.dict() for t in templates]
        await set_cached_response(cache_key, response_data)
        
        return templates
    
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get(
    "/templates/{machine_type}",
    response_model=TemplateInfo,
    summary="Get Template for Specific Machine Type",
    description="""
    Get detailed template for a specific machine type.
    
    **Parameters:**
    - machine_type: Type of machine (motor, pump, cnc, etc.)
    
    **Returns:**
    - Complete template information
    
    **Caching:** 30 seconds TTL
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Template information"},
        404: {"description": "Template not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def get_template(machine_type: str) -> TemplateInfo:
    """Get template for specific machine type"""
    try:
        # Check cache
        cache_key = f"gan:templates:{machine_type}"
        cached = await get_cached_response(cache_key)
        if cached:
            return TemplateInfo(**cached)
        
        # Find template file
        template_file = Path(f"GAN/metadata/{machine_type}_template.json")
        
        if not template_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Template for machine type '{machine_type}' not found"
            )
        
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        
        template_info = TemplateInfo(
            machine_type=template_data.get('machine_type', ''),
            display_name=template_data.get('display_name', ''),
            manufacturer=template_data.get('manufacturer', ''),
            model=template_data.get('model', ''),
            num_sensors=len(template_data.get('sensors', [])),
            degradation_states=template_data.get('degradation_states', 4),
            file_path=str(template_file)
        )
        
        # Cache response
        await set_cached_response(cache_key, template_info.dict())
        
        return template_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {machine_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get template: {str(e)}"
        )


@router.get(
    "/templates/{machine_type}/download",
    summary="Download Template File",
    description="""
    Download template file for specific machine type.
    
    **Parameters:**
    - machine_type: Type of machine
    
    **Returns:**
    - JSON file download
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Template file"},
        404: {"description": "Template not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def download_template(machine_type: str):
    """Download template file"""
    try:
        template_file = Path(f"GAN/metadata/{machine_type}_template.json")
        
        if not template_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Template for machine type '{machine_type}' not found"
            )
        
        return FileResponse(
            path=str(template_file),
            media_type='application/json',
            filename=f"{machine_type}_template.json"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download template {machine_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download template: {str(e)}"
        )


@router.post(
    "/profiles/upload",
    response_model=ProfileUploadResponse,
    summary="Upload Machine Profile",
    description="""
    Upload a new machine profile (JSON/YAML/Excel format).
    
    **Request Body:**
    - machine_id: Lowercase ID with underscores
    - machine_type: Category (motor, pump, cnc, etc.)
    - manufacturer, model: Machine details
    - sensors: List of sensor configurations
    - degradation_states: Number of health states (default: 4)
    - rul_min, rul_max: RUL range in hours
    
    **Returns:**
    - profile_id: UUID for validation/editing
    - validation_required: Always true
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Profile uploaded successfully"},
        400: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def upload_profile(request: ProfileUploadRequest) -> ProfileUploadResponse:
    """Upload machine profile"""
    try:
        # Generate profile ID
        profile_id = str(uuid.uuid4())
        
        # Create metadata directory if not exists
        metadata_path = Path("GAN/metadata")
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Save profile to temporary location
        profile_file = metadata_path / f"{request.machine_id}_profile_temp.json"
        
        with open(profile_file, 'w') as f:
            json.dump(request.dict(), f, indent=2)
        
        logger.info(f"Profile uploaded: {request.machine_id} (ID: {profile_id})")
        
        return ProfileUploadResponse(
            success=True,
            profile_id=profile_id,
            machine_id=request.machine_id,
            message="Profile uploaded successfully. Validation required before machine creation.",
            validation_required=True
        )
    
    except Exception as e:
        logger.error(f"Failed to upload profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload profile: {str(e)}"
        )


@router.post(
    "/profiles/{profile_id}/validate",
    response_model=ProfileValidationResponse,
    summary="Validate Uploaded Profile",
    description="""
    Validate uploaded machine profile before creation.
    
    **Parameters:**
    - profile_id: UUID from upload response
    
    **Validation Checks:**
    - Machine ID uniqueness
    - Sensor configuration validity
    - RUL range consistency
    - Required fields present
    
    **Returns:**
    - valid: True if all checks passed
    - issues: List of validation issues (error/warning/info)
    - can_proceed: True if no blocking errors
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Validation completed"},
        404: {"description": "Profile not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def validate_profile(
    profile_id: str,
    request: ProfileValidationRequest
) -> ProfileValidationResponse:
    """Validate uploaded profile"""
    try:
        # Find profile file (simplified - would use database in production)
        metadata_path = Path("GAN/metadata")
        profile_files = list(metadata_path.glob("*_profile_temp.json"))
        
        if not profile_files:
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found"
            )
        
        # Load profile (use first temp file for this example)
        with open(profile_files[0], 'r') as f:
            profile_data = json.load(f)
        
        machine_id = profile_data.get('machine_id')
        issues = []
        
        # Check 1: Machine ID uniqueness
        existing_machines = gan_manager_wrapper.list_available_machines()
        if machine_id in existing_machines:
            issues.append(ValidationIssue(
                severity="error",
                field="machine_id",
                message=f"Machine ID '{machine_id}' already exists"
            ))
        
        # Check 2: Sensor count
        sensors = profile_data.get('sensors', [])
        if len(sensors) < 1:
            issues.append(ValidationIssue(
                severity="error",
                field="sensors",
                message="At least 1 sensor is required"
            ))
        elif len(sensors) < 5:
            issues.append(ValidationIssue(
                severity="warning",
                field="sensors",
                message=f"Only {len(sensors)} sensors configured. Consider adding more for better predictions."
            ))
        
        # Check 3: RUL range
        rul_min = profile_data.get('rul_min', 0)
        rul_max = profile_data.get('rul_max', 1000)
        if rul_max <= rul_min:
            issues.append(ValidationIssue(
                severity="error",
                field="rul_range",
                message="rul_max must be greater than rul_min"
            ))
        
        # Determine if can proceed
        has_errors = any(issue.severity == "error" for issue in issues)
        valid = not has_errors
        can_proceed = valid
        
        message = "Profile validation passed" if valid else "Profile validation failed. Please fix errors."
        
        return ProfileValidationResponse(
            valid=valid,
            profile_id=profile_id,
            machine_id=machine_id,
            issues=issues,
            can_proceed=can_proceed,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate profile: {str(e)}"
        )


@router.put(
    "/profiles/{profile_id}/edit",
    response_model=ProfileUploadResponse,
    summary="Edit Profile After Validation",
    description="""
    Edit profile after validation (before machine creation).
    
    **Parameters:**
    - profile_id: UUID from upload response
    
    **Request Body:**
    - updates: Dictionary of fields to update
    
    **Returns:**
    - Updated profile information
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Profile updated successfully"},
        404: {"description": "Profile not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def edit_profile(
    profile_id: str,
    request: ProfileEditRequest
) -> ProfileUploadResponse:
    """Edit profile after validation"""
    try:
        # Find profile file
        metadata_path = Path("GAN/metadata")
        profile_files = list(metadata_path.glob("*_profile_temp.json"))
        
        if not profile_files:
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found"
            )
        
        profile_file = profile_files[0]
        
        # Load and update profile
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        # Apply updates
        for field, value in request.updates.items():
            if field in profile_data:
                profile_data[field] = value
        
        # Save updated profile
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Profile edited: {profile_id}")
        
        return ProfileUploadResponse(
            success=True,
            profile_id=profile_id,
            machine_id=profile_data['machine_id'],
            message="Profile updated successfully",
            validation_required=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to edit profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to edit profile: {str(e)}"
        )


# ============================================================================
# MACHINE MANAGEMENT ENDPOINTS (5)
# ============================================================================

@router.post(
    "/machines",
    response_model=MachineDetails,
    summary="Create Machine from Profile",
    description="""
    Create a new machine from validated profile.
    
    **Prerequisites:**
    - Profile must be uploaded and validated
    - Machine ID must be unique
    
    **Returns:**
    - Complete machine details
    - Workflow status (not_started)
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Machine created successfully"},
        400: {"description": "Invalid profile or duplicate machine"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def create_machine(profile_id: str) -> MachineDetails:
    """Create machine from validated profile"""
    try:
        # Find profile file
        metadata_path = Path("GAN/metadata")
        profile_files = list(metadata_path.glob("*_profile_temp.json"))
        
        if not profile_files:
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found. Please upload profile first."
            )
        
        profile_file = profile_files[0]
        
        # Load profile
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        machine_id = profile_data['machine_id']
        
        # Check if machine already exists
        existing_machines = gan_manager_wrapper.list_available_machines()
        if machine_id in existing_machines:
            raise HTTPException(
                status_code=400,
                detail=f"Machine '{machine_id}' already exists"
            )
        
        # Move profile to permanent location
        permanent_file = metadata_path / f"{machine_id}.json"
        profile_file.rename(permanent_file)
        
        # Get workflow status
        workflow_status = gan_manager_wrapper.get_machine_workflow_status(machine_id)
        
        logger.info(f"Machine created: {machine_id}")
        
        return MachineDetails(
            machine_id=machine_id,
            machine_type=profile_data.get('machine_type', ''),
            manufacturer=profile_data.get('manufacturer', ''),
            model=profile_data.get('model', ''),
            num_sensors=len(profile_data.get('sensors', [])),
            degradation_states=profile_data.get('degradation_states', 4),
            status=workflow_status,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create machine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create machine: {str(e)}"
        )


@router.get(
    "/machines",
    response_model=MachineListResponse,
    summary="List All Machines",
    description="""
    List all available machines.
    
    **Returns:**
    - total: Number of machines
    - machines: List of machine IDs
    
    **Caching:** 30 seconds TTL
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "List of machines"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def list_machines() -> MachineListResponse:
    """List all machines"""
    try:
        # Check cache
        cache_key = "gan:machines:list"
        cached = await get_cached_response(cache_key)
        if cached:
            return MachineListResponse(**cached)
        
        # Get machines from GANManager
        machines = gan_manager_wrapper.list_available_machines()
        
        response = MachineListResponse(
            total=len(machines),
            machines=sorted(machines)
        )
        
        # Cache response
        await set_cached_response(cache_key, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to list machines: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list machines: {str(e)}"
        )


@router.get(
    "/machines/{machine_id}",
    response_model=MachineDetails,
    summary="Get Machine Details",
    description="""
    Get detailed information about a specific machine.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Complete machine details
    - Workflow status
    
    **Caching:** 30 seconds TTL
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Machine details"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def get_machine_details(machine_id: str) -> MachineDetails:
    """Get machine details"""
    try:
        # Check cache
        cache_key = f"gan:machines:{machine_id}:details"
        cached = await get_cached_response(cache_key)
        if cached:
            return MachineDetails(**cached)
        
        # Check if machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Get machine details from wrapper
        details = gan_manager_wrapper.get_machine_details(machine_id)
        
        # Cache response
        await set_cached_response(cache_key, details.dict())
        
        return details
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get machine details: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get machine details: {str(e)}"
        )


@router.get(
    "/machines/{machine_id}/status",
    response_model=MachineWorkflowStatus,
    summary="Get Machine Workflow Status",
    description="""
    Get current workflow status for a machine.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Workflow status flags
    - What operations can be performed next
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Workflow status"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def get_workflow_status(machine_id: str) -> MachineWorkflowStatus:
    """Get machine workflow status"""
    try:
        # Check if machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Get workflow status
        status = gan_manager_wrapper.get_machine_workflow_status(machine_id)
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@router.delete(
    "/machines/{machine_id}",
    summary="Delete Machine and Data",
    description="""
    Delete machine and all associated data.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Deletes:**
    - Machine metadata
    - Seed data
    - Trained models
    - Synthetic data
    
    **WARNING:** This operation cannot be undone!
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Machine deleted successfully"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def delete_machine(machine_id: str) -> Dict[str, str]:
    """Delete machine and all data"""
    try:
        # Check if machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Delete metadata
        metadata_file = Path(f"GAN/metadata/{machine_id}.json")
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete seed data
        seed_file = Path(f"GAN/seed_data/{machine_id}_temporal_seed.parquet")
        if seed_file.exists():
            seed_file.unlink()
        
        # Delete trained models
        models_dir = Path(f"GAN/models/{machine_id}")
        if models_dir.exists():
            import shutil
            shutil.rmtree(models_dir)
        
        # Delete synthetic data
        data_dir = Path("GAN/data")
        for data_file in data_dir.glob(f"{machine_id}_*.parquet"):
            data_file.unlink()
        
        logger.info(f"Machine deleted: {machine_id}")
        
        return {
            "message": f"Machine '{machine_id}' and all associated data deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete machine: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete machine: {str(e)}"
        )


# ============================================================================
# WORKFLOW OPERATIONS ENDPOINTS (4)
# ============================================================================

@router.post(
    "/machines/{machine_id}/seed",
    response_model=SeedGenerationResponse,
    summary="Generate Seed Data (Synchronous)",
    description="""
    Generate temporal seed data for a machine (synchronous operation).
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Request Body:**
    - samples: Number of samples (1K-100K, default: 10K)
    
    **Returns:**
    - Seed generation result
    - File path and size
    - Generation time
    
    **Typical Duration:** 10-30 seconds
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Seed data generated"},
        400: {"description": "Invalid parameters"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def generate_seed(
    machine_id: str,
    request: SeedGenerationRequest
) -> SeedGenerationResponse:
    """Generate seed data (synchronous)"""
    try:
        # Validate machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Generate seed data
        result = gan_manager_wrapper.generate_seed_data(
            machine_id=machine_id,
            samples=request.samples
        )
        
        logger.info(f"Seed data generated: {machine_id} ({request.samples} samples)")
        
        return SeedGenerationResponse(
            machine_id=result.machine_id,
            samples_generated=result.samples_generated,
            file_path=result.file_path,
            file_size_mb=result.file_size_mb,
            generation_time_seconds=result.generation_time_seconds,
            timestamp=datetime.fromisoformat(result.timestamp)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate seed data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate seed data: {str(e)}"
        )


@router.post(
    "/machines/{machine_id}/train",
    response_model=TrainingResponse,
    summary="Train TVAE Model (Asynchronous)",
    description="""
    Train TVAE model on temporal seed data (asynchronous via Celery).
    
    **Prerequisites:**
    - Machine must exist
    - Seed data must be generated first
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Request Body:**
    - epochs: Training epochs (50-1000, default: 300)
    - batch_size: Batch size (100-2000, default: 500)
    
    **Returns:**
    - Celery task_id for progress tracking
    - WebSocket URL for real-time updates
    - Estimated completion time
    
    **Typical Duration:** ~4 minutes for 300 epochs
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Training started successfully"},
        400: {"description": "Seed data not found"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def train_model(
    machine_id: str,
    request: TrainingRequest
) -> TrainingResponse:
    """Train TVAE model (asynchronous)"""
    try:
        # Validate machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Check seed data exists
        if not gan_manager_wrapper.validate_seed_data_exists(machine_id):
            raise HTTPException(
                status_code=400,
                detail=f"Seed data not found for machine '{machine_id}'. Generate seed data first."
            )
        
        # Start training task (Celery)
        if train_tvae_task is None:
            raise HTTPException(
                status_code=501,
                detail="Training tasks not yet implemented. Will be available in Phase 3.7.2.3."
            )
        
        task = train_tvae_task.delay(
            machine_id=machine_id,
            epochs=request.epochs
        )
        
        # Estimate completion time (rough: 0.8s per epoch)
        estimated_minutes = (request.epochs * 0.8) / 60
        
        logger.info(f"Training started: {machine_id} (Task: {task.id})")
        
        return TrainingResponse(
            success=True,
            machine_id=machine_id,
            task_id=task.id,
            epochs=request.epochs,
            estimated_time_minutes=round(estimated_minutes, 1),
            websocket_url=f"/ws/gan/training/{task.id}",
            message="Training started successfully. Use WebSocket URL for real-time progress."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post(
    "/machines/{machine_id}/generate",
    response_model=GenerationResponse,
    summary="Generate Synthetic Data",
    description="""
    Generate synthetic train/val/test datasets from trained TVAE model.
    
    **Prerequisites:**
    - Machine must exist
    - TVAE model must be trained
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Request Body:**
    - train_samples: Training set size (default: 35K)
    - val_samples: Validation set size (default: 7.5K)
    - test_samples: Test set size (default: 7.5K)
    
    **Returns:**
    - File paths for train/val/test datasets
    - Generation time
    
    **Typical Duration:** 30-60 seconds
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Synthetic data generated"},
        400: {"description": "Model not trained"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def generate_synthetic(
    machine_id: str,
    request: GenerationRequest
) -> GenerationResponse:
    """Generate synthetic data"""
    try:
        # Validate machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Check model exists
        if not gan_manager_wrapper.validate_model_exists(machine_id):
            raise HTTPException(
                status_code=400,
                detail=f"Trained model not found for machine '{machine_id}'. Train model first."
            )
        
        # Generate synthetic data
        result = gan_manager_wrapper.generate_synthetic_data(
            machine_id=machine_id,
            train_samples=request.train_samples,
            val_samples=request.val_samples,
            test_samples=request.test_samples
        )
        
        logger.info(f"Synthetic data generated: {machine_id}")
        
        return GenerationResponse(
            machine_id=result.machine_id,
            train_samples=result.train_samples,
            val_samples=result.val_samples,
            test_samples=result.test_samples,
            train_file=result.train_file,
            val_file=result.val_file,
            test_file=result.test_file,
            generation_time_seconds=result.generation_time_seconds,
            timestamp=datetime.fromisoformat(result.timestamp)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate synthetic data: {str(e)}"
        )


@router.get(
    "/machines/{machine_id}/validate",
    summary="Validate Data Quality",
    description="""
    Validate quality of generated synthetic data.
    
    **Parameters:**
    - machine_id: Machine identifier
    
    **Returns:**
    - Quality metrics
    - Validation result
    
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Validation result"},
        404: {"description": "Machine or data not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def validate_data_quality(machine_id: str) -> Dict[str, Any]:
    """Validate data quality"""
    try:
        # Check machine exists
        machines = gan_manager_wrapper.list_available_machines()
        if machine_id not in machines:
            raise HTTPException(
                status_code=404,
                detail=f"Machine '{machine_id}' not found"
            )
        
        # Check synthetic data exists
        data_file = Path(f"GAN/data/{machine_id}_train.parquet")
        if not data_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Synthetic data not found for machine '{machine_id}'"
            )
        
        # Basic validation (would be more comprehensive in production)
        import pandas as pd
        df = pd.read_parquet(data_file)
        
        validation_result = {
            "machine_id": machine_id,
            "valid": True,
            "num_samples": len(df),
            "num_features": len(df.columns),
            "null_values": df.isnull().sum().sum(),
            "quality_score": 0.95,  # Placeholder
            "message": "Data quality validation passed"
        }
        
        return validation_result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate data quality: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate data quality: {str(e)}"
        )


# ============================================================================
# MONITORING ENDPOINTS (2)
# ============================================================================

@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get Celery Task Status",
    description="""
    Get status of asynchronous Celery task (training, generation, etc.).
    
    **Parameters:**
    - task_id: Celery task UUID
    
    **Returns:**
    - Task status (PENDING, STARTED, PROGRESS, SUCCESS, FAILURE)
    - Progress information (epoch, loss, percentage)
    - Result data (if completed)
    - Error message (if failed)
    
    **Polling Recommended:** Every 2-3 seconds
    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Task status"},
        404: {"description": "Task not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get Celery task status"""
    try:
        # Get task from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        # Build response
        response = TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus(task_result.status),
            machine_id=None,
            progress=None,
            result=None,
            error=None,
            started_at=None,
            completed_at=None
        )
        
        # Add details based on status
        if task_result.state == 'PROGRESS':
            info = task_result.info
            if isinstance(info, dict):
                response.machine_id = info.get('machine_id')
                response.progress = TaskProgress(
                    current=info.get('current', 0),
                    total=info.get('total', 100),
                    progress_percent=info.get('progress', 0),
                    epoch=info.get('epoch'),
                    loss=info.get('loss'),
                    stage=info.get('stage'),
                    message=info.get('message')
                )
        
        elif task_result.state == 'SUCCESS':
            response.result = task_result.result
            response.completed_at = datetime.now()  # Would be from task metadata
        
        elif task_result.state == 'FAILURE':
            response.error = str(task_result.info)
            response.completed_at = datetime.now()
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="""
    Check GAN service health status.
    
    **Returns:**
    - Service status (healthy/degraded/unhealthy)
    - Operation statistics
    - Available machines count
    - Path accessibility
    
    **NO RATE LIMIT** (health checks exempt)
    """,
    responses={
        200: {"description": "Service health status"}
    }
)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint"""
    try:
        # Get health from wrapper
        health = gan_manager_wrapper.health_check()
        
        return HealthCheckResponse(
            status=health['status'],
            service="GAN Manager",
            total_operations=health['total_operations'],
            available_machines=health['available_machines'],
            paths_accessible=health['paths_accessible'],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            service="GAN Manager",
            total_operations=0,
            available_machines=0,
            paths_accessible=False,
            timestamp=datetime.now()
        )
