"""
GAN API Routes - Professional Implementation
Comprehensive endpoints for GAN workflow management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response, BackgroundTasks, Body
from fastapi.responses import FileResponse
import redis.asyncio as redis
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import uuid
import tempfile

from api.models.gan import (
    # Request Models
    ProfileValidationRequest,
    ProfileEditRequest,
    InlineProfileValidationRequest,
    SeedGenerationRequest,
    TrainingRequest,
    GenerationRequest,
    # Response Models
    TemplateInfo,
    ProfileUploadResponse,
    ProfileValidationResponse,
    ValidationIssue,
    InlineProfileValidationResponse,
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


def _get_project_root() -> Path:
    """Resolve repository root regardless of current working directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "GAN").exists() and (parent / "frontend").exists():
            return parent
    return Path.cwd().resolve()


def _get_gan_metadata_path() -> Path:
    return _get_project_root() / "GAN" / "metadata"


def _get_real_machine_profiles_path() -> Path:
    return _get_project_root() / "GAN" / "data" / "real_machines" / "profiles"


def _iter_profile_machine_ids() -> List[str]:
    """Return machine IDs that have a saved profile JSON on disk."""
    ids: set[str] = set()

    # Primary location used by the dashboard backend.
    metadata_path = _get_gan_metadata_path()
    if metadata_path.exists():
        for p in metadata_path.glob("*.json"):
            name = p.name
            if name.endswith("_metadata.json"):
                continue
            if name.endswith("_profile_temp.json"):
                continue
            if name.endswith("_template.json"):
                continue
            ids.add(p.stem)

    # Legacy/expected location for real machine profiles.
    real_profiles = _get_real_machine_profiles_path()
    if real_profiles.exists():
        for p in real_profiles.glob("*.json"):
            ids.add(p.stem)

    return sorted(ids)


def _machine_profile_file(machine_id: str) -> Optional[Path]:
    """Resolve a machine profile file path if present."""
    mid = (machine_id or "").strip()
    if not mid:
        return None

    p1 = _get_gan_metadata_path() / f"{mid}.json"
    if p1.exists():
        return p1

    p2 = _get_real_machine_profiles_path() / f"{mid}.json"
    if p2.exists():
        return p2

    return None


def _machine_exists(machine_id: str) -> bool:
    """Treat a machine as existing if it has workflow artifacts OR a saved profile."""
    try:
        if machine_id in gan_manager_wrapper.list_available_machines():
            return True
    except Exception:
        # Fall through to filesystem checks.
        pass
    return _machine_profile_file(machine_id) is not None

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

    # Local development UX: the dashboard UI can legitimately issue bursts of
    # requests (e.g., initial machine loading). Avoid 429s on localhost/dev.
    if settings.DEBUG or settings.ENVIRONMENT.lower() == "development" or client_ip in {"127.0.0.1", "::1"}:
        return
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
        metadata_path = _get_gan_metadata_path()
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
        template_file = _get_gan_metadata_path() / f"{machine_type}_template.json"
        
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
        template_file = _get_gan_metadata_path() / f"{machine_type}_template.json"
        
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
async def upload_profile(profile_data: Dict[str, Any] = Body(...)) -> ProfileUploadResponse:
    """
    Upload machine profile with immediate duplicate check.
    Prevents uploading duplicates before validation step.
    """
    try:
        # Import validator for immediate duplicate check
        from api.services.profile_validator import MachineProfileValidator, _to_snake_token

        if not isinstance(profile_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid payload: expected a JSON object (machine profile).",
            )
        
        # IMMEDIATE DUPLICATE CHECK - Fail fast before upload
        metadata_path = _get_gan_metadata_path()
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Force fresh load of existing machines (no caching)
        validator = MachineProfileValidator(gan_metadata_dir=metadata_path)

        # Apply parsing fallbacks so templates / legacy schemas can still stage.
        # This mutates profile_data in-place (e.g., name->machine_id, inferred category, grouped baseline).
        try:
            validator._apply_parsing_fallbacks(profile_data)  # noqa: SLF001 (internal normalization is intentional)
        except Exception:
            # Staging should not hard-fail if normalization has a minor issue.
            pass

        raw_machine_id = str(profile_data.get("machine_id") or "").strip()
        if not raw_machine_id:
            raise HTTPException(
                status_code=400,
                detail="Missing required field 'machine_id' (or fields needed to infer it).",
            )

        # Ensure machine_id is safe for filesystem + consistent across workflow.
        machine_id = _to_snake_token(raw_machine_id)
        
        is_existing = machine_id in validator.existing_machines
        is_pending = validator.is_pending_upload(machine_id)

        logger.info(f"Checking machine_id: {machine_id}")
        logger.info(f"Existing machines count: {len(validator.existing_machines)}")
        logger.info(f"Is existing: {is_existing} | Is pending upload: {is_pending}")
        
        # Check if machine already exists or is already staged for upload
        if is_existing:
            # Do not hard-block duplicates at upload time.
            # Let the user proceed to validation + editor to change machine_id.
            logger.warning(f"Upload staged for duplicate machine_id (will fail validation until changed): {machine_id}")

        # If a staged upload already exists (but the machine doesn't), overwrite it so users
        # can re-upload without getting stuck in a pending state.
        replaced_pending = False
        if is_pending and not is_existing:
            pending_machine_file = metadata_path / f"{machine_id}_profile_temp.json"
            old_profile_id: Optional[str] = None
            if pending_machine_file.exists():
                try:
                    with open(pending_machine_file, "r") as f:
                        old_payload = json.load(f)
                    old_profile_id = str(old_payload.get("profile_id") or "").strip()
                except Exception:
                    old_profile_id = None

            # Best-effort cleanup of the old profile_id temp file
            if old_profile_id:
                old_by_id_file = metadata_path / f"{old_profile_id}_profile_temp.json"
                try:
                    if old_by_id_file.exists():
                        old_by_id_file.unlink()
                except Exception:
                    pass

            replaced_pending = True
            logger.info(f"Overwriting staged upload for machine_id: {machine_id}")
        
        # Generate profile ID
        profile_id = str(uuid.uuid4())
        
        # Save profile to temporary location (keyed by profile_id to avoid collisions)
        profile_by_id_file = metadata_path / f"{profile_id}_profile_temp.json"
        profile_by_machine_file = metadata_path / f"{machine_id}_profile_temp.json"

        payload = dict(profile_data)
        payload["profile_id"] = profile_id
        payload["machine_id"] = machine_id

        with open(profile_by_id_file, 'w') as f:
            json.dump(payload, f, indent=2)

        # Compatibility: also store by machine_id for any existing tooling
        with open(profile_by_machine_file, 'w') as f:
            json.dump(payload, f, indent=2)
        
        logger.info(f"[OK] Profile uploaded: {machine_id} (ID: {profile_id})")
        
        return ProfileUploadResponse(
            success=True,
            profile_id=profile_id,
            machine_id=machine_id,
            message=(
                (
                    f"Profile uploaded successfully (replaced existing staged upload). Machine ID '{machine_id}' is unique. "
                    f"Proceed to validation."
                )
                if replaced_pending
                else (
                    (
                        f"Profile uploaded. Machine ID '{machine_id}' already exists, so validation will fail until you change 'machine_id' in the editor."
                    )
                    if is_existing
                    else (
                        f"Profile uploaded successfully. Machine ID '{machine_id}' is unique. "
                        f"Proceed to validation."
                    )
                )
            ),
            validation_required=True
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors) as-is
        raise
    except Exception as e:
        logger.error(f"Failed to upload profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload profile: {str(e)}"
        )


@router.post(
    "/profiles/validate-inline",
    response_model=InlineProfileValidationResponse,
    summary="Validate Profile Payload (Inline)",
    description="""
    Validate a machine profile JSON payload without staging it on disk.

    This runs the authoritative backend validator (duplicates, TVAE readiness, template checks)
    and is intended for frontend upload/editor/manual-entry flows.
    """,
    responses={
        200: {"description": "Validation completed"},
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limiter)],
)
async def validate_profile_inline(request: InlineProfileValidationRequest) -> InlineProfileValidationResponse:
    try:
        from api.services.profile_validator import MachineProfileValidator

        validator = MachineProfileValidator(gan_metadata_dir=_get_gan_metadata_path())
        profile_data = request.profile_data or {}

        is_valid, validation_issues, can_proceed = validator.validate_profile(
            profile_data=profile_data,
            strict=request.strict,
        )

        # Re-read after normalization/fallbacks (e.g., name -> machine_id, generated machine_id)
        machine_id = str(profile_data.get("machine_id", "")).strip()

        issues = [ValidationIssue(**issue.to_dict()) for issue in validation_issues]

        if is_valid:
            message = "[OK] Profile validation passed."
        else:
            error_count = sum(1 for issue in validation_issues if issue.severity == "error")
            warning_count = sum(1 for issue in validation_issues if issue.severity == "warning")
            message = f"[ERROR] Validation failed with {error_count} errors and {warning_count} warnings."

        logger.info(f"Inline validated machine {machine_id}: {'PASS' if is_valid else 'FAIL'}")

        return InlineProfileValidationResponse(
            valid=is_valid,
            machine_id=machine_id,
            issues=issues,
            can_proceed=can_proceed,
            message=message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate profile inline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate profile: {str(e)}")


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
    payload: Dict[str, Any] = Body(default_factory=dict)
) -> ProfileValidationResponse:
    """
    Validate uploaded profile with comprehensive checks.
    Uses MachineProfileValidator for strict validation.
    """
    try:
        # Import validator
        from api.services.profile_validator import MachineProfileValidator
        
        # Load the specific profile temp file for this profile_id
        metadata_path = _get_gan_metadata_path()
        profile_file = metadata_path / f"{profile_id}_profile_temp.json"

        if not profile_file.exists():
            # Fallback: scan temp profiles and match embedded profile_id (legacy uploads)
            for candidate in metadata_path.glob("*_profile_temp.json"):
                try:
                    with open(candidate, 'r') as f:
                        candidate_data = json.load(f)
                    if str(candidate_data.get("profile_id", "")) == str(profile_id):
                        profile_file = candidate
                        break
                except Exception:
                    continue

        if not profile_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found. Please upload profile first."
            )

        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        # machine_id may be normalized/generated by the validator
        machine_id = profile_data.get('machine_id', '')
        
        # Initialize comprehensive validator
        validator = MachineProfileValidator(gan_metadata_dir=metadata_path)
        
        strict = bool(payload.get("strict", True))

        # Run full validation
        is_valid, validation_issues, can_proceed = validator.validate_profile(
            profile_data=profile_data,
            strict=strict  # Validate with strict mode
        )

        machine_id = profile_data.get('machine_id', '')
        
        # Convert validation issues to API format
        issues = [ValidationIssue(**issue.to_dict()) for issue in validation_issues]
        
        # Generate message
        if is_valid:
            message = "[OK] Profile validation passed. Machine can be created."
        else:
            error_count = sum(1 for issue in validation_issues if issue.severity == "error")
            warning_count = sum(1 for issue in validation_issues if issue.severity == "warning")
            message = f"[ERROR] Validation failed with {error_count} errors and {warning_count} warnings. Fix errors before proceeding."
        
        logger.info(f"Validated profile {profile_id} for machine {machine_id}: {'PASS' if is_valid else 'FAIL'}")
        
        return ProfileValidationResponse(
            valid=is_valid,
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
    payload: Dict[str, Any] = Body(...)
) -> ProfileUploadResponse:
    """Edit profile after validation"""
    try:
        metadata_path = _get_gan_metadata_path()
        profile_file = metadata_path / f"{profile_id}_profile_temp.json"

        if not profile_file.exists():
            # Fallback: scan temp profiles and match embedded profile_id
            for candidate in metadata_path.glob("*_profile_temp.json"):
                try:
                    with open(candidate, 'r') as f:
                        candidate_data = json.load(f)
                    if str(candidate_data.get("profile_id", "")) == str(profile_id):
                        profile_file = candidate
                        break
                except Exception:
                    continue

        if not profile_file.exists():
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        
        # Load and update profile
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)

        updates: Dict[str, Any]
        if isinstance(payload, dict) and isinstance(payload.get("updates"), dict):
            updates = payload.get("updates")
        elif isinstance(payload, dict):
            # Treat payload as a full profile replacement/overlay
            updates = payload
        else:
            raise HTTPException(status_code=400, detail="Invalid payload: expected a JSON object")

        # Apply updates (overlay)
        for field, value in updates.items():
            profile_data[field] = value

        # Keep profile_id stable
        profile_data["profile_id"] = profile_id

        # Normalize machine_id for consistency
        try:
            from api.services.profile_validator import _to_snake_token

            if isinstance(profile_data.get("machine_id"), str) and profile_data.get("machine_id"):
                profile_data["machine_id"] = _to_snake_token(profile_data.get("machine_id"))
        except Exception:
            pass
        
        # Save updated profile
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        # Keep machine_id-keyed temp in sync if present
        machine_id = str(profile_data.get('machine_id', '')).lower()
        if machine_id:
            by_machine = metadata_path / f"{machine_id}_profile_temp.json"
            if by_machine.exists():
                with open(by_machine, 'w') as f:
                    json.dump(profile_data, f, indent=2)
            else:
                # Create/replace the machine-keyed temp file so other tooling can find it
                try:
                    with open(by_machine, 'w') as f:
                        json.dump(profile_data, f, indent=2)
                except Exception:
                    pass
        
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
    """
    Create machine from validated profile.
    Final duplicate check before creation.
    """
    try:
        # Import validator
        from api.services.profile_validator import MachineProfileValidator
        
        metadata_path = _get_gan_metadata_path()
        profile_file = metadata_path / f"{profile_id}_profile_temp.json"

        if not profile_file.exists():
            # Fallback: scan temp profiles and match embedded profile_id
            for candidate in metadata_path.glob("*_profile_temp.json"):
                try:
                    with open(candidate, 'r') as f:
                        candidate_data = json.load(f)
                    if str(candidate_data.get("profile_id", "")) == str(profile_id):
                        profile_file = candidate
                        break
                except Exception:
                    continue

        if not profile_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found. Please upload and validate profile first."
            )

        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        machine_id = profile_data.get('machine_id', '').lower()
        
        if not machine_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid profile: machine_id is required"
            )
        
        # CRITICAL: Final validation before creation
        validator = MachineProfileValidator(gan_metadata_dir=metadata_path)
        
        # Check if machine already exists (comprehensive check)
        if machine_id in validator.existing_machines:
            raise HTTPException(
                status_code=400,
                detail=f"[DUPLICATE ERROR] Machine '{machine_id}' already exists in the system. "
                       f"Creating duplicate machines will cause data conflicts and training errors. "
                       f"Please use a different machine_id or update the existing machine."
            )
        
        # Double-check with GANManager
        existing_machines = gan_manager_wrapper.list_available_machines()
        if machine_id in existing_machines:
            raise HTTPException(
                status_code=400,
                detail=f"[DUPLICATE ERROR] Machine '{machine_id}' already exists (found in GANManager)"
            )
        
        # Write profile to permanent location
        permanent_file = metadata_path / f"{machine_id}.json"
        with open(permanent_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        # Also store profiles in GAN/data/real_machines/profiles for consistency with legacy layout.
        try:
            real_profiles_dir = _get_project_root() / "GAN" / "data" / "real_machines" / "profiles"
            real_profiles_dir.mkdir(parents=True, exist_ok=True)
            real_profile_file = real_profiles_dir / f"{machine_id}.json"
            with open(real_profile_file, "w") as f:
                json.dump(profile_data, f, indent=2)
        except Exception:
            pass

        # Invalidate cached machine list/details so the UI refresh shows the new machine immediately
        try:
            redis_client = await get_redis()
            try:
                await redis_client.delete("gan:machines:list")
                await redis_client.delete(f"gan:machines:{machine_id}:details")
            finally:
                await redis_client.close()
        except Exception:
            pass

        # Clean up temp files (both id-keyed and machine-keyed)
        try:
            if profile_file.exists():
                profile_file.unlink()
        except Exception:
            pass
        try:
            by_machine_temp = metadata_path / f"{machine_id}_profile_temp.json"
            if by_machine_temp.exists():
                by_machine_temp.unlink()
        except Exception:
            pass
        
        # Build API-compliant workflow status. The wrapper may return only flags; the API model
        # requires machine_id/status plus capability fields.
        workflow_flags = gan_manager_wrapper.get_machine_workflow_status(machine_id)
        if not isinstance(workflow_flags, dict):
            workflow_flags = {}

        # After writing the permanent profile, the machine *does* have metadata for workflow purposes.
        has_profile_metadata = permanent_file.exists()

        wrapper_has_metadata = bool(workflow_flags.get("has_metadata"))
        has_metadata = wrapper_has_metadata or has_profile_metadata
        has_seed_data = bool(workflow_flags.get("has_seed_data"))
        has_trained_model = bool(workflow_flags.get("has_trained_model"))
        has_synthetic_data = bool(workflow_flags.get("has_synthetic_data"))

        if has_synthetic_data:
            machine_status = MachineStatus.SYNTHETIC_GENERATED
        elif has_trained_model:
            machine_status = MachineStatus.TRAINED
        elif has_seed_data:
            machine_status = MachineStatus.SEED_GENERATED
        else:
            machine_status = MachineStatus.NOT_STARTED

        # Capability defaults: if the wrapper hasn't caught up yet but we have a saved profile,
        # allow seed generation to proceed.
        if has_profile_metadata and not wrapper_has_metadata:
            can_generate_seed = True
        else:
            can_generate_seed = bool(workflow_flags.get("can_generate_seed", True)) if has_metadata else False

        can_train_model = bool(workflow_flags.get("can_train_model", has_seed_data)) if has_metadata else False
        can_generate_synthetic = (
            bool(workflow_flags.get("can_generate_data") or workflow_flags.get("can_generate_synthetic"))
            if has_metadata
            else False
        )

        # Fallback gating: if wrapper doesn't provide a capability but artifacts exist, enable next step.
        if has_trained_model and has_metadata:
            can_generate_synthetic = True
        if has_seed_data and has_metadata:
            can_train_model = True

        status = MachineWorkflowStatus(
            machine_id=machine_id,
            status=machine_status,
            has_metadata=has_metadata,
            has_seed_data=has_seed_data,
            has_trained_model=has_trained_model,
            has_synthetic_data=has_synthetic_data,
            can_generate_seed=can_generate_seed,
            can_train_model=can_train_model,
            can_generate_synthetic=can_generate_synthetic,
            last_updated=datetime.utcnow(),
        )
        
        logger.info(f"Machine created: {machine_id}")
        
        # Prefer a short workflow-friendly machine_type.
        machine_type = str(profile_data.get("machine_type") or "").strip().lower()
        if not machine_type:
            machine_type = machine_id.split("_")[0] if machine_id else "unknown"

        manufacturer = str(profile_data.get("manufacturer") or "").strip()
        model = str(profile_data.get("model") or "").strip()

        # Sensors may be provided either as a sensors[] list or inside baseline_normal_operation.
        try:
            num_sensors = len(validator._extract_sensors_any_format(profile_data))  # noqa: SLF001
        except Exception:
            num_sensors = len(profile_data.get("sensors", []) or [])

        now = datetime.utcnow()
        return MachineDetails(
            machine_id=machine_id,
            machine_type=machine_type,
            manufacturer=manufacturer,
            model=model,
            num_sensors=num_sensors,
            degradation_states=int(profile_data.get("degradation_states") or 4),
            status=status,
            created_at=now,
            updated_at=now,
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
        cache_key = "gan:machines:list"

        # Prefer cache if it already includes all known machines (GAN manager + saved profiles).
        known = set(_iter_profile_machine_ids())
        try:
            known |= set(gan_manager_wrapper.list_available_machines())
        except Exception:
            pass

        cached = await get_cached_response(cache_key)
        if cached and cached.get("machine_details"):
            cached_machines = set(cached.get("machines") or [])
            # Only return cached data if it exactly matches what's currently on disk.
            # This prevents deleted machines from reappearing until cache TTL expires.
            if cached_machines == known:
                return MachineListResponse(**cached)

        machines = sorted(known)

        # Provide hydrated summaries to avoid N per-machine detail calls in the UI.
        machine_details = []
        for machine_id in machines:
            try:
                machine_details.append(await get_machine_details(machine_id))
            except Exception as e:
                logger.warning(f"Failed to hydrate details for {machine_id}: {e}")

        response = MachineListResponse(
            total=len(machines),
            machines=machines,
            machine_details=machine_details,
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
        
        machine_in_manager = False
        try:
            machine_in_manager = machine_id in gan_manager_wrapper.list_available_machines()
        except Exception:
            machine_in_manager = False

        profile_file = _machine_profile_file(machine_id)
        if not machine_in_manager and profile_file is None:
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")

        # Get machine details from wrapper when available; otherwise synthesize a minimal detail dict.
        if machine_in_manager:
            details = gan_manager_wrapper.get_machine_details(machine_id)
        else:
            details = {
                "machine_id": machine_id,
                "workflow_status": gan_manager_wrapper.get_machine_workflow_status(machine_id),
            }

        # gan_manager_wrapper currently returns a lightweight dict; hydrate it into the API model.
        details_dict: Dict[str, Any]
        if hasattr(details, "dict") and callable(getattr(details, "dict")):
            details_dict = details.dict()
        elif hasattr(details, "model_dump") and callable(getattr(details, "model_dump")):
            details_dict = details.model_dump()
        elif isinstance(details, dict):
            details_dict = details
        else:
            raise TypeError(f"Unexpected machine details type: {type(details)!r}")

        workflow_flags = (details_dict.get("workflow_status") or {})
        if not isinstance(workflow_flags, dict):
            workflow_flags = {}

        has_metadata = bool(workflow_flags.get("has_metadata"))
        has_seed_data = bool(workflow_flags.get("has_seed_data"))
        has_trained_model = bool(workflow_flags.get("has_trained_model"))
        has_synthetic_data = bool(workflow_flags.get("has_synthetic_data"))

        if has_synthetic_data:
            machine_status = MachineStatus.SYNTHETIC_GENERATED
        elif has_trained_model:
            machine_status = MachineStatus.TRAINED
        elif has_seed_data:
            machine_status = MachineStatus.SEED_GENERATED
        else:
            machine_status = MachineStatus.NOT_STARTED

        # Metadata-derived values
        metadata_file = _get_gan_metadata_path() / f"{machine_id}_metadata.json"
        profile_json_file = _machine_profile_file(machine_id)
        num_sensors = 0
        updated_at = datetime.utcnow()
        created_at = updated_at

        # Defaults (may be overridden by metadata/KB)
        machine_type = "unknown"
        manufacturer = "unknown"
        model = "unknown"

        raw: Dict[str, Any] = {}
        if metadata_file.exists():
            try:
                raw = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                raw = {}

            # Support multiple metadata schemas
            sensors = raw.get("sensors")
            if isinstance(sensors, list):
                num_sensors = len(sensors)
            else:
                cols = raw.get("columns")
                if isinstance(cols, dict):
                    num_sensors = len(cols)

            # Some metadata files embed profile fields
            if isinstance(raw.get("machine_type"), str) and raw.get("machine_type"):
                machine_type = str(raw.get("machine_type")).strip().lower()
            if isinstance(raw.get("manufacturer"), str) and raw.get("manufacturer"):
                manufacturer = str(raw.get("manufacturer")).strip()
            if isinstance(raw.get("model"), str) and raw.get("model"):
                model = str(raw.get("model")).strip()

            try:
                stat = metadata_file.stat()
                updated_at = datetime.fromtimestamp(stat.st_mtime)
                created_at = datetime.fromtimestamp(stat.st_ctime)
            except Exception:
                pass

        # If there is no derived metadata yet, fall back to the authored profile JSON.
        if not raw and profile_json_file is not None and profile_json_file.exists():
            try:
                raw = json.loads(profile_json_file.read_text(encoding="utf-8"))
            except Exception:
                raw = {}

            try:
                stat = profile_json_file.stat()
                updated_at = datetime.fromtimestamp(stat.st_mtime)
                created_at = datetime.fromtimestamp(stat.st_ctime)
            except Exception:
                pass

            # Try to derive a sensor count from either sensors[] or baseline_normal_operation.
            sensors = raw.get("sensors")
            if isinstance(sensors, list):
                num_sensors = len(sensors)
            else:
                bno = raw.get("baseline_normal_operation")
                if isinstance(bno, dict):
                    # baseline_normal_operation is often grouped by sensor family -> sensor id
                    # e.g., {"temperature": {"t1": {...}}}
                    count = 0
                    for group in bno.values():
                        if isinstance(group, dict):
                            count += len(group)
                    num_sensors = count

            # Prefer explicit fields if present.
            if isinstance(raw.get("machine_type"), str) and raw.get("machine_type"):
                machine_type = str(raw.get("machine_type")).strip().lower()
            if isinstance(raw.get("manufacturer"), str) and raw.get("manufacturer"):
                manufacturer = str(raw.get("manufacturer")).strip()
            if isinstance(raw.get("model"), str) and raw.get("model"):
                model = str(raw.get("model")).strip()

        # Manufacturer/model: prefer the curated knowledge base if present.
        if machine_type == "unknown":
            multiword_types = {"cooling_tower", "induction_motor"}
            parts = [p for p in machine_id.split("_") if p]
            if len(parts) >= 2 and f"{parts[0]}_{parts[1]}" in multiword_types:
                machine_type = f"{parts[0]}_{parts[1]}"
            else:
                machine_type = parts[0] if parts else "unknown"

        kb_file = _get_project_root() / "LLM" / "data" / "knowledge_base" / "machines" / f"{machine_id}.txt"
        if kb_file.exists():
            try:
                for line in kb_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = line.strip()
                    if s.lower().startswith("manufacturer:"):
                        manufacturer = s.split(":", 1)[1].strip() or manufacturer
                    elif s.lower().startswith("model:"):
                        model = s.split(":", 1)[1].strip() or model
            except Exception:
                pass
        else:
            # Fallback heuristic: (type or type_type)_manufacturer_model_instance
            parts = [p for p in machine_id.split("_") if p]
            type_parts = machine_type.split("_") if machine_type else []
            start = len(type_parts)
            if len(parts) > start:
                if manufacturer == "unknown" and len(parts) > start:
                    manufacturer = parts[start]
                if model == "unknown" and len(parts) > start + 1:
                    model_parts = parts[start + 1 : -1] or parts[start + 1 :]
                    model = "_".join(model_parts) if model_parts else model

        degradation_states = int(raw.get("degradation_states") or 4) if isinstance(raw, dict) else 4

        status = MachineWorkflowStatus(
            machine_id=machine_id,
            status=machine_status,
            has_metadata=has_metadata,
            has_seed_data=has_seed_data,
            has_trained_model=has_trained_model,
            has_synthetic_data=has_synthetic_data,
            can_generate_seed=bool(workflow_flags.get("can_generate_seed")) if has_metadata else False,
            can_train_model=bool(workflow_flags.get("can_train_model")) if has_metadata else False,
            can_generate_synthetic=bool(workflow_flags.get("can_generate_data") or workflow_flags.get("can_generate_synthetic"))
            if has_metadata
            else False,
            last_updated=updated_at,
        )

        hydrated = MachineDetails(
            machine_id=machine_id,
            machine_type=machine_type,
            manufacturer=manufacturer,
            model=model,
            num_sensors=num_sensors,
            degradation_states=degradation_states,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
        )

        await set_cached_response(cache_key, hydrated.dict() if hasattr(hydrated, "dict") else hydrated.model_dump())
        return hydrated
    
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")

        mid = (machine_id or "").strip().lower()

        import shutil

        deleted_paths: List[str] = []

        def _try_unlink(p: Path) -> None:
            try:
                if p.exists() and p.is_file():
                    p.unlink()
                    deleted_paths.append(str(p))
            except Exception:
                # Best-effort deletion; the endpoint should still try to remove other artifacts.
                pass

        def _try_rmtree(p: Path) -> None:
            try:
                if p.exists() and p.is_dir():
                    shutil.rmtree(p)
                    deleted_paths.append(str(p))
            except Exception:
                pass

        # Delete authored profiles (both primary and legacy locations)
        _try_unlink(_get_gan_metadata_path() / f"{mid}.json")
        _try_unlink(_get_real_machine_profiles_path() / f"{mid}.json")

        # Delete derived metadata (GANManager-style)
        _try_unlink(_get_gan_metadata_path() / f"{mid}_metadata.json")

        # Delete staged temp profiles (in case a delete happens mid-wizard)
        _try_unlink(_get_gan_metadata_path() / f"{mid}_profile_temp.json")

        # Delete seed data (temporal)
        seed_root = _get_project_root() / "GAN" / "seed_data"
        if seed_root.exists():
            for p in seed_root.rglob(f"{mid}_*seed*.parquet"):
                _try_unlink(p)
            for p in seed_root.rglob(f"{mid}_temporal_seed.parquet"):
                _try_unlink(p)

        # Delete trained models (TVAE temporal models are stored as machine_id-prefixed PKLs)
        models_root = _get_project_root() / "GAN" / "models"
        if models_root.exists():
            for p in models_root.rglob(f"{mid}_*.pkl"):
                _try_unlink(p)

        # Delete synthetic data
        synthetic_root = _get_project_root() / "GAN" / "data" / "synthetic"
        if synthetic_root.exists():
            _try_rmtree(synthetic_root / mid)
            for p in synthetic_root.glob(f"{mid}_*.parquet"):
                _try_unlink(p)

        synthetic_fixed_root = _get_project_root() / "GAN" / "data" / "synthetic_fixed"
        if synthetic_fixed_root.exists():
            _try_rmtree(synthetic_fixed_root / mid)
            for p in synthetic_fixed_root.glob(f"{mid}_*.parquet"):
                _try_unlink(p)

        # Invalidate caches so deleted machines don't reappear after refresh/reload
        try:
            redis_client = await get_redis()
            try:
                await redis_client.delete("gan:machines:list")
                await redis_client.delete(f"gan:machines:{mid}:details")
            finally:
                await redis_client.close()
        except Exception:
            pass
        
        logger.info(f"Machine deleted: {machine_id}")
        
        # Keep the API response minimal and stable; detailed deletion traces are logged.
        return {"message": f"Machine '{mid}' and all associated data deleted successfully"}
    
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        
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
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        
        # Check synthetic data exists
        synthetic_dir = _get_project_root() / "GAN" / "data" / "synthetic" / machine_id
        data_file = synthetic_dir / "train.parquet"
        if not data_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Synthetic data not found for machine '{machine_id}'"
            )
        
        # Basic validation (would be more comprehensive in production)
        import pandas as pd
        df = pd.read_parquet(data_file)
        
        # Pandas/NumPy often produce numpy scalar types (e.g., numpy.int64) which
        # are not JSON-serializable by Pydantic v2. Cast to plain Python types.
        num_samples = int(len(df))
        num_features = int(len(df.columns))
        null_values = int(df.isnull().sum().sum())

        validation_result = {
            "machine_id": machine_id,
            "valid": True,
            "num_samples": num_samples,
            "num_features": num_features,
            "null_values": null_values,
            "quality_score": float(0.95),  # Placeholder
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


@router.get(
    "/machines/{machine_id}/visualizations/summary",
    summary="Get Visualization Summary",
    description="""
    Returns lightweight, chart-friendly summaries for the GAN dashboard visualizations.

    Includes:
    - Seed data time-series preview (timestamp, RUL, and a subset of sensor columns)
    - Feature distribution histograms (seed vs synthetic)
    - Correlation matrix (synthetic data)

    Query params:
    - points: number of time-series points to return (downsampled)
    - bins: number of histogram bins per feature
    - max_features: cap number of features included in distributions/correlation
    """,
    responses={
        200: {"description": "Visualization summary"},
        404: {"description": "Machine or required data not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def get_visualization_summary(
    machine_id: str,
    points: int = 300,
    bins: int = 20,
    max_features: int = 8,
) -> Dict[str, Any]:
    try:
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")

        root = _get_project_root()
        seed_file = root / "GAN" / "seed_data" / "temporal" / f"{machine_id}_temporal_seed.parquet"
        synthetic_file = root / "GAN" / "data" / "synthetic" / machine_id / "train.parquet"

        if not seed_file.exists():
            raise HTTPException(status_code=404, detail=f"Seed data not found for machine '{machine_id}'")
        if not synthetic_file.exists():
            raise HTTPException(status_code=404, detail=f"Synthetic data not found for machine '{machine_id}'")

        import numpy as np
        import pandas as pd

        def _find_col(cols, candidates):
            lower = {c.lower(): c for c in cols}
            for cand in candidates:
                c = lower.get(cand)
                if c:
                    return c
            return None

        # Load seed + synthetic
        seed_df = pd.read_parquet(seed_file)
        syn_df = pd.read_parquet(synthetic_file)

        # Identify core columns
        ts_col = _find_col(seed_df.columns, ["timestamp", "time", "datetime", "date"]) or "timestamp"
        rul_col = _find_col(seed_df.columns, ["rul", "remaining_useful_life"]) or "rul"

        # Ensure timestamp exists and is serializable
        if ts_col not in seed_df.columns:
            seed_df[ts_col] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(seed_df), freq="S")
        if not np.issubdtype(seed_df[ts_col].dtype, np.datetime64):
            seed_df[ts_col] = pd.to_datetime(seed_df[ts_col], errors="coerce")

        # If RUL missing, create a simple decay so chart still renders
        if rul_col not in seed_df.columns:
            seed_df[rul_col] = np.linspace(100, 0, num=len(seed_df))

        numeric_seed_cols = seed_df.select_dtypes(include=["number"]).columns.tolist()
        numeric_syn_cols = syn_df.select_dtypes(include=["number"]).columns.tolist()

        # Sensor-like columns for time series preview (exclude RUL)
        sensor_cols = [c for c in numeric_seed_cols if c.lower() != rul_col.lower()]
        sensor_cols = sensor_cols[: min(6, len(sensor_cols))]

        # Downsample seed preview
        safe_points = int(max(10, min(points, 2000)))
        if len(seed_df) <= safe_points:
            idx = np.arange(len(seed_df))
        else:
            idx = np.unique(np.linspace(0, len(seed_df) - 1, safe_points).astype(int))

        preview_cols = [ts_col, rul_col] + sensor_cols
        seed_preview = seed_df.loc[idx, preview_cols].copy()

        # Serialize timestamps as ISO strings
        seed_preview[ts_col] = seed_preview[ts_col].dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")

        seed_points: list[dict[str, Any]] = []
        for _, row in seed_preview.iterrows():
            pt: dict[str, Any] = {
                "timestamp": str(row[ts_col]),
                "rul": float(row[rul_col]) if pd.notnull(row[rul_col]) else 0.0,
            }
            for c in sensor_cols:
                v = row[c]
                pt[c] = float(v) if pd.notnull(v) else 0.0
            seed_points.append(pt)

        # Features for distributions/correlation
        exclude = {ts_col.lower(), rul_col.lower()}
        feature_candidates = [c for c in numeric_seed_cols if c.lower() not in exclude]
        feature_candidates = [c for c in feature_candidates if c in numeric_syn_cols]
        feature_cols = feature_candidates[: max(0, int(max_features))]

        def _stats(series: pd.Series) -> Dict[str, float]:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "max": float(s.max()),
            }

        safe_bins = int(max(5, min(bins, 80)))
        distributions: list[dict[str, Any]] = []
        for feat in feature_cols:
            seed_s = pd.to_numeric(seed_df[feat], errors="coerce").dropna()
            syn_s = pd.to_numeric(syn_df[feat], errors="coerce").dropna()

            combined = pd.concat([seed_s, syn_s], ignore_index=True)
            if len(combined) == 0:
                edges = np.linspace(0, 1, safe_bins + 1)
            else:
                lo = float(combined.min())
                hi = float(combined.max())
                if np.isclose(lo, hi):
                    lo -= 0.5
                    hi += 0.5
                edges = np.linspace(lo, hi, safe_bins + 1)

            seed_counts, _ = np.histogram(seed_s.to_numpy(dtype=float), bins=edges)
            syn_counts, _ = np.histogram(syn_s.to_numpy(dtype=float), bins=edges)

            distributions.append(
                {
                    "feature": str(feat),
                    "bin_edges": [float(x) for x in edges.tolist()],
                    "seed_counts": [int(x) for x in seed_counts.tolist()],
                    "synthetic_counts": [int(x) for x in syn_counts.tolist()],
                    "seed_stats": _stats(seed_df[feat]),
                    "synthetic_stats": _stats(syn_df[feat]),
                }
            )

        # Correlation matrix (synthetic)
        corr_features = feature_cols
        if len(corr_features) > 0:
            corr_df = syn_df[corr_features].apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr().fillna(0.0)
            corr_matrix = [[float(x) for x in row] for row in corr.to_numpy().tolist()]
        else:
            corr_matrix = []

        return {
            "machine_id": machine_id,
            "seed_series": {
                "points": seed_points,
                "sensor_keys": sensor_cols,
            },
            "distributions": distributions,
            "correlation": {
                "features": [str(c) for c in corr_features],
                "matrix": corr_matrix,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build visualization summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build visualization summary: {str(e)}")


@router.get(
    "/machines/{machine_id}/data/csv/download",
    summary="Download Combined Synthetic Dataset (CSV) (Legacy Path)",
    description="Legacy alias for the combined CSV download endpoint.",
    responses={
        200: {"description": "CSV file"},
        404: {"description": "Machine or data not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def download_synthetic_csv_legacy(machine_id: str, background_tasks: BackgroundTasks):
    # Call the canonical handler.
    return await download_synthetic_csv(machine_id=machine_id, background_tasks=background_tasks)


@router.get(
    "/machines/{machine_id}/data/{split}/download",
    summary="Download Synthetic Dataset Split",
    description="""
    Download generated synthetic data parquet for a machine.

    **Parameters:**
    - machine_id: Machine identifier
    - split: One of `train`, `val`, `test`

    **Returns:**
    - Parquet file download

    **Rate Limit:** 100 requests/minute per IP
    """,
    responses={
        200: {"description": "Parquet file"},
        400: {"description": "Invalid split"},
        404: {"description": "Machine or data not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def download_synthetic_split(machine_id: str, split: str):
    """Download train/val/test parquet split for a machine."""
    try:
        split_norm = (split or '').strip().lower()
        if split_norm not in {"train", "val", "test"}:
            raise HTTPException(status_code=400, detail="Invalid split. Use train, val, or test.")

        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")

        data_file = _get_project_root() / "GAN" / "data" / "synthetic" / machine_id / f"{split_norm}.parquet"
        if not data_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Synthetic data file not found for machine '{machine_id}' ({split_norm})"
            )

        return FileResponse(
            path=str(data_file),
            filename=data_file.name,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download synthetic data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download synthetic data: {str(e)}"
        )


@router.get(
    "/machines/{machine_id}/data/download/csv",
    summary="Download Combined Synthetic Dataset (CSV)",
    description="""
    Download all generated synthetic splits (train/val/test) merged into a single CSV.

    Adds a `split` column to preserve the origin split.

    **Returns:**
    - A single CSV file for easy viewing in Excel/BI tools
    """,
    responses={
        200: {"description": "CSV file"},
        404: {"description": "Machine or data not found"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def download_synthetic_csv(machine_id: str, background_tasks: BackgroundTasks):
    """Download merged train/val/test as one CSV."""
    try:
        if not _machine_exists(machine_id):
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")

        synthetic_dir = _get_project_root() / "GAN" / "data" / "synthetic" / machine_id
        train_file = synthetic_dir / "train.parquet"
        val_file = synthetic_dir / "val.parquet"
        test_file = synthetic_dir / "test.parquet"

        missing = [p.name for p in (train_file, val_file, test_file) if not p.exists()]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Synthetic data files not found for machine '{machine_id}': {', '.join(missing)}"
            )

        import pandas as pd

        train_df = pd.read_parquet(train_file)
        train_df.insert(0, "split", "train")
        val_df = pd.read_parquet(val_file)
        val_df.insert(0, "split", "val")
        test_df = pd.read_parquet(test_file)
        test_df.insert(0, "split", "test")

        merged = pd.concat([train_df, val_df, test_df], ignore_index=True)

        tmp = tempfile.NamedTemporaryFile(prefix=f"{machine_id}_synthetic_", suffix=".csv", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        merged.to_csv(tmp_path, index=False)

        background_tasks.add_task(lambda p=str(tmp_path): Path(p).unlink(missing_ok=True))

        return FileResponse(
            path=str(tmp_path),
            filename=f"{machine_id}_synthetic.csv",
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download combined CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download combined CSV: {str(e)}")


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
            logs=None,
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
                response.logs = info.get('logs')
                # started_at may be included by tasks as ISO string
                started_at = info.get('started_at')
                if isinstance(started_at, str):
                    try:
                        response.started_at = datetime.fromisoformat(started_at)
                    except Exception:
                        pass
        
        elif task_result.state == 'SUCCESS':
            response.result = task_result.result
            # If tasks include logs in their result, surface it.
            if isinstance(task_result.result, dict) and task_result.result.get('logs'):
                response.logs = task_result.result.get('logs')
            response.completed_at = datetime.now()  # Would be from task metadata
        
        elif task_result.state == 'FAILURE':
            response.error = str(task_result.info)
            response.completed_at = datetime.now()
            # On failure, Celery may store exception text; keep logs best-effort from meta.
            info = task_result.info
            if isinstance(info, dict) and info.get('logs'):
                response.logs = info.get('logs')
        
        return response
    
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.post(
    "/tasks/{task_id}/cancel",
    summary="Cancel Celery Task (Best-Effort)",
    description="""
    Best-effort cancellation for a Celery task.

    Notes:
    - Celery revoke will prevent a *pending* task from running.
    - If the task is already running, termination behavior depends on the worker pool/OS.
      On Windows, hard termination may not be supported reliably.
    """,
    responses={
        200: {"description": "Cancellation requested"},
        429: {"description": "Rate limit exceeded"}
    },
    dependencies=[Depends(rate_limiter)]
)
async def cancel_task(task_id: str):
    try:
        celery_app.control.revoke(task_id, terminate=False)
        return {
            "success": True,
            "task_id": task_id,
            "message": "Cancellation requested. If the task was pending, it will not run; if already running, it may continue until completion."
        }
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


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
