"""
CRUD (Create, Read, Update, Delete) operations for database models.
Phase 3.7.1.2: Database Setup (Days 3-4)

Provides reusable database operations for all models.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

from . import models


# ==================== Machine CRUD ====================

def create_machine(
    db: Session,
    machine_id: str,
    machine_type: str,
    manufacturer: str,
    model: str,
    metadata_path: Optional[str] = None,
) -> models.Machine:
    """Create a new machine record."""
    db_machine = models.Machine(
        machine_id=machine_id,
        machine_type=machine_type,
        manufacturer=manufacturer,
        model=model,
        metadata_path=metadata_path,
    )
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine


def get_machine(db: Session, machine_id: str) -> Optional[models.Machine]:
    """Get machine by machine_id."""
    return db.query(models.Machine).filter(models.Machine.machine_id == machine_id).first()


def get_machine_by_uuid(db: Session, uuid_id: uuid.UUID) -> Optional[models.Machine]:
    """Get machine by UUID."""
    return db.query(models.Machine).filter(models.Machine.id == uuid_id).first()


def get_machines(db: Session, skip: int = 0, limit: int = 100) -> List[models.Machine]:
    """Get all machines with pagination."""
    return db.query(models.Machine).offset(skip).limit(limit).all()


def update_machine(
    db: Session,
    machine_id: str,
    updates: Dict[str, Any],
) -> Optional[models.Machine]:
    """Update machine record."""
    db_machine = get_machine(db, machine_id)
    if not db_machine:
        return None
    
    for key, value in updates.items():
        if hasattr(db_machine, key):
            setattr(db_machine, key, value)
    
    db.commit()
    db.refresh(db_machine)
    return db_machine


def delete_machine(db: Session, machine_id: str) -> bool:
    """Delete machine record."""
    db_machine = get_machine(db, machine_id)
    if not db_machine:
        return False
    
    db.delete(db_machine)
    db.commit()
    return True


# ==================== GAN Training Job CRUD ====================

def create_gan_training_job(
    db: Session,
    machine_id: str,
    epochs: int,
    status: str = "pending",
) -> models.GANTrainingJob:
    """Create a new GAN training job."""
    db_job = models.GANTrainingJob(
        machine_id=machine_id,
        epochs=epochs,
        status=status,
        started_at=datetime.utcnow() if status == "running" else None,
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def get_gan_training_job(db: Session, job_id: uuid.UUID) -> Optional[models.GANTrainingJob]:
    """Get GAN training job by UUID."""
    return db.query(models.GANTrainingJob).filter(models.GANTrainingJob.id == job_id).first()


def get_gan_training_jobs(
    db: Session,
    machine_id: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[models.GANTrainingJob]:
    """Get GAN training jobs with optional filters."""
    query = db.query(models.GANTrainingJob)
    
    if machine_id:
        query = query.filter(models.GANTrainingJob.machine_id == machine_id)
    if status:
        query = query.filter(models.GANTrainingJob.status == status)
    
    return query.order_by(desc(models.GANTrainingJob.started_at)).offset(skip).limit(limit).all()


def update_gan_training_job(
    db: Session,
    job_id: uuid.UUID,
    updates: Dict[str, Any],
) -> Optional[models.GANTrainingJob]:
    """Update GAN training job."""
    db_job = get_gan_training_job(db, job_id)
    if not db_job:
        return None
    
    for key, value in updates.items():
        if hasattr(db_job, key):
            setattr(db_job, key, value)
    
    # Auto-set completed_at if status changes to completed/failed
    if "status" in updates and updates["status"] in ["completed", "failed"]:
        db_job.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_job)
    return db_job


# ==================== Prediction CRUD ====================

def create_prediction(
    db: Session,
    machine_id: str,
    prediction_type: str,
    input_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    confidence: Optional[float] = None,
) -> models.Prediction:
    """Create a new prediction record."""
    db_prediction = models.Prediction(
        machine_id=machine_id,
        prediction_type=prediction_type,
        input_data=input_data,
        prediction_result=prediction_result,
        confidence=confidence,
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def get_prediction(db: Session, prediction_id: uuid.UUID) -> Optional[models.Prediction]:
    """Get prediction by UUID."""
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()


def get_predictions(
    db: Session,
    machine_id: Optional[str] = None,
    prediction_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[models.Prediction]:
    """Get predictions with optional filters."""
    query = db.query(models.Prediction)
    
    if machine_id:
        query = query.filter(models.Prediction.machine_id == machine_id)
    if prediction_type:
        query = query.filter(models.Prediction.prediction_type == prediction_type)
    
    return query.order_by(desc(models.Prediction.timestamp)).offset(skip).limit(limit).all()


def get_recent_predictions(
    db: Session,
    machine_id: str,
    limit: int = 10,
) -> List[models.Prediction]:
    """Get most recent predictions for a machine."""
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.machine_id == machine_id)
        .order_by(desc(models.Prediction.timestamp))
        .limit(limit)
        .all()
    )


# ==================== Explanation CRUD ====================

def create_explanation(
    db: Session,
    prediction_id: uuid.UUID,
    explanation_text: str,
    recommendations: Optional[Dict[str, Any]] = None,
) -> models.Explanation:
    """Create a new explanation for a prediction."""
    db_explanation = models.Explanation(
        prediction_id=prediction_id,
        explanation_text=explanation_text,
        recommendations=recommendations,
    )
    db.add(db_explanation)
    db.commit()
    db.refresh(db_explanation)
    return db_explanation


def get_explanation(db: Session, explanation_id: uuid.UUID) -> Optional[models.Explanation]:
    """Get explanation by UUID."""
    return db.query(models.Explanation).filter(models.Explanation.id == explanation_id).first()


def get_explanation_by_prediction(
    db: Session,
    prediction_id: uuid.UUID,
) -> Optional[models.Explanation]:
    """Get explanation for a specific prediction."""
    return (
        db.query(models.Explanation)
        .filter(models.Explanation.prediction_id == prediction_id)
        .first()
    )


# ==================== Model Version CRUD ====================

def create_model_version(
    db: Session,
    model_type: str,
    version: str,
    file_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    is_active: bool = True,
) -> models.ModelVersion:
    """Create a new model version record."""
    # If this is set as active, deactivate other versions of same type
    if is_active:
        db.query(models.ModelVersion).filter(
            models.ModelVersion.model_type == model_type,
            models.ModelVersion.is_active == True,
        ).update({"is_active": False})
    
    db_model_version = models.ModelVersion(
        model_type=model_type,
        version=version,
        file_path=file_path,
        metrics=metrics,
        is_active=is_active,
    )
    db.add(db_model_version)
    db.commit()
    db.refresh(db_model_version)
    return db_model_version


def get_model_version(db: Session, version_id: uuid.UUID) -> Optional[models.ModelVersion]:
    """Get model version by UUID."""
    return db.query(models.ModelVersion).filter(models.ModelVersion.id == version_id).first()


def get_active_model_version(db: Session, model_type: str) -> Optional[models.ModelVersion]:
    """Get currently active model version for a type."""
    return (
        db.query(models.ModelVersion)
        .filter(
            models.ModelVersion.model_type == model_type,
            models.ModelVersion.is_active == True,
        )
        .first()
    )


def get_model_versions(
    db: Session,
    model_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[models.ModelVersion]:
    """Get model versions with optional filter."""
    query = db.query(models.ModelVersion)
    
    if model_type:
        query = query.filter(models.ModelVersion.model_type == model_type)
    
    return query.order_by(desc(models.ModelVersion.trained_at)).offset(skip).limit(limit).all()


def activate_model_version(db: Session, version_id: uuid.UUID) -> Optional[models.ModelVersion]:
    """Activate a model version (deactivates others of same type)."""
    db_version = get_model_version(db, version_id)
    if not db_version:
        return None
    
    # Deactivate other versions of same type
    db.query(models.ModelVersion).filter(
        models.ModelVersion.model_type == db_version.model_type,
        models.ModelVersion.is_active == True,
    ).update({"is_active": False})
    
    # Activate this version
    db_version.is_active = True
    db.commit()
    db.refresh(db_version)
    return db_version


# ==================== User CRUD (Phase 3.7.1.3) ====================

def create_user(
    db: Session,
    username: str,
    email: str,
    hashed_password: str,
    role: str = "viewer",
    is_approved: bool = True,
) -> models.User:
    """Create a new user."""
    db_user = models.User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        role=role,
        is_approved=is_approved,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
    """Get user by username."""
    return db.query(models.User).filter(models.User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    """Get user by email."""
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[models.User]:
    """Get user by UUID."""
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
    """Get all users with pagination."""
    return db.query(models.User).offset(skip).limit(limit).all()


def get_pending_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
    """Get all users pending approval."""
    return db.query(models.User).filter(
        models.User.is_approved == False,
        models.User.is_active == True,
    ).offset(skip).limit(limit).all()


def approve_user(db: Session, user_id: uuid.UUID) -> Optional[models.User]:
    """Approve a pending user."""
    user = get_user_by_id(db, user_id)
    if user:
        user.is_approved = True
        db.commit()
        db.refresh(user)
    return user


def reject_user(db: Session, user_id: uuid.UUID) -> bool:
    """Reject and delete a pending user."""
    user = get_user_by_id(db, user_id)
    if user and not user.is_approved:
        db.delete(user)
        db.commit()
        return True
    return False


def authenticate_user(db: Session, username: str, password: str) -> Optional[models.User]:
    """
    Authenticate user with username and password.
    
    Args:
        db: Database session
        username: Username
        password: Plain text password
    
    Returns:
        User object if authentication successful, None otherwise
    """
    from utils.security import verify_password
    
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def update_user(
    db: Session,
    user_id: uuid.UUID,
    updates: Dict[str, Any],
) -> Optional[models.User]:
    """Update user record."""
    db_user = get_user_by_id(db, user_id)
    if not db_user:
        return None
    
    for key, value in updates.items():
        if hasattr(db_user, key) and key != "id":  # Don't allow ID updates
            setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: uuid.UUID) -> bool:
    """Delete user record."""
    db_user = get_user_by_id(db, user_id)
    if not db_user:
        return False
    
    db.delete(db_user)
    db.commit()
    return True
