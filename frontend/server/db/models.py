"""
SQLAlchemy database models for Predictive Maintenance Dashboard.
Phase 3.7.1.2: Database Setup (Days 3-4)
Phase 3.7.1.3: Authentication Setup (Days 5-6)

All tables use UUID primary keys and include timestamps.
Foreign key relationships established between related tables.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Base


class User(Base):
    """
    User model for authentication and authorization.
    Phase 3.7.1.3: Authentication Setup
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        String(50),
        nullable=False,
        default="viewer",
        index=True,
        # Valid roles: admin, operator, viewer
    )
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Machine(Base):
    """
    Table 1: machines
    Stores CNC machine profiles and metadata.
    """
    __tablename__ = "machines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    machine_id = Column(String(255), unique=True, nullable=False, index=True)
    machine_type = Column(String(255), nullable=False)
    manufacturer = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    metadata_path = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    gan_training_jobs = relationship("GANTrainingJob", back_populates="machine", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="machine", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Machine(id={self.id}, machine_id='{self.machine_id}', model='{self.model}')>"


class GANTrainingJob(Base):
    """
    Table 2: gan_training_jobs
    Tracks GAN (TVAE) training sessions for synthetic data generation.
    """
    __tablename__ = "gan_training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    machine_id = Column(String(255), ForeignKey("machines.machine_id", ondelete="CASCADE"), nullable=False, index=True)
    epochs = Column(Integer, nullable=False)
    status = Column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
        # Valid statuses: pending, running, completed, failed
    )
    loss_history = Column(JSON, nullable=True)  # Stores training loss progression
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    machine = relationship("Machine", back_populates="gan_training_jobs")

    def __repr__(self):
        return f"<GANTrainingJob(id={self.id}, machine_id='{self.machine_id}', status='{self.status}')>"


class Prediction(Base):
    """
    Table 3: predictions
    Stores ML model predictions for predictive maintenance.
    """
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    machine_id = Column(String(255), ForeignKey("machines.machine_id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_type = Column(
        String(50),
        nullable=False,
        index=True,
        # Valid types: classification, rul (Remaining Useful Life), anomaly, timeseries
    )
    input_data = Column(JSON, nullable=False)  # Input features used for prediction
    prediction_result = Column(JSON, nullable=False)  # Model output (class, RUL value, anomaly score, etc.)
    confidence = Column(Float, nullable=True)  # Confidence score (0.0 - 1.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    machine = relationship("Machine", back_populates="predictions")
    explanations = relationship("Explanation", back_populates="prediction", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Prediction(id={self.id}, machine_id='{self.machine_id}', type='{self.prediction_type}')>"


class Explanation(Base):
    """
    Table 4: explanations
    Stores LLM-generated explanations for predictions.
    """
    __tablename__ = "explanations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    explanation_text = Column(Text, nullable=False)  # Human-readable explanation
    recommendations = Column(JSON, nullable=True)  # Actionable recommendations (list of strings or structured data)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="explanations")

    def __repr__(self):
        return f"<Explanation(id={self.id}, prediction_id={self.prediction_id})>"


class ModelVersion(Base):
    """
    Table 5: model_versions
    Tracks ML/GAN model versions and their performance metrics.
    """
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    model_type = Column(
        String(50),
        nullable=False,
        index=True,
        # Valid types: gan, classification, rul, anomaly, timeseries
    )
    version = Column(String(50), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)  # Path to saved model file
    metrics = Column(JSON, nullable=True)  # Performance metrics (accuracy, f1, loss, etc.)
    trained_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)  # Whether this version is currently in use

    def __repr__(self):
        return f"<ModelVersion(id={self.id}, type='{self.model_type}', version='{self.version}', active={self.is_active})>"
