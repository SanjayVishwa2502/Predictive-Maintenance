"""Configuration management.

Phase 3.7.1.1: Project Initialization - Industrial Grade

Important: this project is often started from different working directories
(repo root, `frontend/server`, etc). We must load the correct `.env` reliably
regardless of CWD so DB/auth settings (e.g. PostgreSQL port) are applied.
"""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        # Always load the `.env` that sits next to this config module.
        # This avoids subtle misconfiguration when uvicorn is started from
        # the repo root (or any other CWD).
        env_file=str(Path(__file__).resolve().parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_VERSION: str = "3.7.0"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/predictive_maintenance"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_DB: int = 3  # DB 3 for API response caching
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    # JWT
    JWT_SECRET_KEY: str = "dev-jwt-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"]
    # Allow any localhost/127.0.0.1 port during dev (Vite may choose 5174+, etc.).
    # Use env var CORS_ORIGIN_REGEX to override.
    CORS_ORIGIN_REGEX: Optional[str] = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    
    # Paths
    GAN_ROOT_PATH: str = "GAN"
    ML_MODELS_PATH: str = "ml_models"
    LLM_PATH: str = "LLM"
    
    # Celery
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_TASK_ACKS_LATE: bool = True
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = 1
    
    # Monitoring
    FLOWER_PORT: int = 5555
    PROMETHEUS_PORT: int = 9090
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Prediction history audit (CSV)
    # Stores snapshots/runs/LLM updates to a machine-specific CSV for permanent offline usage.
    AUDIT_CSV_ENABLED: bool = True
    AUDIT_CSV_DIR: str = "reports/audit_csv"


# Singleton settings instance
settings = Settings()
