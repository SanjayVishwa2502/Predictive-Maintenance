"""
FastAPI Main Application
Phase 3.7.1.1: Project Initialization - Industrial Grade

Central application factory with:
- Lifespan management
- Middleware configuration
- Router registration
- Health check endpoints
- Comprehensive logging
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 80)
    logger.info(f"Starting Predictive Maintenance Dashboard API v{settings.API_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info("=" * 80)
    
    # TODO: Initialize database connections
    # TODO: Initialize Redis connections
    # TODO: Load ML models
    
    yield
    
    # Shutdown
    logger.info("Shutting down Predictive Maintenance Dashboard API...")
    # TODO: Close database connections
    # TODO: Close Redis connections


# Create FastAPI application
app = FastAPI(
    title="Predictive Maintenance Dashboard API",
    description="""
    # Predictive Maintenance Dashboard - Industrial Grade
    
    Unified API for predictive maintenance operations integrating:
    - **GAN Module**: TVAE model training and synthetic data generation
    - **ML Module**: Health classification, RUL prediction, anomaly detection, timeseries forecasting
    - **LLM Module**: AI-powered explanations and maintenance recommendations
    - **Authentication**: JWT-based role-based access control
    
    ## Features
    - Real-time training progress via WebSocket
    - Async task processing with Celery
    - PostgreSQL database for metadata
    - Redis caching and pub/sub
    - Prometheus metrics
    
    ## Architecture
    ```
    React Frontend ↔ FastAPI Backend ↔ [GAN/ML/LLM Systems]
          ↓              ↓                      ↓
      Browser        Celery Workers      File System (Models/Data)
                          ↓
                  Redis + PostgreSQL
    ```
    """,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS - Allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================================================
# Router Registration
# ============================================================================

# Import routers
from api.routes import auth

# Include routers
app.include_router(auth.router)

# TODO: Add remaining routers as implemented
# from api.routes import gan, ml, llm, dashboard, websocket
# app.include_router(gan.router)
# app.include_router(ml.router)
# app.include_router(llm.router)
# app.include_router(dashboard.router)
# app.include_router(websocket.router)

# app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
# app.include_router(gan.router, prefix="/api/gan", tags=["GAN"])
# app.include_router(ml.router, prefix="/api/ml", tags=["ML"])
# app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
# app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
# app.include_router(websocket.router, tags=["WebSocket"])

# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - API health check
    Returns basic API information
    """
    return {
        "status": "healthy",
        "service": "Predictive Maintenance Dashboard API",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.DEBUG else None
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint
    Checks status of all subsystems
    """
    health_status = {
        "status": "healthy",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "services": {
            "api": True,
            "database": "not_checked",  # TODO: Add actual DB health check
            "redis": "not_checked",     # TODO: Add actual Redis health check
            "celery": "not_checked",    # TODO: Add actual Celery health check
        }
    }
    
    return health_status


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint
    Returns application metrics in Prometheus format
    """
    # TODO: Implement Prometheus metrics
    return {
        "message": "Prometheus metrics endpoint - to be implemented",
        "todo": "Integrate prometheus_client library"
    }


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
