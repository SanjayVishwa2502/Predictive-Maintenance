"""
API Configuration Endpoint
Phase 3.7.9: Common endpoint for service discovery

Returns server configuration info including:
- API version
- Server port and host
- Available services/endpoints
- Environment info
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from config import settings

router = APIRouter(prefix="/api/config", tags=["Configuration"])


class ServiceInfo(BaseModel):
    """Information about an available service"""
    name: str
    available: bool
    base_path: str
    description: str


class ServerConfig(BaseModel):
    """Server configuration response"""
    api_version: str
    environment: str
    host: str
    port: int
    base_url: str
    debug: bool
    services: List[ServiceInfo]
    cors_origins: List[str]
    features: Dict[str, bool]


@router.get("/", response_model=ServerConfig)
async def get_server_config():
    """
    Get server configuration and available services.
    
    This is a common endpoint that provides service discovery info
    for clients to understand what's available on this backend.
    """
    # Get host/port from environment or defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    base_url = os.environ.get("BASE_URL", f"http://localhost:{port}")
    
    # Define available services
    services = [
        ServiceInfo(
            name="ML Predictions",
            available=True,
            base_path="/api/ml",
            description="Machine learning predictions (classification, RUL, anomaly, timeseries)"
        ),
        ServiceInfo(
            name="GAN Training",
            available=True,
            base_path="/api/gan",
            description="Synthetic data generation and machine profile management"
        ),
        ServiceInfo(
            name="LLM Explanations",
            available=True,
            base_path="/api/llm",
            description="AI-powered explanations and maintenance recommendations"
        ),
        ServiceInfo(
            name="Authentication",
            available=True,
            base_path="/api/auth",
            description="JWT-based authentication and user management"
        ),
        ServiceInfo(
            name="WebSocket",
            available=True,
            base_path="/ws",
            description="Real-time updates via WebSocket connections"
        ),
        ServiceInfo(
            name="ML Training",
            available=True,
            base_path="/api/ml/training",
            description="Model training and management"
        ),
    ]
    
    # Feature flags
    features = {
        "vlm_integration": False,  # VLM is external (Jetson)
        "realtime_sensors": True,
        "auto_prediction": True,
        "llm_explanations": True,
        "gan_augmentation": True,
        "model_training": True,
        "audit_logging": settings.AUDIT_CSV_ENABLED,
    }
    
    return ServerConfig(
        api_version=settings.API_VERSION,
        environment=settings.ENVIRONMENT,
        host=host,
        port=port,
        base_url=base_url,
        debug=settings.DEBUG,
        services=services,
        cors_origins=settings.CORS_ORIGINS,
        features=features
    )


@router.get("/endpoints")
async def get_available_endpoints():
    """
    Get a list of all available API endpoints with their methods.
    
    Useful for client-side service discovery.
    """
    endpoints = {
        "config": {
            "GET /api/config": "Get server configuration",
            "GET /api/config/endpoints": "List all endpoints",
        },
        "health": {
            "GET /": "Root health check",
            "GET /health": "Detailed health check",
            "GET /metrics": "Prometheus metrics",
        },
        "auth": {
            "POST /api/auth/login": "User login",
            "POST /api/auth/register": "User registration",
            "POST /api/auth/refresh": "Refresh access token",
            "GET /api/auth/me": "Get current user info",
            "POST /api/auth/verify-password": "Verify user password",
        },
        "ml": {
            "GET /api/ml/machines": "List available machines",
            "GET /api/ml/machines/{id}/status": "Get machine sensor status",
            "GET /api/ml/machines/{id}/snapshots": "Get prediction snapshots",
            "POST /api/ml/machines/{id}/auto/run_once": "Run prediction",
            "GET /api/ml/runs/{run_id}": "Get run details",
            "GET /api/ml/health": "ML service health check",
        },
        "gan": {
            "GET /api/gan/machines": "List GAN machines",
            "POST /api/gan/machines/{id}/profile": "Upload machine profile",
            "GET /api/gan/machines/{id}/baseline": "Get machine baseline",
            "POST /api/gan/augment": "Start augmentation task",
            "GET /api/gan/tasks/{task_id}": "Get task status",
        },
        "llm": {
            "GET /api/llm/info": "Get LLM service info",
            "POST /api/llm/explain": "Generate explanation",
        },
        "websocket": {
            "WS /ws/llm/events": "LLM event stream",
            "WS /ws/training/progress": "Training progress stream",
        },
    }
    
    return {
        "total_categories": len(endpoints),
        "endpoints": endpoints
    }


@router.get("/vlm")
async def get_vlm_config():
    """
    Get VLM (Vision Language Model) integration configuration.
    
    VLM runs on external hardware (e.g., Jetson Orin Nano).
    This endpoint provides the expected VLM API structure.
    """
    return {
        "vlm_available": False,
        "note": "VLM runs on external Jetson device, not on this server",
        "expected_endpoints": {
            "health": "GET {vlm_host}/health",
            "stream": "GET {vlm_host}/stream (WebRTC/HLS/MJPEG)",
            "latest": "GET {vlm_host}/latest (JSON with labels, bbox, anomaly)",
            "start_session": "POST {vlm_host}/start_session",
            "stop_session": "POST {vlm_host}/stop_session",
        },
        "recommended_protocols": [
            "WebRTC (lowest latency, browser-native)",
            "HLS (HTTP Live Streaming, wide compatibility)",
            "MJPEG (simple, higher bandwidth)",
        ],
        "configuration": {
            "storage_key": "pm_vlm_endpoint",
            "default_port": 8080,
            "configure_at": "/settings → VLM Settings",
        }
    }
