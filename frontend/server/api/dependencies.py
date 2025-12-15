"""
FastAPI dependencies for authentication and authorization.
Phase 3.7.1.3: Authentication Setup (Days 5-6)
"""
from typing import Optional, List
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db
from db import crud, models
from utils.security import decode_token

# HTTP Bearer token security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> models.User:
    """
    Dependency to get current authenticated user from JWT token.
    
    Validates the JWT token and retrieves the user from the database.
    Raises HTTPException if token is invalid or user not found.
    
    Usage:
        @app.get("/protected")
        def protected_route(current_user: models.User = Depends(get_current_user)):
            return {"user": current_user.username}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Extract token from credentials
    token = credentials.credentials
    
    # Decode and verify token
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    # Extract user ID from payload
    user_id_str: str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception
    
    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise credentials_exception
    
    # Get user from database
    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise credentials_exception
    
    return user


def get_current_active_user(
    current_user: models.User = Depends(get_current_user),
) -> models.User:
    """
    Dependency to get current active user.
    
    Extends get_current_user to also check if user is active.
    Raises HTTPException if user is inactive.
    
    Usage:
        @app.get("/protected")
        def protected_route(user: models.User = Depends(get_current_active_user)):
            return {"user": user.username}
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


def require_role(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control.
    
    Creates a dependency that checks if current user has one of the allowed roles.
    
    Args:
        allowed_roles: List of allowed role names (e.g., ["admin", "operator"])
    
    Returns:
        FastAPI dependency function
    
    Usage:
        @app.post("/admin/users")
        def create_user(
            user_data: UserCreate,
            current_user: models.User = Depends(require_role(["admin"]))
        ):
            # Only admins can access this endpoint
            return create_new_user(user_data)
        
        @app.get("/data")
        def get_data(
            current_user: models.User = Depends(require_role(["admin", "operator"]))
        ):
            # Admins and operators can access this endpoint
            return get_all_data()
    """
    def check_role(current_user: models.User = Depends(get_current_active_user)) -> models.User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(allowed_roles)}",
            )
        return current_user
    
    return check_role


# Convenience dependencies for common role checks
require_admin = require_role(["admin"])
require_operator = require_role(["admin", "operator"])
require_viewer = require_role(["admin", "operator", "viewer"])
