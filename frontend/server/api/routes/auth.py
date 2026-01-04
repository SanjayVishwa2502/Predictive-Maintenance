"""
Authentication API routes.
Phase 3.7.1.3: Authentication Setup (Days 5-6)

Endpoints:
- POST /api/auth/register - Create new user
- POST /api/auth/login - Get JWT token
- POST /api/auth/refresh - Refresh access token
- GET /api/auth/me - Get current user info
"""
import logging
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.dependencies import get_current_active_user, require_admin
from api.schemas.auth import (
    LoginRequest,
    RefreshTokenRequest,
    Token,
    UserCreate,
    UserResponse,
    SecureDeleteRequest,
    PasswordVerifyRequest,
    UserApprovalRequest,
    PendingUsersResponse,
)
from config import settings
from database import get_db
from db import crud, models
from utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

# NOTE: The FastAPI app mounts this router under the `/api/auth` prefix.
# Keeping the router prefix empty prevents accidental double-prefixes like
# `/api/auth/api/auth/*` if the app is also configured with a prefix.
router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
) -> Any:
    """
    Register a new user.
    
    Creates a new user account with hashed password.
    Default role is 'viewer'. Admin role can only be assigned by existing admins.
    
    **Request Body:**
    ```json
    {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "SecurePass123",
        "role": "viewer"
    }
    ```
    
    **Response:**
    ```json
    {
        "id": "uuid",
        "username": "johndoe",
        "email": "john@example.com",
        "role": "viewer",
        "is_active": true,
        "created_at": "2025-12-15T10:00:00Z",
        "updated_at": "2025-12-15T10:00:00Z"
    }
    ```
    
    **Errors:**
    - 400: Username or email already exists
    """
    logger.info(f"Registration attempt for username: {user_data.username}")
    
    # Check if username already exists
    if crud.get_user_by_username(db, user_data.username):
        logger.warning(f"Registration failed: username '{user_data.username}' already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    
    # Check if email already exists
    if crud.get_user_by_email(db, user_data.email):
        logger.warning(f"Registration failed: email '{user_data.email}' already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Validate role
    valid_roles = ["admin", "operator", "viewer"]
    if user_data.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}",
        )
    
    # Admin registration requires approval from an existing admin
    # Exception: the very first user can be admin without approval
    if user_data.role == "admin":
        # Check if any users exist (first user exception)
        existing_users = crud.get_users(db, skip=0, limit=1)
        if existing_users:
            # Not the first user - require admin approval password
            if not user_data.admin_approval_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Admin registration requires approval: provide an existing admin's password",
                )
            
            # Find any admin user and verify the approval password
            admins = [u for u in crud.get_users(db, skip=0, limit=100) if u.role == "admin"]
            if not admins:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No existing admin found. Contact system administrator.",
                )
            
            # Try to verify against any admin's password
            admin_approved = False
            for admin in admins:
                if verify_password(user_data.admin_approval_password, admin.hashed_password):
                    admin_approved = True
                    logger.info(f"Admin registration approved by admin user: {admin.username}")
                    break
            
            if not admin_approved:
                logger.warning(f"Admin registration failed: invalid admin approval password for '{user_data.username}'")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid admin approval password. Must match an existing admin's password.",
                )
        else:
            # First user - allow admin without approval (becomes super admin)
            logger.info(f"First user registration as admin (super admin): {user_data.username}")
    
    # Determine approval status based on role:
    # - Viewers: auto-approved
    # - Admins: auto-approved (they already passed admin password check)
    # - Operators: require admin approval (is_approved=False)
    is_approved = True  # Default for viewers and admins
    if user_data.role == "operator":
        # Check if any admins exist - if not, first operator is auto-approved
        existing_users = crud.get_users(db, skip=0, limit=1)
        if existing_users:
            admins = [u for u in crud.get_users(db, skip=0, limit=100) if u.role == "admin"]
            if admins:
                is_approved = False  # Needs admin approval
                logger.info(f"Operator '{user_data.username}' registered, pending admin approval")
            else:
                # No admins exist, auto-approve first operator
                is_approved = True
                logger.info(f"First operator '{user_data.username}' auto-approved (no admins exist)")
        else:
            # First user ever as operator - auto-approve
            is_approved = True
            logger.info(f"First user registration as operator: {user_data.username}")
    
    # Hash password and create user
    hashed_password = hash_password(user_data.password)
    user = crud.create_user(
        db=db,
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        role=user_data.role,
        is_approved=is_approved,
    )
    
    logger.info(f"User registered successfully: {user.username} (ID: {user.id}, approved: {is_approved})")
    return user


@router.post("/login", response_model=Token)
def login(
    credentials: LoginRequest,
    db: Session = Depends(get_db),
) -> Any:
    """
    Login and get JWT tokens.
    
    Authenticates user with username and password.
    Returns access token (30 min) and refresh token (7 days).
    
    **Request Body:**
    ```json
    {
        "username": "johndoe",
        "password": "SecurePass123"
    }
    ```
    
    **Response:**
    ```json
    {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
        "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
        "token_type": "bearer"
    }
    ```
    
    **Token Payload:**
    ```json
    {
        "sub": "user_id",
        "username": "johndoe",
        "role": "admin",
        "exp": 1732723200
    }
    ```
    
    **Errors:**
    - 401: Invalid username or password
    - 403: User account is inactive
    """
    logger.info(f"Login attempt for username: {credentials.username}")
    
    # Authenticate user
    user = crud.authenticate_user(db, credentials.username, credentials.password)
    if not user:
        logger.warning(f"Login failed: invalid credentials for username '{credentials.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        logger.warning(f"Login failed: user '{credentials.username}' is inactive")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    # Create token payload
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    
    # Generate tokens
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user.id), "username": user.username})
    
    logger.info(f"User logged in successfully: {user.username} (ID: {user.id})")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token)
def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
) -> Any:
    """
    Refresh access token using refresh token.
    
    Validates refresh token and issues new access and refresh tokens.
    
    **Request Body:**
    ```json
    {
        "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
    }
    ```
    
    **Response:**
    ```json
    {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
        "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
        "token_type": "bearer"
    }
    ```
    
    **Errors:**
    - 401: Invalid or expired refresh token
    - 404: User not found
    - 403: User account is inactive
    """
    logger.info("Token refresh attempt")
    
    # Decode refresh token
    payload = decode_token(refresh_data.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        logger.warning("Token refresh failed: invalid refresh token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user ID from token
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    # Get user from database
    from uuid import UUID
    user = crud.get_user_by_id(db, UUID(user_id_str))
    if not user:
        logger.warning(f"Token refresh failed: user not found (ID: {user_id_str})")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check if user is active
    if not user.is_active:
        logger.warning(f"Token refresh failed: user '{user.username}' is inactive")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    # Generate new tokens
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token({"sub": str(user.id), "username": user.username})
    
    logger.info(f"Token refreshed successfully for user: {user.username} (ID: {user.id})")
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Get current authenticated user information.
    
    Returns the profile of the currently logged-in user.
    Requires valid JWT access token in Authorization header.
    
    **Headers:**
    ```
    Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
    ```
    
    **Response:**
    ```json
    {
        "id": "uuid",
        "username": "johndoe",
        "email": "john@example.com",
        "role": "admin",
        "is_active": true,
        "created_at": "2025-12-15T10:00:00Z",
        "updated_at": "2025-12-15T10:00:00Z"
    }
    ```
    
    **Errors:**
    - 401: Invalid or expired token
    - 403: User account is inactive
    """
    logger.info(f"User info requested: {current_user.username} (ID: {current_user.id})")
    return current_user


# Admin-only endpoint example
@router.get("/users", response_model=list[UserResponse])
def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Any:
    """
    List all users (Admin only).
    
    Returns paginated list of all users.
    Only accessible to users with 'admin' role.
    
    **Query Parameters:**
    - skip: Number of records to skip (default: 0)
    - limit: Maximum number of records to return (default: 100)
    
    **Headers:**
    ```
    Authorization: Bearer <admin_token>
    ```
    
    **Response:**
    ```json
    [
        {
            "id": "uuid",
            "username": "johndoe",
            "email": "john@example.com",
            "role": "admin",
            "is_active": true,
            "created_at": "2025-12-15T10:00:00Z",
            "updated_at": "2025-12-15T10:00:00Z"
        }
    ]
    ```
    
    **Errors:**
    - 401: Invalid or expired token
    - 403: Insufficient permissions (not admin)
    """
    logger.info(f"Admin user list requested by: {current_user.username}")
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@router.post("/verify-password")
def verify_user_password(
    verify_data: PasswordVerifyRequest,
    current_user: models.User = Depends(get_current_active_user),
) -> Any:
    """
    Verify the current user's password.
    
    Used for secure operations that require password re-entry (e.g., delete operations).
    
    **Request Body:**
    ```json
    {
        "password": "UserPassword123"
    }
    ```
    
    **Response:**
    ```json
    {
        "verified": true
    }
    ```
    
    **Errors:**
    - 401: Invalid password
    """
    if not verify_password(verify_data.password, current_user.hashed_password):
        logger.warning(f"Password verification failed for user: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )
    
    logger.info(f"Password verified for user: {current_user.username}")
    return {"verified": True}


# ==================== Operator Approval Endpoints (Admin Only) ====================

@router.get("/pending-users", response_model=PendingUsersResponse)
def get_pending_users(
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Any:
    """
    Get list of users pending approval (Admin only).
    
    Returns operators who have registered but not yet been approved.
    
    **Response:**
    ```json
    {
        "pending_users": [...],
        "total": 5
    }
    ```
    """
    logger.info(f"Pending users list requested by admin: {current_user.username}")
    pending = crud.get_pending_users(db, skip=skip, limit=limit)
    return PendingUsersResponse(pending_users=pending, total=len(pending))


@router.post("/approve-user", response_model=UserResponse)
def approve_user(
    approval: UserApprovalRequest,
    current_user: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Any:
    """
    Approve or reject a pending user (Admin only).
    
    **Request Body:**
    ```json
    {
        "user_id": "uuid",
        "approve": true
    }
    ```
    
    If approve=true, the user's is_approved is set to True.
    If approve=false, the user is deleted from the system.
    
    **Errors:**
    - 404: User not found
    """
    user = crud.get_user_by_id(db, approval.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    if approval.approve:
        updated_user = crud.approve_user(db, approval.user_id)
        logger.info(f"User '{user.username}' approved by admin '{current_user.username}'")
        return updated_user
    else:
        crud.reject_user(db, approval.user_id)
        logger.info(f"User '{user.username}' rejected and deleted by admin '{current_user.username}'")
        # Return a representation of the deleted user
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail=f"User '{user.username}' has been rejected and removed",
        )
