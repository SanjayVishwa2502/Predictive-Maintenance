"""
Pydantic schemas for authentication.
Phase 3.7.1.3: Authentication Setup (Days 5-6)
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base user schema with common fields."""
    username: str = Field(..., min_length=3, max_length=100, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    role: str = Field(default="viewer", description="User role: admin, operator, or viewer")


class UserCreate(UserBase):
    """Schema for user registration.
    
    When registering as admin role, admin_approval_password is required
    (must be the password of an existing admin). The first user ever
    registered can be admin without approval.
    """
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    admin_approval_password: Optional[str] = Field(
        None,
        description="Required when registering as admin: password of an existing admin for approval"
    )


class UserUpdate(BaseModel):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response (without password)."""
    id: UUID
    is_active: bool
    is_approved: bool = True  # Default True for backward compatibility
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2 (was orm_mode in v1)


# ==================== Authentication Schemas ====================

class LoginRequest(BaseModel):
    """Schema for login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenPayload(BaseModel):
    """Schema for decoded JWT token payload."""
    sub: str = Field(..., description="User ID (subject)")
    username: str = Field(..., description="Username")
    role: str = Field(..., description="User role")
    exp: int = Field(..., description="Expiration timestamp")


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class PasswordChange(BaseModel):
    """Schema for password change."""
    old_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class SecureDeleteRequest(BaseModel):
    """Schema for secure delete operations requiring admin password verification."""
    password: str = Field(..., description="Admin password to confirm deletion")


class PasswordVerifyRequest(BaseModel):
    """Schema for password verification."""
    password: str = Field(..., description="Password to verify")


# ==================== Approval Schemas ====================

class UserApprovalRequest(BaseModel):
    """Schema for approving or rejecting a pending user."""
    user_id: UUID = Field(..., description="ID of the user to approve/reject")
    approve: bool = Field(..., description="True to approve, False to reject/delete")


class PendingUsersResponse(BaseModel):
    """Schema for listing pending users awaiting approval."""
    pending_users: list[UserResponse] = Field(default_factory=list)
    total: int = Field(default=0)
