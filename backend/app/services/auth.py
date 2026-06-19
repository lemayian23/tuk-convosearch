"""
Authentication Service
Location: backend/app/services/auth.py

Handles:
  - Password hashing and verification (bcrypt direct — already in requirements.txt)
  - JWT token creation and decoding (python-jose)
  - FastAPI dependency for protecting admin routes
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

SECRET_KEY = os.environ.get(
    "TUK_SECRET_KEY",
    "tuk-convosearch-secret-key-change-in-production-2026"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

# ------------------------------------------------------------------ #
# Password hashing  (bcrypt direct — no passlib)
# ------------------------------------------------------------------ #

def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

# ------------------------------------------------------------------ #
# JWT tokens
# ------------------------------------------------------------------ #

def create_access_token(user_id: int, email: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return {}

# ------------------------------------------------------------------ #
# FastAPI dependency — use this on any admin route
# ------------------------------------------------------------------ #

bearer_scheme = HTTPBearer(auto_error=False)


def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    """
    FastAPI dependency that verifies the JWT in the Authorization header.
    Usage:  async def my_route(admin=Depends(require_admin)):
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload