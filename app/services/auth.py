"""
Authentication: password hashing, JWT issuing and the admin dependency.

The previous version fell back to a literal secret baked into the source when
`TUK_SECRET_KEY` was unset, and seeded an `admin@tuk.ac.ke / Admin2026!`
account on first boot and printed it to stdout. Both are gone: the signing
key must come from the environment, and the first admin is created either
explicitly with scripts/create_admin.py or from the bootstrap pair in .env.

Location: app/services/auth.py
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.services import database

logger = logging.getLogger(__name__)
ALGORITHM = "HS256"

# bcrypt silently truncates at 72 bytes; reject rather than mislead.
MAX_PASSWORD_BYTES = 72


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def password_policy_error(password: str) -> Optional[str]:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return "Password must be at most 72 bytes long."
    if len(password) < 12:
        return "Admin password must be at least 12 characters."
    return None


def create_access_token(user_id: int, email: str, role: str) -> str:
    settings = get_settings()
    issued = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(issued.timestamp()),
        "exp": issued + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        logger.debug("Rejected token: %s", exc)
        return {}


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    user = database.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return user


bearer_scheme = HTTPBearer(auto_error=False)


def require_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def ensure_bootstrap_admin() -> None:
    """
    Create the first admin only when both bootstrap credentials are supplied.
    With neither set, the instance boots with no account and the operator must
    run scripts/create_admin.py — which is the safe default.
    """
    settings = get_settings()
    if database.user_count() > 0:
        return
    if not (settings.bootstrap_admin_email and settings.bootstrap_admin_password):
        logger.warning(
            "No admin accounts exist. Create one with: python scripts/create_admin.py"
        )
        return

    error = password_policy_error(settings.bootstrap_admin_password)
    if error:
        logger.error("Refusing to bootstrap admin: %s", error)
        return

    database.create_user(
        full_name="Administrator",
        email=settings.bootstrap_admin_email,
        password_hash=hash_password(settings.bootstrap_admin_password),
        role="admin",
    )
    logger.info("Bootstrapped admin account for %s", settings.bootstrap_admin_email)
