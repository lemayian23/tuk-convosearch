"""
Admin authentication routes. Location: app/api/auth_routes.py
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import enforce_rate_limit
from app.api.schemas import LoginRequest, TokenResponse
from app.config import get_settings
from app.services.auth import (
    authenticate_user,
    create_access_token,
    require_admin,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, _: None = Depends(enforce_rate_limit)) -> TokenResponse:
    settings = get_settings()

    user = authenticate_user(payload.email, payload.password)
    if not user:
        logger.info("Failed login for %s from rate-limited window", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(user["user_id"], user["email"], user["role"])
    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.access_token_expire_minutes,
        user={"user_id": user["user_id"], "email": user["email"], "full_name": user["full_name"], "role": user["role"]},
    )


@router.get("/me")
async def me(admin: Dict = Depends(require_admin)) -> Dict:
    return admin
