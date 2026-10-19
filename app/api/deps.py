"""Shared FastAPI dependencies. Location: app/api/deps.py"""

from fastapi import HTTPException, Request, status

from app.config import get_settings
from app.services.rate_limit import get_limiter

__all__ = ["client_ip", "enforce_rate_limit"]


def client_ip(request: Request) -> str:
    """
    Best-effort client address.

    X-Forwarded-For is only trusted when the app sits behind a reverse proxy
    that overwrites the header — Caddy does, in the shipped deployment, with
    the app port unreachable from outside. If the API is ever exposed
    directly, set TUK_TRUST_FORWARDED_FOR=false or the header can be spoofed
    to dodge the rate limit.
    """
    if get_settings().trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """Route dependency: 429 with Retry-After when a client exceeds its budget."""
    allowed, retry_after = get_limiter().check(client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You are asking too quickly. Please wait a moment.",
            headers={"Retry-After": str(retry_after)},
        )
