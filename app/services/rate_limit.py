"""
In-process fixed-window rate limiting.

Deliberately not Redis: the deployment runs a single uvicorn worker because
one Ollama instance is the bottleneck anyway, so per-process counters cover
every request. The comment matters if that ever changes — with several
workers each one gets its own budget.

Location: app/services/rate_limit.py
"""

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Optional, Tuple

from app.config import get_settings


class RateLimiter:
    def __init__(self, per_minute: int, burst: int):
        self.per_minute = per_minute
        self.burst = burst
        self._hits: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, seconds_until_next_slot)."""
        now = time.monotonic()
        window = 60.0
        limit = self.per_minute + self.burst

        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                if not hits:
                    return False, 1
                retry_after = int(window - (now - hits[0])) + 1
                return False, max(1, retry_after)
            hits.append(now)
            # Opportunistic cleanup so the dict cannot grow without bound.
            if len(self._hits) > 10_000:
                for stale in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(stale, None)
            return True, 0


_limiter: Optional[RateLimiter] = None
_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        with _lock:
            if _limiter is None:
                settings = get_settings()
                _limiter = RateLimiter(
                    per_minute=settings.rate_limit_per_minute,
                    burst=settings.rate_limit_burst,
                )
    return _limiter


def reset_limiter() -> None:
    global _limiter
    with _lock:
        _limiter = None
