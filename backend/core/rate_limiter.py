"""
core/rate_limiter.py
In-memory sliding window rate limiter for DocIntel.

Implementation: sliding window counter per (key, endpoint).
  - Window: 60 seconds (1 minute)
  - State: stored in a module-level dict — resets on process restart
  - Thread-safe: protected by a threading.Lock

Known gap (documented, not a bug):
  This limiter is per-process. If Railway scales DocIntel to multiple
  instances, each instance has its own counter — a single user could
  make N * limit requests across N instances before being blocked.
  Fix: replace _store with a Redis-backed counter. The interface is
  identical — only _get_count() and _increment() need to change.
  Tracked in: TODO(rate-limiter-redis)

Usage:
    from core.rate_limiter import check_rate_limit

    # In a router:
    check_rate_limit(user_id=uid, endpoint="login")
    # Raises HTTP 429 if limit exceeded, otherwise returns None.

    # With a custom limit (overrides config):
    check_rate_limit(user_id=uid, endpoint="upload", limit=5)
"""

import time
import threading
from collections import defaultdict, deque

from fastapi import HTTPException, status

from core.config import config as app_config


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_lock:  threading.Lock = threading.Lock()

# { key: deque of timestamps (float, seconds since epoch) }
_store: dict[str, deque] = defaultdict(deque)

_WINDOW_SECONDS = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_key(user_id: str, endpoint: str) -> str:
    return f"{endpoint}:{user_id}"


def _clean_window(timestamps: deque, now: float) -> None:
    """Remove timestamps older than the window."""
    cutoff = now - _WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()


def _get_count(key: str, now: float) -> int:
    timestamps = _store[key]
    _clean_window(timestamps, now)
    return len(timestamps)


def _increment(key: str, now: float) -> None:
    _store[key].append(now)


def _get_limit(endpoint: str, override: int | None) -> int:
    """Resolve the rate limit for an endpoint."""
    if override is not None:
        return override

    limits = {
        "login":    app_config.rate_limit_login_per_minute,
        "signup":   app_config.rate_limit_login_per_minute,   # same as login
        "upload":   app_config.rate_limit_upload_per_minute,
        "query":    app_config.rate_limit_query_per_minute,
    }
    return limits.get(endpoint, 60)  # default: 60/min


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_rate_limit(
    user_id: str,
    endpoint: str,
    limit: int | None = None,
) -> None:
    """
    Check rate limit for a user+endpoint combination.

    Args:
        user_id:  The authenticated user's ID (or IP for unauthenticated).
        endpoint: Endpoint label — 'login', 'signup', 'upload', 'query'.
        limit:    Override the configured limit for this call.

    Raises:
        HTTP 429 Too Many Requests if the limit is exceeded.

    Returns None if within limit (also increments the counter).
    """
    key       = _make_key(user_id, endpoint)
    max_calls = _get_limit(endpoint, limit)

    with _lock:
        now   = time.time()
        count = _get_count(key, now)

        if count >= max_calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":       "Rate limit exceeded.",
                    "code":        "RATE_LIMIT_EXCEEDED",
                    "limit":       max_calls,
                    "window":      f"{_WINDOW_SECONDS}s",
                    "endpoint":    endpoint,
                    "retry_after": _WINDOW_SECONDS,
                },
                headers={"Retry-After": str(_WINDOW_SECONDS)},
            )

        _increment(key, now)


def get_rate_limit_status(user_id: str, endpoint: str) -> dict:
    """
    Return current rate limit status for a user+endpoint (for debugging).
    Not exposed as an API endpoint — internal use only.
    """
    key       = _make_key(user_id, endpoint)
    max_calls = _get_limit(endpoint, None)

    with _lock:
        now   = time.time()
        count = _get_count(key, now)

    return {
        "user_id":   user_id,
        "endpoint":  endpoint,
        "count":     count,
        "limit":     max_calls,
        "remaining": max(0, max_calls - count),
        "window_s":  _WINDOW_SECONDS,
    }


def reset_rate_limit(user_id: str, endpoint: str) -> None:
    """
    Reset rate limit for a user+endpoint. For testing only.
    """
    key = _make_key(user_id, endpoint)
    with _lock:
        if key in _store:
            del _store[key]