"""
Supabase JWT authentication for DocIntel.
"""

import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client

_bearer = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    user_id: str
    email:   str | None = None
    is_dev:  bool = False


def get_user_id(user) -> str:
    if user is None:
        return "anonymous"
    if isinstance(user, UserContext):
        return user.user_id or "anonymous"
    if isinstance(user, dict):
        return user.get("user_id", "anonymous")
    return "anonymous"


def _verify_supabase_token(token: str) -> UserContext:
    """
    Validate token by calling Supabase — works with both HS256 and ES256.
    """
    url         = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SUPABASE_URL or SUPABASE_SERVICE_KEY not configured.",
        )

    try:
        sb       = create_client(url, service_key)
        response = sb.auth.get_user(token)
        user     = response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return UserContext(user_id=user.id, email=user.email, is_dev=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserContext:
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()

    # Dev mode — auth not configured
    if not jwt_secret:
        return UserContext(user_id="dev_user", is_dev=True)

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _verify_supabase_token(credentials.credentials)