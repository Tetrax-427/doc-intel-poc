"""
Supabase JWT authentication for DocIntel.

Usage in FastAPI routes:
    from core.auth import get_current_user, get_user_id, UserContext

    @router.get("/my-endpoint")
    def my_endpoint(user: UserContext = Depends(get_current_user)):
        docs = get_all_documents(user_id=user.user_id)
        ...

Behaviour:
- SUPABASE_JWT_SECRET set   → validates Bearer JWT from Supabase Auth; 401 on invalid
- SUPABASE_JWT_SECRET unset → returns dev_user fallback so app runs locally
                              without auth configured

Never raises on missing env var — degrades gracefully to dev mode.
"""

import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

_bearer = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    user_id: str
    email:   str | None = None
    is_dev:  bool = False   # True when running without Supabase Auth configured


def get_user_id(user) -> str:
    """
    Extract user_id from a UserContext or dict safely.
    Returns 'anonymous' for None or missing key.
    """
    if user is None:
        return "anonymous"
    if isinstance(user, UserContext):
        return user.user_id or "anonymous"
    if isinstance(user, dict):
        return user.get("user_id", "anonymous")
    return "anonymous"


# ── Supabase JWT verification ─────────────────────────────────────────────────

def _verify_supabase_token(token: str) -> UserContext:
    """
    Decode and verify a Supabase-issued JWT.

    Supabase tokens are signed with HS256 using the project's JWT secret.
    Found in: Supabase Dashboard → Project Settings → API → JWT Secret

    Raises HTTPException(401) on any verification failure.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()

    if not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SUPABASE_JWT_SECRET not configured on server.",
        )

    try:
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase sets aud="authenticated"
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim.",
        )

    return UserContext(
        user_id=user_id,
        email=payload.get("email"),
        is_dev=False,
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserContext:
    """
    FastAPI dependency — inject into any route that needs the current user.

    - No SUPABASE_JWT_SECRET configured → dev_user (local / pre-auth deploy)
    - SUPABASE_JWT_SECRET set, no token  → 401
    - SUPABASE_JWT_SECRET set, valid token → UserContext with Supabase user_id
    - SUPABASE_JWT_SECRET set, invalid/expired → 401
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()

    # Dev mode — auth not configured yet
    if not jwt_secret:
        return UserContext(user_id="dev_user", is_dev=True)

    # Auth required — no token provided
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _verify_supabase_token(credentials.credentials)