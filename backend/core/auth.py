"""
core/auth.py
Clerk JWT authentication for DocIntel.

Usage in FastAPI routes:
    from core.auth import get_current_user, UserContext

    @router.get("/my-endpoint")
    def my_endpoint(user: UserContext = Depends(get_current_user)):
        docs = get_all_documents(user_id=user.user_id)
        ...

Behaviour:
- CLERK_SECRET_KEY set   → validates Bearer JWT from Clerk; 401 on invalid token
- CLERK_SECRET_KEY unset → returns dev_user fallback so the app runs locally
  and on Railway without auth configured yet

Never raises on missing env var — degrades gracefully to dev mode.
"""

import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


@dataclass
class UserContext:
    user_id: str
    email: str | None = None
    is_dev: bool = False   # True when running without Clerk configured


# ── Clerk JWT verification ────────────────────────────────────────────────────

def _verify_clerk_token(token: str) -> UserContext:
    """
    Decode and verify a Clerk-issued JWT.

    Clerk tokens are standard JWTs signed with RS256.
    We use PyJWT + Clerk's JWKS endpoint to verify the signature.

    Raises HTTPException(401) on any verification failure so the
    caller can return immediately without extra error handling.
    """
    import jwt                          # PyJWT — already in requirements
    import requests as _requests        # standard requests

    clerk_secret = os.getenv("CLERK_SECRET_KEY", "")
    jwks_url = "https://api.clerk.dev/v1/jwks"

    try:
        # Fetch JWKS — Clerk rotates keys occasionally so we fetch live
        jwks_resp = _requests.get(jwks_url, timeout=5)
        jwks_resp.raise_for_status()
        jwks = jwks_resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not fetch Clerk JWKS: {exc}",
        )

    try:
        # Let PyJWT pick the right key from the JWKS set
        jwks_client = jwt.PyJWKClient(jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},   # Clerk doesn't always set aud
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
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

    - No CLERK_SECRET_KEY configured → dev_user (local / pre-auth deploy)
    - CLERK_SECRET_KEY configured, no token → 401
    - CLERK_SECRET_KEY configured, valid token → UserContext with Clerk user_id
    - CLERK_SECRET_KEY configured, invalid token → 401
    """
    clerk_secret = os.getenv("CLERK_SECRET_KEY", "").strip()

    # Dev mode — Clerk not configured yet
    if not clerk_secret:
        return UserContext(user_id="dev_user", is_dev=True)

    # Auth required — no token provided
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _verify_clerk_token(credentials.credentials)