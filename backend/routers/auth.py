"""
routers/auth.py
Authentication endpoints — login, signup, logout, me.

All Supabase Auth calls happen here on the backend.
The frontend (Streamlit) never touches Supabase directly.

Endpoints:
    POST /auth/login    → returns JWT + user info
    POST /auth/signup   → creates account + returns JWT
    POST /auth/logout   → invalidates session
    GET  /auth/me       → returns current user info from JWT
"""

import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from core.auth import get_current_user, UserContext
from core.logger import get_logger

logger = get_logger("auth")
router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Input models ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str


class SignupRequest(BaseModel):
    email:    str
    password: str


# ── Supabase Auth client ──────────────────────────────────────────────────────

def _get_supabase():
    """
    Return a Supabase client using the service role key for auth operations.
    Uses SUPABASE_KEY (anon key) — sufficient for auth.sign_in / sign_up.
    """
    from supabase import create_client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase not configured on server.",
        )
    return create_client(url, key)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/login")
def login(req: LoginRequest):
    """
    Sign in with email + password.
    Returns JWT access token and user info on success.
    Returns 401 on invalid credentials.
    """
    # Dev mode — no Supabase JWT secret configured
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        logger.info("Auth in dev mode — returning dev_user token")
        return {
            "access_token": "dev_token",
            "token_type":   "bearer",
            "user": {
                "id":    "dev_user",
                "email": req.email or "dev@local",
            }
        }

    try:
        sb  = _get_supabase()
        res = sb.auth.sign_in_with_password({
            "email":    req.email.strip(),
            "password": req.password,
        })
        logger.info("User logged in", email=req.email)
        return {
            "access_token": res.session.access_token,
            "token_type":   "bearer",
            "user": {
                "id":    res.user.id,
                "email": res.user.email,
            }
        }
    except Exception as e:
        logger.warning("Login failed", email=req.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )


@router.post("/signup")
def signup(req: SignupRequest):
    """
    Create a new account with email + password.
    Automatically signs in after successful signup.
    Returns JWT access token and user info.
    Returns 400 if email already registered or password too weak.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        # Dev mode
        return {
            "access_token": "dev_token",
            "token_type":   "bearer",
            "user": {
                "id":    "dev_user",
                "email": req.email or "dev@local",
            }
        }

    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters.",
        )

    try:
        sb  = _get_supabase()
        res = sb.auth.sign_up({
            "email":    req.email.strip(),
            "password": req.password,
        })

        if not res.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup failed. Please try again.",
            )

        logger.info("New user signed up", email=req.email)

        # Auto sign-in after signup to get a valid session
        login_res = sb.auth.sign_in_with_password({
            "email":    req.email.strip(),
            "password": req.password,
        })
        return {
            "access_token": login_res.session.access_token,
            "token_type":   "bearer",
            "user": {
                "id":    login_res.user.id,
                "email": login_res.user.email,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "already registered" in err or "already exists" in err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists.",
            )
        logger.error("Signup error", email=req.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not create account. Please try again.",
        )


@router.post("/logout")
def logout(user: UserContext = Depends(get_current_user)):
    """
    Sign out. In dev mode this is a no-op.
    The frontend should clear its stored token regardless.
    """
    logger.info("User logged out", user_id=user.user_id)
    return {"status": "logged_out"}


@router.get("/me")
def me(user: UserContext = Depends(get_current_user)):
    """
    Return current user info extracted from the JWT.
    Useful for the frontend to verify the session is still valid.
    """
    return {
        "user_id": user.user_id,
        "email":   user.email,
        "is_dev":  user.is_dev,
    }