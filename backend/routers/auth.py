"""
Authentication endpoints — login, signup, logout, me.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from core.auth import get_current_user, UserContext
from core.logger import get_logger
from supabase import create_client
    
logger = get_logger("auth")
router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    email:    str
    password: str


class SignupRequest(BaseModel):
    email:    str
    password: str


def _get_supabase():
    """Supabase client using anon key — for login."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    return create_client(url, key)


def _get_supabase_admin():
    """
    Supabase admin client using service role key.
    Used for signup so we can auto-confirm emails without
    requiring users to click a verification link.
    """
    url         = os.getenv("SUPABASE_URL", "")
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not service_key:
        return None  # fall back to regular client if not configured
    return create_client(url, service_key)


@router.post("/login")
def login(req: LoginRequest):
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        logger.info("Auth in dev mode — returning dev_user token")
        return {
            "access_token": "dev_token",
            "token_type":   "bearer",
            "user": {"id": "dev_user", "email": req.email or "dev@local"}
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
            "user": {"id": res.user.id, "email": res.user.email}
        }
    except Exception as e:
        logger.warning("Login failed", email=req.email, error=str(e))
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@router.post("/signup")
def signup(req: SignupRequest):
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        return {
            "access_token": "dev_token",
            "token_type":   "bearer",
            "user": {"id": "dev_user", "email": req.email or "dev@local"}
        }

    if len(req.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters."
        )

    email = req.email.strip()

    try:
        # Use admin client to create user with email already confirmed
        admin_sb = _get_supabase_admin()

        if admin_sb:
            # Admin signup — auto-confirms email, no verification link needed
            res = admin_sb.auth.admin.create_user({
                "email":            email,
                "password":         req.password,
                "email_confirm":    True,   # ← skip email verification
            })

            if not res.user:
                raise HTTPException(status_code=400, detail="Signup failed.")

            logger.info("New user created (admin, auto-confirmed)", email=email)

        else:
            # Fallback — regular signup (will send verification email)
            logger.warning("SUPABASE_SERVICE_KEY not set — falling back to regular signup")
            sb  = _get_supabase()
            res = sb.auth.sign_up({"email": email, "password": req.password})
            if not res.user:
                raise HTTPException(status_code=400, detail="Signup failed.")
            logger.info("New user signed up (email confirmation required)", email=email)

        # Sign in immediately to get session token
        sb       = _get_supabase()
        login_res = sb.auth.sign_in_with_password({
            "email":    email,
            "password": req.password,
        })
        return {
            "access_token": login_res.session.access_token,
            "token_type":   "bearer",
            "user": {"id": login_res.user.id, "email": login_res.user.email}
        }

    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "already registered" in err or "already exists" in err:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists."
            )
        logger.error("Signup error", email=email, error=str(e))
        raise HTTPException(status_code=400, detail="Could not create account. Please try again.")


@router.post("/logout")
def logout(user: UserContext = Depends(get_current_user)):
    logger.info("User logged out", user_id=user.user_id)
    return {"status": "logged_out"}


@router.get("/me")
def me(user: UserContext = Depends(get_current_user)):
    return {
        "user_id": user.user_id,
        "email":   user.email,
        "is_dev":  user.is_dev,
    }