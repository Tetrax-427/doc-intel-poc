"""
Authentication endpoints — login, signup, logout, me, refresh, password reset.

Additions:
  POST /auth/refresh          — exchange a refresh token for a new session
  POST /auth/forgot-password   — send password reset email
  POST /auth/reset-password    — apply new password from reset token

F3 note: CORS is locked in main.py via CORS_ALLOWED_ORIGINS env var.
         These endpoints themselves need no CORS-specific code.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, validator
from core.auth import get_current_user, UserContext
from core.logger import get_logger
from supabase import create_client

logger = get_logger("auth")
router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Supabase client helpers ───────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str


class SignupRequest(BaseModel):
    email:    str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str

    @validator("refresh_token")
    def token_not_empty(cls, v):
        if not v.strip():
            raise ValueError("refresh_token cannot be empty")
        return v.strip()


class PasswordResetRequestBody(BaseModel):
    email: str

    @validator("email")
    def email_not_empty(cls, v):
        if not v.strip():
            raise ValueError("email cannot be empty")
        return v.strip().lower()


class PasswordResetApply(BaseModel):
    access_token: str
    new_password: str

    @validator("new_password")
    def password_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


def _get_supabase():
    """Supabase client using anon key — for login / user-facing auth."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    return create_client(url, key)


def _get_supabase_admin():
    """
    Supabase admin client using service role key.
    Used for signup (auto-confirm) and password updates.
    """
    url         = os.getenv("SUPABASE_URL", "")
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not service_key:
        return None
    return create_client(url, service_key)



def _supabase_with_token(access_token: str):
    """
    Supabase client authenticated with a user's access token.
    Used for password reset — allows calling update_user() as that user
    rather than as the admin, which is what Supabase expects for the
    reset-password flow.
    """
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    client = create_client(url, key)
    client.auth.set_session(access_token=access_token, refresh_token="")
    return client

# ── Existing endpoints ────────────────────────────────────────────────────────

@router.post("/login")
def login(req: LoginRequest):
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        logger.info("Auth in dev mode — returning dev_user token")
        return {
            "access_token":  "dev_token",
            "refresh_token": "dev_refresh_token",
            "token_type":    "bearer",
            "user":          {"id": "dev_user", "email": req.email or "dev@local"},
        }
    try:
        sb  = _get_supabase()
        res = sb.auth.sign_in_with_password({
            "email":    req.email.strip(),
            "password": req.password,
        })
        logger.info("User logged in", email=req.email)
        return {
            "access_token":  res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "token_type":    "bearer",
            "user":          {"id": res.user.id, "email": res.user.email},
        }
    except Exception as e:
        logger.warning("Login failed", email=req.email, error=str(e))
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@router.post("/signup")
def signup(req: SignupRequest):
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        return {
            "access_token":  "dev_token",
            "refresh_token": "dev_refresh_token",
            "token_type":    "bearer",
            "user":          {"id": "dev_user", "email": req.email or "dev@local"},
        }

    if len(req.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )

    email = req.email.strip()

    try:
        admin_sb = _get_supabase_admin()

        if admin_sb:
            res = admin_sb.auth.admin.create_user({
                "email":          email,
                "password":       req.password,
                "email_confirm":  True,
            })
            if not res.user:
                raise HTTPException(status_code=400, detail="Signup failed.")
            logger.info("New user created (admin, auto-confirmed)", email=email)
        else:
            logger.warning("SUPABASE_SERVICE_KEY not set — falling back to regular signup")
            sb  = _get_supabase()
            res = sb.auth.sign_up({"email": email, "password": req.password})
            if not res.user:
                raise HTTPException(status_code=400, detail="Signup failed.")
            logger.info("New user signed up (email confirmation required)", email=email)

        sb        = _get_supabase()
        login_res = sb.auth.sign_in_with_password({
            "email":    email,
            "password": req.password,
        })
        return {
            "access_token":  login_res.session.access_token,
            "refresh_token": login_res.session.refresh_token,
            "token_type":    "bearer",
            "user":          {"id": login_res.user.id, "email": login_res.user.email},
        }

    except HTTPException:
        raise
    except Exception as e:
        err = str(e).lower()
        if "already registered" in err or "already exists" in err:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists.",
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


# ──────────────────────────────────────────────────────────

@router.post("/refresh")
def refresh_session(req: RefreshRequest):
    """
    F2 — Exchange a refresh token for a new access + refresh token pair.

    The frontend should call this proactively when the access token is
    within ~60 seconds of expiry (check the exp claim in the JWT).
    On 401 from any other endpoint, the frontend should call this once
    and retry; if this also returns 401, redirect to login.

    Dev mode: returns a fresh dev token without touching Supabase.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        return {
            "access_token":  "dev_token",
            "refresh_token": "dev_refresh_token",
            "token_type":    "bearer",
        }

    try:
        sb  = _get_supabase()
        res = sb.auth.refresh_session(req.refresh_token)

        if not res.session:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")

        logger.info("Session refreshed", user_id=res.user.id if res.user else "unknown")
        return {
            "access_token":  res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "token_type":    "bearer",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Session refresh failed", error=str(e))
        raise HTTPException(status_code=401, detail="Could not refresh session. Please log in again.")


@router.post("/forgot-password")
def request_password_reset(req: PasswordResetRequestBody):
    """
    F3 — Send a password reset email.

    Always returns 200 regardless of whether the email exists — this
    prevents email enumeration.  Supabase sends the reset link.

    The redirect_to URL must be listed in Supabase's allowed redirect URLs
    (Authentication → URL Configuration in the Supabase dashboard).
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        # Dev mode — just acknowledge
        logger.info("Dev mode password reset requested", email=req.email)
        return {"status": "reset_email_sent"}

    redirect_url = os.getenv("PASSWORD_RESET_REDIRECT_URL", "http://localhost:8501")

    try:
        sb = _get_supabase()
        sb.auth.reset_password_email(req.email, options={"redirect_to": redirect_url})
        logger.info("Password reset email sent", email=req.email)
    except Exception as e:
        # Log but don't surface — prevents email enumeration
        logger.warning("Password reset email failed (non-fatal)", email=req.email, error=str(e))

    return {"status": "reset_email_sent"}


@router.post("/reset-password")
def apply_password_reset(req: PasswordResetApply):
    """
    F3 — Apply a new password using the access token from the reset email link.

    Flow:
      1. User clicks reset link in email → Supabase redirects to frontend
         with access_token + refresh_token as URL hash params.
      2. Frontend extracts access_token from URL and POSTs here with the
         new password.
      3. This endpoint calls the admin API to update the password.

    Requires SUPABASE_SERVICE_KEY (admin client).
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        return {"status": "password_updated"}

    admin_sb = _get_supabase_admin()
    if not admin_sb:
        raise HTTPException(
            status_code=503,
            detail="Password reset requires SUPABASE_SERVICE_KEY to be configured.",
        )

    try:
        # Authenticate as the user using their reset access token,
        # then call update_user() — this is the Supabase-recommended flow
        client = _supabase_with_token(req.access_token)
        client.auth.update_user({"password": req.new_password})

        logger.info("Password reset applied")
        return {"status": "password_updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password reset failed", error=str(e))
        raise HTTPException(status_code=400, detail="Could not reset password. The link may have expired.")