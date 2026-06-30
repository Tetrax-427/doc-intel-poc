"""
routers/auth.py
Authentication endpoints.

Changes in this phase:
  - Rate limiting on /login and /signup
  - Email verification check on /login (when REQUIRE_EMAIL_VERIFICATION=true)
  - DELETE /auth/account — hard delete all user data + Supabase auth user

Existing endpoints unchanged in behavior:
  POST /auth/login, /auth/signup, /auth/logout, /auth/me,
  /auth/refresh, /auth/forgot-password, /auth/reset-password
"""

import os
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, validator
from core.auth import get_current_user_context, get_current_user, UserContext
from core.rate_limiter import check_rate_limit
from core.config import config as app_config
from core.logger import get_logger
from db_audit import log_audit
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
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    return create_client(url, key)


def _get_supabase_admin():
    url         = os.getenv("SUPABASE_URL", "")
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not service_key:
        return None
    return create_client(url, service_key)


def _supabase_with_token(access_token: str):
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
    client = create_client(url, key)
    client.auth.set_session(access_token=access_token, refresh_token="")
    return client


# ── Existing endpoints ────────────────────────────────────────────────────────

@router.post("/login")
def login(req: LoginRequest, request: Request):
    # Rate limit by IP for unauthenticated endpoint
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(user_id=client_ip, endpoint="login")

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

        # Email verification check
        if app_config.require_email_verification:
            user = res.user
            if user and not user.email_confirmed_at:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Email not verified. Please check your inbox and verify your email before logging in.",
                )

        logger.info("User logged in", email=req.email)
        return {
            "access_token":  res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "token_type":    "bearer",
            "user":          {"id": res.user.id, "email": res.user.email},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Login failed", email=req.email, error=str(e))
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@router.post("/signup")
def signup(req: SignupRequest, request: Request):
    # Rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(user_id=client_ip, endpoint="signup")

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
                "email":         email,
                "password":      req.password,
                "email_confirm": True,
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
def me(user: UserContext = Depends(get_current_user_context)):
    return {
        "user_id":   user.user_id,
        "email":     user.email,
        "is_dev":    user.is_dev,
        "org_id":    user.org_id,
        "org_role":  user.org_role,
        "team_id":   user.team_id,
        "team_role": user.team_role,
    }


@router.post("/refresh")
def refresh_session(req: RefreshRequest):
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
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not jwt_secret:
        return {"status": "reset_email_sent"}

    redirect_url = os.getenv("PASSWORD_RESET_REDIRECT_URL", "http://localhost:8501")

    try:
        sb = _get_supabase()
        sb.auth.reset_password_email(req.email, options={"redirect_to": redirect_url})
        logger.info("Password reset email sent", email=req.email)
    except Exception as e:
        logger.warning("Password reset email failed (non-fatal)", email=req.email, error=str(e))

    return {"status": "reset_email_sent"}


@router.post("/reset-password")
def apply_password_reset(req: PasswordResetApply):
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
        client = _supabase_with_token(req.access_token)
        client.auth.update_user({"password": req.new_password})
        logger.info("Password reset applied")
        return {"status": "password_updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password reset failed", error=str(e))
        raise HTTPException(status_code=400, detail="Could not reset password. The link may have expired.")


# ── Account deletion ──────────────────────────────────────────────────────────

@router.delete("/account")
def delete_account(user: UserContext = Depends(get_current_user_context)):
    """
    Hard delete all user data and the Supabase auth account.

    Cascade order (respects FK dependencies):
      1. chunks               (document_id FK)
      2. extraction_results   (document_id FK)
      3. lineage_logs         (document_id + user_id)
      4. llm_cache            (user_id + document_id)
      5. llm_calls            (user_id)
      6. documents            (user_id)
      7. api_keys             (user_id)
      8. review_corrections   (via document_id already deleted — cleanup)
      9. team_members         (user_id)
      10. org_members         (user_id)
      11. audit_logs          (actor_id — optional, keep for compliance)
      12. Supabase auth user  (hard delete via admin API)

    Non-reversible. Returns 200 on success.
    Dev mode: clears data but skips Supabase auth deletion.
    """
    uid = user.user_id

    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    admin_sb   = _get_supabase_admin()

    if not admin_sb and jwt_secret:
        raise HTTPException(
            status_code=503,
            detail="Account deletion requires SUPABASE_SERVICE_KEY.",
        )

    try:
        sb = admin_sb or _get_supabase()

        # 1. Get all user's document IDs for cascade
        doc_resp = sb.table("documents").select("id").eq("user_id", uid).execute()
        doc_ids  = [d["id"] for d in (doc_resp.data or [])]

        # 2. Delete chunks for all documents
        if doc_ids:
            sb.table("chunks").delete().in_("document_id", doc_ids).execute()

        # 3. Delete extraction_results
        if doc_ids:
            sb.table("extraction_results").delete().in_("document_id", doc_ids).execute()

        # 4. Delete lineage_logs by user_id (covers all events)
        sb.table("lineage_logs").delete().eq("user_id", uid).execute()

        # 5. Delete llm_cache by user_id
        sb.table("llm_cache").delete().eq("user_id", uid).execute()

        # 6. Delete llm_calls by user_id
        sb.table("llm_calls").delete().eq("user_id", uid).execute()

        # 7. Delete documents
        sb.table("documents").delete().eq("user_id", uid).execute()

        # 8. Delete api_keys
        sb.table("api_keys").delete().eq("user_id", uid).execute()

        # 9. Delete review_corrections
        sb.table("review_corrections").delete().eq("user_id", uid).execute()

        # 10. Delete team memberships
        sb.table("team_members").delete().eq("user_id", uid).execute()

        # 11. Delete org memberships
        sb.table("org_members").delete().eq("user_id", uid).execute()

        # 12. Hard delete Supabase auth user
        if jwt_secret and admin_sb:
            admin_sb.auth.admin.delete_user(uid)
            logger.info("Auth user hard deleted", user_id=uid)

        log_audit(
            actor_id=uid,
            actor_role=user.org_role or "member",
            action="account_deleted",
            resource_type="user",
            resource_id=uid,
            org_id=str(user.org_id) if user.org_id else None,
            details={"document_count": len(doc_ids)},
        )

        logger.info("Account deletion complete", user_id=uid, docs_deleted=len(doc_ids))
        return {
            "status":          "deleted",
            "user_id":         uid,
            "documents_deleted": len(doc_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Account deletion failed", user_id=uid, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Account deletion failed: {e}",
        )