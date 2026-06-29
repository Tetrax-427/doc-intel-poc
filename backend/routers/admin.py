"""
routers/admin.py
Developer-only admin endpoints.

Protected by DEVELOPER_API_KEY header — not user JWT.
Only the developer (you) can call these endpoints.

Endpoints:
    POST /admin/orgs        — create a new org + enroll first admin
    GET  /admin/orgs        — list all orgs (developer overview)
"""

import os
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, validator

from core.config import config as app_config
from core.logger import get_logger
from db_orgs import create_org, list_orgs, get_org_by_slug
from db_audit import log_audit

logger = get_logger("routers.admin")
router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Developer key auth
# ---------------------------------------------------------------------------

def _verify_developer_key(x_developer_key: str | None) -> None:
    """
    Verify the X-Developer-Key header matches DEVELOPER_API_KEY in config.
    Raises HTTP 401 if missing or wrong.
    """
    expected = app_config.developer_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Developer API key not configured on server.",
        )
    if not x_developer_key or x_developer_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Developer-Key header.",
        )


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class CreateOrgRequest(BaseModel):
    name:       str
    slug:       str
    admin_user_id: str   # Supabase user_id of the first org admin

    @validator("slug")
    def slug_valid(cls, v):
        import re
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9][a-z0-9\-]{1,48}[a-z0-9]$", v):
            raise ValueError(
                "slug must be 3-50 chars, lowercase letters/numbers/hyphens, "
                "start and end with letter or number"
            )
        return v

    @validator("name")
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    @validator("admin_user_id")
    def admin_user_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("admin_user_id cannot be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/orgs", status_code=status.HTTP_201_CREATED)
def create_org_endpoint(
    req: CreateOrgRequest,
    x_developer_key: str | None = Header(default=None),
):
    """
    Create a new org and enroll the first org admin.

    Protected by X-Developer-Key header — developer use only.
    Call this when onboarding a new customer.

    The admin_user_id must be a valid Supabase auth user — the user
    must have already signed up before you can make them an org admin.
    """
    _verify_developer_key(x_developer_key)

    # Check slug uniqueness
    existing = get_org_by_slug(req.slug)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Org with slug '{req.slug}' already exists.",
        )

    try:
        org = create_org(
            name=req.name,
            slug=req.slug,
            created_by=req.admin_user_id,
        )
    except Exception as e:
        logger.error("Org creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create org: {e}",
        )

    log_audit(
        actor_id=req.admin_user_id,
        actor_role="developer",
        action="org_created",
        resource_type="org",
        resource_id=org["id"],
        org_id=org["id"],
        details={"name": req.name, "slug": req.slug},
    )

    logger.info("Org created", org_id=org["id"], slug=req.slug, admin=req.admin_user_id)

    return {
        "org_id":         org["id"],
        "name":           org["name"],
        "slug":           org["slug"],
        "admin_user_id":  req.admin_user_id,
        "created_at":     org["created_at"],
        "message":        f"Org '{req.name}' created. Admin enrolled.",
    }


@router.get("/orgs")
def list_orgs_endpoint(
    x_developer_key: str | None = Header(default=None),
):
    """List all orgs. Developer overview only."""
    _verify_developer_key(x_developer_key)
    return list_orgs()