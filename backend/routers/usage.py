"""
routers/usage.py
Usage and quota management endpoints.

Endpoints:
    GET  /usage/me                    — personal usage + quota status
    GET  /usage/team/{team_id}        — team usage (team lead + org admin)
    GET  /usage/org/{org_id}          — org usage (org admin only)
    POST /usage/quotas                — set a quota (org admin only)
    DELETE /usage/quotas/{quota_id}   — remove a quota (org admin only)
    GET  /usage/org/{org_id}/quotas   — list all quotas for an org (org admin)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, validator

from core.auth import get_current_user_context, UserContext
from core.permissions import require_org_admin, require_team_lead, require_same_org, assert_team_access
from core.quota_checker import get_quota_status
from core.logger import get_logger
from db_usage import get_user_usage, get_team_usage, get_org_usage, get_user_daily_breakdown
from db_quotas import set_quota, delete_quota, list_quotas_for_org, get_quota_by_id, VALID_QUOTA_TYPES
from db_audit import log_audit

logger = get_logger("routers.usage")
router = APIRouter(prefix="/usage", tags=["Usage"])


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class SetQuotaRequest(BaseModel):
    quota_type:    str
    limit_value:   float
    is_hard_limit: bool = True
    # Scope — exactly one must be set
    user_id:  str | None = None
    team_id:  str | None = None
    org_id:   str | None = None

    @validator("quota_type")
    def quota_type_valid(cls, v):
        if v not in VALID_QUOTA_TYPES:
            raise ValueError(
                f"quota_type must be one of: {', '.join(sorted(VALID_QUOTA_TYPES))}"
            )
        return v

    @validator("limit_value")
    def limit_positive(cls, v):
        if v <= 0:
            raise ValueError("limit_value must be positive")
        return v

    @validator("org_id", always=True)
    def exactly_one_scope(cls, org_id, values):
        scopes = [
            x for x in [values.get("user_id"), values.get("team_id"), org_id]
            if x is not None
        ]
        if len(scopes) != 1:
            raise ValueError("Exactly one of user_id, team_id, org_id must be provided.")
        return org_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/me")
def get_my_usage(user: UserContext = Depends(get_current_user_context)):
    """
    Return personal usage summary + quota status for the current user.
    Includes monthly LLM cost, daily uploads, daily queries, document count.
    """
    usage  = get_user_usage(user.user_id)
    quotas = get_quota_status(user)
    breakdown = get_user_daily_breakdown(user.user_id, days=30)

    return {
        "user_id":         user.user_id,
        "org_id":          user.org_id,
        "team_id":         user.team_id,
        "usage":           usage,
        "quota_status":    quotas,
        "daily_breakdown": breakdown,
    }


@router.get("/team/{team_id}")
def get_team_usage_endpoint(
    team_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Return usage summary for a team (current month).
    Accessible by: team lead of that team, org admin.
    """
    assert_team_access(user, team_id)

    usage = get_team_usage(team_id)
    return usage


@router.get("/org/{org_id}")
def get_org_usage_endpoint(
    org_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Return usage summary for an org (current month).
    Includes per-team and per-member breakdown.
    Org admin only.
    """
    require_org_admin(user)
    require_same_org(user, org_id)

    usage = get_org_usage(org_id)
    return usage


@router.get("/org/{org_id}/quotas")
def list_org_quotas(
    org_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """
    List all quotas configured for an org (org-level, team-level, user-level).
    Org admin only.
    """
    require_org_admin(user)
    require_same_org(user, org_id)

    quotas = list_quotas_for_org(org_id)
    return {"org_id": org_id, "quotas": quotas}


@router.post("/quotas", status_code=status.HTTP_201_CREATED)
def set_quota_endpoint(
    req: SetQuotaRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Create or update a quota. Org admin only.

    Quota resolution order at enforcement time:
      user-specific → team → org → system default

    Setting a quota with the same (scope, quota_type) as an existing one
    will update it (upsert behavior).
    """
    require_org_admin(user)

    # If org-scoped, verify it's the admin's own org
    if req.org_id:
        require_same_org(user, req.org_id)

    try:
        quota = set_quota(
            quota_type=req.quota_type,
            limit_value=req.limit_value,
            set_by=user.user_id,
            user_id=req.user_id,
            team_id=req.team_id,
            org_id=req.org_id,
            is_hard_limit=req.is_hard_limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="quota_set",
        resource_type="quota",
        resource_id=quota["id"],
        org_id=user.org_id,
        details={
            "quota_type":    req.quota_type,
            "limit_value":   req.limit_value,
            "is_hard_limit": req.is_hard_limit,
            "scope_user":    req.user_id,
            "scope_team":    req.team_id,
            "scope_org":     req.org_id,
        },
    )

    return quota


@router.delete("/quotas/{quota_id}")
def delete_quota_endpoint(
    quota_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    """Remove a quota. Org admin only."""
    require_org_admin(user)

    quota = get_quota_by_id(quota_id)
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found.")

    delete_quota(quota_id)

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="quota_removed",
        resource_type="quota",
        resource_id=quota_id,
        org_id=user.org_id,
        details={"quota_type": quota.get("quota_type")},
    )

    return {"status": "deleted", "quota_id": quota_id}