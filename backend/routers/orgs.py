"""
routers/orgs.py
Org and team management endpoints.

All endpoints require JWT auth (get_current_user_context).
Role enforcement via core/permissions.py.

Endpoints:
    GET    /orgs/me                              — my org + team info
    GET    /orgs/{org_id}/members               — list org members (org admin)
    POST   /orgs/{org_id}/members               — add member (org admin)
    DELETE /orgs/{org_id}/members/{user_id}     — remove member (org admin)
    PATCH  /orgs/{org_id}/members/{user_id}     — update role/permissions (org admin)

    GET    /orgs/{org_id}/teams                 — list teams (org member)
    POST   /orgs/{org_id}/teams                 — create team (org admin)
    DELETE /orgs/{org_id}/teams/{team_id}       — delete team (org admin)

    GET    /orgs/{org_id}/teams/{team_id}/members        — list team members
    POST   /orgs/{org_id}/teams/{team_id}/members        — add team member (team lead / org admin)
    DELETE /orgs/{org_id}/teams/{team_id}/members/{uid}  — remove team member
    PATCH  /orgs/{org_id}/teams/{team_id}/members/{uid}  — update team role

    GET    /orgs/{org_id}/audit                 — audit log (org admin)
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, validator

from core.auth import get_current_user_context, UserContext
from core.permissions import (
    require_org_admin, require_team_lead,
    require_same_org, assert_org_isolation,
)
from core.logger import get_logger
from db_orgs import (
    get_org_by_id, list_teams_for_org, create_team, delete_team, get_team_by_id,
    list_org_members, add_org_member, remove_org_member,
    update_org_member_role, update_org_member_permissions, get_org_member,
    suspend_org_member, reactivate_org_member,
    list_team_members, add_team_member, remove_team_member,
    update_team_member_role, get_team_member,
)
from db_audit import log_audit, get_audit_logs

logger = get_logger("routers.orgs")
router = APIRouter(prefix="/orgs", tags=["Orgs"])


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class AddMemberRequest(BaseModel):
    user_id: str
    role:    str = "member"

    @validator("role")
    def role_valid(cls, v):
        if v not in ("org_admin", "member"):
            raise ValueError("role must be 'org_admin' or 'member'")
        return v


class UpdateMemberRequest(BaseModel):
    role:                    str | None = None
    can_read_team_documents: bool | None = None
    can_read_all_usage:      bool | None = None

    @validator("role")
    def role_valid(cls, v):
        if v is not None and v not in ("org_admin", "member"):
            raise ValueError("role must be 'org_admin' or 'member'")
        return v


class CreateTeamRequest(BaseModel):
    name: str

    @validator("name")
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()


class AddTeamMemberRequest(BaseModel):
    user_id: str
    role:    str = "member"

    @validator("role")
    def role_valid(cls, v):
        if v not in ("team_lead", "member"):
            raise ValueError("role must be 'team_lead' or 'member'")
        return v


class UpdateTeamMemberRequest(BaseModel):
    role: str

    @validator("role")
    def role_valid(cls, v):
        if v not in ("team_lead", "member"):
            raise ValueError("role must be 'team_lead' or 'member'")
        return v


# ---------------------------------------------------------------------------
# My org info
# ---------------------------------------------------------------------------

@router.get("/me")
def get_my_org(user: UserContext = Depends(get_current_user_context)):
    """Return the current user's org + team membership info."""
    if not user.has_org:
        return {"org": None, "team": None, "role": None, "message": "Not a member of any org"}

    org = get_org_by_id(user.org_id)
    team = get_team_by_id(user.team_id) if user.team_id else None

    return {
        "org":       org,
        "team":      team,
        "org_role":  user.org_role,
        "team_role": user.team_role,
        "permissions": {
            "can_read_team_documents": user.can_read_team_documents,
            "can_read_all_usage":      user.can_read_all_usage,
        },
    }


# ---------------------------------------------------------------------------
# Org members
# ---------------------------------------------------------------------------

@router.get("/{org_id}/members")
def list_members(
    org_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)
    return list_org_members(org_id)


@router.post("/{org_id}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    org_id: str,
    req: AddMemberRequest,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    existing = get_org_member(org_id, req.user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this org.",
        )

    member = add_org_member(org_id=org_id, user_id=req.user_id, role=req.role)

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="member_added",
        resource_type="org_member",
        resource_id=req.user_id,
        org_id=org_id,
        details={"role": req.role},
    )

    return member


@router.delete("/{org_id}/members/{target_user_id}", status_code=status.HTTP_200_OK)
def remove_member(
    org_id: str,
    target_user_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    if target_user_id == user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself from the org.",
        )

    remove_org_member(org_id=org_id, user_id=target_user_id)

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="member_removed",
        resource_type="org_member",
        resource_id=target_user_id,
        org_id=org_id,
    )

    return {"status": "removed", "user_id": target_user_id}


@router.patch("/{org_id}/members/{target_user_id}")
def update_member(
    org_id: str,
    target_user_id: str,
    req: UpdateMemberRequest,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    if req.role is not None:
        update_org_member_role(org_id=org_id, user_id=target_user_id, role=req.role)

    if req.can_read_team_documents is not None or req.can_read_all_usage is not None:
        update_org_member_permissions(
            org_id=org_id,
            user_id=target_user_id,
            can_read_team_documents=req.can_read_team_documents,
            can_read_all_usage=req.can_read_all_usage,
        )

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="permissions_updated",
        resource_type="org_member",
        resource_id=target_user_id,
        org_id=org_id,
        details=req.model_dump(exclude_none=True),
    )

    return {"status": "updated", "user_id": target_user_id}


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

@router.get("/{org_id}/teams")
def list_teams(
    org_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_same_org(user, org_id)
    return list_teams_for_org(org_id)


@router.post("/{org_id}/teams", status_code=status.HTTP_201_CREATED)
def create_team_endpoint(
    org_id: str,
    req: CreateTeamRequest,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    try:
        team = create_team(org_id=org_id, name=req.name, created_by=user.user_id)
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A team named '{req.name}' already exists in this org.",
            )
        raise HTTPException(status_code=500, detail=str(e))

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="team_created",
        resource_type="team",
        resource_id=team["id"],
        org_id=org_id,
        details={"name": req.name},
    )

    return team


@router.delete("/{org_id}/teams/{team_id}")
def delete_team_endpoint(
    org_id: str,
    team_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    team = get_team_by_id(team_id)
    if not team or str(team["org_id"]) != str(org_id):
        raise HTTPException(status_code=404, detail="Team not found in this org.")

    delete_team(team_id)

    log_audit(
        actor_id=user.user_id,
        actor_role=user.org_role or "org_admin",
        action="team_deleted",
        resource_type="team",
        resource_id=team_id,
        org_id=org_id,
        details={"name": team.get("name", "")},
    )

    return {"status": "deleted", "team_id": team_id}


# ---------------------------------------------------------------------------
# Team members
# ---------------------------------------------------------------------------

@router.get("/{org_id}/teams/{team_id}/members")
def list_team_members_endpoint(
    org_id: str,
    team_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_same_org(user, org_id)
    return list_team_members(team_id)


@router.post("/{org_id}/teams/{team_id}/members", status_code=status.HTTP_201_CREATED)
def add_team_member_endpoint(
    org_id: str,
    team_id: str,
    req: AddTeamMemberRequest,
    user: UserContext = Depends(get_current_user_context),
):
    require_team_lead(user)
    require_same_org(user, org_id)

    existing = get_team_member(team_id, req.user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this team.",
        )

    member = add_team_member(
        team_id=team_id, org_id=org_id,
        user_id=req.user_id, role=req.role,
    )

    log_audit(
        actor_id=user.user_id,
        actor_role=user.team_role or user.org_role or "team_lead",
        action="team_member_added",
        resource_type="team_member",
        resource_id=req.user_id,
        org_id=org_id,
        details={"team_id": team_id, "role": req.role},
    )

    return member


@router.delete("/{org_id}/teams/{team_id}/members/{target_user_id}")
def remove_team_member_endpoint(
    org_id: str,
    team_id: str,
    target_user_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    require_team_lead(user)
    require_same_org(user, org_id)

    remove_team_member(team_id=team_id, user_id=target_user_id)

    log_audit(
        actor_id=user.user_id,
        actor_role=user.team_role or user.org_role or "team_lead",
        action="team_member_removed",
        resource_type="team_member",
        resource_id=target_user_id,
        org_id=org_id,
        details={"team_id": team_id},
    )

    return {"status": "removed", "user_id": target_user_id}


@router.patch("/{org_id}/teams/{team_id}/members/{target_user_id}")
def update_team_member_endpoint(
    org_id: str,
    team_id: str,
    target_user_id: str,
    req: UpdateTeamMemberRequest,
    user: UserContext = Depends(get_current_user_context),
):
    require_team_lead(user)
    require_same_org(user, org_id)

    updated = update_team_member_role(
        team_id=team_id, user_id=target_user_id, role=req.role,
    )

    log_audit(
        actor_id=user.user_id,
        actor_role=user.team_role or user.org_role or "team_lead",
        action="team_member_role_updated",
        resource_type="team_member",
        resource_id=target_user_id,
        org_id=org_id,
        details={"team_id": team_id, "new_role": req.role},
    )

    return {"status": "updated", "user_id": target_user_id, "role": req.role}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@router.get("/{org_id}/audit")
def get_org_audit(
    org_id: str,
    limit:  int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    action: str | None = Query(default=None),
    actor:  str | None = Query(default=None),
    user: UserContext = Depends(get_current_user_context),
):
    require_org_admin(user)
    require_same_org(user, org_id)

    logs = get_audit_logs(
        org_id=org_id,
        limit=limit,
        offset=offset,
        action_filter=action,
        actor_id_filter=actor,
    )
    return {"org_id": org_id, "count": len(logs), "logs": logs}