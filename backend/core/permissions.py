"""
core/permissions.py
RBAC permission checks for DocIntel.

All permission checks go through this module — never inline in routers.

Permission model:
  Developer   → can create orgs (DEVELOPER_API_KEY)
  Org Admin   → can manage teams, members, quotas, org-level API keys
  Team Lead   → can manage team members, see team usage
  Member      → can upload, query, extract their own documents

Document visibility:
  private → owner only
  team    → team members + org admin (if can_read_team_documents)
  org     → all active org members
"""

from fastapi import HTTPException, status
from core.auth import UserContext


# ---------------------------------------------------------------------------
# Role checks — raise 403 if check fails
# ---------------------------------------------------------------------------

def require_org_admin(user: UserContext) -> None:
    """Raise 403 if user is not an org admin."""
    if user.is_dev:
        return  # dev mode has full access
    if not user.is_org_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires org admin privileges.",
        )


def require_team_lead(user: UserContext) -> None:
    """Raise 403 if user is not a team lead or org admin."""
    if user.is_dev:
        return
    if not (user.is_team_lead or user.is_org_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires team lead or org admin privileges.",
        )


def require_org_member(user: UserContext) -> None:
    """Raise 403 if user has no org membership."""
    if user.is_dev:
        return
    if not user.has_org:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires org membership.",
        )


def require_same_org(user: UserContext, target_org_id: str) -> None:
    """Raise 403 if user is not in the target org."""
    if user.is_dev:
        return
    if str(user.org_id) != str(target_org_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied — resource belongs to a different org.",
        )


# ---------------------------------------------------------------------------
# Document access checks
# ---------------------------------------------------------------------------

def can_access_document(user: UserContext, document: dict) -> bool:
    """
    Check if a user can read a document based on visibility rules.

    Args:
        user:     Resolved UserContext for the requesting user.
        document: Document row dict with user_id, org_id, team_id, visibility.

    Returns True if access is allowed, False otherwise.

    This is a defense-in-depth check on top of RLS — RLS handles DB-level
    enforcement, this handles business-logic enforcement at the router level.
    """
    if user.is_dev:
        return True

    visibility = document.get("visibility", "private")
    doc_owner  = document.get("user_id")
    doc_org_id = document.get("org_id")
    doc_team_id= document.get("team_id")

    # Owner always has access
    if doc_owner == user.user_id:
        return True

    if visibility == "private":
        return False

    if visibility == "team":
        # Must be in the same team
        if doc_team_id and str(user.team_id) == str(doc_team_id):
            return True
        # Org admin with can_read_team_documents
        if user.is_org_admin and user.can_read_team_documents:
            if doc_org_id and str(user.org_id) == str(doc_org_id):
                return True
        return False

    if visibility == "org":
        # Any active member of the same org
        if doc_org_id and str(user.org_id) == str(doc_org_id):
            return True
        return False

    return False


def assert_document_access(user: UserContext, document: dict) -> None:
    """Raise 403 if user cannot access the document."""
    if not can_access_document(user, document):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied — you do not have permission to access this document.",
        )


# ---------------------------------------------------------------------------
# Org isolation assertion — defense-in-depth
# ---------------------------------------------------------------------------

def assert_org_isolation(user: UserContext, resource_org_id: str | None) -> None:
    """
    Assert that a resource belongs to the same org as the requesting user.

    Called as a second layer of protection after RLS — if RLS fails or is
    bypassed, this ensures org data never leaks across org boundaries.

    Non-blocking for personal resources (resource_org_id=None) —
    those are user-scoped and handled by user_id checks elsewhere.
    """
    if user.is_dev:
        return
    if resource_org_id is None:
        return  # personal resource, not org-scoped
    if user.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied — resource is org-scoped but you have no org membership.",
        )
    if str(user.org_id) != str(resource_org_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied — org isolation violation.",
        )


# ---------------------------------------------------------------------------
# Team access checks
# ---------------------------------------------------------------------------

def can_access_team(user: UserContext, team_id: str) -> bool:
    """Check if user can access team data (usage, members etc.)."""
    if user.is_dev:
        return True
    if user.is_org_admin:
        return True
    if user.is_team_lead and str(user.team_id) == str(team_id):
        return True
    return False


def assert_team_access(user: UserContext, team_id: str) -> None:
    if not can_access_team(user, team_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied — you do not have access to this team.",
        )