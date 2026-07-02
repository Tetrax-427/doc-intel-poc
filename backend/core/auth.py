"""
core/auth.py
Supabase JWT authentication for DocIntel.

Changes in this phase (Security + Org/Team):
  - UserContext extended with org_id, team_id, role, org_role, permissions
  - get_current_user_context() resolves org/team membership on every request
  - Dev mode returns a full UserContext with safe defaults
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import create_client

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------

@dataclass
class UserContext:
    # Core identity (always set)
    user_id: str
    email:   Optional[str] = None
    is_dev:  bool = False

    # Org membership (None if user has no org)
    org_id:   Optional[str] = None   # UUID as string
    org_role: Optional[str] = None   # 'org_admin' | 'member'

    # Team membership (None if user is not in any team)
    team_id:   Optional[str] = None  # UUID as string
    team_role: Optional[str] = None  # 'team_lead' | 'member'

    # Permission toggles (resolved from org_members row)
    can_read_team_documents: bool = False
    can_read_all_usage:      bool = True

    # Computed helpers
    @property
    def is_org_admin(self) -> bool:
        return self.org_role == "org_admin"

    @property
    def is_team_lead(self) -> bool:
        return self.team_role == "team_lead"

    @property
    def has_org(self) -> bool:
        return self.org_id is not None

    @property
    def org_id_str(self) -> str | None:
        return str(self.org_id) if self.org_id else None

    @property
    def team_id_str(self) -> str | None:
        return str(self.team_id) if self.team_id else None
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_user_id(user) -> str:
    if user is None:
        return "anonymous"
    if isinstance(user, UserContext):
        return user.user_id or "anonymous"
    if isinstance(user, dict):
        return user.get("user_id", "anonymous")
    return "anonymous"



def _verify_supabase_token(token: str) -> UserContext:
    """
    Validate token by calling Supabase.
    Returns a base UserContext — org/team fields populated separately
    by _resolve_org_context().
    """
    url         = os.getenv("SUPABASE_URL", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SUPABASE_URL or SUPABASE_SERVICE_KEY not configured.",
        )

    try:
        sb       = create_client(url, service_key)
        response = sb.auth.get_user(token)
        user     = response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return UserContext(user_id=user.id, email=user.email, is_dev=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


def _resolve_org_context(user_id: str) -> dict:
    """
    Resolve org/team membership for a user from the database.

    Returns a dict with org_id, org_role, team_id, team_role,
    can_read_team_documents, can_read_all_usage.

    Strategy:
    - Picks the first active org membership (alphabetical by org_id for
      determinism). Org switching is handled by the frontend passing an
      explicit org_id header in future — for now single-org is the norm.
    - Picks the first team membership within that org.
    - Returns empty defaults if user has no org.

    Non-blocking: any DB error returns empty defaults so auth never fails
    due to an org lookup error.
    """
    defaults = {
        "org_id":                   None,
        "org_role":                 None,
        "team_id":                  None,
        "team_role":                None,
        "can_read_team_documents":  False,
        "can_read_all_usage":       True,
    }

    try:
        url         = os.getenv("SUPABASE_URL", "").strip()
        service_key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
        if not url or not service_key:
            return defaults

        sb = create_client(url, service_key)

        # Fetch first active org membership
        org_resp = (
            sb.table("org_members")
            .select("org_id, role, can_read_team_documents, can_read_all_usage")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("org_id")
            .limit(1)
            .execute()
        )

        if not org_resp.data:
            return defaults

        org_row = org_resp.data[0]
        org_id  = org_row["org_id"]

        # Fetch first team membership within this org
        team_resp = (
            sb.table("team_members")
            .select("team_id, role")
            .eq("user_id", user_id)
            .eq("org_id", org_id)
            .order("team_id")
            .limit(1)
            .execute()
        )

        team_id   = None
        team_role = None
        if team_resp.data:
            team_id   = team_resp.data[0]["team_id"]
            team_role = team_resp.data[0]["role"]

        return {
            "org_id":                  org_id,
            "org_role":                org_row["role"],
            "team_id":                 team_id,
            "team_role":               team_role,
            "can_read_team_documents": org_row.get("can_read_team_documents", False),
            "can_read_all_usage":      org_row.get("can_read_all_usage", True),
        }

    except Exception:
        return defaults


def _build_user_context(base: UserContext) -> UserContext:
    """Enrich a base UserContext with org/team fields."""
    if base.is_dev:
        # Dev mode — return with safe defaults, no DB lookup
        return UserContext(
            user_id="dev_user",
            email="dev@local",
            is_dev=True,
            org_id=None,
            org_role="org_admin",   # dev gets full permissions
            team_id=None,
            team_role=None,
            can_read_team_documents=True,
            can_read_all_usage=True,
        )

    ctx = _resolve_org_context(base.user_id)
    return UserContext(
        user_id=base.user_id,
        email=base.email,
        is_dev=False,
        org_id=ctx["org_id"],
        org_role=ctx["org_role"],
        team_id=ctx["team_id"],
        team_role=ctx["team_role"],
        can_read_team_documents=ctx["can_read_team_documents"],
        can_read_all_usage=ctx["can_read_all_usage"],
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------



def get_current_user_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserContext:
    """
    Primary auth dependency for this phase.
    Validates JWT, resolves org/team membership, returns full UserContext.

    Dev mode (no SUPABASE_JWT_SECRET): returns dev UserContext with
    org_admin permissions — no DB lookup, no token required.
    """
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()

    if not jwt_secret:
        return _build_user_context(
            UserContext(user_id="dev_user", is_dev=True)
        )

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    base = _verify_supabase_token(credentials.credentials)
    return _build_user_context(base)