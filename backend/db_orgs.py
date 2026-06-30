"""
db_orgs.py
Database helpers for orgs, teams, org members, team members, and quotas.

quota management is small (6
functions) and lives in the same org/team domain, so it was folded in here
to reduce file sprawl.

All writes use the service-role client (bypasses RLS) since org/team/quota
management is performed by trusted server-side code after permission
checks in the router layer.
"""

from db import get_supabase_admin


VALID_QUOTA_TYPES = {
    "max_documents",
    "max_uploads_per_day",
    "max_llm_cost_month",
    "max_queries_per_day",
}


# ---------------------------------------------------------------------------
# Orgs
# ---------------------------------------------------------------------------

def create_org(name: str, slug: str, created_by: str) -> dict:
    """
    Create a new org. Developer-only — called from POST /admin/orgs.
    Also creates an org_member row for created_by as org_admin.
    """
    sb = get_supabase_admin()

    result = sb.table("orgs").insert({
        "name":       name,
        "slug":       slug,
        "created_by": created_by,
    }).execute()

    org = result.data[0]

    # Auto-enroll creator as org_admin
    sb.table("org_members").insert({
        "org_id":  org["id"],
        "user_id": created_by,
        "role":    "org_admin",
        "status":  "active",
    }).execute()

    return org


def get_org_by_id(org_id: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("orgs").select("*").eq("id", org_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def get_org_by_slug(slug: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("orgs").select("*").eq("slug", slug).limit(1).execute()
    return resp.data[0] if resp.data else None


def list_orgs() -> list[dict]:
    """List all orgs. Developer-only."""
    sb = get_supabase_admin()
    return sb.table("orgs").select("*").order("created_at", desc=True).execute().data or []


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def create_team(org_id: str, name: str, created_by: str) -> dict:
    sb     = get_supabase_admin()
    result = sb.table("teams").insert({
        "org_id":     org_id,
        "name":       name,
        "created_by": created_by,
    }).execute()
    return result.data[0]


def get_team_by_id(team_id: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("teams").select("*").eq("id", team_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def list_teams_for_org(org_id: str) -> list[dict]:
    sb = get_supabase_admin()
    return (
        sb.table("teams")
        .select("*")
        .eq("org_id", org_id)
        .order("name")
        .execute()
        .data or []
    )


def delete_team(team_id: str) -> None:
    """Delete a team and cascade removes team_members via FK."""
    get_supabase_admin().table("teams").delete().eq("id", team_id).execute()


# ---------------------------------------------------------------------------
# Org Members
# ---------------------------------------------------------------------------

def add_org_member(
    org_id: str,
    user_id: str,
    role: str = "member",
) -> dict:
    sb     = get_supabase_admin()
    result = sb.table("org_members").insert({
        "org_id":  org_id,
        "user_id": user_id,
        "role":    role,
        "status":  "active",
    }).execute()
    return result.data[0]


def remove_org_member(org_id: str, user_id: str) -> None:
    get_supabase_admin().table("org_members").delete()\
        .eq("org_id", org_id).eq("user_id", user_id).execute()


def update_org_member_role(org_id: str, user_id: str, role: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("org_members").update({"role": role})\
        .eq("org_id", org_id).eq("user_id", user_id).execute()
    return resp.data[0] if resp.data else None


def update_org_member_permissions(
    org_id: str,
    user_id: str,
    can_read_team_documents: bool | None = None,
    can_read_all_usage: bool | None = None,
) -> dict | None:
    sb      = get_supabase_admin()
    updates = {}
    if can_read_team_documents is not None:
        updates["can_read_team_documents"] = can_read_team_documents
    if can_read_all_usage is not None:
        updates["can_read_all_usage"] = can_read_all_usage
    if not updates:
        return None
    resp = sb.table("org_members").update(updates)\
        .eq("org_id", org_id).eq("user_id", user_id).execute()
    return resp.data[0] if resp.data else None


def list_org_members(org_id: str) -> list[dict]:
    sb = get_supabase_admin()
    return (
        sb.table("org_members")
        .select("*")
        .eq("org_id", org_id)
        .order("joined_at")
        .execute()
        .data or []
    )


def get_org_member(org_id: str, user_id: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("org_members").select("*")\
        .eq("org_id", org_id).eq("user_id", user_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def suspend_org_member(org_id: str, user_id: str) -> None:
    get_supabase_admin().table("org_members").update({"status": "suspended"})\
        .eq("org_id", org_id).eq("user_id", user_id).execute()


def reactivate_org_member(org_id: str, user_id: str) -> None:
    get_supabase_admin().table("org_members").update({"status": "active"})\
        .eq("org_id", org_id).eq("user_id", user_id).execute()


# ---------------------------------------------------------------------------
# Team Members
# ---------------------------------------------------------------------------

def add_team_member(
    team_id: str,
    org_id: str,
    user_id: str,
    role: str = "member",
) -> dict:
    sb     = get_supabase_admin()
    result = sb.table("team_members").insert({
        "team_id": team_id,
        "org_id":  org_id,
        "user_id": user_id,
        "role":    role,
    }).execute()
    return result.data[0]


def remove_team_member(team_id: str, user_id: str) -> None:
    get_supabase_admin().table("team_members").delete()\
        .eq("team_id", team_id).eq("user_id", user_id).execute()


def update_team_member_role(team_id: str, user_id: str, role: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("team_members").update({"role": role})\
        .eq("team_id", team_id).eq("user_id", user_id).execute()
    return resp.data[0] if resp.data else None


def list_team_members(team_id: str) -> list[dict]:
    sb = get_supabase_admin()
    return (
        sb.table("team_members")
        .select("*")
        .eq("team_id", team_id)
        .order("joined_at")
        .execute()
        .data or []
    )


def get_team_member(team_id: str, user_id: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("team_members").select("*")\
        .eq("team_id", team_id).eq("user_id", user_id).limit(1).execute()
    return resp.data[0] if resp.data else None


# ---------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------

def set_quota(
    quota_type: str,
    limit_value: float,
    set_by: str,
    user_id: str | None = None,
    team_id: str | None = None,
    org_id: str | None = None,
    is_hard_limit: bool = True,
) -> dict:
    """
    Create or update a quota. Exactly one of user_id, team_id, org_id must
    be provided. Uses upsert on (scope, quota_type) unique constraint.

    Args:
        quota_type:    One of VALID_QUOTA_TYPES.
        limit_value:   The limit (count or USD amount).
        set_by:        user_id of the org_admin setting this quota.
        user_id:       Set for user-scoped quota.
        team_id:       Set for team-scoped quota.
        org_id:        Set for org-scoped quota.
        is_hard_limit: True = block on exceed, False = warn only.

    Returns the created/updated quota row.
    Raises ValueError if quota_type is invalid or scope is ambiguous.
    """
    if quota_type not in VALID_QUOTA_TYPES:
        raise ValueError(
            f"Invalid quota_type '{quota_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_QUOTA_TYPES))}"
        )

    scopes = [x for x in [user_id, team_id, org_id] if x is not None]
    if len(scopes) != 1:
        raise ValueError("Exactly one of user_id, team_id, org_id must be provided.")

    sb = get_supabase_admin()

    # Build the row
    row: dict = {
        "quota_type":    quota_type,
        "limit_value":   limit_value,
        "set_by":        set_by,
        "is_hard_limit": is_hard_limit,
    }
    if user_id:
        row["user_id"] = user_id
    if team_id:
        row["team_id"] = team_id
    if org_id:
        row["org_id"] = org_id

    # Upsert — ON CONFLICT updates the existing row
    result = sb.table("quotas").upsert(row).execute()
    return result.data[0]


def delete_quota(quota_id: str) -> None:
    """Delete a quota by ID. Org admin only — enforced at router."""
    get_supabase_admin().table("quotas").delete().eq("id", quota_id).execute()


def delete_quota_for_user(user_id: str, quota_type: str) -> None:
    get_supabase_admin().table("quotas").delete()\
        .eq("user_id", user_id).eq("quota_type", quota_type).execute()


def delete_quota_for_team(team_id: str, quota_type: str) -> None:
    get_supabase_admin().table("quotas").delete()\
        .eq("team_id", team_id).eq("quota_type", quota_type).execute()


def delete_quota_for_org(org_id: str, quota_type: str) -> None:
    get_supabase_admin().table("quotas").delete()\
        .eq("org_id", org_id).eq("quota_type", quota_type).execute()


def list_quotas_for_org(org_id: str) -> list[dict]:
    """
    List all quotas scoped to an org: org-level + team-level + user-level
    quotas within the org. Used by org admin to see all quota configuration.
    """
    sb = get_supabase_admin()

    # Get all teams in org first
    teams_resp = sb.table("teams").select("id").eq("org_id", org_id).execute()
    team_ids   = [t["id"] for t in (teams_resp.data or [])]

    # Get all members in org
    members_resp = sb.table("org_members").select("user_id").eq("org_id", org_id).execute()
    user_ids     = [m["user_id"] for m in (members_resp.data or [])]

    all_quotas = []

    # Org-level quotas
    try:
        resp = sb.table("quotas").select("*").eq("org_id", org_id).execute()
        all_quotas.extend(resp.data or [])
    except Exception:
        pass

    # Team-level quotas within this org
    if team_ids:
        try:
            resp = sb.table("quotas").select("*").in_("team_id", team_ids).execute()
            all_quotas.extend(resp.data or [])
        except Exception:
            pass

    # User-level quotas for org members
    if user_ids:
        try:
            resp = sb.table("quotas").select("*").in_("user_id", user_ids).execute()
            all_quotas.extend(resp.data or [])
        except Exception:
            pass

    return all_quotas


def get_quota_by_id(quota_id: str) -> dict | None:
    sb   = get_supabase_admin()
    resp = sb.table("quotas").select("*").eq("id", quota_id).limit(1).execute()
    return resp.data[0] if resp.data else None