"""
db_quotas.py
Quota CRUD helpers for DocIntel.

All writes use the service-role client — quota management is
performed by trusted server-side code after permission checks
in the router layer (org admin only).

Usage:
    from db_quotas import set_quota, delete_quota, list_quotas_for_org
"""

from db import get_supabase_admin

VALID_QUOTA_TYPES = {
    "max_documents",
    "max_uploads_per_day",
    "max_llm_cost_month",
    "max_queries_per_day",
}




# ---------------------------------------------------------------------------
# Upsert (create or update)
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


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

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