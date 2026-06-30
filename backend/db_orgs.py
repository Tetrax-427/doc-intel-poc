"""
db_orgs.py
Database helpers for orgs, teams, org members, and team members.

All writes use the service-role client (bypasses RLS) since org
management is performed by trusted server-side code after permission
checks in the router layer.

Reads use the anon client so RLS is enforced for user-facing queries.
"""

from db import get_supabase_admin




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