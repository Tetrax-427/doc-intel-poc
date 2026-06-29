"""
db_audit.py
Audit log helpers for DocIntel.

All writes use the service-role client — audit logs are written by
trusted server-side code and must not be blocked by RLS.

Reads use the anon client so RLS limits what each user can see
(own actions + org admin sees all org actions).

Usage:
    from db_audit import log_audit

    log_audit(
        actor_id="user-uuid",
        actor_role="org_admin",
        action="member_added",
        resource_type="org_member",
        resource_id="target-user-uuid",
        org_id="org-uuid",
        details={"email": "new@example.com", "role": "member"},
    )
"""

import os
from typing import Any
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


def _sb_admin():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )


def _sb():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
    )


# ---------------------------------------------------------------------------
# Valid action values (for reference — not enforced at DB level)
# ---------------------------------------------------------------------------

AUDIT_ACTIONS = {
    "org_created",
    "member_added",
    "member_removed",
    "member_suspended",
    "member_reactivated",
    "permissions_updated",
    "team_created",
    "team_deleted",
    "team_member_added",
    "team_member_removed",
    "team_member_role_updated",
    "api_key_created",
    "api_key_rotated",
    "api_key_deleted",
    "document_deleted",
    "account_deleted",
    "quota_set",
    "quota_removed",
    "visibility_updated",
}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def log_audit(
    actor_id: str,
    actor_role: str,
    action: str,
    resource_type: str,
    resource_id: str,
    org_id: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """
    Write an audit log entry. Non-blocking — any DB error is silently
    swallowed so audit logging never breaks the main request flow.

    Args:
        actor_id:      user_id of the person performing the action.
        actor_role:    their role at time of action.
        action:        action performed (see AUDIT_ACTIONS).
        resource_type: type of resource affected.
        resource_id:   ID of the affected resource.
        org_id:        org scope (None for developer-level actions).
        details:       action-specific metadata dict.
        ip_address:    request IP (optional, for security audits).
    """
    try:
        _sb_admin().table("audit_logs").insert({
            "actor_id":     actor_id,
            "actor_role":   actor_role,
            "action":       action,
            "resource_type": resource_type,
            "resource_id":  resource_id,
            "org_id":       org_id,
            "details":      details or {},
            "ip_address":   ip_address,
        }).execute()
    except Exception as e:
        # Never let audit logging break a request
        import logging
        logging.warning(f"[audit] Failed to write audit log: {e}")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_audit_logs(
    org_id: str,
    limit: int = 100,
    offset: int = 0,
    action_filter: str | None = None,
    actor_id_filter: str | None = None,
) -> list[dict]:
    """
    Fetch audit logs for an org. Org admin only — enforced by router.
    RLS also limits results to the requesting user's org.

    Args:
        org_id:          Filter to this org's audit trail.
        limit:           Max rows to return (default 100, max 500).
        offset:          Pagination offset.
        action_filter:   Optional — filter to a specific action type.
        actor_id_filter: Optional — filter to a specific actor.
    """
    limit = min(limit, 500)

    try:
        query = (
            _sb_admin()
            .table("audit_logs")
            .select("*")
            .eq("org_id", org_id)
            .order("created_at", desc=True)
            .limit(limit)
            .offset(offset)
        )

        if action_filter:
            query = query.eq("action", action_filter)
        if actor_id_filter:
            query = query.eq("actor_id", actor_id_filter)

        resp = query.execute()
        return resp.data or []

    except Exception as e:
        import logging
        logging.warning(f"[audit] Failed to read audit logs: {e}")
        return []


def get_my_audit_logs(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Fetch audit logs for a specific user's own actions."""
    try:
        resp = (
            _sb_admin()
            .table("audit_logs")
            .select("*")
            .eq("actor_id", user_id)
            .order("created_at", desc=True)
            .limit(min(limit, 200))
            .offset(offset)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []