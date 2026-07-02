"""
db_audit.py
Audit log helpers (audit_logs table) and document lineage helpers
(lineage_logs table) for DocIntel.

Merged from db_audit.py + db_lineage.py — both are append-only event logs
and were combined to reduce file sprawl.

Access pattern differs intentionally between the two halves:
  - audit_logs:    writes use the service-role client (get_supabase_admin) —
                    trusted server-side writes must not be blocked by RLS.
  - lineage_logs:  uses the plain anon client (db.supabase) — RLS scopes
                    results naturally, and writes go through core/lineage.py
                    only, never called directly from application code.

Usage:
    from db_audit import log_audit
    from db_audit import store_lineage_event, get_lineage_for_document, get_lineage_summary
"""

from __future__ import annotations

from typing import Any

from db import get_supabase_admin, supabase
from core.logger import get_logger

logger = get_logger("db_audit")


# ===========================================================================
# audit_logs
# ===========================================================================

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
        get_supabase_admin().table("audit_logs").insert({
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
        logger.warning("Failed to write audit log", error=str(e))


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
            get_supabase_admin()
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
        logger.warning("Failed to read audit logs", error=str(e))
        return []


def get_my_audit_logs(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Fetch audit logs for a specific user's own actions."""
    try:
        resp = (
            get_supabase_admin()
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


# ===========================================================================
# lineage_logs
# ===========================================================================
# F1 — DB helpers for the lineage_logs table.
#
# store_lineage_event() is called exclusively by core/lineage.log_event().
# Do not call it directly from application code.

def store_lineage_event(
    document_id: str,
    user_id: str,
    event_type: str,
    event_data: dict | None = None,
    duration_ms: int | None = None,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    """
    Insert one event row into lineage_logs.

    Called exclusively by core/lineage.log_event().
    Do not call directly from application code.
    """
    payload: dict = {
        "document_id": document_id,
        "user_id":     user_id,
        "event_type":  event_type,
        "event_data":  event_data or {},
        "status":      status,
    }
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if error_message:
        payload["error_message"] = error_message

    get_supabase_admin().table("lineage_logs").insert(payload).execute()


def get_lineage_for_document(
    document_id: str,
    user_id: str,
    limit: int = 100,
    event_type_filter: str | None = None,
) -> list[dict]:
    """
    Return all lineage events for a document, newest-first.

    Args:
        document_id:        Document to fetch events for.
        user_id:            Owner — enforces data isolation at query level
                            (defense in depth on top of RLS).
        limit:              Max rows (default 100, cap 500).
        event_type_filter:  Optional filter to a single event_type string.
    """
    effective_limit = min(limit, 500)

    query = (
        supabase.table("lineage_logs")
        .select("id, document_id, user_id, event_type, event_data, duration_ms, status, error_message, created_at")
        .eq("document_id", document_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(effective_limit)
    )

    if event_type_filter:
        query = query.eq("event_type", event_type_filter)

    result = query.execute()
    return result.data or []


def get_lineage_summary(document_id: str, user_id: str) -> dict[str, int]:
    """
    Return event counts grouped by event_type for a document.
    """
    result = (
        supabase.table("lineage_logs")
        .select("event_type")
        .eq("document_id", document_id)
        .eq("user_id", user_id)
        .execute()
    )
    events = result.data or []

    counts: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "unknown")
        counts[et] = counts.get(et, 0) + 1

    return counts