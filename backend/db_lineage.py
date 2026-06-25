"""
F1 — DB helpers for lineage_logs table.

Separated from core/lineage.py so DB concerns stay out of core/ and
tests can mock this module independently.

Public API:
    store_lineage_event()       — insert one row into lineage_logs
    get_lineage_for_document()  — fetch events for a document (newest first)
    get_lineage_summary()       — event counts grouped by type
"""

from __future__ import annotations

from db import supabase
from core.logger import get_logger

logger = get_logger("db_lineage")


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

    supabase.table("lineage_logs").insert(payload).execute()


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