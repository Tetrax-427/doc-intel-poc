"""
F1 — Lineage API endpoints.

Mounted under the documents router prefix so URLs match the spec:
    GET  /documents/{id}/lineage          — full audit trail
    GET  /documents/{id}/lineage/summary  — event counts by type

Register this router in main.py BEFORE documents.router, or add these
routes directly into routers/documents.py (both approaches work).
"""

from fastapi import APIRouter, Depends, Query
from core.responses import internal_error
from core.auth import get_current_user, get_user_id
from db_lineage import get_lineage_for_document, get_lineage_summary

router = APIRouter(prefix="/documents", tags=["Lineage"])


@router.get("/{document_id}/lineage")
def get_document_lineage(
    document_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    event_type: str | None = Query(default=None, description="Filter by event_type"),
    user=Depends(get_current_user),
):
    """
    Return the full audit trail for a document, newest-first.

    Query params:
        limit       — max rows to return (1–500, default 100)
        event_type  — optional filter, e.g. "classified"
    """
    try:
        uid = get_user_id(user)
        events = get_lineage_for_document(
            document_id,
            user_id=uid,
            limit=limit,
            event_type_filter=event_type,
        )
        return {
            "document_id": document_id,
            "events":      events,
            "total":       len(events),
        }
    except Exception as exc:
        return internal_error(f"Lineage fetch failed: {exc}")


@router.get("/{document_id}/lineage/summary")
def get_document_lineage_summary(
    document_id: str,
    user=Depends(get_current_user),
):
    """
    Return event counts grouped by event_type for a document.

    Response:
        {
            "document_id": "...",
            "counts": {
                "upload_received": 1,
                "classified": 1,
                "extraction_run": 3,
                ...
            }
        }
    """
    try:
        uid = get_user_id(user)
        counts = get_lineage_summary(document_id, user_id=uid)
        return {"document_id": document_id, "counts": counts}
    except Exception as exc:
        return internal_error(f"Lineage summary failed: {exc}")