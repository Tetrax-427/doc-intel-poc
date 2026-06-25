"""
DB helpers for extraction_results table.

E2: store_extraction_result() now accepts the enriched {value, bbox} shape
    per field.  get_extraction_result_by_id() is new — used by the
    GET /extract/{extraction_id} endpoint.
"""

from __future__ import annotations
from datetime import datetime, timezone
from db import supabase
from core.logger import get_logger

logger = get_logger("db_extraction")


def store_extraction_result(
    document_id: str,
    template_id: str,
    results: dict,
    user_id: str = "",
) -> str:
    """
    Store extraction results (E2 enriched shape) in extraction_results table.

    Args:
        document_id:  Document UUID.
        template_id:  Template used ("custom" for ad-hoc).
        results:      {field_name: {"value": str|None, "bbox": dict|None}}
        user_id:      Owner of the extraction (for RLS / lineage).

    Returns:
        The new extraction_results row UUID.
    """
    payload = {
        "document_id": document_id,
        "template_id": template_id,
        "result":       results,          # JSONB — stores new {value,bbox} shape
        "user_id":      user_id,
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }
    response = supabase.table("extraction_results").insert(payload).execute()
    row_id = response.data[0]["id"] if response.data else ""
    logger.info(
        "Extraction result stored",
        document_id=document_id,
        extraction_id=row_id,
        field_count=len(results),
    )
    return row_id


def get_extraction_result_by_id(
    extraction_id: str,
    user_id: str = "",
) -> dict | None:
    """
    Fetch one extraction result row by UUID.

    Ownership check: if user_id is provided, the row must belong to that user
    (via document ownership join).  Service-role backend calls can pass ""
    to bypass (RLS already enforces at the DB level when using anon key).

    Returns the row dict or None if not found / not owned.
    """
    query = (
        supabase.table("extraction_results")
        .select("*")
        .eq("id", extraction_id)
    )
    if user_id:
        # Filter by ownership via documents join
        query = query.eq("user_id", user_id)

    result = query.limit(1).execute()
    return result.data[0] if result.data else None


def get_latest_extraction_for_document(
    document_id: str,
    template_id: str | None = None,
) -> dict | None:
    """
    Return the most recent extraction result for a document.
    Optionally filter by template_id.
    """
    query = (
        supabase.table("extraction_results")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .limit(1)
    )
    if template_id:
        query = query.eq("template_id", template_id)

    result = query.execute()
    return result.data[0] if result.data else None