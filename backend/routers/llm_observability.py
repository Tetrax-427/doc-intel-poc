"""
routers/llm_observability.py
Endpoints:
    GET /llm/calls
    GET /llm/calls/summary
    GET /llm/calls/{call_id}
    GET /llm/cache/stats

All endpoints are user-scoped via get_current_user/get_user_id, matching the
convention used elsewhere (e.g. routers/documents.py list_documents()).

Route order matters: /llm/calls/summary is registered before
/llm/calls/{call_id} so FastAPI doesn't try to match "summary" as a call_id
path parameter. See FINAL_PLAN.md — this was checked against the real
routing behavior, not assumed.
"""

from core.auth import get_current_user_context, get_user_id
from db_llm import get_calls, get_call_by_id, get_summary, get_cache_stats
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/llm", tags=["LLM Observability"])


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/calls")
def list_calls(
    call_type: str | None = None,
    document_id: str | None = None,
    success: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    user=Depends(get_current_user_context),
):
    """
    Return recent llm_calls rows for the authenticated user, most recent
    first. Optional filters: call_type, document_id, success.
    """
    uid = get_user_id(user)
    return get_calls(
        user_id=uid,
        call_type=call_type,
        document_id=document_id,
        success=success,
        limit=limit,
        offset=offset,
    )


@router.get("/calls/summary")
def calls_summary(
    document_id: str | None = None,
    user=Depends(get_current_user_context),
):
    """
    Aggregate usage summary for the authenticated user — total calls,
    tokens, estimated cost, average latency, cache hit rate, broken down
    by call_type. Optionally scoped to a single document_id.
    """
    uid = get_user_id(user)
    return get_summary(user_id=uid, document_id=document_id)


@router.get("/calls/{call_id}")
def get_call(call_id: str, user=Depends(get_current_user_context)):
    """
    Return a single llm_calls row by id, scoped to the authenticated user.
    Returns null (not 404) if not found or not owned — matches the existing
    soft-failure convention used by db.py's get_document()-style helpers
    rather than introducing a new error shape for this one endpoint.
    """
    uid = get_user_id(user)
    return get_call_by_id(call_id, user_id=uid)


@router.get("/cache/stats")
def cache_stats(user=Depends(get_current_user_context)):
    """
    Aggregate Layer 2 cache stats for the authenticated user — total
    entries, total hits, estimated USD saved.
    """
    uid = get_user_id(user)
    return get_cache_stats(user_id=uid)