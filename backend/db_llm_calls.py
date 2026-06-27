"""
db_llm_calls.py
Supabase table helpers for llm_calls (tracing).

Mirrors db.py's conventions:
- Module-level `supabase` client (service role — bypasses RLS).
- Reads are manually scoped by user_id (RLS policy exists as defense-in-depth
  only; the backend never uses a user-scoped key for these calls).
- Read helpers never raise — return safe empty defaults on any DB error so a
  tracing/observability failure never breaks an API response.
- insert_llm_call() is allowed to raise; llm/tracer.py is responsible for
  catching it (logging-over-correctness — never break the actual LLM call
  over a tracing write failure).
"""

from __future__ import annotations

from db import supabase


# ── Insert ────────────────────────────────────────────────────────────────────

def insert_llm_call(record: dict) -> None:
    """
    Insert one row into llm_calls.

    `record` is expected to already match the table's column shape (built by
    llm/tracer.py). This function does no validation/defaulting — it's a thin
    write. Raises on failure; caller (tracer.py) decides how to handle that.
    """
    supabase.table("llm_calls").insert(record).execute()


# ── Reads — all manually scoped by user_id ───────────────────────────────────

def get_calls(
    user_id: str,
    call_type: str | None = None,
    document_id: str | None = None,
    success: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return recent llm_calls rows for a user, most recent first.
    Optional filters: call_type, document_id, success.
    Never raises — returns [] on any error.
    """
    try:
        query = (
            supabase.table("llm_calls")
            .select("*")
            .eq("user_id", user_id)
        )
        if call_type:
            query = query.eq("call_type", call_type)
        if document_id:
            query = query.eq("document_id", document_id)
        if success is not None:
            query = query.eq("success", success)

        result = (
            query.order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data or []
    except Exception:
        return []


def get_call_by_id(call_id: str, user_id: str) -> dict | None:
    """
    Return a single llm_calls row by id, scoped to user_id.
    Returns None if not found, not owned by user_id, or on any DB error.
    """
    try:
        result = (
            supabase.table("llm_calls")
            .select("*")
            .eq("id", call_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = result.data
        return data[0] if data else None
    except Exception:
        return None


def get_summary(user_id: str, document_id: str | None = None) -> dict:
    """
    Aggregate usage summary for a user (optionally scoped to one document).

    Returns:
        {
            "total_calls": int,
            "total_tokens": int,
            "total_cost_usd": float,
            "avg_latency_ms": float,
            "cache_hit_rate": float,        # 0.0-1.0
            "by_call_type": {
                call_type: {"calls": int, "tokens": int, "cost_usd": float}
            },
        }

    Never raises — returns a zeroed-out summary shape on any error so the
    endpoint always returns valid JSON rather than a 500.
    """
    empty = {
        "total_calls": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "avg_latency_ms": 0.0,
        "cache_hit_rate": 0.0,
        "by_call_type": {},
    }

    try:
        query = (
            supabase.table("llm_calls")
            .select("call_type, total_tokens, estimated_cost_usd, latency_ms, cache_hit")
            .eq("user_id", user_id)
        )
        if document_id:
            query = query.eq("document_id", document_id)

        result = query.execute()
        rows = result.data or []

        if not rows:
            return empty

        total_calls = len(rows)
        total_tokens = sum(r.get("total_tokens") or 0 for r in rows)
        total_cost = sum(float(r.get("estimated_cost_usd") or 0.0) for r in rows)
        total_latency = sum(r.get("latency_ms") or 0 for r in rows)
        cache_hits = sum(1 for r in rows if r.get("cache_hit"))

        by_call_type: dict[str, dict] = {}
        for r in rows:
            ct = r.get("call_type") or "unknown"
            bucket = by_call_type.setdefault(ct, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            bucket["calls"] += 1
            bucket["tokens"] += r.get("total_tokens") or 0
            bucket["cost_usd"] += float(r.get("estimated_cost_usd") or 0.0)

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(total_latency / total_calls, 1),
            "cache_hit_rate": round(cache_hits / total_calls, 4),
            "by_call_type": by_call_type,
        }
    except Exception:
        return empty