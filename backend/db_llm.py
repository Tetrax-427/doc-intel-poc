"""
db_llm.py
Supabase table helpers for LLM observability — llm_calls (tracing) and
llm_cache (Layer 2 exact-match cache).
related LLM observability tables and were combined to reduce file sprawl.

Conventions (shared across both halves):
- Module-level `supabase` client (service role — bypasses RLS).
- Reads are manually scoped by user_id (RLS policy exists as defense-in-depth
  only; the backend never uses a user-scoped key for these calls).
- Read helpers never raise — return safe empty defaults on any DB error so a
  tracing/observability/cache failure never breaks an API response.
- Write helpers (insert_llm_call, set_cached) ARE allowed to raise — callers
  (llm/tracer.py, llm/cache.py) are responsible for catching them
  (logging-over-correctness — never break the actual LLM call over a
  tracing/cache write failure).
"""

from __future__ import annotations

from datetime import datetime, timezone

from db import supabase, get_supabase_admin


# ===========================================================================
# llm_calls — tracing
# ===========================================================================

# ── Insert ────────────────────────────────────────────────────────────────────

def insert_llm_call(record: dict) -> None:
    """
    Insert one row into llm_calls.

    `record` is expected to already match the table's column shape (built by
    llm/tracer.py). This function does no validation/defaulting — it's a thin
    write. Raises on failure; caller (tracer.py) decides how to handle that.
    """
    get_supabase_admin().table("llm_calls").insert(record).execute()


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


# ===========================================================================
# llm_cache — Layer 2 exact-match cache
# ===========================================================================

# ── Lookup ────────────────────────────────────────────────────────────────────

def get_cached(user_id: str, provider: str, model: str, cache_key: str) -> dict | None:
    """
    Return the cached row for this exact (user_id, provider, model, cache_key)
    combination, or None on a miss or any DB error.

    Does NOT increment hit_count itself — call record_hit() separately after
    a confirmed hit, so a lookup-only caller (if one ever exists) doesn't
    accidentally inflate hit stats.
    """
    try:
        result = (
            supabase.table("llm_cache")
            .select("*")
            .eq("user_id", user_id)
            .eq("provider", provider)
            .eq("model", model)
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        data = result.data
        return data[0] if data else None
    except Exception:
        return None


def record_hit(cache_row_id: str) -> None:
    """
    Increment hit_count and update last_hit_at for a cache row.
    Best-effort — failure here must never affect the cached response being
    returned to the caller. Swallows all exceptions.
    """
    try:
        # Supabase python client has no atomic increment helper here without
        # an RPC function; read-modify-write is acceptable for this use case
        # (hit_count is a stats nicety, not a correctness-critical counter —
        # a lost increment under rare concurrent access is fine).
        current = (
            supabase.table("llm_cache")
            .select("hit_count")
            .eq("id", cache_row_id)
            .limit(1)
            .execute()
        )
        if not current.data:
            return
        new_count = (current.data[0].get("hit_count") or 0) + 1
        supabase.table("llm_cache").update({
            "hit_count": new_count,
            "last_hit_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", cache_row_id).execute()
    except Exception:
        pass


# ── Write ─────────────────────────────────────────────────────────────────────

def set_cached(record: dict) -> None:
    """
    Insert or overwrite a cache entry. `record` must already match the
    llm_cache column shape (built by llm/cache.py).

    Uses upsert on the (user_id, provider, model, cache_key) unique index —
    if a row with this exact key already exists (e.g. a race between two
    concurrent identical requests), the newer write wins rather than erroring.
    Raises on failure; caller (llm/cache.py) decides how to handle that.
    """
    supabase.table("llm_cache").upsert(
        record,
        on_conflict="user_id,provider,model,cache_key",
    ).execute()


# ── Invalidation ──────────────────────────────────────────────────────────────

def invalidate_for_document(document_id: str, user_id: str) -> int:
    """
    Delete all cache entries for a document, scoped to user_id (a user can
    only invalidate their own document's cache entries — mirrors the
    ownership scoping added to DELETE /documents/{id} itself).

    Returns the number of rows deleted (0 on any error — never raises,
    since this is called from the document-delete path and a cache
    invalidation failure should not block or fail the actual delete).
    """
    try:
        result = (
            supabase.table("llm_cache")
            .delete()
            .eq("document_id", document_id)
            .eq("user_id", user_id)
            .execute()
        )
        return len(result.data or [])
    except Exception:
        return 0


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_cache_stats(user_id: str) -> dict:
    """
    Aggregate cache stats for a user: total entries, total hits, estimated
    USD saved (hit_count * original_cost_usd, summed, treating NULL
    original_cost_usd as 0 for this aggregate only — display concern, not a
    stored-value concern, consistent with get_summary() above).

    Never raises — returns a zeroed shape on any error.
    """
    empty = {"total_entries": 0, "total_hits": 0, "estimated_saved_usd": 0.0}
    try:
        result = (
            supabase.table("llm_cache")
            .select("hit_count, original_cost_usd")
            .eq("user_id", user_id)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return empty

        total_hits = sum(r.get("hit_count") or 0 for r in rows)
        saved = sum(
            (r.get("hit_count") or 0) * float(r.get("original_cost_usd") or 0.0)
            for r in rows
        )
        return {
            "total_entries": len(rows),
            "total_hits": total_hits,
            "estimated_saved_usd": round(saved, 6),
        }
    except Exception:
        return empty