"""
db_llm_cache.py
Supabase table helpers for llm_cache (Layer 2 exact-match cache).

Same conventions as db_llm_calls.py: module-level service-role `supabase`
client, manual user_id scoping, reads never raise.

set_cached() IS allowed to raise on failure — llm/cache.py's store() call
site wraps it in try/except (a failed cache write should never break the
LLM response that's about to be returned to the user; it just means the
next identical call won't get a cache hit, which is a performance miss,
not a correctness issue).
"""

from __future__ import annotations

from datetime import datetime, timezone

from db import supabase


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
    stored-value concern, consistent with db_llm_calls.get_summary()).

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