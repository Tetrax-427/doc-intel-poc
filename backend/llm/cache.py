"""
llm/cache.py — Layer 2 exact-match cache.

Design (see FINAL_PLAN.md §0 for the full reasoning trail):
- Cache key = sha256(user_id | actual_provider | actual_model | system | user)
  — reuses llm.hashing.compute_prompt_hash, the SAME function llm/tracer.py
  uses, so a trace row and its corresponding cache row always agree on what
  "the same call" means.
- "actual_provider"/"actual_model" — ALWAYS whoever really answered. Never
  the nominal chain[0]/primary. This is the resolution to the fallback-vs-
  cache-poisoning discussion: lookup happens per-provider, inside the
  fallback loop (engine.py calls is_cacheable() + lookup() once per chain
  member, in order, BEFORE attempting a live call to that member).
- No TTL. Invalidated only by document delete (invalidate_for_document).
- chat/query/query_stream are NEVER cacheable — excluded via is_cacheable().
- Streaming is NEVER cacheable, full stop, regardless of call_type — engine.py
  enforces this by never calling cache lookup/store from the streaming path
  at all (not just by checking is_cacheable()), since streaming responses
  can't be captured as a single response_text the way this cache expects.
"""

from __future__ import annotations

from core.logger import get_logger
from db_llm_cache import get_cached, set_cached, record_hit
from llm.hashing import compute_prompt_hash

logger = get_logger("llm.cache")


# ---------------------------------------------------------------------------
# Cacheability allowlist — see FINAL_PLAN.md §1 for the full call-type table
# and the reasoning behind each inclusion/exclusion.
# ---------------------------------------------------------------------------

_CACHEABLE_CALL_TYPES = {
    "expand",             # expand_query
    "evidence_extract",   # get_exact_sentence
    "extract",            # extract_fields
    "extract_tables",     # extract_tables
    "summarize",          # generate_summary
    "nl_to_schema",       # nl_to_schema
    "classify_document",  # classify_document / classify_document_from_text
    "hyde",               # generate_hyde_passage
    "multiquery",         # generate_query_variants
    "split_detection",    # detect_boundaries_llm
    "vision",             # call_vision_llm (separate key scheme — see below)
}

# Explicitly NOT cacheable, listed for clarity even though absence from the
# set above already excludes them: classify, general, compress, query,
# query_stream. Do not add these without re-reading FINAL_PLAN.md §1 — each
# was excluded for a specific reason (live history, conversational, etc.),
# not by oversight.


def is_cacheable(call_type: str) -> bool:
    """Whether this call_type is eligible for Layer 2 caching at all."""
    return call_type in _CACHEABLE_CALL_TYPES


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def build_cache_key(user_id: str, provider: str, model: str, system: str, user: str) -> str:
    """
    Thin wrapper over the shared hash function — kept here (rather than
    having every caller import llm.hashing directly) so the cache module is
    the single place that documents "this hash IS the cache key", while
    llm/tracer.py's use of the same function is documented there as
    "this hash is stored for grouping/debugging, not used for lookup".
    Same function, two different roles — see llm/hashing.py docstring.
    """
    return compute_prompt_hash(user_id, provider, model, system, user)


# ---------------------------------------------------------------------------
# Lookup / store — text-shaped calls (system + user)
# ---------------------------------------------------------------------------

def lookup(user_id: str, provider: str, model: str, system: str, user: str) -> dict | None:
    """
    Check the cache for an exact match on (user_id, provider, model, system,
    user). Returns the cache row dict on a hit (caller reads response_text
    off it), or None on a miss.

    Intended call site: inside engine.py's fallback loop, once per
    (provider, model) chain entry, BEFORE attempting a live call to that
    entry — see FINAL_PLAN.md §0/§3 Phase F step 20.

    Never raises (get_cached() already swallows DB errors) — a cache lookup
    failure degrades to "treat as a miss", never blocks the real call.
    """
    cache_key = build_cache_key(user_id, provider, model, system, user)
    row = get_cached(user_id, provider, model, cache_key)
    if row is not None:
        record_hit(row["id"])
    return row


def store(
    *,
    user_id: str,
    provider: str,
    model: str,
    system: str,
    user: str,
    call_type: str,
    response_text: str,
    document_id: str | None = None,
    response_model_name: str | None = None,
    original_provider_call_id: str | None = None,
    original_cost_usd: float | None = None,
) -> None:
    """
    Store a successful, cacheable response. Only called after a real
    provider call succeeds (never on a call that was itself a cache hit —
    re-storing a cache hit would just overwrite the row with identical
    content for no benefit, and engine.py's call site naturally avoids this
    by only calling store() in the "made a live call" branch).

    Best-effort — failures are logged and swallowed. A failed cache write
    must never affect the response already being returned to the user.
    """
    if not is_cacheable(call_type):
        return

    cache_key = build_cache_key(user_id, provider, model, system, user)
    record = {
        "user_id": user_id,
        "document_id": document_id,
        "call_type": call_type,
        "provider": provider,
        "model": model,
        "cache_key": cache_key,
        "system_text": system,
        "user_text": user,
        "response_text": response_text,
        "response_model_name": response_model_name,
        "original_provider_call_id": original_provider_call_id,
        "original_cost_usd": original_cost_usd,
    }
    try:
        set_cached(record)
    except Exception as exc:
        logger.warning(
            "Failed to write cache entry — continuing without it",
            call_type=call_type, provider=provider, model=model, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Vision cache — separate key scheme (image bytes, not system/user text)
# ---------------------------------------------------------------------------

def build_vision_cache_key(user_id: str, provider: str, model: str, image_data_b64: str, prompt: str) -> str:
    """
    Vision calls hash the (already base64-encoded) image bytes + prompt
    instead of system/user text — there is no meaningful system/user split
    for an image call. Reuses the same hash function with image_data_b64
    standing in for "system" and prompt standing in for "user"; this is
    purely a parameter-naming convenience, not a claim that image bytes are
    semantically a system prompt.
    """
    return compute_prompt_hash(user_id, provider, model, image_data_b64, prompt)


def lookup_vision(user_id: str, provider: str, model: str, image_data_b64: str, prompt: str) -> dict | None:
    """Vision-specific lookup — same underlying table/columns, different key inputs."""
    cache_key = build_vision_cache_key(user_id, provider, model, image_data_b64, prompt)
    row = get_cached(user_id, provider, model, cache_key)
    if row is not None:
        record_hit(row["id"])
    return row


def store_vision(
    *,
    user_id: str,
    provider: str,
    model: str,
    image_data_b64: str,
    prompt: str,
    response_text: str,
    document_id: str | None = None,
    original_cost_usd: float | None = None,
) -> None:
    """Vision-specific store — see store() above for the general-case docstring; same semantics."""
    cache_key = build_vision_cache_key(user_id, provider, model, image_data_b64, prompt)
    record = {
        "user_id": user_id,
        "document_id": document_id,
        "call_type": "vision",
        "provider": provider,
        "model": model,
        "cache_key": cache_key,
        "system_text": None,
        "user_text": prompt,
        "response_text": response_text,
        "response_model_name": None,
        "original_provider_call_id": None,
        "original_cost_usd": original_cost_usd,
    }
    try:
        set_cached(record)
    except Exception as exc:
        logger.warning(
            "Failed to write vision cache entry — continuing without it",
            provider=provider, model=model, error=str(exc),
        )