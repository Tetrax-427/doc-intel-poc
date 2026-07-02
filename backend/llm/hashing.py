"""
llm/hashing.py — shared prompt-hash derivation.

Used by both llm/tracer.py (Phase A — stores the hash for debugging/grouping,
does not use it for any lookup) and llm/cache.py (Phase F — uses the identical
hash as the actual Layer-2 cache key).

Defined once, here, so the two phases can never drift into computing the hash
two different ways.
"""

from __future__ import annotations

import hashlib


def compute_prompt_hash(user_id: str, provider: str, model: str, system: str, user: str) -> str:
    """
    Deterministic hash over the exact inputs that determine whether two calls
    are "the same call" for caching/grouping purposes.

    Includes user_id — cache entries (and trace groupings) never cross users.
    Includes provider + model — see FINAL_PLAN.md §0: a cache key always
    reflects the provider/model that actually answered, never a nominal
    "primary" provider, so a fallback never silently mislabels or poisons
    another provider's cache entry.

    Uses sha256 (not md5, unlike the unrelated TTLCache in core/cache.py —
    that cache is for embeddings/vision/local stuff with no security framing;
    this one backs a cross-user-isolation guarantee, so the stronger hash is
    deliberate even though collision resistance isn't really the concern —
    it's just good hygiene to not reuse md5 for anything resembling an
    isolation boundary).
    """
    raw = f"{user_id}|{provider}|{model}|{system}|{user}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()