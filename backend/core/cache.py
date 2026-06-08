# backend/core/cache.py

import time
import hashlib
import threading
from typing import Any

from core.logger import get_logger

logger = get_logger("cache")


# ---------------------------------------------------------------------------
# TTLCache — in-memory key/value store with per-entry expiry
# ---------------------------------------------------------------------------

class TTLCache:
    """
    Simple in-memory cache with TTL expiry.

    Thread-safe for single-process use via a reentrant lock.
    Each entry stores (value, expires_at) — expired entries are evicted
    lazily on read rather than via a background sweep thread.

    Upgrade path:
        Replace _store with a Redis client for multi-process / multi-server
        deployments. The public API (get / set / delete / clear / size)
        stays identical so callers need no changes.
    """

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """
        Return the cached value for key, or None if missing / expired.
        Expired entries are deleted on read (lazy eviction).
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Store value under key with a TTL (default 1 hour)."""
        with self._lock:
            self._store[key] = (value, time.time() + ttl_seconds)

    def delete(self, key: str):
        """Remove a key. No-op if key does not exist."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        """Remove all entries. Used in tests."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return the number of entries currently in the store (including expired)."""
        with self._lock:
            return len(self._store)

    def evict_expired(self) -> int:
        """
        Proactively remove all expired entries.
        Returns the number of entries removed.
        Call periodically if memory usage is a concern.
        """
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._store.items() if now > exp
            ]
            for k in expired_keys:
                del self._store[k]
        return len(expired_keys)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache = TTLCache()


# ---------------------------------------------------------------------------
# Key builder
# ---------------------------------------------------------------------------

def cache_key(*parts) -> str:
    """
    Build a deterministic cache key from one or more parts.
    Uses MD5 for a short, fixed-length key — not used for security.

    Example:
        cache_key("emb", "hello world")  → "a1b2c3..."
        cache_key("vis", "/tmp/x.png", "invoice") → "d4e5f6..."
    """
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache  (TTL: 24 hours)
# ---------------------------------------------------------------------------
# Embeddings are deterministic for the same model — safe to cache long-term.
# Re-uploading the same document won't call the embedding model again.

_EMBEDDING_TTL = 86_400  # 24 hours


def get_embedding(text: str) -> list[float] | None:
    """Return cached embedding for text, or None on cache miss."""
    return _cache.get(cache_key("emb", text))


def set_embedding(text: str, embedding: list[float]):
    """Cache an embedding for 24 hours."""
    _cache.set(cache_key("emb", text), embedding, ttl_seconds=_EMBEDDING_TTL)


# ---------------------------------------------------------------------------
# Vision description cache  (TTL: 7 days)
# ---------------------------------------------------------------------------
# Image files don't change between requests — long TTL is safe.
# Key includes doc_type so reclassifying a document gets a fresh description.

_VISION_TTL = 604_800  # 7 days


def get_vision_description(image_path: str, doc_type: str) -> str | None:
    """Return cached vision description, or None on cache miss."""
    return _cache.get(cache_key("vis", image_path, doc_type))


def set_vision_description(image_path: str, doc_type: str, description: str):
    """Cache a vision description for 7 days."""
    _cache.set(cache_key("vis", image_path, doc_type), description, ttl_seconds=_VISION_TTL)


# ---------------------------------------------------------------------------
# Classification cache  (TTL: 1 hour)
# ---------------------------------------------------------------------------
# Classifications can change if the LLM or prompt changes — shorter TTL.
# Key is a hash of the document text so different content always gets a fresh call.

_CLASSIFICATION_TTL = 3_600  # 1 hour


def get_classification(text_hash: str) -> dict | None:
    """Return cached classification result, or None on cache miss."""
    return _cache.get(cache_key("cls", text_hash))


def set_classification(text_hash: str, result: dict):
    """Cache a classification result for 1 hour."""
    _cache.set(cache_key("cls", text_hash), result, ttl_seconds=_CLASSIFICATION_TTL)


def make_text_hash(text: str) -> str:
    """
    Hash document text for use as a classification cache key.
    Truncate to first 2000 chars — that's all classify_document() uses.
    """
    return hashlib.md5(text[:2000].encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cache stats — exposed via GET /admin/cache or logs
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """
    Return a snapshot of current cache state.
    Useful for debugging and monitoring endpoints.
    """
    return {
        "total_entries": _cache.size(),
    }


def clear_all():
    """
    Wipe the entire cache.
    Used in tests and can be wired to an admin endpoint.
    """
    _cache.clear()
    logger.info("Cache cleared")