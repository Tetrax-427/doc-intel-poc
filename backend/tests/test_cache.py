# tests/test_cache.py

import time
import pytest


# ---------------------------------------------------------------------------
# TTLCache unit tests
# ---------------------------------------------------------------------------

def test_cache_hit_returns_value():
    """set() then get() returns the stored value."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("key1", "hello", ttl_seconds=60)

    result = cache.get("key1")
    assert result == "hello"


def test_cache_miss_returns_none():
    """get() on a key that was never set returns None."""
    from core.cache import TTLCache

    cache = TTLCache()
    result = cache.get("nonexistent_key")
    assert result is None


def test_cache_ttl_expires():
    """Entries become None after their TTL has elapsed."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("expiring_key", "value", ttl_seconds=1)

    # Confirm it's there immediately
    assert cache.get("expiring_key") == "value"

    # Wait for expiry
    time.sleep(1.1)

    assert cache.get("expiring_key") is None, \
        "Expired entry should return None"


def test_cache_delete_removes_key():
    """delete() removes an existing key."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("to_delete", "value", ttl_seconds=60)
    cache.delete("to_delete")

    assert cache.get("to_delete") is None


def test_cache_delete_nonexistent_is_noop():
    """delete() on a missing key does not raise."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.delete("never_existed")  # should not raise


def test_cache_clear_removes_all():
    """clear() wipes all entries."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("a", 1, ttl_seconds=60)
    cache.set("b", 2, ttl_seconds=60)
    cache.set("c", 3, ttl_seconds=60)

    assert cache.size() == 3
    cache.clear()
    assert cache.size() == 0
    assert cache.get("a") is None


def test_cache_size_counts_entries():
    """size() returns the current number of stored entries."""
    from core.cache import TTLCache

    cache = TTLCache()
    assert cache.size() == 0

    cache.set("x", 1, ttl_seconds=60)
    assert cache.size() == 1

    cache.set("y", 2, ttl_seconds=60)
    assert cache.size() == 2

    cache.delete("x")
    assert cache.size() == 1


def test_cache_overwrites_existing_key():
    """set() on an existing key overwrites the value and resets TTL."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("key", "original", ttl_seconds=60)
    cache.set("key", "updated", ttl_seconds=60)

    assert cache.get("key") == "updated"


def test_cache_stores_various_types():
    """Cache can store lists, dicts, floats, booleans."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("list",  [1, 2, 3],           ttl_seconds=60)
    cache.set("dict",  {"a": 1},             ttl_seconds=60)
    cache.set("float", 3.14,                 ttl_seconds=60)
    cache.set("bool",  False,                ttl_seconds=60)

    assert cache.get("list")  == [1, 2, 3]
    assert cache.get("dict")  == {"a": 1}
    assert cache.get("float") == 3.14
    assert cache.get("bool")  is False


def test_evict_expired_removes_stale_entries():
    """evict_expired() removes all expired entries and returns the count."""
    from core.cache import TTLCache

    cache = TTLCache()
    cache.set("live",    "stays",  ttl_seconds=60)
    cache.set("expired", "gone",   ttl_seconds=1)

    time.sleep(1.1)

    evicted = cache.evict_expired()
    assert evicted == 1
    assert cache.get("live")    == "stays"
    assert cache.get("expired") is None


# ---------------------------------------------------------------------------
# cache_key helper
# ---------------------------------------------------------------------------

def test_cache_key_is_deterministic():
    """Same inputs always produce the same key."""
    from core.cache import cache_key

    assert cache_key("emb", "hello") == cache_key("emb", "hello")
    assert cache_key("vis", "/path/img.png", "invoice") == \
           cache_key("vis", "/path/img.png", "invoice")


def test_cache_key_differs_for_different_inputs():
    """Different inputs produce different keys."""
    from core.cache import cache_key

    assert cache_key("emb", "hello") != cache_key("emb", "world")
    assert cache_key("emb", "text")  != cache_key("vis", "text")
    assert cache_key("vis", "/img.png", "invoice") != \
           cache_key("vis", "/img.png", "cv_resume")


# ---------------------------------------------------------------------------
# Domain-specific accessor tests
# ---------------------------------------------------------------------------

def test_embedding_cache_roundtrip():
    """get_embedding / set_embedding roundtrip works correctly."""
    from core.cache import get_embedding, set_embedding, clear_all

    clear_all()

    text      = "the quick brown fox"
    embedding = [0.1, 0.2, 0.3, 0.4]

    assert get_embedding(text) is None, "Should be None before set"

    set_embedding(text, embedding)
    result = get_embedding(text)

    assert result == embedding
    clear_all()


def test_vision_description_cache_roundtrip():
    """get_vision_description / set_vision_description roundtrip works."""
    from core.cache import get_vision_description, set_vision_description, clear_all

    clear_all()

    image_path  = "/uploads/invoice_scan.png"
    doc_type    = "invoice"
    description = "This is an invoice from Acme Ltd dated January 2024."

    assert get_vision_description(image_path, doc_type) is None

    set_vision_description(image_path, doc_type, description)
    result = get_vision_description(image_path, doc_type)

    assert result == description
    clear_all()


def test_vision_cache_is_doc_type_specific():
    """
    Vision cache keys include doc_type — same image with different
    doc_type returns different cached values.
    """
    from core.cache import get_vision_description, set_vision_description, clear_all

    clear_all()

    image_path = "/uploads/document.png"

    set_vision_description(image_path, "invoice",   "Invoice description")
    set_vision_description(image_path, "cv_resume", "CV description")

    assert get_vision_description(image_path, "invoice")   == "Invoice description"
    assert get_vision_description(image_path, "cv_resume") == "CV description"
    assert get_vision_description(image_path, "contract")  is None

    clear_all()


def test_classification_cache_roundtrip():
    """get_classification / set_classification roundtrip works."""
    from core.cache import get_classification, set_classification, clear_all

    clear_all()

    text_hash = "abc123hash"
    cls_result = {
        "doc_type":   "invoice",
        "confidence": 0.95,
        "reasoning":  "Contains invoice number and line items",
    }

    assert get_classification(text_hash) is None

    set_classification(text_hash, cls_result)
    result = get_classification(text_hash)

    assert result == cls_result
    clear_all()


def test_make_text_hash_is_deterministic():
    """make_text_hash returns the same hash for the same text."""
    from core.cache import make_text_hash

    text = "This is a sample document with invoice details."
    assert make_text_hash(text) == make_text_hash(text)


def test_make_text_hash_uses_first_2000_chars():
    """make_text_hash produces the same hash regardless of text after 2000 chars."""
    from core.cache import make_text_hash

    base   = "x" * 2000
    long1  = base + "AAAA"
    long2  = base + "BBBB"

    # Both have the same first 2000 chars — should produce same hash
    assert make_text_hash(long1) == make_text_hash(long2)


# ---------------------------------------------------------------------------
# cache_stats and clear_all
# ---------------------------------------------------------------------------

def test_cache_stats_returns_dict():
    """cache_stats() returns a dict with at least total_entries."""
    from core.cache import cache_stats, clear_all, set_embedding

    clear_all()
    set_embedding("test text", [0.1, 0.2])

    stats = cache_stats()

    assert isinstance(stats, dict)
    assert "total_entries" in stats
    assert stats["total_entries"] >= 1

    clear_all()


def test_clear_all_wipes_singleton_cache():
    """clear_all() empties the module-level singleton cache."""
    from core.cache import set_embedding, get_embedding, clear_all

    set_embedding("some text", [1.0, 2.0])
    assert get_embedding("some text") is not None

    clear_all()
    assert get_embedding("some text") is None


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------

def test_cache_thread_safety():
    """
    Concurrent reads and writes from multiple threads do not raise or corrupt data.
    Not a strict correctness test — checks for absence of crashes or exceptions.
    """
    import threading
    from core.cache import TTLCache

    cache  = TTLCache()
    errors = []

    def writer(n):
        try:
            for i in range(50):
                cache.set(f"key_{n}_{i}", f"value_{n}_{i}", ttl_seconds=60)
        except Exception as e:
            errors.append(str(e))

    def reader(n):
        try:
            for i in range(50):
                cache.get(f"key_{n}_{i}")
        except Exception as e:
            errors.append(str(e))

    threads = (
        [threading.Thread(target=writer, args=(i,)) for i in range(5)] +
        [threading.Thread(target=reader, args=(i,)) for i in range(5)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread safety errors: {errors}"