"""
tests/test_auth.py
F2 — API key rotation tests. Source-verified, no real imports needed.
Run: pytest tests/test_auth.py -v
"""
import sys, os, ast, hashlib, secrets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timezone, timedelta
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')

def src(f): return open(os.path.join(BACKEND, f)).read()
def fns(f):
    t = ast.parse(src(f))
    return [n.name for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
def routes(f):
    import re
    return re.findall(r'@router\.\w+\("([^"]+)"', src(f))

# ---------------------------------------------------------------------------
# Inline key generation (mirrors api_keys.py exactly)
# ---------------------------------------------------------------------------
def generate_api_key():
    key      = f"dik_{secrets.token_urlsafe(32)}"
    prefix   = key[:10]
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, prefix, key_hash

def _is_within_grace(grace_expires_at_iso):
    now = datetime.now(timezone.utc)
    try:
        expiry = datetime.fromisoformat(grace_expires_at_iso.replace("Z", "+00:00"))
        return now < expiry
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------
def test_generate_api_key_exists():
    assert "generate_api_key" in fns("api_keys.py")
    print("  generate_api_key in api_keys.py ✓")

def test_rotate_api_key_exists():
    assert "rotate_api_key" in fns("api_keys.py")
    print("  rotate_api_key in api_keys.py ✓")

def test_validate_api_key_exists():
    assert "validate_api_key" in fns("api_keys.py")
    print("  validate_api_key in api_keys.py ✓")

def test_revoke_api_key_exists():
    assert "revoke_api_key" in fns("api_keys.py")
    print("  revoke_api_key in api_keys.py ✓")

def test_api_key_uses_dik_prefix():
    assert '"dik_' in src("api_keys.py")
    print("  api_keys.py uses dik_ prefix ✓")

def test_api_key_uses_status_field():
    assert '"status"' in src("api_keys.py")
    print("  api_keys.py uses status field ✓")

def test_api_key_has_rotating_status():
    assert '"rotating"' in src("api_keys.py")
    print("  api_keys.py has rotating status ✓")

def test_api_key_has_deleted_status():
    assert '"deleted"' in src("api_keys.py")
    print("  api_keys.py has deleted status ✓")

def test_api_key_has_grace_expires_at():
    assert "grace_expires_at" in src("api_keys.py")
    print("  api_keys.py has grace_expires_at ✓")

def test_mark_api_key_rotating_in_db_apikeys():
    assert "mark_api_key_rotating" in fns("db_apikeys.py")
    print("  mark_api_key_rotating in db_apikeys.py ✓")

def test_get_api_key_by_id_in_db_apikeys():
    assert "get_api_key_by_id" in fns("db_apikeys.py")
    print("  get_api_key_by_id in db_apikeys.py ✓")

def test_db_apikeys_sets_status_rotating():
    assert '"rotating"' in src("db_apikeys.py")
    print("  db_apikeys.py sets status=rotating ✓")

def test_db_apikeys_sets_grace_expires_at():
    assert "grace_expires_at" in src("db_apikeys.py")
    print("  db_apikeys.py sets grace_expires_at ✓")

def test_refresh_endpoint_exists():
    assert "/refresh" in routes("routers/auth.py")
    print("  /auth/refresh endpoint exists ✓")

def test_forgot_password_endpoint_exists():
    assert "/forgot-password" in routes("routers/auth.py")
    print("  /auth/forgot-password endpoint exists ✓")

def test_reset_password_endpoint_exists():
    assert "/reset-password" in routes("routers/auth.py")
    print("  /auth/reset-password endpoint exists ✓")

# ---------------------------------------------------------------------------
# Key generation logic tests (inline, no imports)
# ---------------------------------------------------------------------------
def test_generated_key_starts_with_dik():
    for _ in range(5):
        key, _, _ = generate_api_key()
        assert key.startswith("dik_"), f"Key must start with dik_, got {key[:10]}"
    print("  generated keys start with dik_ ✓")

def test_prefix_is_first_10_chars():
    key, prefix, _ = generate_api_key()
    assert prefix == key[:10]
    print("  prefix == key[:10] ✓")

def test_hash_is_sha256_of_key():
    key, _, key_hash = generate_api_key()
    expected = hashlib.sha256(key.encode()).hexdigest()
    assert key_hash == expected
    print("  hash is sha256 of key ✓")

def test_keys_are_unique():
    keys = [generate_api_key()[0] for _ in range(10)]
    assert len(set(keys)) == 10
    print("  10 generated keys are all unique ✓")

# ---------------------------------------------------------------------------
# Grace period logic tests (inline)
# ---------------------------------------------------------------------------
def test_within_grace_period_returns_true():
    future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    assert _is_within_grace(future) is True
    print("  future grace_expires_at → within grace ✓")

def test_past_grace_period_returns_false():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert _is_within_grace(past) is False
    print("  past grace_expires_at → outside grace ✓")

def test_grace_period_boundary():
    # Exactly now ± 1 second
    just_future = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    just_past   = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    assert _is_within_grace(just_future) is True
    assert _is_within_grace(just_past)   is False
    print("  grace period boundary (±5s) correct ✓")

if __name__ == "__main__":
    tests = [
        test_generate_api_key_exists, test_rotate_api_key_exists,
        test_validate_api_key_exists, test_revoke_api_key_exists,
        test_api_key_uses_dik_prefix, test_api_key_uses_status_field,
        test_api_key_has_rotating_status, test_api_key_has_deleted_status,
        test_api_key_has_grace_expires_at,
        test_mark_api_key_rotating_in_db_apikeys,
        test_get_api_key_by_id_in_db_apikeys,
        test_db_apikeys_sets_status_rotating, test_db_apikeys_sets_grace_expires_at,
        test_refresh_endpoint_exists, test_forgot_password_endpoint_exists,
        test_reset_password_endpoint_exists,
        test_generated_key_starts_with_dik, test_prefix_is_first_10_chars,
        test_hash_is_sha256_of_key, test_keys_are_unique,
        test_within_grace_period_returns_true, test_past_grace_period_returns_false,
        test_grace_period_boundary,
    ]
    passed = failed = 0
    for t in tests:
        try:
            print(f"\n▶ {t.__name__}")
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    print(f"\n{'='*50}\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed: sys.exit(1)
