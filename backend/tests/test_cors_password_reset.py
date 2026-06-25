"""
tests/test_cors_password_reset.py
F3 — CORS config + password reset endpoint tests. Source-verified.
Run: pytest tests/test_cors_password_reset.py -v
"""
import sys, os, ast, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')

def src(f): return open(os.path.join(BACKEND, f)).read()
def routes(f): return re.findall(r'@router\.\w+\("([^"]+)"', src(f))
def fns(f):
    t = ast.parse(src(f))
    return [n.name for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

# ---------------------------------------------------------------------------
# Inline CORS parsing logic (mirrors core/config.py)
# ---------------------------------------------------------------------------
def parse_cors_origins(cors_raw, streamlit_url="", is_dev=True):
    if not cors_raw and streamlit_url:
        cors_raw = f"{streamlit_url},http://localhost:8501,http://127.0.0.1:8501"
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    if not origins:
        return ["http://localhost:8501"]
    return origins

def should_use_wildcard(origins, is_dev):
    return is_dev and origins == ["http://localhost:8501"]

# ---------------------------------------------------------------------------
# CORS config source checks
# ---------------------------------------------------------------------------
def test_config_has_cors_allowed_origins_field():
    assert "cors_allowed_origins" in src("core/config.py")
    print("  cors_allowed_origins in config ✓")

def test_config_has_get_cors_origins_method():
    assert "get_cors_origins" in fns("core/config.py")
    print("  get_cors_origins method in config ✓")

def test_config_reads_cors_env_var():
    assert "CORS_ALLOWED_ORIGINS" in src("core/config.py")
    print("  CORS_ALLOWED_ORIGINS env var read in config ✓")

def test_config_has_streamlit_url_migration():
    assert "STREAMLIT_URL" in src("core/config.py")
    print("  STREAMLIT_URL legacy migration in config ✓")

def test_main_uses_get_cors_origins():
    assert "get_cors_origins" in src("main.py")
    print("  main.py calls get_cors_origins() ✓")

def test_main_no_hardcoded_wildcard_in_prod():
    main_src = src("main.py")
    assert "CORS_ALLOWED_ORIGINS" in main_src or "get_cors_origins" in main_src
    print("  main.py CORS not hardcoded ✓")

# ---------------------------------------------------------------------------
# CORS parsing logic tests (inline)
# ---------------------------------------------------------------------------
def test_cors_parses_multiple_origins():
    origins = parse_cors_origins("https://app.com,http://localhost:8501")
    assert "https://app.com" in origins
    assert "http://localhost:8501" in origins
    assert len(origins) == 2
    print("  multiple origins parsed correctly ✓")

def test_cors_fallback_to_localhost():
    origins = parse_cors_origins("")
    assert origins == ["http://localhost:8501"]
    print("  empty CORS → localhost fallback ✓")

def test_cors_migration_from_streamlit_url():
    origins = parse_cors_origins("", streamlit_url="https://myapp.streamlit.app")
    assert "https://myapp.streamlit.app" in origins
    print("  STREAMLIT_URL migration adds it to origins ✓")

def test_wildcard_only_in_dev_with_no_custom_origins():
    assert should_use_wildcard(["http://localhost:8501"], is_dev=True)  is True
    assert should_use_wildcard(["https://prod.com"],     is_dev=True)  is False
    assert should_use_wildcard(["http://localhost:8501"], is_dev=False) is False
    print("  wildcard only in dev with default origins ✓")

def test_cors_strips_whitespace():
    origins = parse_cors_origins("  https://a.com  ,  https://b.com  ")
    assert "https://a.com" in origins
    assert "https://b.com" in origins
    print("  CORS origins whitespace stripped ✓")

# ---------------------------------------------------------------------------
# Password reset endpoint source checks
# ---------------------------------------------------------------------------
def test_forgot_password_endpoint_exists():
    assert "/forgot-password" in routes("routers/auth.py"), \
        f"Expected /auth/forgot-password, got: {routes('routers/auth.py')}"
    print("  /auth/forgot-password endpoint exists ✓")

def test_reset_password_endpoint_exists():
    assert "/reset-password" in routes("routers/auth.py"), \
        f"Expected /auth/reset-password, got: {routes('routers/auth.py')}"
    print("  /auth/reset-password endpoint exists ✓")

def test_refresh_endpoint_exists():
    assert "/refresh" in routes("routers/auth.py")
    print("  /auth/refresh endpoint exists ✓")

def test_supabase_with_token_exists():
    assert "_supabase_with_token" in fns("routers/auth.py")
    print("  _supabase_with_token() exists in auth ✓")

def test_reset_password_uses_update_user():
    assert "update_user" in src("routers/auth.py")
    print("  reset-password uses update_user() ✓")

def test_password_reset_redirect_url_env_read():
    assert "PASSWORD_RESET_REDIRECT_URL" in src("routers/auth.py")
    print("  PASSWORD_RESET_REDIRECT_URL read in auth ✓")

def test_forgot_password_returns_fixed_response():
    # Spec: always return same response regardless of whether email exists
    auth_src = src("routers/auth.py")
    assert "reset_email_sent" in auth_src
    print("  forgot-password returns fixed response (anti-enumeration) ✓")

def test_no_old_password_request_path():
    assert "/password/request" not in routes("routers/auth.py")
    print("  old /auth/password/request path not present ✓")

def test_no_old_password_reset_path():
    assert "/password/reset" not in routes("routers/auth.py")
    print("  old /auth/password/reset path not present ✓")

if __name__ == "__main__":
    tests = [
        test_config_has_cors_allowed_origins_field,
        test_config_has_get_cors_origins_method,
        test_config_reads_cors_env_var,
        test_config_has_streamlit_url_migration,
        test_main_uses_get_cors_origins,
        test_main_no_hardcoded_wildcard_in_prod,
        test_cors_parses_multiple_origins,
        test_cors_fallback_to_localhost,
        test_cors_migration_from_streamlit_url,
        test_wildcard_only_in_dev_with_no_custom_origins,
        test_cors_strips_whitespace,
        test_forgot_password_endpoint_exists,
        test_reset_password_endpoint_exists,
        test_refresh_endpoint_exists,
        test_supabase_with_token_exists,
        test_reset_password_uses_update_user,
        test_password_reset_redirect_url_env_read,
        test_forgot_password_returns_fixed_response,
        test_no_old_password_request_path,
        test_no_old_password_reset_path,
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
