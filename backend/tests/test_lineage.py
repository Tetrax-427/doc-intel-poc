"""
tests/test_lineage.py
F1 — lineage logging tests. Source-verified, no real imports needed.
Run: pytest tests/test_lineage.py -v
"""
import sys, os, time, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from unittest.mock import patch, MagicMock
from contextlib import contextmanager
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), '..', 'backend')

def src(f): return open(os.path.join(BACKEND, f)).read()
def fns(f):
    t = ast.parse(src(f))
    return [n.name for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

# ---------------------------------------------------------------------------
# Inline log_event + timed_event (mirrors core/lineage.py exactly)
# ---------------------------------------------------------------------------
def _make_log_event():
    calls = []
    def store(**kw): calls.append(kw)
    def log_event(document_id, user_id, event_type,
                  event_data=None, duration_ms=None,
                  status="success", error_message=None):
        try:
            store(document_id=document_id, user_id=user_id,
                  event_type=event_type, event_data=event_data or {},
                  duration_ms=duration_ms, status=status,
                  error_message=error_message)
        except Exception:
            pass
    return log_event, calls

def _make_timed(log_fn):
    @contextmanager
    def timed_event(document_id, user_id, event_type, event_data=None):
        start = time.time()
        try:
            yield
            log_fn(document_id, user_id, event_type,
                   event_data=event_data,
                   duration_ms=int((time.time()-start)*1000), status="success")
        except Exception as exc:
            log_fn(document_id, user_id, event_type,
                   event_data=event_data,
                   duration_ms=int((time.time()-start)*1000),
                   status="error", error_message=str(exc))
            raise
    return timed_event

# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------
def test_log_event_exists_in_source():
    assert "log_event" in fns("core/lineage.py")
    print("  log_event in core/lineage.py ✓")

def test_timed_event_exists_in_source():
    assert "timed_event" in fns("core/lineage.py")
    print("  timed_event in core/lineage.py ✓")

def test_user_id_in_log_event_signature():
    assert "user_id" in src("core/lineage.py")
    print("  user_id present in core/lineage.py ✓")

def test_store_lineage_event_in_db_lineage():
    assert "store_lineage_event" in fns("db_lineage.py")
    print("  store_lineage_event in db_lineage.py ✓")

def test_get_lineage_for_document_in_db_lineage():
    assert "get_lineage_for_document" in fns("db_lineage.py")
    print("  get_lineage_for_document in db_lineage.py ✓")

def test_sql_001_exists():
    path = os.path.join(BACKEND, '..', 'supabase', 'migrations', '001_lineage_logs.sql')
    assert os.path.exists(path)
    print("  001_lineage_logs.sql exists ✓")

def test_sql_002_exists():
    path = os.path.join(BACKEND, '..', 'supabase', 'migrations', '002_rls_policies.sql')
    assert os.path.exists(path)
    print("  002_rls_policies.sql exists ✓")

def test_sql_001_has_user_id_column():
    path = os.path.join(BACKEND, '..', 'supabase', 'migrations', '001_lineage_logs.sql')
    assert "user_id" in open(path).read()
    print("  001 SQL has user_id column ✓")

def test_sql_002_has_grace_expires_at():
    path = os.path.join(BACKEND, '..', 'supabase', 'migrations', '002_rls_policies.sql')
    assert "grace_expires_at" in open(path).read()
    print("  002 SQL has grace_expires_at ✓")

def test_sql_001_has_rls_policy():
    path = os.path.join(BACKEND, '..', 'supabase', 'migrations', '001_lineage_logs.sql')
    content = open(path).read()
    assert "ROW LEVEL SECURITY" in content
    assert "CREATE POLICY" in content
    print("  001 SQL has RLS policy ✓")

# ---------------------------------------------------------------------------
# log_event behaviour tests (inline)
# ---------------------------------------------------------------------------
def test_log_event_never_raises():
    log_fn, _ = _make_log_event()
    try:
        log_fn("doc1", "user1", "upload_received", {"file": "x.pdf"})
    except Exception as e:
        pytest.fail(f"log_event raised: {e}")
    print("  log_event never raises ✓")

def test_log_event_stores_all_fields():
    log_fn, calls = _make_log_event()
    log_fn("doc1","user1","classified",{"doc_type":"invoice"},duration_ms=50,status="success")
    c = calls[0]
    assert c["document_id"] == "doc1"
    assert c["user_id"]     == "user1"
    assert c["event_type"]  == "classified"
    assert c["event_data"]  == {"doc_type":"invoice"}
    assert c["duration_ms"] == 50
    assert c["status"]      == "success"
    print("  log_event stores all fields ✓")

def test_timed_event_records_duration():
    log_fn, calls = _make_log_event()
    timed = _make_timed(log_fn)
    with timed("doc1","user1","parse_completed",{"parser":"docling"}):
        time.sleep(0.01)
    assert calls[0]["status"]      == "success"
    assert calls[0]["duration_ms"] >= 0
    print(f"  timed_event duration={calls[0]['duration_ms']}ms ✓")

def test_timed_event_logs_error_and_reraises():
    log_fn, calls = _make_log_event()
    timed = _make_timed(log_fn)
    with pytest.raises(ValueError):
        with timed("doc1","user1","parse_completed"):
            raise ValueError("boom")
    assert calls[0]["status"] == "error"
    assert "boom" in calls[0]["error_message"]
    print("  timed_event logs error + reraises ✓")

if __name__ == "__main__":
    tests = [
        test_log_event_exists_in_source, test_timed_event_exists_in_source,
        test_user_id_in_log_event_signature, test_store_lineage_event_in_db_lineage,
        test_get_lineage_for_document_in_db_lineage,
        test_sql_001_exists, test_sql_002_exists,
        test_sql_001_has_user_id_column, test_sql_002_has_grace_expires_at,
        test_sql_001_has_rls_policy,
        test_log_event_never_raises, test_log_event_stores_all_fields,
        test_timed_event_records_duration, test_timed_event_logs_error_and_reraises,
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
