"""
routers/system.py
Endpoints: GET /   GET /health   GET /usage

Health check verifies every critical subsystem and returns a single
"healthy" bool so load balancers and monitoring tools can act on it.
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter(tags=["System"])


@router.get("/")
def root():
    return {"status": "DocIntel API running", "version": "1.0.0"}


@router.get("/health")
def health_check():
    """
    Comprehensive health check.
    Returns HTTP 200 with healthy=true when all critical systems are up.
    Returns HTTP 503 with healthy=false if database or embeddings are down.
    LLM is config-only (we don't make a live call — too slow for a health probe).
    """
    checks: dict = {}

    # ── Supabase ──────────────────────────────────────────────────────────────
    try:
        from db import supabase
        supabase.table("documents").select("id").limit(1).execute()
        checks["database"] = {"status": "healthy"}
    except Exception as exc:
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    # ── Embedding model ───────────────────────────────────────────────────────
    try:
        from ingestion import get_embed_model
        get_embed_model()  # cached after first call — fast on subsequent checks
        checks["embeddings"] = {"status": "healthy"}
    except Exception as exc:
        checks["embeddings"] = {"status": "unhealthy", "error": str(exc)}

    # ── LLM (config check only) ───────────────────────────────────────────────
    try:
        from llm.engine import LLM_PROVIDER, LLM_MODEL
        checks["llm"] = {
            "status": "configured",
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
        }
    except Exception as exc:
        checks["llm"] = {"status": "misconfigured", "error": str(exc)}

    all_healthy = all(
        v.get("status") in ("healthy", "configured")
        for v in checks.values()
    )

    status_code = 200 if all_healthy else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "healthy": all_healthy,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": checks,
        },
    )


@router.get("/usage")
def get_usage():
    """Return LLM token usage summary from usage_logs table."""
    from llm.usage import get_usage_summary
    return get_usage_summary()
