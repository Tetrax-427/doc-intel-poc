"""
Endpoints: GET /   GET /health   GET /usage   GET /tasks/{task_id}
"""

from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from db import supabase
from ingestion import get_embed_model
from llm.engine import LLM_PROVIDER, LLM_MODEL
from llm.usage import get_usage_summary
from core.queue import task_queue

router = APIRouter(tags=["System"])


@router.get("/")
def root():
    return {"status": "DocIntel API running", "version": "1.0.0"}


@router.get("/health")
def health_check():
    checks: dict = {}
    try:
        supabase.table("documents").select("id").limit(1).execute()
        checks["database"] = {"status": "healthy"}
    except Exception as exc:
        checks["database"] = {"status": "unhealthy", "error": str(exc)}

    try:
        get_embed_model()
        checks["embeddings"] = {"status": "healthy"}
    except Exception as exc:
        checks["embeddings"] = {"status": "unhealthy", "error": str(exc)}

    try:
        checks["llm"] = {"status": "configured", "provider": LLM_PROVIDER, "model": LLM_MODEL}
    except Exception as exc:
        checks["llm"] = {"status": "misconfigured", "error": str(exc)}

    all_healthy = all(
        v.get("status") in ("healthy", "configured") for v in checks.values()
    )

    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "healthy": all_healthy,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": checks,
        },
    )


@router.get("/usage")
def get_usage():
    return get_usage_summary()


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Poll for async task completion.
    Used by the upload UI to check ingestion progress (Contract 5).
    Returns 404 if task_id not found.
    Returns 503 if task queue is not running (Dev 1 not deployed yet).
    """

    status = task_queue.get_status(task_id)
    if not status:
        return JSONResponse(
            status_code=404,
            content={"error": True, "message": "Task not found", "code": "TASK_001"}
        )
    return status