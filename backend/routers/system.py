"""
Endpoints: GET /   GET /health   GET /usage   GET /tasks/{task_id}
           GET /llm/available-models 
"""

from datetime import datetime
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from db import supabase
from ingestion import get_embed_model
from llm.engine import LLM_PROVIDER, LLM_MODEL
from llm.fallback import SUPPORTED_PROVIDERS, get_fallback_chain
from core.config import config as app_config
from core.queue import task_queue
from core.auth import get_current_user_context, get_user_id, UserContext
from db_usage import get_user_usage
router = APIRouter(tags=["System"])

# ---------------------------------------------------------------------------
# Model catalogue — update as providers release new models
# ---------------------------------------------------------------------------

_MODEL_CATALOGUE: dict[str, list[str]] = {
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
def get_usage(user: UserContext = Depends(get_current_user_context)):
    return get_user_usage(get_user_id(user))


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Poll for async task completion.
    Used by the upload UI to check ingestion progress (Contract 5).
    Returns 404 if task_id not found.
    Returns 503 if task queue is not running.
    """
    status = task_queue.get_status(task_id)
    if not status:
        return JSONResponse(
            status_code=404,
            content={"error": True, "message": "Task not found", "code": "TASK_001"}
        )
    return status


@router.get("/llm/available-models")
def get_available_models():
    """
    Return all providers from the fallback chain with their available models.

    Each provider entry shows:
    - provider:   name string
    - configured: True (all providers here have a valid API key — chain skips unconfigured ones)
    - active:     True if this provider is in the resolved fallback chain
    - models:     list of known model strings for this provider
    - current_model: the model configured for this provider in the fallback chain
                     (None if provider is not in the active chain)

    Also returns:
    - fallback_chain: ordered list of {provider, model} showing the active chain
    - primary:        {provider, model} — the first entry in the chain

    This endpoint is always safe to call — errors resolve to empty lists,
    never 500s.
    """
    # Resolve the active fallback chain
    try:
        active_chain = get_fallback_chain()  # [(provider, model), ...]
    except Exception:
        active_chain = []

    # Build a lookup: provider -> model from the active chain
    chain_model_map: dict[str, str] = {p: m for p, m in active_chain}

    # Key lookup: which providers have API keys in config
    key_map = {
        "groq":      app_config.groq_api_key,
        "openai":    app_config.openai_api_key,
        "anthropic": app_config.anthropic_api_key,
    }

    providers = []
    for provider in sorted(SUPPORTED_PROVIDERS):
        has_key = bool(key_map.get(provider, ""))
        in_chain = provider in chain_model_map
        providers.append({
            "provider":      provider,
            "configured":    has_key,
            "active":        in_chain,
            "current_model": chain_model_map.get(provider),
            "models":        _MODEL_CATALOGUE.get(provider, []),
        })

    fallback_chain = [
        {"provider": p, "model": m} for p, m in active_chain
    ]

    primary = fallback_chain[0] if fallback_chain else None

    return {
        "providers":      providers,
        "fallback_chain": fallback_chain,
        "primary":        primary,
    }