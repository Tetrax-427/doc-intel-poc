"""
FastAPI application entry point.

F2: Added POST /api-keys/{key_id}/rotate endpoint.
F3: CORS origins now driven by CORS_ALLOWED_ORIGINS env var (replaces
    wildcard "*" in production).  Falls back to ["*"] in dev mode
    (no CORS_ALLOWED_ORIGINS set AND no SUPABASE_JWT_SECRET set).
"""

import os
from fastapi import FastAPI, Security, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from core.queue import task_queue
from ingestion import get_embed_model
from routers.auth import router as auth_router
from routers import system, documents, query, extraction, export, integration
from routers.lineage import router as lineage_router   # F1 — /documents/{id}/lineage

load_dotenv()

# ── Auth ──────────────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        return None
    from api_keys import validate_api_key
    is_valid, reason = validate_api_key(api_key)
    if not is_valid:
        raise HTTPException(status_code=401, detail=reason)
    return api_key


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocIntel API",
    description="Document intelligence — upload, query, extract, classify.",
    version="1.0.0",
)

# ── F3 — CORS hardening ───────────────────────────────────────────────────────
#
# Priority order for determining allowed origins:
#   1. CORS_ALLOWED_ORIGINS  — explicit comma-separated list (recommended).
#   2. STREAMLIT_URL         — legacy single-origin var (backward compat).
#   3. Dev fallback           — wildcard "*" when JWT auth is not configured.
#
# Production deployments MUST set CORS_ALLOWED_ORIGINS.
# Example (Railway env var):
#   CORS_ALLOWED_ORIGINS=https://myapp.railway.app,https://myapp.com
#
# Wildcard ("*") with allow_credentials=True is invalid per the CORS spec
# and rejected by browsers.  We only set allow_credentials=True when
# specific origins are configured.

from core.config import config as app_config

_cors_origins = app_config.get_cors_origins()

# Detect dev mode — no JWT secret configured
_is_dev_mode = not os.getenv("SUPABASE_JWT_SECRET", "").strip()

# In strict dev mode with no origins configured fall back to wildcard
_use_wildcard = _is_dev_mode and _cors_origins == ["http://localhost:8501"]

ALLOWED_ORIGINS    = ["*"] if _use_wildcard else _cors_origins
_allow_credentials = not _use_wildcard   # credentials require specific origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def warmup():
    try:
        task_queue.start()
        print("[startup] Task queue started.")
    except ImportError:
        print("[startup] Task queue not available yet — skipping.")
    except Exception as exc:
        print(f"[startup] Task queue failed to start: {exc} — continuing.")

    print("[startup] Warming up embedding model...")
    get_embed_model()
    print("[startup] Embedding model ready.")

    print(f"[startup] CORS — origins: {ALLOWED_ORIGINS}")
    print(f"[startup] Dev mode: {_is_dev_mode}")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(extraction.router)
app.include_router(export.router)
app.include_router(integration.router)
app.include_router(auth_router)
app.include_router(lineage_router)    # F1 — /documents/{id}/lineage


# ── F2 — API key rotation endpoint ───────────────────────────────────────────
#
# Placed here (not in a dedicated router) because it's a one-off management
# endpoint that doesn't justify a new router file.

class CreateApiKeyRequest(BaseModel):
    name: str
    rate_limit: int = 100


@app.post("/api-keys", tags=["API Keys"])
def create_api_key_endpoint(
    req: CreateApiKeyRequest,
    user=Depends(verify_api_key),
):
    """Create a new API key."""
    from api_keys import create_api_key
    return create_api_key(req.name, req.rate_limit)


@app.get("/api-keys", tags=["API Keys"])
def list_api_keys_endpoint(user=Depends(verify_api_key)):
    """List all API keys (hashes excluded)."""
    from api_keys import list_api_keys
    return list_api_keys()


@app.post("/api-keys/{key_id}/rotate", tags=["API Keys"])
def rotate_api_key_endpoint(key_id: str, user=Depends(verify_api_key)):
    """
    F2 — Rotate an API key.

    Generates a new key, marks the old one as inactive but keeps it valid
    for the configured grace period (default 24 hours) so existing
    integrations have time to update.

    Response includes the new key (shown once only) and the grace expiry.
    """
    from api_keys import rotate_api_key
    try:
        return rotate_api_key(key_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api-keys/{key_id}", tags=["API Keys"])
def revoke_api_key_endpoint(key_id: str, user=Depends(verify_api_key)):
    """Immediately revoke an API key. No grace period."""
    from api_keys import revoke_api_key
    revoke_api_key(key_id)
    return {"status": "revoked", "key_id": key_id}