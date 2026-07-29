"""
FastAPI application entry point.

Changes in this phase (Security + Org/Team):
  - Startup env var check — server refuses to start if REQUIRED_ENV_VARS missing
  - Registered new routers: admin, orgs, usage
  - Org-scoped API key support in existing api-keys endpoints

Existing:
  F2: POST /api-keys/{key_id}/rotate
  F3: CORS origins driven by CORS_ALLOWED_ORIGINS env var
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
from routers.lineage import router as lineage_router
from routers.admin import router as admin_router
from routers.orgs import router as orgs_router
from routers.usage import router as usage_router
from routers.llm_observability import router as llm_observability_router
from routers import comparison
from routers.agents import router as agents_router


load_dotenv()

# ── Startup env check ─────────────────────────────────────────────────────────
# Fail fast if required env vars are missing.
# This runs at import time so Railway deployment fails visibly rather than
# serving broken requests.

from core.config import REQUIRED_ENV_VARS as _REQUIRED_ENV_VARS

_missing = [k for k in _REQUIRED_ENV_VARS if not os.getenv(k, "").strip()]
if _missing:
    raise RuntimeError(
        f"[startup] Missing required environment variables: {_missing}. "
        f"Server cannot start. Set these in your .env / Railway env vars."
    )

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

# ── CORS ──────────────────────────────────────────────────────────────────────

from core.config import config as app_config

_cors_origins = app_config.get_cors_origins()
_is_dev_mode  = not os.getenv("SUPABASE_JWT_SECRET", "").strip()
_use_wildcard = _is_dev_mode and _cors_origins == ["http://localhost:8501"]

ALLOWED_ORIGINS    = ["*"] if _use_wildcard else _cors_origins
_allow_credentials = not _use_wildcard

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Developer-Key"],
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
    print(f"[startup] Required env vars: OK")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(extraction.router)
app.include_router(export.router)
app.include_router(integration.router)
app.include_router(auth_router)
app.include_router(lineage_router)
app.include_router(admin_router)
app.include_router(orgs_router)
app.include_router(usage_router)
app.include_router(llm_observability_router)
app.include_router(comparison.router)
app.include_router(agents_router)
# ── API Key endpoints ─────────────────────────────────────────────────────────

class CreateApiKeyRequest(BaseModel):
    name:       str
    rate_limit: int = 100
    scope:      str = "personal"   # 'personal' | 'org'
    org_id:     str | None = None  # required when scope='org'


@app.post("/api-keys", tags=["API Keys"])
def create_api_key_endpoint(
    req: CreateApiKeyRequest,
    user=Depends(verify_api_key),
):
    """Create a new API key. Set scope='org' + org_id for org-scoped keys."""
    from api_keys import create_api_key
    return create_api_key(
        name=req.name,
        rate_limit=req.rate_limit,
        scope=req.scope,
        org_id=req.org_id,
    )


@app.get("/api-keys", tags=["API Keys"])
def list_api_keys_endpoint(user=Depends(verify_api_key)):
    from api_keys import list_api_keys
    return list_api_keys()


@app.post("/api-keys/{key_id}/rotate", tags=["API Keys"])
def rotate_api_key_endpoint(key_id: str, user=Depends(verify_api_key)):
    """
    Rotate an API key. Old key stays valid for the configured grace period.
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