"""
main.py
FastAPI application entry point.

Responsibilities:
- Create the app instance
- Register API-key auth middleware
- Mount all routers
- Run startup warmup (embedding model pre-load)

All route logic lives in routers/. Nothing else belongs here.
"""

import os
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from routers import system, documents, query, extraction, export, integration

load_dotenv()

# ── Auth ──────────────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key when provided.
    No key = UI/browser mode — allowed through.
    Invalid key = 401.
    """
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


@app.on_event("startup")
async def warmup():
    """Pre-load the embedding model so the first request isn't slow."""
    print("[startup] Warming up embedding model...")
    from ingestion import get_embed_model
    get_embed_model()
    print("[startup] Embedding model ready.")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(extraction.router)
app.include_router(export.router)
app.include_router(integration.router)
