"""
main.py
FastAPI application entry point.

Responsibilities:
- Create the app instance
- Register CORS middleware
- Register API-key auth middleware
- Mount all routers
- Run startup warmup (embedding model pre-load + task queue start)

All route logic lives in routers/. Nothing else belongs here.
"""

import os
from fastapi import FastAPI, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# CORS — open during initial deploy; tighten to Streamlit Cloud domain after
# first successful end-to-end test on live URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warmup():
    """
    Server startup tasks:
    1. Start the async task queue.
       Fails silently so the server starts regardless of whether queue is ready.
    2. Pre-load the embedding model so the first request isn't slow.
    """
    # Task queue — guarded import; queue.py may not exist yet
    try:
        from core.queue import task_queue
        task_queue.start()
        print("[startup] Task queue started.")
    except ImportError:
        print("[startup] Task queue not available yet — skipping.")
    except Exception as exc:
        print(f"[startup] Task queue failed to start: {exc} — continuing.")

    # Embedding model warmup
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