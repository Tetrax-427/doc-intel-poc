"""
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
from core.queue import task_queue
from ingestion import get_embed_model
    
from routers import system, documents, query, extraction, export, integration

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

# Fix 5 — CORS: specific domain in prod, wildcard only in dev
_streamlit_url = os.getenv("STREAMLIT_URL", "").strip()
ALLOWED_ORIGINS = (
    [
        _streamlit_url,
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    if _streamlit_url
    else ["*"]   # dev fallback — set STREAMLIT_URL in Railway to lock down
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=bool(_streamlit_url),  # only send credentials in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warmup():
    # Task queue — guarded import; queue.py may not exist yet
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


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(system.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(extraction.router)
app.include_router(export.router)
app.include_router(integration.router)