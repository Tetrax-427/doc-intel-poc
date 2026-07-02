"""
routers/query.py
Query endpoints.

Changes in this phase:
  - Switch to get_current_user_context()
  - Rate limiting on /query and /query/stream
  - org_id/team_id threaded through to retrieval layer
  - LLM output already sanitized in retrieval.py — no extra step needed here
"""

import base64

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from core.responses import bad_request, internal_error
from core.auth import get_current_user_context, get_user_id, UserContext
from core.rate_limiter import check_rate_limit
from core.logger import get_logger
from retrieval import query_document, query_document_stream, compress_history
from db import get_chat_history, save_message
from hyde import VALID_RETRIEVAL_MODES

logger = get_logger("routers.query")
router = APIRouter(tags=["Query"])


# ── Input models ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:        str
    document_id:     str | None = None
    document_ids:    list[str] | None = None
    history:         list[dict] = []
    history_summary: str = ""
    provider:        str | None = None
    model:           str | None = None
    retrieval_mode:  str = "standard"

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @validator("model", always=True)
    def provider_and_model_must_be_paired(cls, model, values):
        provider = values.get("provider")
        if bool(provider) != bool(model):
            raise ValueError(
                "provider and model must be supplied together — "
                "set both or neither for a per-call override."
            )
        return model

    @validator("retrieval_mode", always=True)
    def retrieval_mode_valid(cls, v):
        normalised = v.strip().lower() if v else "standard"
        if normalised == "none":
            return "standard"
        if normalised not in VALID_RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of: {', '.join(sorted(VALID_RETRIEVAL_MODES))}. "
                f"Got '{v}'."
            )
        return normalised


class SaveChatRequest(BaseModel):
    role:    str
    content: str
    sources: list[dict] = []

    @validator("role")
    def valid_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @validator("content")
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("content cannot be empty")
        return v


class CompressRequest(BaseModel):
    messages: list[dict]

    @validator("messages")
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query")
def query(
    req: QueryRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    # Rate limit
    check_rate_limit(user_id=uid, endpoint="query")

    try:
        return query_document(
            req.question,
            req.document_id,
            req.document_ids,
            req.history,
            req.history_summary,
            provider=req.provider,
            model=req.model,
            retrieval_mode=req.retrieval_mode,
            user_id=uid,
            org_id=str(user.org_id)  if user.org_id  else None,
            team_id=str(user.team_id) if user.team_id else None,
        )
    except Exception as exc:
        return internal_error(f"Query failed: {exc}")


@router.post("/query/stream")
def query_stream(
    req: QueryRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    # Rate limit — same limit as /query
    check_rate_limit(user_id=uid, endpoint="query")

    def event_stream():
        try:
            for token in query_document_stream(
                req.question,
                req.document_id,
                req.document_ids,
                req.history,
                req.history_summary,
                provider=req.provider,
                model=req.model,
                retrieval_mode=req.retrieval_mode,
                user_id=uid,
                org_id=str(user.org_id)  if user.org_id  else None,
                team_id=str(user.team_id) if user.team_id else None,
            ):
                encoded = base64.b64encode(token.encode()).decode()
                yield f"data: {encoded}\n\n"
        except GeneratorExit:
            pass
        except Exception as exc:
            logger.error("Stream error", error=str(exc))
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chats/{document_id}")
def get_chats(document_id: str):
    """Return full chat history for a document."""
    return get_chat_history(document_id)


@router.post("/chats/{document_id}")
def save_chat(document_id: str, body: SaveChatRequest):
    """Persist a single chat message."""
    save_message(document_id, body.role, body.content, body.sources)
    return {"status": "saved"}


@router.post("/compress")
def compress(
    req: CompressRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)
    try:
        summary = compress_history(req.messages, user_id=uid)
        return {"summary": summary}
    except Exception as exc:
        return internal_error(f"Compression failed: {exc}")