import base64

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from core.responses import bad_request, internal_error
from core.logger import get_logger
from retrieval import query_document, query_document_stream, compress_history
from db import get_chat_history, save_message 

 
logger = get_logger("routers.query")

router = APIRouter(tags=["Query"])


# ── Input models ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None
    document_ids: list[str] | None = None
    history: list[dict] = []
    history_summary: str = ""

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SaveChatRequest(BaseModel):
    role: str
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
def query(req: QueryRequest):
    """
    Answer a question, optionally grounded in one or more documents.
    Uses hybrid search + Cohere reranking + LLM generation.
    Returns answer text and source chunk references.
    """
    try:
        return query_document(
            req.question,
            req.document_id,
            req.document_ids,
            req.history,
            req.history_summary,
        )
    except Exception as exc:
        return internal_error(f"Query failed: {exc}")


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Streaming version of /query.
    Tokens are base64-encoded SSE events; ends with data: [DONE].
    """
    def event_stream():
        try:
            for token in query_document_stream(
                req.question,
                req.document_id,
                req.document_ids,
                req.history,
                req.history_summary,
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
def compress(req: CompressRequest):
    """
    Summarise a conversation history into a compact string.
    Used by the frontend to manage context-window size.
    """
    try:
        summary = compress_history(req.messages)
        return {"summary": summary}
    except Exception as exc:
        return internal_error(f"Compression failed: {exc}")