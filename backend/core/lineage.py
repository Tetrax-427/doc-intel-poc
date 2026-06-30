"""
F1 — Document lineage logging.

Every meaningful state change on a document is recorded to the
``lineage_logs`` Supabase table.  The log is append-only.

Public API:
    log_event()       — fire-and-forget, never raises
    timed_event()     — context manager, records duration_ms + status
    Convenience wrappers: log_uploaded, log_parsed, log_classified,
    log_chunked, log_queried, log_deleted, log_extraction_started,
    log_extraction_completed, log_corrected, log_classification_overridden,
    log_split, log_export
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from enum import Enum
from core.logger import get_logger

logger = get_logger("lineage")


# ---------------------------------------------------------------------------
# Event type catalogue
# ---------------------------------------------------------------------------

class LineageEvent(str, Enum):
    DOCUMENT_UPLOADED         = "upload_received"
    DOCUMENT_PARSED           = "parse_completed"
    DOCUMENT_CLASSIFIED       = "classified"
    DOCUMENT_CHUNKED          = "chunks_stored"
    DOCUMENT_QUERIED          = "document.queried"
    DOCUMENT_DELETED          = "deleted"
    DOCUMENT_REPROCESSED      = "reprocessed"
    EXTRACTION_STARTED        = "extraction.started"
    EXTRACTION_COMPLETED      = "extraction_run"
    EXTRACTION_CORRECTED      = "extraction.corrected"
    CLASSIFICATION_OVERRIDDEN = "classification.overridden"
    SPLIT_COMPLETED           = "split_completed"
    EXPORT_RUN                = "export_run"
    SUMMARIZED                = "summarized"


# ---------------------------------------------------------------------------
# Internal store — lazy import to avoid circular deps
# ---------------------------------------------------------------------------

_store_fn = None


def _get_store():
    global _store_fn
    if _store_fn is None:
        from db_audit import store_lineage_event
        _store_fn = store_lineage_event
    return _store_fn


# ---------------------------------------------------------------------------
# Core log_event — fire-and-forget, never raises
# ---------------------------------------------------------------------------

def log_event(
    document_id: str,
    user_id: str,
    event_type: LineageEvent | str,
    event_data: dict | None = None,
    duration_ms: int | None = None,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    """
    Append a lineage event to the ``lineage_logs`` table.

    Never raises — a lineage failure must never break the main request flow.

    Args:
        document_id:   Document this event belongs to.
        user_id:       User who triggered the event ("system" for automated steps).
        event_type:    LineageEvent enum value or raw string.
        event_data:    Small JSON-serialisable dict of extra context.
        duration_ms:   Elapsed time in ms (set automatically by timed_event).
        status:        "success" | "error"
        error_message: Error detail (set automatically by timed_event on exception).
    """
    try:
        _get_store()(
            document_id=document_id,
            user_id=user_id,
            event_type=str(event_type),
            event_data=event_data or {},
            duration_ms=duration_ms,
            status=status,
            error_message=error_message,
        )
        logger.info(
            "Lineage event recorded",
            document_id=document_id,
            user_id=user_id,
            event_type=str(event_type),
            status=status,
        )
    except Exception as exc:
        logger.warning(
            "Lineage log write failed (non-fatal)",
            document_id=document_id,
            event_type=str(event_type),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# timed_event — context manager for duration-tracked events
# ---------------------------------------------------------------------------

@contextmanager
def timed_event(
    document_id: str,
    user_id: str,
    event_type: LineageEvent | str,
    event_data: dict | None = None,
):
    """
    Context manager that wraps a block of work and records:
      - duration_ms: elapsed wall time in milliseconds
      - status: "success" on normal exit, "error" on exception
      - error_message: str(exception) on failure

    The exception is always re-raised.

    Usage:
        with timed_event(doc_id, user_id, LineageEvent.DOCUMENT_PARSED,
                         event_data={"parser_used": "docling"}):
            document = parser.parse(file_path)
    """
    start = time.perf_counter()
    try:
        yield
        duration_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            document_id, user_id, event_type,
            event_data=event_data,
            duration_ms=duration_ms,
            status="success",
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            document_id, user_id, event_type,
            event_data=event_data,
            duration_ms=duration_ms,
            status="error",
            error_message=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Convenience wrappers — all accept user_id
# ---------------------------------------------------------------------------

def log_uploaded(
    document_id: str,
    user_id: str,
    filename: str,
    file_size_bytes: int = 0,
    content_type: str = "",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_UPLOADED,
        event_data={
            "file_name":       filename,
            "file_size_bytes": file_size_bytes,
            "content_type":    content_type,
        },
    )


def log_parsed(
    document_id: str,
    user_id: str = "system",
    page_count: int = 0,
    parser_used: str = "",
    is_scanned: bool = False,
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_PARSED,
        event_data={
            "page_count":  page_count,
            "parser_used": parser_used,
            "is_scanned":  is_scanned,
        },
    )


def log_classified(
    document_id: str,
    user_id: str = "system",
    doc_type: str = "general",
    confidence: float = 0.0,
    stage_used: str = "stage2",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_CLASSIFIED,
        event_data={
            "doc_type":   doc_type,
            "confidence": round(confidence, 3),
            "stage_used": stage_used,
        },
    )


def log_chunked(
    document_id: str,
    user_id: str = "system",
    chunk_count: int = 0,
    chunk_mode: str = "flat",
    doc_type: str = "general",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_CHUNKED,
        event_data={
            "chunk_count":   chunk_count,
            "chunking_mode": chunk_mode,
            "doc_type":      doc_type,
        },
    )


def log_queried(
    document_id: str,
    user_id: str,
    question_preview: str = "",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_QUERIED,
        event_data={"question_preview": question_preview[:100]},
    )


def log_deleted(
    document_id: str,
    user_id: str,
    filename: str = "",
    doc_type: str = "",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.DOCUMENT_DELETED,
        event_data={"file_name": filename, "doc_type": doc_type},
    )


def log_extraction_started(
    document_id: str,
    user_id: str,
    field_count: int,
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.EXTRACTION_STARTED,
        event_data={"field_count": field_count},
    )


def log_extraction_completed(
    document_id: str,
    user_id: str,
    field_count: int,
    fields_with_value: int = 0,
    fields_with_bbox: int = 0,
    template_id: str = "custom",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.EXTRACTION_COMPLETED,
        event_data={
            "template_id":       template_id,
            "field_count":       field_count,
            "fields_with_value": fields_with_value,
            "fields_with_bbox":  fields_with_bbox,
        },
    )


def log_corrected(
    document_id: str,
    user_id: str,
    field_name: str,
    action: str,
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.EXTRACTION_CORRECTED,
        event_data={"field_name": field_name, "action": action},
    )


def log_classification_overridden(
    document_id: str,
    user_id: str,
    old_type: str,
    new_type: str,
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.CLASSIFICATION_OVERRIDDEN,
        event_data={"old_type": old_type, "new_type": new_type},
    )


def log_split(
    document_id: str,
    user_id: str = "system",
    total_parts: int = 0,
    boundary_pages: list | None = None,
    use_llm: bool = False,
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.SPLIT_COMPLETED,
        event_data={
            "total_parts":    total_parts,
            "boundary_pages": boundary_pages or [],
            "use_llm":        use_llm,
        },
    )


def log_export(
    document_id: str,
    user_id: str,
    export_format: str = "",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.EXPORT_RUN,
        event_data={"export_format": export_format},
    )


def log_summarized(
    document_id: str,
    user_id: str = "system",
) -> None:
    log_event(
        document_id, user_id,
        LineageEvent.SUMMARIZED,
    )