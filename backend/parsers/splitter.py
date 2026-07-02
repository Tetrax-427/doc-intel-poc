"""
Document Splitter — detect logical document boundaries inside a multi-document PDF.
F1: logs split event via log_split() after sub-documents are created.
"""

import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.document import Document, DocumentPage
from core.logger import get_logger
from core.lineage import log_split
from llm.engine import call_llm

logger = get_logger("splitter")

BOUNDARY_SIGNALS = [
    "--- document", "--- page 1 ---", "document 1", "exhibit a",
    "attachment", "annex", "schedule",
    "invoice", "purchase order", "delivery note", "bill of lading",
    "loan application", "credit agreement", "deed of",
    "form no", "form no.", "application no", "reference no",
    "gst invoice", "tax invoice", "proforma invoice",
    "salary slip", "payslip", "pay slip",
    "bank statement", "account statement",
]

MIN_PAGE_GAP = 2

_SPLIT_DETECTION_SYSTEM = (
    "A PDF has been uploaded that may contain multiple documents bundled together. "
    "Review the page excerpts the user provides and confirm which ones start a genuinely "
    "new document. A new document boundary is indicated by a distinct document type change, "
    "a form header, an exhibit marker, or a page numbering reset. "
    "Do NOT mark boundaries for section headings within a single document. "
    "Return the confirmed page numbers that start a new document. Always include page 1."
)


class SplitBoundaries(BaseModel):
    page_numbers: list[int] = Field(
        default_factory=lambda: [1],
        description="Confirmed page numbers that start a new document. Must always include 1.",
    )


def detect_boundaries_fast(document: Document) -> list[int]:
    boundaries = [1]
    for page in document.pages[1:]:
        page_text_lower = page.text[:500].lower()
        for signal in BOUNDARY_SIGNALS:
            if signal in page_text_lower:
                if page.page_num - boundaries[-1] >= MIN_PAGE_GAP:
                    boundaries.append(page.page_num)
                    logger.info("Keyword boundary detected",
                                file=document.file_name, page=page.page_num, signal=signal)
                break
        if "page 1 of" in page_text_lower or "page 1\n" in page_text_lower:
            if page.page_num not in boundaries:
                if page.page_num - boundaries[-1] >= MIN_PAGE_GAP:
                    boundaries.append(page.page_num)
                    logger.info("Page numbering reset boundary detected",
                                file=document.file_name, page=page.page_num)
    return sorted(boundaries)


def detect_boundaries_llm(
    document: Document,
    fast_boundaries: list[int],
    user_id: str = "system",
) -> list[int]:
    if len(fast_boundaries) <= 1:
        return fast_boundaries

    candidate_pages = [
        f"Page {page.page_num}: {page.text[:200]}"
        for page in document.pages
        if page.page_num in fast_boundaries[1:]
    ]
    if not candidate_pages:
        return fast_boundaries

    try:
        result: SplitBoundaries = call_llm(
            system=_SPLIT_DETECTION_SYSTEM,
            user="Candidate pages:\n" + "\n".join(candidate_pages),
            temperature=0.0,
            call_type="split_detection",
            response_model=SplitBoundaries,
            user_id=user_id,
            document_id=document.id if hasattr(document, "id") else None,
        )
        confirmed = sorted(set([1] + [p for p in result.page_numbers if isinstance(p, int)]))
        logger.info("LLM boundary refinement complete", file=document.file_name,
                    fast=fast_boundaries, confirmed=confirmed)
        return confirmed
    except Exception as e:
        logger.warning("LLM boundary detection failed — using fast boundaries", error=str(e))
        return fast_boundaries


def split_document(
    document: Document,
    use_llm: bool = True,
    user_id: str = "system",
    document_id: str = "",
) -> list[Document]:
    boundaries = detect_boundaries_fast(document)
    if use_llm and len(boundaries) > 1:
        boundaries = detect_boundaries_llm(document, boundaries, user_id=user_id)
    if len(boundaries) <= 1:
        logger.info("No document boundaries — returning as single document",
                    file=document.file_name)
        return [document]

    logger.info("Splitting document", file=document.file_name,
                parts=len(boundaries), boundaries=boundaries)

    sub_docs = []
    for i, start_page in enumerate(boundaries):
        end_page  = boundaries[i + 1] if i + 1 < len(boundaries) else None
        sub_pages = [
            p for p in document.pages
            if p.page_num >= start_page and (end_page is None or p.page_num < end_page)
        ]
        if not sub_pages:
            continue
        sub_text   = "\n\n".join(p.text for p in sub_pages if p.text.strip())
        sub_tables = [
            t for t in document.tables
            if t.page_num >= start_page and (end_page is None or t.page_num < end_page)
        ]
        sub_doc = Document(
            id=str(uuid.uuid4()),
            file_name=f"{document.file_name}_part{i + 1}",
            file_type=document.file_type,
            file_path=document.file_path,
            pages=sub_pages,
            full_text=sub_text,
            tables=sub_tables,
            entities={},
            metadata={
                **document.metadata,
                "page_count": len(sub_pages), "is_split": True,
                "split_index": i + 1, "split_total": len(boundaries),
                "parent_file": document.file_name,
                "split_start_page": start_page, "split_end_page": end_page,
            },
            classifications=[], summary="", version=1,
            parent_id=document.id,
            created_at=datetime.now(timezone.utc).isoformat(),
            parser_used=document.parser_used, vision_used=False,
        )
        sub_docs.append(sub_doc)

    # F1 — log split completed
    if document_id:
        log_split(
            document_id, user_id=user_id,
            total_parts=len(sub_docs),
            boundary_pages=boundaries,
            use_llm=use_llm,
        )

    return sub_docs


def preview_split(document: Document, use_llm: bool = True, user_id: str = "system") -> dict:
    """Preview only — non-destructive, no lineage logged."""
    boundaries = detect_boundaries_fast(document)
    if use_llm and len(boundaries) > 1:
        boundaries = detect_boundaries_llm(document, boundaries, user_id=user_id)
    page_count = document.metadata.get("page_count") or len(document.pages)
    parts = []
    for i, start_page in enumerate(boundaries):
        end_page = boundaries[i + 1] if i + 1 < len(boundaries) else page_count
        parts.append({"part": i + 1, "start_page": start_page, "end_page": end_page})
    return {"would_split": len(boundaries) > 1, "total_parts": len(boundaries), "parts": parts}