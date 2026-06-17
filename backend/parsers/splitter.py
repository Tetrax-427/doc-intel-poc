"""
Document Splitter — detect logical document boundaries inside a multi-document PDF
and split it into separate Document objects.

Common use cases:
  - Loan package: application + KYC + income proof + bank statements = 4 docs
  - Due diligence bundle: NDA + term sheet + financials + cap table = N docs
  - Contract with exhibits: base agreement + Exhibit A + Exhibit B = 3 docs

This is a USER-TRIGGERED, OPTIONAL flow — never called automatically on upload.
The default upload behaviour (single-document ingestion) is completely unchanged.

Flow:
  1. User uploads PDF normally — one document, as always.
  2. User explicitly calls POST /documents/{id}/split/preview — sees boundaries,
     no data is mutated.
  3. User confirms → POST /documents/{id}/split — sub-documents are created,
     ingested, classified, summarised independently.
  4. Original document remains queryable (non-destructive).

Detection strategy:
  Fast keyword scan first (zero LLM cost, runs in milliseconds).
  Optional LLM refinement pass only when fast scan finds candidates.
  Minimum 2-page gap between boundaries prevents false positives on short docs.
"""

import uuid
from datetime import datetime, timezone

from core.document import Document, DocumentPage
from core.logger import get_logger
from llm.engine import call_llm

logger = get_logger("splitter")

# ---------------------------------------------------------------------------
# Boundary signal vocabulary
# ---------------------------------------------------------------------------

# Words/phrases that strongly suggest a new document is starting.
# Checked against the first 500 characters of each page (lowercased).
# Tune this list if false positives appear for your document corpus.
BOUNDARY_SIGNALS = [
    # Explicit structural markers
    "--- document", "--- page 1 ---", "document 1", "exhibit a",
    "attachment", "annex", "schedule",
    # Document type headers
    "invoice", "purchase order", "delivery note", "bill of lading",
    "loan application", "credit agreement", "deed of",
    # Form / reference identifiers
    "form no", "form no.", "application no", "reference no",
    # Indian document types (DocIntel's primary vertical)
    "gst invoice", "tax invoice", "proforma invoice",
    "salary slip", "payslip", "pay slip",
    "bank statement", "account statement",
]

# Minimum number of pages between two consecutive boundaries.
# Prevents splitting a 3-page document into [1], [2], [3].
MIN_PAGE_GAP = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_boundaries_fast(document: Document) -> list[int]:
    """
    Fast keyword-based boundary detection. No LLM, no I/O.

    Returns a list of page numbers where new documents start.
    Page 1 is always included (first document always starts at page 1).

    Args:
        document: Parsed Document object.

    Returns:
        Sorted list of page numbers, e.g. [1, 5, 12].
        A list with only [1] means no split was detected.
    """
    boundaries = [1]

    for page in document.pages[1:]:  # page 1 is always a boundary — skip it
        page_text_lower = page.text[:500].lower()

        # Check keyword signals
        for signal in BOUNDARY_SIGNALS:
            if signal in page_text_lower:
                if page.page_num - boundaries[-1] >= MIN_PAGE_GAP:
                    boundaries.append(page.page_num)
                    logger.info(
                        "Keyword boundary detected",
                        file=document.file_name,
                        page=page.page_num,
                        signal=signal,
                    )
                break  # one signal is enough per page — don't double-count

        # Page numbering reset: "Page 1 of N" appearing after page 1 is a strong signal
        if "page 1 of" in page_text_lower or "page 1\n" in page_text_lower:
            if page.page_num not in boundaries:
                if page.page_num - boundaries[-1] >= MIN_PAGE_GAP:
                    boundaries.append(page.page_num)
                    logger.info(
                        "Page numbering reset boundary detected",
                        file=document.file_name,
                        page=page.page_num,
                    )

    return sorted(boundaries)


def detect_boundaries_llm(document: Document, fast_boundaries: list[int]) -> list[int]:
    """
    LLM-based boundary refinement.

    Only called when fast detection already found candidate boundaries.
    Sends the first 200 chars of each candidate page to the LLM and asks
    it to confirm or reject. Keeps page 1 unconditionally.

    Args:
        document:        The parsed Document.
        fast_boundaries: Output of detect_boundaries_fast().

    Returns:
        Refined list of boundary page numbers (always includes 1).
    """
    if len(fast_boundaries) <= 1:
        # Fast scan found nothing — skip LLM entirely
        return fast_boundaries

    candidate_pages = [
        f"Page {page.page_num}: {page.text[:200]}"
        for page in document.pages
        if page.page_num in fast_boundaries[1:]  # skip page 1 — always a boundary
    ]

    if not candidate_pages:
        return fast_boundaries

    prompt = (
        "A PDF has been uploaded that may contain multiple documents bundled together.\n"
        "Review these page excerpts and confirm which ones start a genuinely new document.\n"
        "A new document boundary is indicated by a distinct document type change, "
        "a form header, an exhibit marker, or a page numbering reset.\n"
        "Do NOT mark boundaries for section headings within a single document.\n\n"
        "Candidate pages:\n"
        + "\n".join(candidate_pages)
        + "\n\nReturn ONLY a JSON array of confirmed page numbers that start a new document.\n"
        "Always include page 1. Example: [1, 5, 12]\n"
        "JSON:"
    )

    try:
        result = call_llm(
            prompt,
            temperature=0.0,
            json_mode=True,
            call_type="split_detection",
        )

        if isinstance(result, list):
            confirmed = sorted(set([1] + [int(p) for p in result if isinstance(p, (int, float))]))
            logger.info(
                "LLM boundary refinement complete",
                file=document.file_name,
                fast=fast_boundaries,
                confirmed=confirmed,
            )
            return confirmed

        # LLM returned a dict (e.g. {"error": ...}) — fall back to fast results
        logger.warning("LLM returned unexpected format — using fast boundaries",
                       result=result)
        return fast_boundaries

    except Exception as e:
        logger.warning("LLM boundary detection failed — using fast boundaries",
                       error=str(e))
        return fast_boundaries


def split_document(document: Document, use_llm: bool = True) -> list[Document]:
    """
    Split a multi-document PDF into individual Document objects.

    Each sub-document gets its own id, file_name (_part1, _part2, ...),
    pages, tables, and metadata. The original document is not modified.

    Args:
        document: The parsed Document to split.
        use_llm:  If True, run LLM refinement on top of fast detection.
                  Set False in tests or when LLM is unavailable.

    Returns:
        List of Document objects.
        Length 1 (the original document) if no split boundaries detected.
    """
    # Step 1: fast keyword detection
    boundaries = detect_boundaries_fast(document)

    # Step 2: optional LLM refinement
    if use_llm and len(boundaries) > 1:
        boundaries = detect_boundaries_llm(document, boundaries)

    # No split needed
    if len(boundaries) <= 1:
        logger.info("No document boundaries — returning as single document",
                    file=document.file_name)
        return [document]

    logger.info("Splitting document",
                file=document.file_name,
                parts=len(boundaries),
                boundaries=boundaries)

    # Step 3: build sub-documents
    sub_docs = []
    for i, start_page in enumerate(boundaries):
        end_page = boundaries[i + 1] if i + 1 < len(boundaries) else None

        sub_pages = [
            p for p in document.pages
            if p.page_num >= start_page and (end_page is None or p.page_num < end_page)
        ]

        if not sub_pages:
            logger.warning("Empty page range for split segment — skipping",
                           start=start_page, end=end_page)
            continue

        sub_text = "\n\n".join(p.text for p in sub_pages if p.text.strip())
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
                "page_count": len(sub_pages),
                "is_split": True,
                "split_index": i + 1,
                "split_total": len(boundaries),
                "parent_file": document.file_name,
                "split_start_page": start_page,
                "split_end_page": end_page,
            },
            classifications=[],
            summary="",
            version=1,
            parent_id=document.id,
            created_at=datetime.now(timezone.utc).isoformat(),
            parser_used=document.parser_used,
            vision_used=False,
        )
        sub_docs.append(sub_doc)

    return sub_docs


def preview_split(document: Document, use_llm: bool = True) -> dict:
    """
    Preview-only split analysis. Returns boundary info WITHOUT creating
    sub-documents or touching any storage.

    Used by POST /documents/{id}/split/preview so the user can review
    before committing. This function is fully non-destructive.

    Args:
        document: The parsed Document to analyse.
        use_llm:  If True, run LLM refinement on fast candidates.

    Returns:
        {
            "would_split": bool,
            "total_parts": int,
            "parts": [{"part": 1, "start_page": 1, "end_page": 4}, ...]
        }
    """
    boundaries = detect_boundaries_fast(document)

    if use_llm and len(boundaries) > 1:
        boundaries = detect_boundaries_llm(document, boundaries)

    page_count = document.metadata.get("page_count") or len(document.pages)

    parts = []
    for i, start_page in enumerate(boundaries):
        end_page = boundaries[i + 1] if i + 1 < len(boundaries) else page_count
        parts.append({
            "part": i + 1,
            "start_page": start_page,
            "end_page": end_page,
        })

    return {
        "would_split": len(boundaries) > 1,
        "total_parts": len(boundaries),
        "parts": parts,
    }