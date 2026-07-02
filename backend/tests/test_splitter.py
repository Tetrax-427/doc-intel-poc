"""
Tests for parsers/splitter.py (Milestone A3 — Document Splitter).

All tests are pure unit tests — no LLM calls, no DB, no file I/O.
use_llm=False is passed everywhere to isolate keyword-based logic.
"""

import pytest
from parsers.splitter import detect_boundaries_fast, split_document, preview_split
from core.document import Document, DocumentPage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_mock_doc(page_texts: list[str]) -> Document:
    """Build a minimal Document from a list of page text strings."""
    pages = [
        DocumentPage(
            page_num=i + 1,
            text=text,
            tables=[],
            images=[],
            layout=[],
            entities=[],
            word_count=len(text.split()),
            ocr_confidence=1.0,
            reading_order=[],
        )
        for i, text in enumerate(page_texts)
    ]
    return Document(
        id="test-doc-id",
        file_name="test.pdf",
        file_type=".pdf",
        file_path="test.pdf",
        pages=pages,
        full_text="\n".join(page_texts),
        tables=[],
        entities={},
        metadata={"page_count": len(pages)},
        classifications=[],
        summary="",
        version=1,
        parent_id=None,
        created_at="2024-01-01T00:00:00+00:00",
        parser_used="docling",
        vision_used=False,
    )


# ---------------------------------------------------------------------------
# detect_boundaries_fast
# ---------------------------------------------------------------------------

class TestDetectBoundariesFast:

    def test_single_document_returns_only_page_one(self):
        doc = make_mock_doc([
            "This is a contract between parties.",
            "The terms are as follows...",
            "Signed on this day...",
        ])
        boundaries = detect_boundaries_fast(doc)
        assert boundaries == [1]

    def test_invoice_keyword_detected(self):
        doc = make_mock_doc([
            "Contract agreement between A and B.",
            "Terms and conditions apply.",
            "INVOICE\nInvoice No: 12345\nDate: 2024-01-01",
            "Total amount: ₹45,000",
        ])
        boundaries = detect_boundaries_fast(doc)
        assert 3 in boundaries

    def test_minimum_page_gap_enforced(self):
        """Boundary not added if fewer than 2 pages since last boundary."""
        doc = make_mock_doc([
            "Start of document",
            "INVOICE\nNew doc on page 2 — too close to page 1",
            "Content continues",
        ])
        boundaries = detect_boundaries_fast(doc)
        # Page 2 is only 1 page after page 1 — below MIN_PAGE_GAP=2
        assert 2 not in boundaries

    def test_page_numbering_reset_detected(self):
        """'Page 1 of N' appearing after page 1 should trigger a boundary."""
        doc = make_mock_doc([
            "Document one content",
            "Document one page 2",
            "Page 1 of 3\nNew document starts here",
            "New document page 2",
        ])
        boundaries = detect_boundaries_fast(doc)
        assert 3 in boundaries

    def test_page_one_always_in_boundaries(self):
        doc = make_mock_doc(["Just one page"])
        boundaries = detect_boundaries_fast(doc)
        assert 1 in boundaries

    def test_multiple_boundaries_detected(self):
        doc = make_mock_doc([
            "Document 1 page 1",
            "Document 1 page 2",
            "Document 1 page 3",
            "INVOICE\nDocument 2 starts",
            "Document 2 page 2",
            "Document 2 page 3",
            "LOAN APPLICATION\nDocument 3 starts",
            "Document 3 page 2",
        ])
        boundaries = detect_boundaries_fast(doc)
        assert 1 in boundaries
        assert 4 in boundaries
        assert 7 in boundaries

    def test_boundary_signal_in_middle_of_text(self):
        """Signal doesn't have to be the first word — appears anywhere in first 500 chars."""
        doc = make_mock_doc([
            "First document about contracts.",
            "Continuation of contract terms.",
            "Please find attached the INVOICE for services rendered.",
            "Payment due within 30 days.",
        ])
        boundaries = detect_boundaries_fast(doc)
        assert 3 in boundaries


# ---------------------------------------------------------------------------
# split_document
# ---------------------------------------------------------------------------

class TestSplitDocument:

    def test_no_split_single_document(self):
        doc = make_mock_doc([
            "This is a contract between parties.",
            "The terms are as follows...",
            "Signed on this day...",
        ])
        result = split_document(doc, use_llm=False)
        assert len(result) == 1

    def test_split_creates_correct_number_of_parts(self):
        doc = make_mock_doc([
            "First document page 1",
            "First document page 2",
            "INVOICE\nSecond document",
            "Second document page 2",
        ])
        result = split_document(doc, use_llm=False)
        assert len(result) == 2

    def test_split_page_ranges_correct(self):
        doc = make_mock_doc([
            "First document page 1",
            "First document page 2",
            "INVOICE\nSecond document",
            "Second document continued",
        ])
        result = split_document(doc, use_llm=False)
        assert result[0].pages[0].page_num == 1
        assert result[0].pages[-1].page_num == 2
        assert result[1].pages[0].page_num == 3

    def test_split_metadata_correct(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        result = split_document(doc, use_llm=False)
        assert result[0].metadata["split_index"] == 1
        assert result[1].metadata["split_index"] == 2
        assert result[0].metadata["split_total"] == 2
        assert result[1].metadata["split_total"] == 2
        assert result[0].metadata["parent_file"] == "test.pdf"
        assert result[1].metadata["parent_file"] == "test.pdf"

    def test_split_is_split_flag_set(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        result = split_document(doc, use_llm=False)
        for sub_doc in result:
            assert sub_doc.metadata["is_split"] is True

    def test_split_sub_docs_have_unique_ids(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        result = split_document(doc, use_llm=False)
        ids = [d.id for d in result]
        assert len(ids) == len(set(ids)), "Sub-document IDs must be unique"

    def test_split_preserves_parser_used(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        result = split_document(doc, use_llm=False)
        for sub_doc in result:
            assert sub_doc.parser_used == "docling"

    def test_split_full_text_contains_only_sub_doc_pages(self):
        doc = make_mock_doc([
            "Alpha content page 1",
            "Alpha content page 2",
            "INVOICE\nBeta content page 3",
            "Beta content page 4",
        ])
        result = split_document(doc, use_llm=False)
        assert "Alpha" in result[0].full_text
        assert "Beta" not in result[0].full_text
        assert "Beta" in result[1].full_text
        assert "Alpha" not in result[1].full_text


# ---------------------------------------------------------------------------
# preview_split
# ---------------------------------------------------------------------------

class TestPreviewSplit:

    def test_preview_would_split_true(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        preview = preview_split(doc, use_llm=False)
        assert preview["would_split"] is True
        assert preview["total_parts"] == 2

    def test_preview_would_split_false_for_single_doc(self):
        doc = make_mock_doc([
            "Just one document.", "Continued content.", "Final page."
        ])
        preview = preview_split(doc, use_llm=False)
        assert preview["would_split"] is False
        assert preview["total_parts"] == 1

    def test_preview_does_not_mutate_document(self):
        """preview_split must not modify the original document."""
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        original_page_count = len(doc.pages)
        original_id = doc.id

        preview_split(doc, use_llm=False)

        assert len(doc.pages) == original_page_count
        assert doc.id == original_id

    def test_preview_parts_structure(self):
        doc = make_mock_doc([
            "Doc 1 page 1", "Doc 1 page 2",
            "INVOICE\nDoc 2", "Doc 2 page 2",
        ])
        preview = preview_split(doc, use_llm=False)
        for part in preview["parts"]:
            assert "part" in part
            assert "start_page" in part
            assert "end_page" in part

    def test_preview_page_ranges_are_contiguous(self):
        """end_page of part N should equal start_page of part N+1."""
        doc = make_mock_doc([
            "Doc 1 p1", "Doc 1 p2", "Doc 1 p3",
            "INVOICE\nDoc 2 p1", "Doc 2 p2",
            "LOAN APPLICATION\nDoc 3 p1", "Doc 3 p2",
        ])
        preview = preview_split(doc, use_llm=False)
        parts = preview["parts"]
        for i in range(len(parts) - 1):
            assert parts[i]["end_page"] == parts[i + 1]["start_page"], \
                f"Part {i+1} end_page should equal part {i+2} start_page"

    def test_preview_single_part_end_page_is_total_pages(self):
        doc = make_mock_doc(["Only page"])
        preview = preview_split(doc, use_llm=False)
        assert preview["parts"][0]["end_page"] == 1