"""
tests/test_classification.py
Unit tests for the classify_document() function and TEMPLATE_MAP.

Run with:
    pytest tests/test_classification.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "document_id": "doc-1",
        "content": "Invoice #1234\nBill To: Acme Corp\nTotal Due: $4,500.00\nDue Date: 2025-01-31",
        "metadata": {"page": 1, "file": "invoice.pdf"},
        "embedding": [0.1] * 1536,
    },
    {
        "document_id": "doc-1",
        "content": "Line items: Consulting services x 3 days @ $1,500/day",
        "metadata": {"page": 1, "file": "invoice.pdf"},
        "embedding": [0.1] * 1536,
    },
]

RESUME_CHUNKS = [
    {
        "document_id": "doc-2",
        "content": "Jane Smith | jane@email.com | LinkedIn: /in/janesmith\nSoftware Engineer with 8 years experience",
        "metadata": {"page": 1, "file": "resume.pdf"},
        "embedding": [0.1] * 1536,
    },
]


# ---------------------------------------------------------------------------
# classify_document — happy paths
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def _run(self, chunks, llm_response, document_id="doc-1"):
        with (
            patch("retrieval.get_all_chunks", return_value=chunks),
            patch("retrieval.call_llm", return_value=llm_response),
        ):
            from retrieval import classify_document
            return classify_document(document_id)

    def test_invoice_classified_correctly(self):
        result = self._run(
            SAMPLE_CHUNKS,
            {
                "doc_type": "invoice",
                "confidence": 0.97,
                "reasoning": "Contains invoice number, billing address, and total due.",
                "key_signals": ["Invoice #1234", "Total Due", "Due Date", "Bill To"],
            },
        )
        assert result["doc_type"] == "invoice"
        assert result["schema_template"] == "invoice"
        assert result["confidence"] == 0.97
        assert result["requires_human_review"] is False
        assert len(result["key_signals"]) == 4

    def test_resume_classified_correctly(self):
        result = self._run(
            RESUME_CHUNKS,
            {
                "doc_type": "resume",
                "confidence": 0.91,
                "reasoning": "Contains name, contact info, and work experience.",
                "key_signals": ["Software Engineer", "years experience"],
            },
            document_id="doc-2",
        )
        assert result["doc_type"] == "resume"
        assert result["schema_template"] == "cv_resume"
        assert result["requires_human_review"] is False

    def test_low_confidence_sets_requires_review(self):
        result = self._run(
            SAMPLE_CHUNKS,
            {
                "doc_type": "general",
                "confidence": 0.45,
                "reasoning": "Could not determine document type clearly.",
                "key_signals": [],
            },
        )
        assert result["requires_human_review"] is True

    def test_confidence_exactly_at_threshold_does_not_require_review(self):
        """Confidence == 0.75 should NOT require review (boundary condition)."""
        result = self._run(
            SAMPLE_CHUNKS,
            {"doc_type": "report", "confidence": 0.75, "reasoning": "x", "key_signals": []},
        )
        assert result["requires_human_review"] is False

    def test_confidence_just_below_threshold_requires_review(self):
        result = self._run(
            SAMPLE_CHUNKS,
            {"doc_type": "report", "confidence": 0.74, "reasoning": "x", "key_signals": []},
        )
        assert result["requires_human_review"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestClassifyDocumentEdgeCases:
    def test_no_chunks_returns_safe_default(self):
        with patch("retrieval.get_all_chunks", return_value=[]):
            from retrieval import classify_document
            result = classify_document("nonexistent-doc")
        assert result["doc_type"] == "general"
        assert result["requires_human_review"] is True

    def test_llm_raises_returns_safe_default(self):
        with (
            patch("retrieval.get_all_chunks", return_value=SAMPLE_CHUNKS),
            patch("retrieval.call_llm", side_effect=Exception("LLM timeout")),
        ):
            from retrieval import classify_document
            result = classify_document("doc-1")
        assert result["doc_type"] == "general"
        assert result["requires_human_review"] is True
        assert "LLM timeout" in result["reasoning"]

    def test_llm_returns_wrong_type_returns_safe_default(self):
        with (
            patch("retrieval.get_all_chunks", return_value=SAMPLE_CHUNKS),
            patch("retrieval.call_llm", return_value="not a dict"),
        ):
            from retrieval import classify_document
            result = classify_document("doc-1")
        assert result["doc_type"] == "general"
        assert result["requires_human_review"] is True

    def test_llm_returns_unknown_doc_type_uses_general_template(self):
        with (
            patch("retrieval.get_all_chunks", return_value=SAMPLE_CHUNKS),
            patch("retrieval.call_llm", return_value={
                "doc_type": "quantum_manifesto",
                "confidence": 0.8,
                "reasoning": "Novel doc type.",
                "key_signals": [],
            }),
        ):
            from retrieval import classify_document
            result = classify_document("doc-1")
        # Unknown type → falls back to general template
        assert result["schema_template"] == "general"


# ---------------------------------------------------------------------------
# TEMPLATE_MAP coverage
# ---------------------------------------------------------------------------

class TestTemplateMap:
    """Ensure every alias in TEMPLATE_MAP maps to a known template bucket."""

    KNOWN_TEMPLATES = {"invoice", "cv_resume", "contract", "report", "financial", "medical", "legal", "general"}

    def test_all_template_map_values_are_known(self):
        from retrieval import TEMPLATE_MAP
        for doc_type, template in TEMPLATE_MAP.items():
            assert template in self.KNOWN_TEMPLATES, (
                f"TEMPLATE_MAP['{doc_type}'] = '{template}' is not a known template bucket"
            )

    def test_receipt_maps_to_invoice(self):
        from retrieval import TEMPLATE_MAP
        assert TEMPLATE_MAP["receipt"] == "invoice"

    def test_cv_and_resume_both_map_to_cv_resume(self):
        from retrieval import TEMPLATE_MAP
        assert TEMPLATE_MAP["cv"] == "cv_resume"
        assert TEMPLATE_MAP["resume"] == "cv_resume"

    def test_nda_maps_to_contract(self):
        from retrieval import TEMPLATE_MAP
        assert TEMPLATE_MAP["nda"] == "contract"
