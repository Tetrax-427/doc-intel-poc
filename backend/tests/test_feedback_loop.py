"""tests/test_feedback_loop.py"""
import pytest
from unittest.mock import patch


def test_build_correction_examples_empty():
    """Returns empty string when no corrections exist."""
    with patch("db.get_corrections_for_doc_type", return_value=[]):
        from retrieval import build_correction_examples
        result = build_correction_examples("invoice", {"total_amount": ""})
        assert result == ""


def test_build_correction_examples_with_corrections():
    """Returns formatted examples when corrections exist."""
    mock_corrections = [{
        "field_name": "total_amount",
        "original_value": "44000",
        "corrected_value": "45000"
    }]
    with patch("db.get_corrections_for_doc_type", return_value=mock_corrections):
        from retrieval import build_correction_examples
        result = build_correction_examples("invoice", {"total_amount": ""})
        assert "total_amount" in result
        assert "44000" in result
        assert "45000" in result


def test_build_correction_examples_skips_unchanged():
    """Skips corrections where original == corrected (approved values)."""
    mock_corrections = [{
        "field_name": "vendor_name",
        "original_value": "Acme",
        "corrected_value": "Acme"   # unchanged — should be skipped
    }]
    with patch("db.get_corrections_for_doc_type", return_value=mock_corrections):
        from retrieval import build_correction_examples
        result = build_correction_examples("invoice", {"vendor_name": ""})
        assert result == ""


def test_build_correction_examples_limits_to_five():
    """Returns at most 5 examples even if more corrections exist."""
    mock_corrections = [
        {"field_name": f"field_{i}",
         "original_value": f"old_{i}",
         "corrected_value": f"new_{i}"}
        for i in range(10)
    ]
    with patch("db.get_corrections_for_doc_type", return_value=mock_corrections):
        from retrieval import build_correction_examples
        fields = {f"field_{i}": "" for i in range(10)}
        result = build_correction_examples("invoice", fields)
        assert result.count("field_") <= 5


def test_extract_fields_includes_business_validation(sample_document_id):
    """extract_fields response always includes business_validation key."""
    from retrieval import extract_fields
    result = extract_fields(
        sample_document_id,
        {"total_amount": "total amount in the document"},
        doc_type="invoice"
    )
    assert "extracted" in result
    assert "validation" in result
    assert "business_validation" in result