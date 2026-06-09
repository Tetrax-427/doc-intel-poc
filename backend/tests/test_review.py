"""tests/test_review.py"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_submit_review_approve(sample_document_id):
    """Submitting approve actions saves to review_corrections table."""
    response = client.post(
        f"/review/{sample_document_id}",
        json=[{
            "field": "total_amount",
            "action": "approve",
            "original_value": "45000",
            "corrected_value": "45000",
            "reviewer_note": ""
        }]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["saved"] == 1
    assert "total_amount" in data["reviewed_fields"]


def test_submit_review_correct(sample_document_id):
    """Submitting correct action saves original and corrected values."""
    response = client.post(
        f"/review/{sample_document_id}",
        json=[{
            "field": "vendor_name",
            "action": "correct",
            "original_value": "Acme Corp",
            "corrected_value": "Acme Corporation",
            "reviewer_note": "Full legal name"
        }]
    )
    assert response.status_code == 200
    assert response.json()["saved"] == 1


def test_submit_review_reject(sample_document_id):
    """Reject action is saved correctly."""
    response = client.post(
        f"/review/{sample_document_id}",
        json=[{
            "field": "phone",
            "action": "reject",
            "original_value": "not found",
            "corrected_value": "",
            "reviewer_note": "Not present in document"
        }]
    )
    assert response.status_code == 200


def test_get_corrections(sample_document_id):
    """GET /review/{id}/corrections returns list."""
    response = client.get(f"/review/{sample_document_id}/corrections")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_submit_review_multiple_fields(sample_document_id):
    """Can submit multiple field actions in one request."""
    response = client.post(
        f"/review/{sample_document_id}",
        json=[
            {"field": "name", "action": "approve",
             "original_value": "John", "corrected_value": "John"},
            {"field": "email", "action": "correct",
             "original_value": "john@old.com", "corrected_value": "john@correct.com"},
        ]
    )
    assert response.status_code == 200
    assert response.json()["saved"] == 2


def test_submit_review_unknown_document():
    """Unknown document_id returns 200 but saves with doc_type 'general'."""
    response = client.post(
        "/review/00000000-0000-0000-0000-000000000000",
        json=[{
            "field": "test",
            "action": "approve",
            "original_value": "x",
            "corrected_value": "x"
        }]
    )
    # Should not crash — doc_type defaults to "general"
    assert response.status_code == 200