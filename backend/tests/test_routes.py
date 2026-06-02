"""
tests/test_routes.py
Smoke tests for all major API routes.

Run with:
    pytest tests/test_routes.py -v

Dependencies are mocked at module level — no live Supabase or LLM calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture — patch heavy dependencies before importing main
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    # Patch everything that touches the network or disk at import/startup time
    with (
        patch("ingestion.get_embed_model", return_value=MagicMock()),
        patch("db.supabase", new_callable=MagicMock),
        patch("llm.engine.call_llm", return_value="mocked answer"),
        patch("llm.usage.get_usage_summary", return_value={"total_tokens": 0}),
    ):
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class TestSystem:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "DocIntel" in r.json()["status"]

    def test_health_returns_checks(self, client):
        r = client.get("/health")
        body = r.json()
        # healthy key must be present
        assert "healthy" in body
        # all three subsystem keys must be present
        assert "database" in body["checks"]
        assert "embeddings" in body["checks"]
        assert "llm" in body["checks"]

    def test_usage(self, client):
        r = client.get("/usage")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_list_documents(self, client):
        with patch("db.supabase") as mock_sb:
            mock_sb.table.return_value.select.return_value \
                .order.return_value.limit.return_value.execute.return_value \
                .data = [{"id": "doc1", "name": "test.pdf"}]
            r = client.get("/documents")
        assert r.status_code == 200

    def test_list_documents_filter_by_doc_type(self, client):
        with patch("db.supabase") as mock_sb:
            chain = mock_sb.table.return_value.select.return_value \
                        .order.return_value.limit.return_value
            chain.eq.return_value.execute.return_value.data = []
            r = client.get("/documents?doc_type=invoice")
        assert r.status_code == 200

    def test_delete_document(self, client):
        with patch("db.delete_document_by_id") as mock_del:
            r = client.delete("/documents/doc-123")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
        mock_del.assert_called_once_with("doc-123")

    def test_upload_unsupported_file_type(self, client):
        r = client.post(
            "/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
            data={"use_llamaparse": "False"},
        )
        assert r.status_code == 415
        assert r.json()["code"] == "UNSUPPORTED_FILE_TYPE"

    def test_get_classification(self, client):
        with patch("db.get_classification", return_value={
            "doc_type": "invoice",
            "classification_confidence": 0.95,
            "classification_data": {},
            "requires_review": False,
        }):
            r = client.get("/documents/doc-123/classification")
        assert r.status_code == 200
        assert r.json()["doc_type"] == "invoice"

    def test_get_classification_not_found_returns_default(self, client):
        with patch("db.get_classification", return_value=None):
            r = client.get("/documents/nonexistent/classification")
        assert r.status_code == 200
        assert r.json()["doc_type"] == "general"

    def test_override_classification(self, client):
        with (
            patch("db.get_classification", return_value={"classification_data": {}}),
            patch("db.save_classification") as mock_save,
        ):
            r = client.post(
                "/documents/doc-123/classification",
                json={"doc_type": "contract"},
            )
        assert r.status_code == 200
        assert r.json()["classification"]["doc_type"] == "contract"
        assert r.json()["classification"]["confidence"] == 1.0
        assert r.json()["classification"]["manually_overridden"] is True
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_empty_question_rejected(self, client):
        r = client.post("/query", json={"question": "  "})
        assert r.status_code == 422  # Pydantic validator

    def test_query_general(self, client):
        with patch("retrieval.query_document", return_value={
            "answer": "42",
            "sources": [],
            "type": "general",
        }):
            r = client.post("/query", json={"question": "What is the meaning of life?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "42"

    def test_compress_empty_messages_rejected(self, client):
        r = client.post("/compress", json={"messages": []})
        assert r.status_code == 422

    def test_compress(self, client):
        with patch("retrieval.compress_history", return_value="Compressed summary."):
            r = client.post("/compress", json={"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]})
        assert r.status_code == 200
        assert r.json()["summary"] == "Compressed summary."


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extract_empty_fields_rejected(self, client):
        r = client.post("/extract", json={"document_id": "doc-1", "fields": {}})
        assert r.status_code == 422

    def test_extract(self, client):
        with (
            patch("retrieval.extract_fields", return_value={
                "extracted": {"name": "John"},
                "validation": {"name": True},
            }),
            patch("webhooks.trigger_webhooks"),
        ):
            r = client.post("/extract", json={
                "document_id": "doc-1",
                "fields": {"name": "full name of the person"},
            })
        assert r.status_code == 200
        assert r.json()["extracted"]["name"] == "John"

    def test_extract_nl_preview_only(self, client):
        with patch("retrieval.nl_to_schema", return_value={"name": "full name"}):
            r = client.post("/extract/nl", json={
                "document_id": "doc-1",
                "instruction": "Get the person's name",
                "preview_only": True,
            })
        assert r.status_code == 200
        assert r.json()["schema"] == {"name": "full name"}
        assert r.json()["extracted"] is None

    def test_batch_extract_no_spec_rejected(self, client):
        r = client.post("/extract/batch", json={
            "document_ids": ["doc-1", "doc-2"],
            "fields": {},
        })
        assert r.status_code == 400
        assert r.json()["code"] == "MISSING_EXTRACTION_SPEC"

    def test_batch_extract(self, client):
        with (
            patch("retrieval.extract_fields", return_value={
                "extracted": {"amount": "$100"},
                "validation": {},
            }),
            patch("webhooks.trigger_webhooks"),
        ):
            r = client.post("/extract/batch", json={
                "document_ids": ["doc-1", "doc-2"],
                "fields": {"amount": "invoice total"},
            })
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert body["succeeded"] == 2

    def test_template_not_found(self, client):
        with patch("schemas.templates.get_template", return_value=None):
            r = client.get("/templates/nonexistent")
        assert r.status_code == 404
        assert r.json()["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_pdf(self, client):
        with patch("export.export_chat_pdf", return_value=b"%PDF-stub"):
            r = client.post("/export/pdf", json={
                "document_id": "doc-1",
                "file_name": "test_chat",
                "messages": [{"role": "user", "content": "Hello"}],
            })
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"

    def test_export_pdf_empty_messages_rejected(self, client):
        r = client.post("/export/pdf", json={
            "document_id": "doc-1",
            "file_name": "test",
            "messages": [],
        })
        assert r.status_code == 422

    def test_file_name_sanitised(self, client):
        with patch("export.export_chat_pdf", return_value=b"%PDF-stub"):
            r = client.post("/export/pdf", json={
                "document_id": "doc-1",
                "file_name": "../../etc/passwd",
                "messages": [{"role": "user", "content": "x"}],
            })
        assert r.status_code == 200
        # Slashes must have been replaced
        assert "/" not in r.headers["content-disposition"]


# ---------------------------------------------------------------------------
# Integration (API keys + webhooks)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_create_api_key(self, client):
        with patch("api_keys.create_api_key", return_value={"id": "k1", "key": "sk-abc"}):
            r = client.post("/api-keys", json={"name": "test-key"})
        assert r.status_code == 200
        assert r.json()["key"] == "sk-abc"

    def test_create_api_key_empty_name_rejected(self, client):
        r = client.post("/api-keys", json={"name": "  "})
        assert r.status_code == 422

    def test_revoke_api_key(self, client):
        with patch("api_keys.revoke_api_key"):
            r = client.delete("/api-keys/k1")
        assert r.status_code == 200
        assert r.json()["status"] == "revoked"

    def test_create_webhook_invalid_url(self, client):
        r = client.post("/webhooks", json={
            "name": "my-hook",
            "url": "not-a-url",
            "events": ["extraction.complete"],
        })
        assert r.status_code == 422

    def test_test_webhook_not_found(self, client):
        with patch("db.supabase") as mock_sb:
            mock_sb.table.return_value.select.return_value \
                .eq.return_value.execute.return_value.data = []
            r = client.post("/webhooks/nonexistent/test")
        assert r.status_code == 404
        assert r.json()["code"] == "NOT_FOUND"

    def test_webhook_logs_route_not_swallowed_as_id(self, client):
        """
        /webhooks/logs must NOT be interpreted as /webhooks/{webhook_id}.
        """
        with patch("db.supabase") as mock_sb:
            mock_sb.table.return_value.select.return_value \
                .order.return_value.limit.return_value.execute.return_value.data = []
            r = client.get("/webhooks/logs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
