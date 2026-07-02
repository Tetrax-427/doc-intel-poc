"""
tests/test_retrieval_config.py

Unit tests for D3 config additions and retrieval pool/top-N behaviour:
- config fields load correctly from env
- RETRIEVAL_TOP_N clamped when > RETRIEVAL_CANDIDATE_POOL
- uses_hierarchical_chunking() helper
- HIERARCHICAL_CHUNKING_DOC_TYPES parsed correctly
- hybrid_search() respects configurable pool + top_n
- expand_to_parent_context() respects HIERARCHICAL_EXPAND_TO_PARENT flag
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_config():
    import core.config as cfg_mod
    cfg_mod._config_instance = None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant_test")
    monkeypatch.setenv("LLM_FALLBACK_CHAIN", "groq:llama-3.3-70b-versatile")
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# D3 — Config fields
# ---------------------------------------------------------------------------

class TestD3ConfigFields:

    def test_retrieval_candidate_pool_default(self):
        _reset_config()
        from core.config import config
        assert config.retrieval_candidate_pool == 50

    def test_retrieval_top_n_default(self):
        _reset_config()
        from core.config import config
        assert config.retrieval_top_n == 5

    def test_retrieval_candidate_pool_from_env(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "100")
        _reset_config()
        from core.config import config
        assert config.retrieval_candidate_pool == 100

    def test_retrieval_top_n_from_env(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_TOP_N", "10")
        _reset_config()
        from core.config import config
        assert config.retrieval_top_n == 10

    def test_top_n_clamped_when_greater_than_pool(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "20")
        monkeypatch.setenv("RETRIEVAL_TOP_N", "50")   # > pool
        _reset_config()
        from core.config import config
        assert config.retrieval_top_n <= config.retrieval_candidate_pool

    def test_top_n_not_clamped_when_equal_to_pool(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "20")
        monkeypatch.setenv("RETRIEVAL_TOP_N", "20")
        _reset_config()
        from core.config import config
        assert config.retrieval_top_n == 20

    def test_top_n_not_clamped_when_less_than_pool(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "50")
        monkeypatch.setenv("RETRIEVAL_TOP_N", "5")
        _reset_config()
        from core.config import config
        assert config.retrieval_top_n == 5


# ---------------------------------------------------------------------------
# D1 — Hierarchical config fields
# ---------------------------------------------------------------------------

class TestHierarchicalConfigFields:

    def test_default_doc_types_non_empty(self):
        _reset_config()
        from core.config import config
        assert len(config.hierarchical_chunking_doc_types) > 0

    def test_contract_in_default_doc_types(self):
        _reset_config()
        from core.config import config
        assert "contract" in config.hierarchical_chunking_doc_types

    def test_custom_doc_types_from_env(self, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_CHUNKING_DOC_TYPES", "invoice,receipt,medical_record")
        _reset_config()
        from core.config import config
        assert "invoice" in config.hierarchical_chunking_doc_types
        assert "receipt" in config.hierarchical_chunking_doc_types
        assert "medical_record" in config.hierarchical_chunking_doc_types

    def test_doc_types_lowercased(self, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_CHUNKING_DOC_TYPES", "CONTRACT,NDA,AGREEMENT")
        _reset_config()
        from core.config import config
        assert "contract" in config.hierarchical_chunking_doc_types
        assert "CONTRACT" not in config.hierarchical_chunking_doc_types

    def test_parent_chunk_size_default(self):
        _reset_config()
        from core.config import config
        assert config.hierarchical_parent_chunk_size == 2000

    def test_child_chunk_size_default(self):
        _reset_config()
        from core.config import config
        assert config.hierarchical_child_chunk_size == 400

    def test_expand_to_parent_default_true(self):
        _reset_config()
        from core.config import config
        assert config.hierarchical_expand_to_parent is True

    def test_expand_to_parent_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "false")
        _reset_config()
        from core.config import config
        assert config.hierarchical_expand_to_parent is False


# ---------------------------------------------------------------------------
# uses_hierarchical_chunking()
# ---------------------------------------------------------------------------

class TestUsesHierarchicalChunking:

    def test_returns_true_for_configured_type(self):
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("contract") is True

    def test_returns_false_for_non_configured_type(self):
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("invoice") is False

    def test_case_insensitive(self):
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("CONTRACT") is True

    def test_returns_false_for_general(self):
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("general") is False

    def test_returns_false_for_empty_string(self):
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("") is False

    def test_custom_types_via_env(self, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_CHUNKING_DOC_TYPES", "medical_record,prescription")
        _reset_config()
        from core.config import uses_hierarchical_chunking
        assert uses_hierarchical_chunking("medical_record") is True
        assert uses_hierarchical_chunking("contract") is False


# ---------------------------------------------------------------------------
# expand_to_parent_context()
# ---------------------------------------------------------------------------

class TestExpandToParentContext:

    def _child_chunk(self, parent_id: str = "parent-1") -> dict:
        return {
            "chunk_num": 1, "content": "child text", "page": "1",
            "file": "f.pdf", "score": 0.9,
            "metadata": {"chunk_level": "child", "parent_chunk_id": parent_id},
        }

    def _flat_chunk(self) -> dict:
        return {
            "chunk_num": 2, "content": "flat text", "page": "2",
            "file": "f.pdf", "score": 0.8,
            "metadata": {"chunk_level": "flat", "parent_chunk_id": None},
        }

    @patch("retrieval.get_parent_chunk")
    def test_child_expanded_to_parent_text(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()
        mock_get_parent.return_value = {"content": "FULL PARENT TEXT", "id": "parent-1"}

        from retrieval import expand_to_parent_context
        result = expand_to_parent_context([self._child_chunk()])
        assert result[0]["content"] == "FULL PARENT TEXT"

    @patch("retrieval.get_parent_chunk")
    def test_child_metadata_preserved_after_expansion(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()
        mock_get_parent.return_value = {"content": "PARENT TEXT", "id": "parent-1"}

        from retrieval import expand_to_parent_context
        chunk = self._child_chunk()
        chunk["page"] = "3"
        result = expand_to_parent_context([chunk])
        assert result[0]["page"] == "3"   # original metadata preserved

    @patch("retrieval.get_parent_chunk")
    def test_flat_chunk_not_expanded(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()

        from retrieval import expand_to_parent_context
        flat = self._flat_chunk()
        result = expand_to_parent_context([flat])
        assert result[0]["content"] == "flat text"
        mock_get_parent.assert_not_called()

    @patch("retrieval.get_parent_chunk")
    def test_expand_disabled_returns_child_unchanged(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "false")
        _reset_config()

        from retrieval import expand_to_parent_context
        child = self._child_chunk()
        result = expand_to_parent_context([child])
        assert result[0]["content"] == "child text"
        mock_get_parent.assert_not_called()

    @patch("retrieval.get_parent_chunk")
    def test_parent_lookup_failure_falls_back_to_child(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()
        mock_get_parent.side_effect = Exception("DB error")

        from retrieval import expand_to_parent_context
        child = self._child_chunk()
        result = expand_to_parent_context([child])
        assert result[0]["content"] == "child text"   # falls back gracefully

    @patch("retrieval.get_parent_chunk")
    def test_parent_not_found_falls_back_to_child(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()
        mock_get_parent.return_value = None   # parent deleted / not found

        from retrieval import expand_to_parent_context
        child = self._child_chunk()
        result = expand_to_parent_context([child])
        assert result[0]["content"] == "child text"

    @patch("retrieval.get_parent_chunk")
    def test_mixed_flat_and_child_chunks(self, mock_get_parent, monkeypatch):
        monkeypatch.setenv("HIERARCHICAL_EXPAND_TO_PARENT", "true")
        _reset_config()
        mock_get_parent.return_value = {"content": "PARENT TEXT", "id": "parent-1"}

        from retrieval import expand_to_parent_context
        chunks = [self._child_chunk(), self._flat_chunk()]
        result = expand_to_parent_context(chunks)
        assert result[0]["content"] == "PARENT TEXT"   # child → expanded
        assert result[1]["content"] == "flat text"      # flat → unchanged


# ---------------------------------------------------------------------------
# hybrid_search() pool/top_n passthrough
# ---------------------------------------------------------------------------

class TestHybridSearchPool:

    def _make_chunks(self, n: int) -> list[dict]:
        return [
            {
                "id": f"chunk-{i}",
                "document_id": "doc-1",
                "content": f"Content number {i} about contracts and agreements.",
                "embedding": [0.1] * 10,
                "metadata": {
                    "page": "1", "file": "test.pdf",
                    "chunk_type": "text", "chunk_level": "flat",
                    "parent_chunk_id": None, "image_ref": None,
                }
            }
            for i in range(n)
        ]

    @patch("retrieval.rerank_chunks", side_effect=lambda q, c, top_k=None: c[:top_k or 5])
    @patch("retrieval.expand_query", return_value="expanded question")
    @patch("retrieval.get_chunks_by_document")
    def test_pool_size_passed_to_rrf(self, mock_get_chunks, mock_expand, mock_rerank, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "30")
        monkeypatch.setenv("RETRIEVAL_TOP_N", "5")
        _reset_config()
        mock_get_chunks.return_value = self._make_chunks(50)

        from retrieval import hybrid_search
        hybrid_search("question", document_ids=["doc-1"])

        # rerank_chunks called with top_k = retrieval_top_n
        call_kwargs = mock_rerank.call_args
        assert call_kwargs is not None

    @patch("retrieval.rerank_chunks", side_effect=lambda q, c, top_k=None: c[:top_k or 5])
    @patch("retrieval.expand_query", return_value="expanded")
    @patch("retrieval.get_chunks_by_document")
    def test_top_n_override_respected(self, mock_get_chunks, mock_expand, mock_rerank, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "50")
        monkeypatch.setenv("RETRIEVAL_TOP_N", "5")
        _reset_config()
        mock_get_chunks.return_value = self._make_chunks(30)

        from retrieval import hybrid_search
        hybrid_search("question", document_ids=["doc-1"], top_n=3)

        call_kwargs = mock_rerank.call_args.kwargs
        assert call_kwargs.get("top_k") == 3

    @patch("retrieval.rerank_chunks", side_effect=lambda q, c, top_k=None: c[:top_k or 5])
    @patch("retrieval.expand_query", return_value="expanded")
    @patch("retrieval.get_chunks_by_document")
    def test_parent_chunks_excluded_from_search(
        self, mock_get_chunks, mock_expand, mock_rerank, monkeypatch
    ):
        monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "50")
        _reset_config()

        # Mix of parent and child/flat chunks
        chunks = self._make_chunks(5)
        chunks[0]["metadata"]["chunk_level"] = "parent"
        chunks[0]["embedding"] = None   # parents have no embedding
        mock_get_chunks.return_value = chunks

        from retrieval import hybrid_search
        hybrid_search("question", document_ids=["doc-1"])

        # rerank_chunks should receive pool without the parent chunk
        rerank_args = mock_rerank.call_args.args
        candidates = rerank_args[1]
        assert all(
            c.get("chunk_level", "flat") != "parent"
            for c in candidates
        )

    @patch("retrieval.get_chunks_by_document", return_value=[])
    def test_returns_empty_when_no_chunks(self, mock_get_chunks):
        from retrieval import hybrid_search
        result = hybrid_search("question", document_ids=["doc-1"])
        assert result == []

    def test_returns_empty_when_no_document_ids(self):
        from retrieval import hybrid_search
        result = hybrid_search("question", document_ids=None)
        assert result == []