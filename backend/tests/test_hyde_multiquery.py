"""
tests/test_hyde_multiquery.py

Unit tests for backend/hyde.py (D2):
- generate_hyde_passage() — success, fallback, doc_type hint
- generate_query_variants() — success, fallback, dedup, min variants
- merge_and_dedupe() — dedup logic, chunk_num renumbering, top_n cap
- normalise_retrieval_mode() — all modes, aliases, unknown values
- hybrid_search_with_mode() — mode dispatch (mocked hybrid_search)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _reset_config():
    import core.config as cfg_mod
    cfg_mod._config_instance = None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("RETRIEVAL_TOP_N", "5")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_POOL", "20")
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# normalise_retrieval_mode
# ---------------------------------------------------------------------------

class TestNormaliseRetrievalMode:

    def test_standard_unchanged(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("standard") == "standard"

    def test_none_alias_maps_to_standard(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("none") == "standard"

    def test_hyde_unchanged(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("hyde") == "hyde"

    def test_multiquery_unchanged(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("multiquery") == "multiquery"

    def test_none_input_defaults_to_standard(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode(None) == "standard"

    def test_empty_string_defaults_to_standard(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("") == "standard"

    def test_unknown_value_defaults_to_standard(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("magic_retrieval") == "standard"

    def test_case_insensitive(self):
        from hyde import normalise_retrieval_mode
        assert normalise_retrieval_mode("HYDE") == "hyde"
        assert normalise_retrieval_mode("MultiQuery") == "multiquery"
        assert normalise_retrieval_mode("NONE") == "standard"


# ---------------------------------------------------------------------------
# generate_hyde_passage
# ---------------------------------------------------------------------------

class TestGenerateHyDEPassage:

    @patch("hyde.call_llm")
    def test_returns_passage_on_success(self, mock_call):
        from hyde import HyDEPassage, generate_hyde_passage
        mock_call.return_value = HyDEPassage(
            passage="The total amount due is five thousand dollars payable within 30 days."
        )
        result = generate_hyde_passage("What is the total amount due?")
        assert "five thousand" in result
        assert isinstance(result, str)

    @patch("hyde.call_llm")
    def test_uses_response_model_hyde_passage(self, mock_call):
        from hyde import HyDEPassage, generate_hyde_passage
        mock_call.return_value = HyDEPassage(passage="some passage")
        generate_hyde_passage("What is the due date?")
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("response_model") == HyDEPassage

    @patch("hyde.call_llm")
    def test_doc_type_included_in_prompt(self, mock_call):
        from hyde import HyDEPassage, generate_hyde_passage
        mock_call.return_value = HyDEPassage(passage="relevant passage")
        generate_hyde_passage("What are the payment terms?", doc_type="contract")
        prompt = mock_call.call_args.args[0]
        assert "contract" in prompt

    @patch("hyde.call_llm")
    def test_falls_back_to_original_question_on_error(self, mock_call):
        mock_call.side_effect = Exception("LLM error")
        from hyde import generate_hyde_passage
        result = generate_hyde_passage("What is the penalty clause?")
        assert result == "What is the penalty clause?"

    @patch("hyde.call_llm")
    def test_falls_back_when_passage_empty(self, mock_call):
        from hyde import HyDEPassage, generate_hyde_passage
        mock_call.return_value = HyDEPassage(passage="   ")
        result = generate_hyde_passage("original question")
        assert result == "original question"

    @patch("hyde.call_llm")
    def test_call_type_is_hyde(self, mock_call):
        from hyde import HyDEPassage, generate_hyde_passage
        mock_call.return_value = HyDEPassage(passage="passage")
        generate_hyde_passage("question")
        assert mock_call.call_args.kwargs.get("call_type") == "hyde"


# ---------------------------------------------------------------------------
# generate_query_variants
# ---------------------------------------------------------------------------

class TestGenerateQueryVariants:

    @patch("hyde.call_llm")
    def test_returns_original_plus_variants(self, mock_call):
        from hyde import QueryVariants, generate_query_variants
        mock_call.return_value = QueryVariants(variants=[
            "What is the payment amount?",
            "How much needs to be paid?",
        ])
        result = generate_query_variants("What is the total due?")
        assert result[0] == "What is the total due?"   # original always first
        assert len(result) >= 2

    @patch("hyde.call_llm")
    def test_deduplicates_if_variant_echoes_original(self, mock_call):
        from hyde import QueryVariants, generate_query_variants
        question = "What is the total due?"
        mock_call.return_value = QueryVariants(variants=[
            question,                         # same as original — should be deduped
            "What is the amount owed?",
        ])
        result = generate_query_variants(question)
        # Original should appear exactly once
        assert result.count(question) == 1

    @patch("hyde.call_llm")
    def test_falls_back_to_original_on_error(self, mock_call):
        mock_call.side_effect = Exception("LLM down")
        from hyde import generate_query_variants
        result = generate_query_variants("How long is the notice period?")
        assert result == ["How long is the notice period?"]

    @patch("hyde.call_llm")
    def test_falls_back_when_variants_empty(self, mock_call):
        from hyde import QueryVariants, generate_query_variants
        mock_call.return_value = QueryVariants(variants=[])
        result = generate_query_variants("original")
        assert result == ["original"]

    @patch("hyde.call_llm")
    def test_call_type_is_multiquery(self, mock_call):
        from hyde import QueryVariants, generate_query_variants
        mock_call.return_value = QueryVariants(variants=["v1", "v2"])
        generate_query_variants("question")
        assert mock_call.call_args.kwargs.get("call_type") == "multiquery"

    @patch("hyde.call_llm")
    def test_uses_response_model_query_variants(self, mock_call):
        from hyde import QueryVariants, generate_query_variants
        mock_call.return_value = QueryVariants(variants=["v1"])
        generate_query_variants("question")
        assert mock_call.call_args.kwargs.get("response_model") == QueryVariants


# ---------------------------------------------------------------------------
# merge_and_dedupe
# ---------------------------------------------------------------------------

class TestMergeAndDedupe:

    def _make_chunk(self, content: str, chunk_num: int = 1) -> dict:
        return {
            "chunk_num": chunk_num,
            "content":   content,
            "page":      "1",
            "file":      "test.pdf",
            "score":     0.9,
        }

    def test_deduplicates_identical_content(self):
        from hyde import merge_and_dedupe
        chunk = self._make_chunk("identical content")
        results = [[chunk], [chunk], [chunk]]
        merged = merge_and_dedupe(results, top_n=10)
        assert len(merged) == 1

    def test_keeps_unique_chunks(self):
        from hyde import merge_and_dedupe
        results = [
            [self._make_chunk("content A"), self._make_chunk("content B")],
            [self._make_chunk("content C"), self._make_chunk("content A")],
        ]
        merged = merge_and_dedupe(results, top_n=10)
        assert len(merged) == 3

    def test_renumbers_chunk_num_sequentially(self):
        from hyde import merge_and_dedupe
        results = [
            [self._make_chunk("A", chunk_num=5), self._make_chunk("B", chunk_num=3)],
        ]
        merged = merge_and_dedupe(results, top_n=10)
        nums = [c["chunk_num"] for c in merged]
        assert nums == list(range(1, len(merged) + 1))

    def test_respects_top_n(self):
        from hyde import merge_and_dedupe
        chunks = [self._make_chunk(f"content {i}") for i in range(20)]
        results = [chunks]
        merged = merge_and_dedupe(results, top_n=5)
        assert len(merged) <= 5

    def test_empty_input_returns_empty(self):
        from hyde import merge_and_dedupe
        merged = merge_and_dedupe([], top_n=5)
        assert merged == []

    def test_first_occurrence_wins(self):
        """When same content appears in multiple result lists, first list wins."""
        from hyde import merge_and_dedupe
        chunk_a1 = {**self._make_chunk("shared"), "score": 0.9, "page": "1"}
        chunk_a2 = {**self._make_chunk("shared"), "score": 0.5, "page": "2"}
        merged = merge_and_dedupe([[chunk_a1], [chunk_a2]], top_n=5)
        assert len(merged) == 1
        assert merged[0]["page"] == "1"   # first occurrence kept


# ---------------------------------------------------------------------------
# hybrid_search_with_mode — dispatch
# ---------------------------------------------------------------------------

class TestHybridSearchWithMode:

    def _sample_chunks(self, n=3):
        return [
            {"chunk_num": i+1, "content": f"chunk {i}", "page": "1",
             "file": "f.pdf", "score": 0.9, "chunk_level": "flat",
             "parent_chunk_id": None}
            for i in range(n)
        ]

    @patch("retrieval.expand_to_parent_context", side_effect=lambda x: x)
    @patch("retrieval.hybrid_search")
    def test_standard_mode_calls_hybrid_search_once(self, mock_hs, mock_expand):
        mock_hs.return_value = self._sample_chunks()
        from retrieval import hybrid_search_with_mode
        hybrid_search_with_mode("question", document_ids=["doc-1"], retrieval_mode="standard")
        assert mock_hs.call_count == 1
        # No dense_query_override for standard mode
        call_kwargs = mock_hs.call_args.kwargs
        assert call_kwargs.get("dense_query_override") is None

    @patch("retrieval.expand_to_parent_context", side_effect=lambda x: x)
    @patch("retrieval.generate_hyde_passage", return_value="hypothetical passage")
    @patch("retrieval.hybrid_search")
    def test_hyde_mode_passes_passage_as_override(self, mock_hs, mock_hyde, mock_expand):
        mock_hs.return_value = self._sample_chunks()
        from retrieval import hybrid_search_with_mode
        hybrid_search_with_mode("question", document_ids=["doc-1"], retrieval_mode="hyde")
        call_kwargs = mock_hs.call_args.kwargs
        assert call_kwargs.get("dense_query_override") == "hypothetical passage"

    @patch("retrieval.expand_to_parent_context", side_effect=lambda x: x)
    @patch("retrieval.merge_and_dedupe", return_value=[])
    @patch("retrieval.generate_query_variants", return_value=["q1", "q2", "q3"])
    @patch("retrieval.hybrid_search")
    def test_multiquery_mode_calls_hybrid_search_per_variant(
        self, mock_hs, mock_variants, mock_merge, mock_expand
    ):
        mock_hs.return_value = self._sample_chunks()
        from retrieval import hybrid_search_with_mode
        hybrid_search_with_mode("question", document_ids=["doc-1"], retrieval_mode="multiquery")
        # hybrid_search called once per variant (3 variants)
        assert mock_hs.call_count == 3

    @patch("retrieval.expand_to_parent_context", side_effect=lambda x: x)
    @patch("retrieval.hybrid_search")
    def test_none_mode_alias_works(self, mock_hs, mock_expand):
        mock_hs.return_value = self._sample_chunks()
        from retrieval import hybrid_search_with_mode
        hybrid_search_with_mode("question", document_ids=["doc-1"], retrieval_mode="none")
        assert mock_hs.call_count == 1

    @patch("retrieval.expand_to_parent_context", side_effect=lambda x: x)
    @patch("retrieval.hybrid_search")
    def test_expand_to_parent_context_always_called(self, mock_hs, mock_expand):
        mock_hs.return_value = self._sample_chunks()
        from retrieval import hybrid_search_with_mode
        hybrid_search_with_mode("question", document_ids=["doc-1"])
        mock_expand.assert_called_once()


# ---------------------------------------------------------------------------
# QueryRequest validator — retrieval_mode
# ---------------------------------------------------------------------------

class TestQueryRequestValidator:

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.query import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @patch("retrieval.query_document", return_value={"answer": "ok", "sources": [], "type": "document"})
    def test_valid_modes_accepted(self, mock_qd):
        client = self._make_app()
        for mode in ["standard", "none", "hyde", "multiquery"]:
            resp = client.post("/query", json={"question": "test", "retrieval_mode": mode})
            assert resp.status_code == 200, f"Mode '{mode}' rejected unexpectedly"

    def test_invalid_mode_returns_422(self):
        client = self._make_app()
        resp = client.post("/query", json={"question": "test", "retrieval_mode": "magic"})
        assert resp.status_code == 422

    @patch("retrieval.query_document", return_value={"answer": "ok", "sources": [], "type": "document"})
    def test_default_mode_is_standard(self, mock_qd):
        client = self._make_app()
        client.post("/query", json={"question": "test"})
        _, kwargs = mock_qd.call_args
        assert kwargs.get("retrieval_mode") == "standard"

    @patch("retrieval.query_document", return_value={"answer": "ok", "sources": [], "type": "document"})
    def test_retrieval_mode_forwarded_to_query_document(self, mock_qd):
        client = self._make_app()
        client.post("/query", json={"question": "test", "retrieval_mode": "hyde"})
        _, kwargs = mock_qd.call_args
        assert kwargs.get("retrieval_mode") == "hyde"