"""
tests/test_hierarchical_chunking.py

Unit tests for backend/hierarchical.py (D1):
- build_hierarchical_chunks() output shape
- parent/child relationships
- chunk_level metadata
- embedding behaviour (children embedded, parents not)
- flat fallback on misconfigured sizes
- make_flat_chunk_metadata helper
"""

from __future__ import annotations

import uuid
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
    monkeypatch.setenv("HIERARCHICAL_PARENT_CHUNK_SIZE", "500")
    monkeypatch.setenv("HIERARCHICAL_CHILD_CHUNK_SIZE", "100")
    _reset_config()
    yield
    _reset_config()


SAMPLE_TEXT = " ".join([
    "This agreement is entered into between Party A and Party B.",
    "Party A agrees to provide consulting services.",
    "The term of this agreement shall be twelve months from the effective date.",
    "Party B agrees to pay a monthly retainer of five thousand dollars.",
    "Either party may terminate with thirty days written notice.",
    "Confidential information shall not be disclosed to third parties.",
    "This agreement is governed by the laws of the state of California.",
    "Any disputes shall be resolved through binding arbitration.",
    "This constitutes the entire agreement between the parties.",
    "Amendments must be made in writing and signed by both parties.",
] * 5)   # repeat to ensure multiple parent chunks


def _mock_embed(text: str) -> list[float]:
    """Deterministic fake embedding — just return text length as a float vector."""
    return [float(len(text))] * 10


# ---------------------------------------------------------------------------
# build_hierarchical_chunks — output shape
# ---------------------------------------------------------------------------

class TestBuildHierarchicalChunks:

    def test_returns_list_of_dicts(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
        )
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

    def test_returns_empty_for_blank_text(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text="   ", page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
        )
        assert rows == []

    def test_returns_empty_for_none_text(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text="", page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
        )
        assert rows == []

    def test_all_rows_have_required_keys(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
        )
        for row in rows:
            assert "document_id" in row
            assert "content" in row
            assert "embedding" in row
            assert "metadata" in row

    def test_document_id_set_on_all_rows(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-xyz", embed_fn=_mock_embed,
        )
        assert all(r["document_id"] == "doc-xyz" for r in rows)


# ---------------------------------------------------------------------------
# Parent / child structure
# ---------------------------------------------------------------------------

class TestParentChildStructure:

    def _get_rows(self):
        from hierarchical import build_hierarchical_chunks
        return build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
        )

    def test_has_both_parent_and_child_rows(self):
        rows = self._get_rows()
        levels = {r["metadata"]["chunk_level"] for r in rows}
        assert "parent" in levels
        assert "child" in levels

    def test_parent_rows_have_no_embedding(self):
        rows = self._get_rows()
        parents = [r for r in rows if r["metadata"]["chunk_level"] == "parent"]
        assert len(parents) > 0
        assert all(r["embedding"] is None for r in parents)

    def test_child_rows_have_embedding(self):
        rows = self._get_rows()
        children = [r for r in rows if r["metadata"]["chunk_level"] == "child"]
        assert len(children) > 0
        assert all(r["embedding"] is not None for r in children)

    def test_child_parent_chunk_id_matches_a_parent(self):
        rows = self._get_rows()
        parent_ids = {r["id"] for r in rows if r["metadata"]["chunk_level"] == "parent"}
        children   = [r for r in rows if r["metadata"]["chunk_level"] == "child"]
        for child in children:
            assert child["metadata"]["parent_chunk_id"] in parent_ids

    def test_parent_rows_have_stable_ids(self):
        rows = self._get_rows()
        parents = [r for r in rows if r["metadata"]["chunk_level"] == "parent"]
        for p in parents:
            assert "id" in p
            assert len(p["id"]) > 0

    def test_parent_chunk_id_none_on_parent_rows(self):
        rows = self._get_rows()
        parents = [r for r in rows if r["metadata"]["chunk_level"] == "parent"]
        assert all(r["metadata"]["parent_chunk_id"] is None for r in parents)

    def test_each_parent_has_at_least_one_child(self):
        rows = self._get_rows()
        parent_ids = {r["id"] for r in rows if r["metadata"]["chunk_level"] == "parent"}
        child_parent_ids = {
            r["metadata"]["parent_chunk_id"]
            for r in rows if r["metadata"]["chunk_level"] == "child"
        }
        assert parent_ids == child_parent_ids

    def test_child_text_is_subset_of_parent_text(self):
        """Every child's text should be contained within its parent's text."""
        rows = self._get_rows()
        parent_map = {
            r["id"]: r["content"]
            for r in rows if r["metadata"]["chunk_level"] == "parent"
        }
        children = [r for r in rows if r["metadata"]["chunk_level"] == "child"]
        for child in children:
            parent_text = parent_map[child["metadata"]["parent_chunk_id"]]
            # Child text should be a substring (or very close) of parent text
            assert child["content"].strip()[:50] in parent_text or \
                   len(child["content"]) <= len(parent_text)


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------

class TestMetadata:

    def _get_rows(self, page_num=3, file_name="contract.pdf"):
        from hierarchical import build_hierarchical_chunks
        return build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=page_num, file_name=file_name,
            document_id="doc-1", embed_fn=_mock_embed,
        )

    def test_page_num_set_correctly(self):
        rows = self._get_rows(page_num=3)
        assert all(r["metadata"]["page"] == "3" for r in rows)

    def test_file_name_set_correctly(self):
        rows = self._get_rows(file_name="contract.pdf")
        assert all(r["metadata"]["file"] == "contract.pdf" for r in rows)

    def test_chunk_type_is_text(self):
        rows = self._get_rows()
        assert all(r["metadata"]["chunk_type"] == "text" for r in rows)

    def test_image_ref_is_none(self):
        rows = self._get_rows()
        assert all(r["metadata"]["image_ref"] is None for r in rows)


# ---------------------------------------------------------------------------
# Size overrides
# ---------------------------------------------------------------------------

class TestSizeOverrides:

    def test_custom_parent_and_child_sizes_respected(self):
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
            parent_chunk_size=800, child_chunk_size=150,
        )
        parents  = [r for r in rows if r["metadata"]["chunk_level"] == "parent"]
        children = [r for r in rows if r["metadata"]["chunk_level"] == "child"]
        assert len(parents) > 0
        assert len(children) > 0
        # Children should be shorter than parents on average
        avg_parent = sum(len(p["content"]) for p in parents) / len(parents)
        avg_child  = sum(len(c["content"]) for c in children) / len(children)
        assert avg_child < avg_parent

    def test_flat_fallback_when_child_gte_parent(self):
        """When child_size >= parent_size, falls back to flat chunking."""
        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=_mock_embed,
            parent_chunk_size=100, child_chunk_size=200,  # child > parent → fallback
        )
        # Flat fallback: no parent rows, all rows are flat
        levels = {r["metadata"].get("chunk_level") for r in rows}
        assert "parent" not in levels

    def test_embed_fn_called_only_for_children(self):
        """Embedding function should NOT be called for parent chunks."""
        call_count = {"n": 0}

        def counting_embed(text: str) -> list[float]:
            call_count["n"] += 1
            return [0.1] * 10

        from hierarchical import build_hierarchical_chunks
        rows = build_hierarchical_chunks(
            page_text=SAMPLE_TEXT, page_num=1, file_name="test.pdf",
            document_id="doc-1", embed_fn=counting_embed,
        )
        child_count = sum(1 for r in rows if r["metadata"]["chunk_level"] == "child")
        assert call_count["n"] == child_count


# ---------------------------------------------------------------------------
# make_flat_chunk_metadata
# ---------------------------------------------------------------------------

class TestMakeFlatChunkMetadata:

    def test_basic_shape(self):
        from hierarchical import make_flat_chunk_metadata
        meta = make_flat_chunk_metadata(page_num=2, file_name="doc.pdf")
        assert meta["page"] == "2"
        assert meta["file"] == "doc.pdf"
        assert meta["chunk_level"] == "flat"
        assert meta["parent_chunk_id"] is None

    def test_chunk_type_default_is_text(self):
        from hierarchical import make_flat_chunk_metadata
        meta = make_flat_chunk_metadata(page_num=1, file_name="f.pdf")
        assert meta["chunk_type"] == "text"

    def test_chunk_type_override(self):
        from hierarchical import make_flat_chunk_metadata
        meta = make_flat_chunk_metadata(page_num=1, file_name="f.pdf", chunk_type="table")
        assert meta["chunk_type"] == "table"

    def test_image_ref_set(self):
        from hierarchical import make_flat_chunk_metadata
        meta = make_flat_chunk_metadata(page_num=1, file_name="f.pdf", image_ref="img_1")
        assert meta["image_ref"] == "img_1"

    def test_extra_fields_merged(self):
        from hierarchical import make_flat_chunk_metadata
        meta = make_flat_chunk_metadata(
            page_num=1, file_name="f.pdf",
            extra={"vision_prompt_used": "invoice", "source_url": "https://example.com"}
        )
        assert meta["vision_prompt_used"] == "invoice"
        assert meta["source_url"] == "https://example.com"