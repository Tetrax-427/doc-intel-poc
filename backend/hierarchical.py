"""
backend/hierarchical.py — D1 Hierarchical Chunking

Produces parent chunks (large, section-level) and child chunks (small,
sentence/paragraph-level) linked via parent_chunk_id.

Retrieval searches child chunks for precise matching but can expand to the
parent chunk's full text as LLM context (controlled by
config.hierarchical_expand_to_parent).

Only called for doc types listed in config.hierarchical_chunking_doc_types.
All other doc types continue to use flat SentenceSplitter chunking in
ingestion.py — this module is never imported for those paths.

Chunk metadata shape (extends flat chunk metadata):
    {
        "page":            str,
        "file":            str,
        "chunk_type":      "text",
        "image_ref":       None,
        "chunk_level":     "parent" | "child" | "flat",
        "parent_chunk_id": str | None,   # set on child chunks only
    }

chunk_level="flat" is used for table and vision description chunks, which
are always stored flat regardless of doc type — they don't split cleanly
into parent/child boundaries.
"""

from __future__ import annotations

import uuid

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument

from core.config import config as app_config
from core.logger import get_logger

logger = get_logger("hierarchical")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hierarchical_chunks(
    page_text: str,
    page_num: int,
    file_name: str,
    document_id: str,
    embed_fn,                      # callable: (text: str) -> list[float]
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
) -> list[dict]:
    """
    Build parent + child chunk rows for a single page of text.

    Strategy:
    1. Split page text into parent chunks (large windows, default 2000 chars).
    2. For each parent chunk, split again into child chunks (small windows,
       default 400 chars).
    3. Each child chunk stores its parent's ID in parent_chunk_id.
    4. Both parent and child rows are returned — both are stored in the DB.
       Retrieval searches child rows; expand_to_parent_context() in
       retrieval.py fetches the parent row when building LLM context.

    Args:
        page_text:         Raw text of the page.
        page_num:          1-based page number (stored in metadata).
        file_name:         Original file name (stored in metadata).
        document_id:       Document UUID (stored on each chunk row).
        embed_fn:          Embedding function — called once per child chunk.
                           Parent chunks are NOT embedded (never searched directly).
        parent_chunk_size: Override for config.hierarchical_parent_chunk_size.
        child_chunk_size:  Override for config.hierarchical_child_chunk_size.

    Returns:
        List of chunk dicts ready for insert_chunks(). Mix of parent and child
        rows. Empty list if page_text is blank.
    """
    if not page_text or not page_text.strip():
        return []

    p_size = parent_chunk_size or app_config.hierarchical_parent_chunk_size
    c_size = child_chunk_size  or app_config.hierarchical_child_chunk_size

    # Sanity: child must be smaller than parent
    if c_size >= p_size:
        logger.warning(
            "child_chunk_size >= parent_chunk_size — falling back to flat chunking",
            child=c_size, parent=p_size, page=page_num,
        )
        return _flat_fallback(page_text, page_num, file_name, document_id, embed_fn)

    parent_splitter = SentenceSplitter(
        chunk_size=p_size,
        chunk_overlap=min(200, p_size // 10),   # 10% overlap, max 200 chars
    )
    child_splitter = SentenceSplitter(
        chunk_size=c_size,
        chunk_overlap=min(50, c_size // 8),     # ~12% overlap, max 50 chars
    )

    llama_doc    = LlamaIndexDocument(text=page_text)
    parent_nodes = parent_splitter.get_nodes_from_documents([llama_doc])

    chunk_rows: list[dict] = []

    for parent_node in parent_nodes:
        parent_text = parent_node.text.replace("\x00", " ").strip()
        if not parent_text:
            continue

        parent_id = str(uuid.uuid4())

        # Parent row — stored but NOT embedded (retrieved only via child lookup)
        chunk_rows.append({
            "id":          parent_id,
            "document_id": document_id,
            "content":     parent_text,
            "embedding":   None,          # no embedding on parent
            "metadata": {
                "page":            str(page_num),
                "file":            file_name,
                "chunk_type":      "text",
                "image_ref":       None,
                "chunk_level":     "parent",
                "parent_chunk_id": None,
            },
        })

        # Child rows — embedded and searched during retrieval
        child_llama_doc = LlamaIndexDocument(text=parent_text)
        child_nodes     = child_splitter.get_nodes_from_documents([child_llama_doc])

        for child_node in child_nodes:
            child_text = child_node.text.replace("\x00", " ").strip()
            if not child_text:
                continue

            embedding = embed_fn(child_text)
            chunk_rows.append({
                "id":          str(uuid.uuid4()),
                "document_id": document_id,
                "content":     child_text,
                "embedding":   embedding,
                "metadata": {
                    "page":            str(page_num),
                    "file":            file_name,
                    "chunk_type":      "text",
                    "image_ref":       None,
                    "chunk_level":     "child",
                    "parent_chunk_id": parent_id,
                },
            })

        logger.debug(
            "Parent chunk built",
            parent_id=parent_id,
            page=page_num,
            parent_chars=len(parent_text),
            children=len([r for r in chunk_rows if r.get("metadata", {}).get("parent_chunk_id") == parent_id]),
        )

    logger.info(
        "Hierarchical chunking complete for page",
        page=page_num,
        file=file_name,
        total_rows=len(chunk_rows),
        parents=sum(1 for r in chunk_rows if r["metadata"]["chunk_level"] == "parent"),
        children=sum(1 for r in chunk_rows if r["metadata"]["chunk_level"] == "child"),
    )

    return chunk_rows


# ---------------------------------------------------------------------------
# Flat-chunk metadata helper — used for table + vision chunks in all modes
# ---------------------------------------------------------------------------

def make_flat_chunk_metadata(
    page_num: int,
    file_name: str,
    chunk_type: str = "text",
    image_ref: str | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Build metadata dict for a flat chunk (table, vision description, or
    flat-mode text chunk). Always sets chunk_level="flat" and
    parent_chunk_id=None so retrieval code can safely call .get() on these.
    """
    base = {
        "page":            str(page_num),
        "file":            file_name,
        "chunk_type":      chunk_type,
        "image_ref":       image_ref,
        "chunk_level":     "flat",
        "parent_chunk_id": None,
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Flat fallback — used when child >= parent (misconfiguration safety net)
# ---------------------------------------------------------------------------

def _flat_fallback(
    page_text: str,
    page_num: int,
    file_name: str,
    document_id: str,
    embed_fn,
) -> list[dict]:
    """
    Emergency flat-chunk fallback when chunk sizes are misconfigured.
    Uses default SentenceSplitter (app_config.chunk_size).
    """
    from llama_index.core.node_parser import SentenceSplitter as _SS
    from core.config import config as _cfg

    splitter = _SS(chunk_size=_cfg.chunk_size, chunk_overlap=_cfg.chunk_overlap)
    nodes    = splitter.get_nodes_from_documents([LlamaIndexDocument(text=page_text)])
    rows     = []

    for node in nodes:
        text = node.text.replace("\x00", " ").strip()
        if not text:
            continue
        rows.append({
            "document_id": document_id,
            "content":     text,
            "embedding":   embed_fn(text),
            "metadata":    make_flat_chunk_metadata(page_num, file_name),
        })

    return rows