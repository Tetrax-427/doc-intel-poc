"""
backend/hyde.py — D2 HyDE + Multi-Query Retrieval

Two retrieval enhancement modes, both off by default (per-session opt-in):

HyDE (Hypothetical Document Embeddings):
    Instead of embedding the user's raw question for dense retrieval, generate
    a hypothetical "ideal answer" passage via LLM and embed *that*. For vague
    questions, the hypothetical passage often uses vocabulary closer to the
    document's actual wording — improving dense retrieval recall.

Multi-Query:
    Generate 2-3 paraphrased variants of the question, run hybrid_search()
    for each, merge and deduplicate results by content hash. Useful when the
    question's phrasing doesn't match the document's terminology.

Both use Instructor (response_model=) via call_llm() and automatically
benefit from Group C's fallback chain. They never use the per-call C3
provider override — they are retrieval-internal calls, not user-facing.

Retrieval modes (matches QueryRequest.retrieval_mode):
    "standard"  — plain hybrid search (default, no extra LLM calls)
    "none"      — alias for "standard"
    "hyde"      — HyDE passage replaces query for dense embedding step
    "multiquery"— 2-3 variants, merged + deduped results
"""

from __future__ import annotations

import hashlib
from pydantic import BaseModel, Field

from llm.engine import call_llm
from core.logger import get_logger

logger = get_logger("hyde")


# ---------------------------------------------------------------------------
# Pydantic response models (Instructor)
# ---------------------------------------------------------------------------

class HyDEPassage(BaseModel):
    """Hypothetical ideal-answer passage for dense retrieval."""
    passage: str = Field(
        description=(
            "A short passage (2-4 sentences) that would be an ideal answer to the "
            "question if it appeared verbatim in the document. Use vocabulary and "
            "phrasing likely found in professional documents of this type. "
            "Do NOT include 'I don't know' or hedging — write as if the answer exists."
        )
    )


class QueryVariants(BaseModel):
    """Paraphrased question variants for multi-query retrieval."""
    variants: list[str] = Field(
        description=(
            "2 to 3 paraphrased versions of the question that use different "
            "vocabulary or phrasing but ask for the same information. "
            "Each variant should be a complete, standalone question."
        ),
        min_length=2,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

def generate_hyde_passage(question: str, doc_type: str = "general") -> str:
    """
    Generate a hypothetical ideal-answer passage for the given question.

    The passage is used in place of the raw question for the dense embedding
    step in hybrid_search(). BM25 still uses the original question.

    Args:
        question: The user's question.
        doc_type: Document type hint — helps the LLM generate vocabulary
                  that matches the document's domain (e.g. "contract" →
                  legal terminology).

    Returns:
        The hypothetical passage string. Falls back to the original question
        on any error so retrieval always proceeds.
    """
    doc_type_hint = f" The document is a {doc_type}." if doc_type and doc_type != "general" else ""

    prompt = (
        f"Question: {question}\n"
        f"{doc_type_hint}\n\n"
        "Generate a short passage (2-4 sentences) that would be an ideal answer "
        "to this question if it appeared verbatim in the document. "
        "Use vocabulary and phrasing typical of professional documents of this type."
    )

    try:
        result: HyDEPassage = call_llm(
            prompt,
            temperature=0.3,       # slight creativity to vary vocabulary
            max_tokens=200,
            call_type="hyde",
            response_model=HyDEPassage,
        )
        passage = result.passage.strip()
        if not passage:
            return question
        logger.info("HyDE passage generated", question_len=len(question), passage_len=len(passage))
        return passage

    except Exception as exc:
        logger.warning("HyDE passage generation failed — using original question", error=str(exc))
        return question


# ---------------------------------------------------------------------------
# Multi-Query
# ---------------------------------------------------------------------------

def generate_query_variants(question: str) -> list[str]:
    """
    Generate 2-3 paraphrased variants of the question for multi-query retrieval.

    The original question is always included in the returned list (prepended),
    so callers always have at least 1 query even if generation fails.

    Args:
        question: The user's original question.

    Returns:
        List of query strings: [original] + [variant1, variant2, ...].
        Falls back to [question] on any error.
    """
    prompt = (
        f"Original question: {question}\n\n"
        "Generate 2 to 3 paraphrased versions of this question that use different "
        "vocabulary or phrasing but ask for the same information. "
        "Each variant should be a complete, standalone question."
    )

    try:
        result: QueryVariants = call_llm(
            prompt,
            temperature=0.4,
            max_tokens=300,
            call_type="multiquery",
            response_model=QueryVariants,
        )

        variants = [v.strip() for v in result.variants if v.strip()]
        if not variants:
            return [question]

        # Original first, then variants — dedup in case LLM echoes original
        all_queries = [question] + [v for v in variants if v.lower() != question.lower()]
        logger.info(
            "Query variants generated",
            original=question[:60],
            variants=len(variants),
        )
        return all_queries

    except Exception as exc:
        logger.warning("Query variant generation failed — using original only", error=str(exc))
        return [question]


# ---------------------------------------------------------------------------
# Merge + deduplicate
# ---------------------------------------------------------------------------

def merge_and_dedupe(
    results_per_query: list[list[dict]],
    top_n: int,
) -> list[dict]:
    """
    Merge chunk result lists from multiple queries and deduplicate by content.

    Deduplication strategy:
    - Hash each chunk's content (MD5 — fast, collision-safe for this use case).
    - First occurrence wins (earlier queries / higher-scored results take priority).
    - After dedup, re-number chunk_num sequentially.

    Args:
        results_per_query: List of result lists, one per query variant.
                           Each inner list is already reranked by Cohere.
        top_n:             Maximum chunks to return after merge/dedup.

    Returns:
        Deduplicated list of up to top_n chunks, renumbered from 1.
    """
    seen_hashes: set[str] = set()
    merged: list[dict] = []

    for result_list in results_per_query:
        for chunk in result_list:
            content_hash = hashlib.md5(
                chunk.get("content", "").encode("utf-8", errors="replace")
            ).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                merged.append(chunk)

            if len(merged) >= top_n * 3:
                # Safety: cap merge buffer to avoid runaway memory on huge result sets
                break

    # Re-number chunk_num sequentially
    final = merged[:top_n]
    for i, chunk in enumerate(final):
        chunk["chunk_num"] = i + 1

    logger.info(
        "Merge and dedup complete",
        total_input=sum(len(r) for r in results_per_query),
        after_dedup=len(merged),
        returned=len(final),
    )
    return final


# ---------------------------------------------------------------------------
# Retrieval mode helpers
# ---------------------------------------------------------------------------

VALID_RETRIEVAL_MODES = {"standard", "none", "hyde", "multiquery"}


def normalise_retrieval_mode(mode: str | None) -> str:
    """
    Normalise retrieval_mode string to one of: standard, hyde, multiquery.
    "none" is an alias for "standard". Unknown values fall back to "standard".
    """
    if not mode:
        return "standard"
    normalised = mode.strip().lower()
    if normalised == "none":
        return "standard"
    if normalised in VALID_RETRIEVAL_MODES:
        return normalised
    logger.warning("Unknown retrieval_mode — defaulting to standard", mode=mode)
    return "standard"