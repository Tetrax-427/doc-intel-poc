"""
backend/hyde.py — D2 HyDE + Multi-Query Retrieval
"""

from __future__ import annotations

import hashlib
from pydantic import BaseModel, Field

from llm.engine import call_llm
from core.logger import get_logger

logger = get_logger("hyde")


# ---------------------------------------------------------------------------
# Pydantic response models
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

_HYDE_SYSTEM = (
    "Generate a short passage (2-4 sentences) that would be an ideal answer "
    "to the question if it appeared verbatim in the document. "
    "Use vocabulary and phrasing typical of professional documents of this type. "
    "Do NOT include 'I don't know' or hedging — write as if the answer exists."
)

_MULTIQUERY_SYSTEM = (
    "Generate 2 to 3 paraphrased versions of the question that use different "
    "vocabulary or phrasing but ask for the same information. "
    "Each variant should be a complete, standalone question."
)


def generate_hyde_passage(
    question: str,
    doc_type: str = "general",
    user_id: str = "system",
) -> str:
    """
    Generate a hypothetical ideal-answer passage for the given question.
    Falls back to the original question on any error.
    """
    doc_type_hint = f" The document is a {doc_type}." if doc_type and doc_type != "general" else ""
    user_content = f"Question: {question}{doc_type_hint}"

    try:
        result: HyDEPassage = call_llm(
            system=_HYDE_SYSTEM,
            user=user_content,
            temperature=0.3,
            max_tokens=200,
            call_type="hyde",
            response_model=HyDEPassage,
            user_id=user_id,
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

def generate_query_variants(
    question: str,
    user_id: str = "system",
) -> list[str]:
    """
    Generate 2-3 paraphrased variants of the question.
    Always includes the original question as the first entry.
    Falls back to [question] on any error.
    """
    try:
        result: QueryVariants = call_llm(
            system=_MULTIQUERY_SYSTEM,
            user=f"Original question: {question}",
            temperature=0.4,
            max_tokens=300,
            call_type="multiquery",
            response_model=QueryVariants,
            user_id=user_id,
        )

        variants = [v.strip() for v in result.variants if v.strip()]
        if not variants:
            return [question]

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
                break

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
    if not mode:
        return "standard"
    normalised = mode.strip().lower()
    if normalised == "none":
        return "standard"
    if normalised in VALID_RETRIEVAL_MODES:
        return normalised
    logger.warning("Unknown retrieval_mode — defaulting to standard", mode=mode)
    return "standard"