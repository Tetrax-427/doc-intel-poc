"""
backend/agents/tools/candidate_data.py
-----------------------------------------
Shared data-access layer for any agent working over documents that already
have Extraction Helper CSV rows (CV Processor today; reusable for others
later). CSV is the fast/cheap path; RAG (via retrieval.query_document) is
the fallback any stage can reach for when a field is missing or looks
inconsistent - never hard-coded per criterion, so every scorer gets the
same verification escape hatch for free.
"""
from __future__ import annotations

from core.logger import get_logger
from retrieval import query_document

logger = get_logger("agents.tools.candidate_data")


def build_candidate_dataset(document_ids: list[str], csv_data: list[dict]) -> list[dict]:
    """
    Merges the batch's document_ids with whatever CSV rows were supplied,
    matched by document_id (CSV rows are expected to carry a "document_id"
    column - see Extraction Helper note below). Every document_id gets an
    entry even if no CSV row matched, so downstream stages always have a
    consistent list to iterate and can fall back to RAG for a candidate
    with no CSV data at all.

    NOTE: Extraction Helper's current CSV export doesn't include
    document_id as a column (it's keyed on filename) - this is a
    prerequisite follow-up on the extraction_UI side before this merge
    is fully wired end-to-end. Until then, callers can pass csv_data=[]
    and every candidate falls back to RAG for everything.
    """
    csv_by_doc_id = {row["document_id"]: row for row in csv_data if row.get("document_id")}
    dataset = []
    for doc_id in document_ids:
        row = csv_by_doc_id.get(doc_id, {})
        candidate = dict(row)
        candidate["document_id"] = doc_id
        candidate["_csv_matched"] = doc_id in csv_by_doc_id
        dataset.append(candidate)
    return dataset


def verify_via_rag(document_id: str, question: str, user_id: str = "system") -> dict:
    """
    The fallback any stage/sub-agent calls when a CSV field is missing or
    looks inconsistent. Thin wrapper over retrieval.query_document() - reuses
    the existing hybrid search + LLM answer pipeline rather than
    reimplementing document Q&A here.

    Returns {"answer": str, "sources": [...], "source": "rag"}.
    """
    try:
        result = query_document(question, document_id=document_id, user_id=user_id)
        return {"answer": result.get("answer", ""), "sources": result.get("sources", []), "source": "rag"}
    except Exception as exc:
        logger.warning("RAG verification failed", document_id=document_id, question=question, error=str(exc))
        return {"answer": "", "sources": [], "source": "rag_error", "error": str(exc)}


def get_field(
    candidate: dict,
    field_name: str,
    fallback_question: str | None = None,
    user_id: str = "system",
) -> dict:
    """
    CSV-first field lookup with an optional RAG fallback. Every scoring
    sub-agent should go through this (or verify_via_rag directly, for
    fields that don't map cleanly to one CSV column) rather than reading
    candidate[field_name] straight off the dict, so the CSV-first/RAG-
    fallback behaviour stays consistent everywhere.

    Returns {"value": ..., "source": "csv" | "rag" | "missing"}.
    """
    value = candidate.get(field_name)
    if value not in (None, "", [], {}):
        return {"value": value, "source": "csv"}

    if fallback_question:
        result = verify_via_rag(candidate["document_id"], fallback_question, user_id=user_id)
        return {"value": result["answer"], "source": result["source"]}

    return {"value": None, "source": "missing"}


def flag_inconsistent(
    candidate: dict,
    field_name: str,
    claimed_value,
    verification_question: str,
    user_id: str = "system",
) -> dict:
    """
    For when a scorer has a CSV value but doesn't trust it (looks
    implausible, contradicts another field, etc.) - re-checks against the
    raw document via RAG and returns BOTH values so the caller/LLM scoring
    step can weigh them, rather than silently overwriting one with the
    other.

    Returns {"csv_value": ..., "rag_value": ..., "rag_sources": [...]}.
    """
    result = verify_via_rag(candidate["document_id"], verification_question, user_id=user_id)
    return {
        "csv_value": claimed_value,
        "rag_value": result["answer"],
        "rag_sources": result["sources"],
    }