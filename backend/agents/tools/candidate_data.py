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

import re

from core.logger import get_logger
from retrieval import query_document

logger = get_logger("agents.tools.candidate_data")


# ---------------------------------------------------------------------------
# Sanitization - strips hidden/invisible-character injection tricks (e.g.
# zero-width unicode used to hide instructions in resume text) before any
# candidate text reaches an LLM prompt.
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_text(text: str) -> tuple[str, bool]:
    """Strips zero-width/control characters often used to hide injected text. Returns (cleaned, was_modified)."""
    if not text:
        return text, False
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    return cleaned, cleaned != text


def sanitize_candidate(candidate: dict) -> tuple[dict, list[str]]:
    """Sanitizes every string field in a candidate dict. Returns (cleaned_candidate, list_of_modified_field_names)."""
    modified_fields = []

    def _clean_value(key, value):
        if isinstance(value, str):
            cleaned, changed = sanitize_text(value)
            if changed:
                modified_fields.append(key)
            return cleaned
        if isinstance(value, list):
            return [
                {k: _clean_value(f"{key}.{k}", v) for k, v in item.items()} if isinstance(item, dict) else _clean_value(key, item)
                for item in value
            ]
        return value

    sanitized = {k: _clean_value(k, v) for k, v in candidate.items()}
    return sanitized, modified_fields


def build_candidate_dataset(document_ids: list[str], csv_data: list[dict]) -> list[dict]:
    """
    Merges the batch's document_ids with whatever CSV rows were supplied,
    matched by document_id. Every document_id gets an entry even if no CSV
    row matched, so downstream stages always have a consistent list to
    iterate and can fall back to RAG for a candidate with no CSV data at all.

    IMPORTANT: the Extraction Helper's merged-cell CSV export puts ONE ROW
    PER LIST-FIELD ITEM (e.g. a candidate with 2 education entries and 3
    work entries spans 3 rows, with scalar columns only filled on the first
    row and "prefix.suffix" columns like "education.institute_name" filled
    only up to each list field's own row count). A naive "one row per
    document_id" merge silently keeps only the LAST row and loses every
    earlier list item. This reconstructs the nested lists properly: groups
    all rows for a document_id, then for each "prefix.suffix" column group
    rebuilds prefix -> [{suffix: value}, ...] (skipping rows with no data
    for that prefix - those are the blank filler rows beyond that field's
    own count), and takes the first non-null value for any non-prefixed
    (scalar) column.

    Every candidate is passed through sanitize_candidate() before being
    added to the returned list - any field with invisible/hidden characters
    stripped gets recorded under candidate["_sanitization_flag"] (a list of
    field names) so the calling stage can log it to the audit trail. This
    key is agent-internal bookkeeping - callers that dump candidate fields
    into an LLM prompt should pop it off first.
    """
    rows_by_doc: dict[str, list[dict]] = {}
    for row in csv_data:
        doc_id = row.get("document_id")
        if doc_id:
            rows_by_doc.setdefault(doc_id, []).append(row)

    dataset = []
    for doc_id in document_ids:
        rows = rows_by_doc.get(doc_id, [])
        candidate = {"document_id": doc_id, "_csv_matched": bool(rows)}

        if not rows:
            dataset.append(candidate)
            continue

        all_columns = {c for row in rows for c in row.keys()}
        list_prefixes = {c.split(".", 1)[0] for c in all_columns if "." in c}
        scalar_columns = {c for c in all_columns if "." not in c and c != "document_id"}

        # Scalar fields: first non-null value across this candidate's rows.
        for col in scalar_columns:
            for row in rows:
                v = row.get(col)
                if v not in (None, ""):
                    candidate[col] = v
                    break

        # List fields: rebuild prefix -> [{suffix: value}, ...], one item per
        # row that actually had non-null data under that prefix.
        for prefix in list_prefixes:
            items = []
            suffix_cols = [c for c in all_columns if c.startswith(prefix + ".")]
            for row in rows:
                item = {}
                for col in suffix_cols:
                    v = row.get(col)
                    if v not in (None, ""):
                        item[col[len(prefix) + 1:]] = v
                if item:
                    items.append(item)
            if items:
                candidate[prefix] = items

        candidate, modified_fields = sanitize_candidate(candidate)
        if modified_fields:
            candidate["_sanitization_flag"] = modified_fields
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

    # Exact key miss - build_candidate_dataset() names top-level keys after
    # THAT SCHEMA's own field name (e.g. "education", "past_companies"),
    # which may not match this agent's field_name guess (e.g.
    # "candidate_education") at all. Before falling back to RAG, try
    # substring-matching candidate's own keys against a normalized version
    # of field_name, so the CSV fast path still works across differently-
    # named schemas instead of silently degrading to RAG for every candidate.
    normalized = field_name.replace("candidate_", "").split(".")[0]
    if normalized:
        matches = {
            k: v for k, v in candidate.items()
            if normalized.lower() in k.lower() and v not in (None, "", [], {})
        }
        if len(matches) == 1:
            return {"value": next(iter(matches.values())), "source": "csv"}
        elif matches:
            return {"value": matches, "source": "csv"}

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