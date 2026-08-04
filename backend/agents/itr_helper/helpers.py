"""
backend/agents/itr_helper/helpers.py
--------------------------------------
Plain helper functions for the ITR agent — not stage functions themselves,
called BY stage functions in agent.py. Kept separate from agent.py so the
stitching logic (and, later, other reusable helpers) can be unit tested
without going through the stage machine.
"""
from __future__ import annotations

from schemas.dynamic import SchemaSpec
from retrieval import extract_dynamic_fields
from db import get_global_schema_template_by_schema_name

# Stable machine keys — must match schema_spec.schema_name in the seeded
# global nested_schema_templates rows (see itr_schema_templates_seed.json).
FORM16_SCHEMA_NAME = "form16"
FD_INTEREST_SCHEMA_NAME = "fd_interest_certificate"
STOCKS_SCHEMA_NAME = "capital_gains_transactions"


class ITRHelperError(Exception):
    """Raised when a required global schema template is missing (deploy/seed problem, not user error)."""


def _load_spec(schema_name: str) -> SchemaSpec:
    row = get_global_schema_template_by_schema_name(schema_name)
    if row is None:
        raise ITRHelperError(
            f"No global schema template found for schema_name='{schema_name}' — "
            f"has the ITR seed migration been run?"
        )
    return SchemaSpec.model_validate(row["schema_spec"])


def extract_tagged_documents(
    form16_doc_ids: list[str],
    fd_doc_ids: list[str],
    stocks_doc_ids: list[str],
    user_id: str,
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    """
    Runs extract_dynamic_fields() against each tagged document with the
    correct schema for its type. Returns results keyed by type, each a
    list (parallel to the input doc_id lists) — a user can have multiple
    Form16s (multiple employers), multiple FD certificates (multiple
    banks), etc.

    Any single document's extraction failing does not abort the others —
    its entry carries an "error" key instead, so one bad document doesn't
    block a run that has other valid documents. The validate stage (next
    in the pipeline) is what decides whether a failure here is fatal.
    """
    form16_spec = _load_spec(FORM16_SCHEMA_NAME)
    fd_spec     = _load_spec(FD_INTEREST_SCHEMA_NAME)
    stocks_spec = _load_spec(STOCKS_SCHEMA_NAME)

    def _extract_all(doc_ids: list[str], spec: SchemaSpec) -> list[dict]:
        results = []
        for doc_id in doc_ids:
            try:
                result = extract_dynamic_fields(
                    doc_id, spec, user_id=user_id, org_id=org_id, team_id=team_id,
                )
                results.append({"document_id": doc_id, **result})
            except Exception as exc:
                results.append({"document_id": doc_id, "error": str(exc)})
        return results

    return {
        "form16": _extract_all(form16_doc_ids, form16_spec),
        "fd_interest": _extract_all(fd_doc_ids, fd_spec),
        "stocks": _extract_all(stocks_doc_ids, stocks_spec),
    }


def _unwrap(value):
    """
    extract_dynamic_fields() wraps flat string fields as {"value":..,"bbox":..}
    but leaves list/object fields as raw nested JSON (see retrieval.py's
    response-shaping comment in extract_dynamic_fields). This normalises
    either shape down to the plain value, recursively, so stitching doesn't
    have to special-case field types.
    """
    if isinstance(value, dict) and set(value.keys()) <= {"value", "bbox"} and "value" in value:
        return value["value"]
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def stitch_taxpayer_profile(extractions: dict) -> dict:
    """
    Merges extract_tagged_documents()'s output into one canonical
    TaxpayerProfile dict — the single object every downstream stage
    (validate, calculate, summarize) and every chat tool reads from.

    Shape:
        {
            "form16_entries":     [ {employer_name, ..., tds_summary: [...], ...}, ... ],
            "fd_interest_entries": [ {bank_name, fd_number, interest_earned, tds_deducted}, ... ],
            "stock_transactions":  [ {symbol, buy_date, sell_date, buy_value, sell_value, quantity}, ... ],
            "extraction_errors":   [ {source, document_id, error}, ... ],
        }

    Multiple Form16s' tds_summary/deductions lists stay nested per-employer
    (not flattened across employers) — total_taxable_income etc. must be
    computed per Form16 first, since combining them naively would double
    count things like standard_deduction. That combination is the
    calculate stage's job, not stitch's — stitch only merges/normalises,
    never sums or computes.
    """
    profile: dict = {
        "form16_entries": [],
        "fd_interest_entries": [],
        "stock_transactions": [],
        "extraction_errors": [],
    }

    for entry in extractions.get("form16", []):
        if "error" in entry:
            profile["extraction_errors"].append(
                {"source": "form16", "document_id": entry.get("document_id"), "error": entry["error"]}
            )
            continue
        extracted = _unwrap(entry.get("extracted", {}))
        extracted["_document_id"] = entry.get("document_id")
        profile["form16_entries"].append(extracted)

    for entry in extractions.get("fd_interest", []):
        if "error" in entry:
            profile["extraction_errors"].append(
                {"source": "fd_interest", "document_id": entry.get("document_id"), "error": entry["error"]}
            )
            continue
        extracted = _unwrap(entry.get("extracted", {}))
        # fd_interest_certificate's schema is a single list field, "fd_entries" —
        # flatten it directly into fd_interest_entries (unlike form16, a
        # certificate has no other identity to preserve per-entry).
        for fd in extracted.get("fd_entries", []):
            fd["_document_id"] = entry.get("document_id")
            profile["fd_interest_entries"].append(fd)

    for entry in extractions.get("stocks", []):
        if "error" in entry:
            profile["extraction_errors"].append(
                {"source": "stocks", "document_id": entry.get("document_id"), "error": entry["error"]}
            )
            continue
        extracted = _unwrap(entry.get("extracted", {}))
        for txn in extracted.get("transactions", []):
            txn["_document_id"] = entry.get("document_id")
            profile["stock_transactions"].append(txn)

    return profile