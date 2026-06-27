"""
Endpoints:
    POST  /extract
    POST  /extract/nl
    POST  /extract/batch
    GET   /extract/{extraction_id}
    GET   /templates
    GET   /templates/{template_id}
    GET   /tables/{document_id}
    POST  /review/{document_id}
    GET   /review/{document_id}/corrections
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, validator

from core.responses import bad_request, error_response, internal_error, not_found
from core.auth import get_current_user, get_user_id
from core.lineage import log_extraction_started, log_extraction_completed, log_corrected
from retrieval import extract_fields, nl_to_schema, extract_nl, extract_tables
from webhooks import trigger_webhooks
from schemas.templates import list_templates
from schemas.templates import get_template as _get_template
from db import get_classification, save_correction, supabase
from db_extraction import (
    store_extraction_result as _store_extraction,
    get_extraction_result_by_id as _get_extraction_by_id,
)

router = APIRouter(tags=["Extraction"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten_extracted(extracted: dict) -> dict:
    flat = {}
    for k, v in extracted.items():
        if isinstance(v, dict) and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    return flat


# ── Input models ──────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    document_id: str
    fields: dict

    @validator("document_id")
    def doc_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("document_id cannot be empty")
        return v.strip()

    @validator("fields")
    def fields_not_empty(cls, v):
        if not v:
            raise ValueError("fields dict cannot be empty")
        return v


class NLExtractRequest(BaseModel):
    document_id: str
    instruction: str
    preview_only: bool = False

    @validator("instruction")
    def instruction_not_empty(cls, v):
        if not v.strip():
            raise ValueError("instruction cannot be empty")
        return v.strip()


class BatchExtractRequest(BaseModel):
    document_ids: list[str]
    fields: dict = {}
    instruction: str | None = None

    @validator("document_ids")
    def ids_not_empty(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        return v

    @validator("fields")
    def fields_or_instruction_required(cls, v, values):
        return v


class ReviewAction(BaseModel):
    field: str
    action: str
    original_value: str = ""
    corrected_value: str = ""
    evidence_used: str = ""
    reviewer_note: str = ""

    @validator("action")
    def action_must_be_valid(cls, v):
        if v not in ("approve", "reject", "correct"):
            raise ValueError("action must be 'approve', 'reject', or 'correct'")
        return v


# ── Extraction routes ─────────────────────────────────────────────────────────

@router.post("/extract")
def extract(req: ExtractRequest, user=Depends(get_current_user)):
    """
    Extract structured fields. Returns per-field {value, bbox} shape + extraction_id.
    """
    uid = get_user_id(user)

    log_extraction_started(req.document_id, user_id=uid, field_count=len(req.fields))

    try:
        result = extract_fields(req.document_id, req.fields, user_id=uid)
    except Exception as exc:
        return internal_error(f"Extraction failed: {exc}")

    extracted = result.get("extracted", {})

    fields_with_value = sum(
        1 for v in extracted.values()
        if isinstance(v, dict) and v.get("value") is not None
    )
    fields_with_bbox = sum(
        1 for v in extracted.values()
        if isinstance(v, dict) and v.get("bbox") is not None
    )
    log_extraction_completed(
        req.document_id,
        user_id=uid,
        field_count=len(extracted),
        fields_with_value=fields_with_value,
        fields_with_bbox=fields_with_bbox,
        template_id="custom",
    )

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "extracted":   _flatten_extracted(extracted),
        "validation":  result.get("validation"),
    })

    extraction_id = _store_extraction(
        document_id=req.document_id,
        template_id="custom",
        results=extracted,
        user_id=uid,
    )
    return {**result, "extraction_id": extraction_id}


@router.post("/extract/nl")
def extract_natural_language(req: NLExtractRequest, user=Depends(get_current_user)):
    uid = get_user_id(user)

    if req.preview_only:
        try:
            schema = nl_to_schema(req.instruction, user_id=uid)
        except Exception as exc:
            return internal_error(f"Schema generation failed: {exc}")
        return {"schema": schema, "extracted": None, "validation": None}

    try:
        result = extract_nl(req.document_id, req.instruction, user_id=uid)
    except Exception as exc:
        return internal_error(f"NL extraction failed: {exc}")

    if result.get("error"):
        return error_response(result["error"], code="NL_EXTRACTION_ERROR")

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "instruction": req.instruction,
        "extracted":   _flatten_extracted(result.get("extracted") or {}),
        "validation":  result.get("validation"),
    })

    return result


@router.post("/extract/batch")
def batch_extract(req: BatchExtractRequest, user=Depends(get_current_user)):
    uid = get_user_id(user)

    if not req.fields and not req.instruction:
        return bad_request(
            "Provide either 'fields' (schema dict) or 'instruction' (natural language).",
            code="MISSING_EXTRACTION_SPEC",
        )

    results = []
    for doc_id in req.document_ids:
        try:
            if req.instruction:
                result = extract_nl(doc_id, req.instruction, user_id=uid)
            else:
                result = extract_fields(doc_id, req.fields, user_id=uid)

            results.append({"document_id": doc_id, "success": True, **result})

            trigger_webhooks("extraction.complete", {
                "document_id": doc_id,
                "extracted":   _flatten_extracted(result.get("extracted") or {}),
                "validation":  result.get("validation"),
            })

        except Exception as exc:
            results.append({"document_id": doc_id, "success": False, "error": str(exc)})

    return {
        "total":     len(req.document_ids),
        "succeeded": sum(1 for r in results if r["success"]),
        "failed":    sum(1 for r in results if not r["success"]),
        "results":   results,
    }


@router.get("/extract/{extraction_id}")
def get_extraction(extraction_id: str, user=Depends(get_current_user)):
    """E2 — Fetch a stored extraction result by UUID."""
    uid    = get_user_id(user)
    result = _get_extraction_by_id(extraction_id, user_id=uid)
    if not result:
        return not_found(f"Extraction result '{extraction_id}'")
    return result


@router.get("/templates")
def get_templates():
    return list_templates()


@router.get("/templates/{template_id}")
def get_template(template_id: str):
    template = _get_template(template_id)
    if not template:
        return not_found(f"Template '{template_id}'")
    return template


@router.get("/tables/{document_id}")
def get_tables(document_id: str, user=Depends(get_current_user)):
    uid = get_user_id(user)
    try:
        tables = extract_tables(document_id, user_id=uid)
        return {"tables": tables}
    except Exception as exc:
        return internal_error(f"Table extraction failed: {exc}")


# ── Review routes ─────────────────────────────────────────────────────────────

@router.post("/review/{document_id}")
def submit_review(
    document_id: str,
    actions: list[ReviewAction],
    user=Depends(get_current_user),
):
    uid      = get_user_id(user)
    cls      = get_classification(document_id)
    doc_type = cls.get("doc_type", "general") if cls else "general"

    for action in actions:
        save_correction(
            document_id=document_id,
            doc_type=doc_type,
            field_name=action.field,
            original=action.original_value,
            corrected=action.corrected_value,
            action=action.action,
            evidence=action.evidence_used,
            note=action.reviewer_note,
        )
        log_corrected(
            document_id,
            user_id=uid,
            field_name=action.field,
            action=action.action,
        )

    return {
        "reviewed_fields": [a.field for a in actions],
        "doc_type":        doc_type,
        "saved":           len(actions),
    }


@router.get("/review/{document_id}/corrections")
def get_corrections(document_id: str):
    result = (
        supabase.table("review_corrections")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []