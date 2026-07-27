"""
routers/extraction.py
Extraction endpoints.

Changes in this phase:
  - Switch to get_current_user_context()
  - org_id/team_id threaded through to retrieval layer

Dynamic/complex schema extraction:
  - ExtractRequest gains an optional `nested_schema` field (a raw dict matching
    schemas.dynamic.SchemaSpec) alongside the existing flat `fields` dict.
    Named `nested_schema` rather than `schema` to avoid colliding with
    pydantic v1's reserved BaseModel.schema() method.
  - extract() branches: nested_schema present -> extract_dynamic_fields();
    otherwise -> the existing flat extract_fields() path. Both return the
    same top-level response shape (extracted/validation/business_validation/
    extraction_id), so no client-facing contract break for existing callers
    using `fields`.
  - NLExtractRequest.preview_only now returns a schemas.dynamic.SchemaSpec
    (nested-capable) instead of the old flat SchemaResult.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, validator

from core.responses import bad_request, error_response, internal_error, not_found
from core.auth import get_current_user_context, get_user_id, UserContext
from core.lineage import log_extraction_started, log_extraction_completed, log_corrected
from retrieval import extract_fields, extract_dynamic_fields, extract_nl, extract_tables
from schemas.dynamic import SchemaSpec, generate_schema_spec
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
    """
    Flatten for webhook payloads. CHANGED: nested list/object field values are
    no longer wrapped in {"value": ...} (see retrieval.extract_dynamic_fields()
    response shaping) — pass them through as-is instead of trying to pull a
    "value" key that won't exist.
    """
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
    fields: dict | None = None
    nested_schema: dict | None = None  # inline SchemaSpec — for nested/complex extraction

    @validator("document_id")
    def doc_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("document_id cannot be empty")
        return v.strip()

    @validator("nested_schema", always=True)
    def fields_or_schema_required(cls, v, values):
        if not v and not values.get("fields"):
            raise ValueError(
                "Provide either 'fields' (flat {name: description} dict) "
                "or 'nested_schema' (schemas.dynamic.SchemaSpec — see GET /schemas/example)"
            )
        return v


class NLExtractRequest(BaseModel):
    document_id:  str
    instruction:  str
    preview_only: bool = False

    @validator("instruction")
    def instruction_not_empty(cls, v):
        if not v.strip():
            raise ValueError("instruction cannot be empty")
        return v.strip()


class BatchExtractRequest(BaseModel):
    document_ids: list[str]
    fields:       dict = {}
    instruction:  str | None = None

    @validator("document_ids")
    def ids_not_empty(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        return v


class ReviewAction(BaseModel):
    field:           str
    action:          str
    original_value:  str = ""
    corrected_value: str = ""
    evidence_used:   str = ""
    reviewer_note:   str = ""

    @validator("action")
    def action_must_be_valid(cls, v):
        if v not in ("approve", "reject", "correct"):
            raise ValueError("action must be 'approve', 'reject', or 'correct'")
        return v


# ── Extraction routes ─────────────────────────────────────────────────────────

@router.post("/extract")
def extract(
    req: ExtractRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid    = get_user_id(user)
    org_id = user.org_id_str
    tid    = user.team_id_str

    if req.nested_schema:
        try:
            spec = SchemaSpec.model_validate(req.nested_schema)
        except Exception as exc:
            return bad_request(f"Invalid nested_schema: {exc}")

        log_extraction_started(req.document_id, user_id=uid, field_count=len(spec.fields))
        try:
            result = extract_dynamic_fields(
                req.document_id, spec, user_id=uid, org_id=org_id, team_id=tid,
            )
        except Exception as exc:
            return internal_error(f"Extraction failed: {exc}")
        template_id = "dynamic"
    else:
        log_extraction_started(req.document_id, user_id=uid, field_count=len(req.fields))
        try:
            result = extract_fields(
                req.document_id, req.fields,
                user_id=uid, org_id=org_id, team_id=tid,
            )
        except Exception as exc:
            return internal_error(f"Extraction failed: {exc}")
        template_id = "custom"

    extracted = result.get("extracted", {})

    fields_with_value = sum(
        1 for v in extracted.values()
        if (isinstance(v, dict) and "value" in v and v.get("value") is not None)
        or (not isinstance(v, dict) and v not in (None, "", []))
        or (isinstance(v, dict) and "value" not in v and v)  # nested object field, non-empty
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
        template_id=template_id,
    )

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "extracted":   _flatten_extracted(extracted),
        "validation":  result.get("validation"),
    })

    extraction_id = _store_extraction(
        document_id=req.document_id,
        template_id=template_id,
        results=extracted,
        user_id=uid,
    )
    return {**result, "extraction_id": extraction_id}


@router.post("/extract/nl")
def extract_natural_language(
    req: NLExtractRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    CHANGED: preview_only now returns a schemas.dynamic.SchemaSpec (nested-capable)
    instead of the old flat SchemaResult — the same schema shape extract_nl()
    below will actually extract against, so a preview accurately reflects what
    the real extraction call will produce (including nested list/object fields).
    """
    uid    = get_user_id(user)
    org_id = user.org_id_str
    tid    = user.team_id_str
    if req.preview_only:
        try:
            spec = generate_schema_spec(req.instruction, user_id=uid)
        except Exception as exc:
            return internal_error(f"Schema generation failed: {exc}")
        return {"schema": spec.model_dump(), "extracted": None, "validation": None}

    try:
        result = extract_nl(
            req.document_id, req.instruction,
            user_id=uid, org_id=org_id, team_id=tid,
        )
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
def batch_extract(
    req: BatchExtractRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid    = get_user_id(user)
    org_id = user.org_id_str
    tid    = user.team_id_str
    
    if not req.fields and not req.instruction:
        return bad_request(
            "Provide either 'fields' (schema dict) or 'instruction' (natural language).",
            code="MISSING_EXTRACTION_SPEC",
        )

    results = []
    for doc_id in req.document_ids:
        try:
            if req.instruction:
                result = extract_nl(
                    doc_id, req.instruction,
                    user_id=uid, org_id=org_id, team_id=tid,
                )
            else:
                result = extract_fields(
                    doc_id, req.fields,
                    user_id=uid, org_id=org_id, team_id=tid,
                )

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
def get_extraction(
    extraction_id: str,
    user: UserContext = Depends(get_current_user_context),
):
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


@router.get("/schemas/example")
def get_schema_example():
    """
    NEW — returns a worked example of a nested_schema payload for /extract,
    so API consumers building a custom schema by hand (rather than via
    /extract/nl) have a concrete shape to copy.
    """
    return {
        "schema_name": "candidate_profile",
        "fields": [
            {"name": "candidate_name", "type": "string", "description": "Full name of the candidate"},
            {
                "name": "past_companies",
                "type": "list",
                "description": "Each company the candidate has worked at",
                "properties": [
                    {"name": "name", "type": "string", "description": "Company name"},
                    {"name": "start_date", "type": "date", "description": "Start date, YYYY-MM"},
                    {"name": "end_date", "type": "date", "description": "End date, YYYY-MM, or null if current"},
                    {"name": "place", "type": "string", "description": "Office location"},
                ],
            },
        ],
    }


@router.get("/tables/{document_id}")
def get_tables(
    document_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    uid    = get_user_id(user)
    org_id = user.org_id_str
    tid    = user.team_id_str
    try:
        tables = extract_tables(
            document_id,
            user_id=uid,
            org_id=org_id,
            team_id=tid,
        )
        return {"tables": tables}
    except Exception as exc:
        return internal_error(f"Table extraction failed: {exc}")


# ── Review routes ─────────────────────────────────────────────────────────────

@router.post("/review/{document_id}")
def submit_review(
    document_id: str,
    actions: list[ReviewAction],
    user: UserContext = Depends(get_current_user_context),
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