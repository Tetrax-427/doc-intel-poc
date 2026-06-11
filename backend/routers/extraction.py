"""
Endpoints:
    POST  /extract
    POST  /extract/nl
    POST  /extract/batch
    GET   /templates
    GET   /templates/{template_id}
    GET   /tables/{document_id}
    POST  /review/{document_id}
    GET   /review/{document_id}/corrections
"""

from fastapi import APIRouter
from pydantic import BaseModel, validator

from core.responses import bad_request, error_response, internal_error, not_found
from retrieval import extract_fields,nl_to_schema, extract_nl,extract_tables
from webhooks import trigger_webhooks, trigger_webhooks
from schemas.templates import list_templates
from schemas.templates import get_template as _get_template
from db import get_classification, save_correction,supabase

router = APIRouter(tags=["Extraction"])


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
    action: str                  # "approve" | "reject" | "correct"
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
def extract(req: ExtractRequest):
    """
    Extract structured fields from a document using a field schema dict.
    Each key is a field name; each value is a description of what to extract.
    """
    try:
        result = extract_fields(req.document_id, req.fields)
    except Exception as exc:
        return internal_error(f"Extraction failed: {exc}")

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "extracted": result.get("extracted"),
        "validation": result.get("validation"),
    })

    return result


@router.post("/extract/nl")
def extract_natural_language(req: NLExtractRequest):
    """
    Natural language extraction.
    Converts a plain-English instruction to a field schema, then extracts.
    Set preview_only=true to return the generated schema without extracting.
    """
    

    if req.preview_only:
        try:
            schema = nl_to_schema(req.instruction)
        except Exception as exc:
            return internal_error(f"Schema generation failed: {exc}")
        return {"schema": schema, "extracted": None, "validation": None}

    try:
        result = extract_nl(req.document_id, req.instruction)
    except Exception as exc:
        return internal_error(f"NL extraction failed: {exc}")

    if result.get("error"):
        return error_response(result["error"], code="NL_EXTRACTION_ERROR")

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "instruction": req.instruction,
        "extracted": result.get("extracted"),
        "validation": result.get("validation"),
    })

    return result


@router.post("/extract/batch")
def batch_extract(req: BatchExtractRequest):
    """
    Extract fields from multiple documents using the same schema or instruction.
    Runs sequentially.
    Provide either `fields` (schema dict) or `instruction` (natural language).
    """

    if not req.fields and not req.instruction:
        return bad_request(
            "Provide either 'fields' (schema dict) or 'instruction' (natural language).",
            code="MISSING_EXTRACTION_SPEC",
        )

    results = []
    for doc_id in req.document_ids:
        try:
            if req.instruction:
                result = extract_nl(doc_id, req.instruction)
            else:
                result = extract_fields(doc_id, req.fields)

            results.append({
                "document_id": doc_id,
                "success": True,
                **result,
            })

            trigger_webhooks("extraction.complete", {
                "document_id": doc_id,
                "extracted": result.get("extracted"),
                "validation": result.get("validation"),
            })

        except Exception as exc:
            results.append({
                "document_id": doc_id,
                "success": False,
                "error": str(exc),
            })

    return {
        "total": len(req.document_ids),
        "succeeded": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }


@router.get("/templates")
def get_templates():
    """List all available extraction templates."""
    return list_templates()


@router.get("/templates/{template_id}")
def get_template(template_id: str):
    """Return a single extraction template by ID."""
    template = _get_template(template_id)
    if not template:
        return not_found(f"Template '{template_id}'")
    return template


@router.get("/tables/{document_id}")
def get_tables(document_id: str):
    """
    Extract and return all tables found in a document as structured JSON.
    Each table includes headers, rows, and a suggested chart type.
    """

    try:
        tables = extract_tables(document_id)
        return {"tables": tables}
    except Exception as exc:
        return internal_error(f"Table extraction failed: {exc}")


# ── Review routes ─────────────────────────────────────────────────────────────

@router.post("/review/{document_id}")
def submit_review(document_id: str, actions: list[ReviewAction]):
    """
    Submit human review decisions for a document's extracted fields.
    Persists each decision to review_corrections so the feedback loop
    can inject past corrections into future extraction prompts.
    """
    
    cls = get_classification(document_id)
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

    return {
        "reviewed_fields": [a.field for a in actions],
        "doc_type": doc_type,
        "saved": len(actions),
    }


@router.get("/review/{document_id}/corrections")
def get_corrections(document_id: str):
    """
    Return all review corrections saved for a document, newest-first.
    Used by the frontend to show review history.
    """
    
    result = (
        supabase.table("review_corrections")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []