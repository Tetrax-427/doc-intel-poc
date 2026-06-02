"""
routers/extraction.py
Endpoints:
    POST  /extract
    POST  /extract/nl
    POST  /extract/batch
    GET   /templates
    GET   /templates/{template_id}
    GET   /tables/{document_id}
"""

from fastapi import APIRouter
from pydantic import BaseModel, validator

from core.responses import bad_request, error_response, internal_error, not_found

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
    instruction: str | None = None  # if provided, NL extraction is used

    @validator("document_ids")
    def ids_not_empty(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        return v

    @validator("fields")
    def fields_or_instruction_required(cls, v, values):
        # We can't cross-validate with instruction here easily in Pydantic v1,
        # so we do it at route level.
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/extract")
def extract(req: ExtractRequest):
    """
    Extract structured fields from a document using a field schema dict.
    Each key is a field name; each value is a description of what to extract.
    """
    from retrieval import extract_fields
    from webhooks import trigger_webhooks

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
    from retrieval import nl_to_schema, extract_nl
    from webhooks import trigger_webhooks

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
    Runs sequentially; returns one result per document_id.
    Provide either `fields` (schema dict) or `instruction` (natural language).
    """
    from retrieval import extract_fields, extract_nl
    from webhooks import trigger_webhooks

    # Validate that at least one extraction method is specified
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
    from schemas.templates import list_templates
    return list_templates()


@router.get("/templates/{template_id}")
def get_template(template_id: str):
    """Return a single extraction template by ID."""
    from schemas.templates import get_template
    template = get_template(template_id)
    if not template:
        return not_found(f"Template '{template_id}'")
    return template


@router.get("/tables/{document_id}")
def get_tables(document_id: str):
    """
    Extract and return all tables found in a document as structured JSON.
    Each table includes headers, rows, and a suggested chart type.
    """
    from retrieval import extract_tables
    try:
        tables = extract_tables(document_id)
        return {"tables": tables}
    except Exception as exc:
        return internal_error(f"Table extraction failed: {exc}")
