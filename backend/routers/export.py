"""
Endpoints:
    POST /export/pdf
    POST /export/docx
"""

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, validator

from core.responses import internal_error
from export import export_chat_pdf,export_chat_docx
 
router = APIRouter(tags=["Export"])


# ── Input models ──────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    document_id: str
    file_name: str
    messages: list[dict]
    summary: dict = {}

    @validator("file_name")
    def file_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("file_name cannot be empty")
        # Sanitise: strip path separators so the filename stays safe
        return v.strip().replace("/", "_").replace("\\", "_")

    @validator("messages")
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/export/pdf")
def export_pdf(req: ExportRequest):
    """Export a chat conversation as a formatted PDF report."""
    try:
        pdf_bytes = export_chat_pdf(req.file_name, req.messages, req.summary)
    except Exception as exc:
        return internal_error(f"PDF export failed: {exc}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="DocIntel_{req.file_name}.pdf"'
        },
    )


@router.post("/export/docx")
def export_docx(req: ExportRequest):
    """Export a chat conversation as a formatted Word document."""
    try:
        docx_bytes = export_chat_docx(req.file_name, req.messages, req.summary)
    except Exception as exc:
        return internal_error(f"DOCX export failed: {exc}")

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="DocIntel_{req.file_name}.docx"'
        },
    )
