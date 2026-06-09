"""
routers/documents.py
Endpoints:
    POST   /upload
    POST   /ingest-url
    GET    /documents
    DELETE /documents/{document_id}
    GET    /summary/{document_id}
    GET    /documents/{document_id}/classification
    POST   /documents/{document_id}/classification

Fix 2: auth dependency added to /upload, /ingest-url, /documents.
       user_id threaded through to db and ingestion calls.
"""

import os
import shutil

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, validator

from core.auth import get_current_user, get_user_id
from core.responses import (
    bad_request, error_response, internal_error,
    not_found, success_response, unsupported_file_type,
)

router = APIRouter(tags=["Documents"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx",
    ".rtf", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tiff",
}


# ── Input models ──────────────────────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str

    @validator("url")
    def url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class ClassificationOverrideRequest(BaseModel):
    doc_type: str
    schema_template: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_post_ingest(document_id: str, result: dict) -> dict:
    """
    After ingestion succeeds, run summary generation and classification.
    Both are non-blocking — failures are logged but don't fail the request.
    """
    try:
        from retrieval import generate_summary
        from db import save_summary
        summary_data = generate_summary(document_id)
        save_summary(document_id, summary_data["summary"], summary_data["summary_short"])
        result["summary_short"] = summary_data["summary_short"]
    except Exception as exc:
        print(f"[documents] Summary generation failed (non-blocking): {exc}")

    try:
        from retrieval import classify_document
        from db import save_classification
        classification = classify_document(document_id)
        save_classification(document_id, classification)
        result["classification"] = classification
    except Exception as exc:
        print(f"[documents] Classification failed (non-blocking): {exc}")
        result["classification"] = {"doc_type": "general", "confidence": 0.0}

    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_llamaparse: str = Form("True"),
    vision_template: str = Form("general"),
    user=Depends(get_current_user),           # Fix 2
):
    """
    Upload and ingest a document file.
    Supported: PDF, DOCX, TXT, CSV, XLSX, RTF, MD, PNG, JPG, JPEG, WEBP, TIFF.
    """
    uid = get_user_id(user)                   # Fix 2 — "anonymous" in dev mode

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return unsupported_file_type(ext)

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        return internal_error(f"Could not save uploaded file: {exc}")

    try:
        from ingestion import ingest_file
        use_lp = use_llamaparse.lower() == "true"
        result = ingest_file(
            temp_path,
            use_llamaparse=use_lp,
            vision_template=vision_template,
            user_id=uid,                      # Fix 2
        )
    except Exception as exc:
        return internal_error(f"Ingestion failed: {exc}")

    if "error" in result:
        return error_response(result["error"], code="INGESTION_ERROR")

    result = _run_post_ingest(result["document_id"], result)
    return result


@router.post("/ingest-url")
async def ingest_from_url(
    req: URLRequest,
    user=Depends(get_current_user),           # Fix 2
):
    """Ingest a document from a public URL."""
    uid = get_user_id(user)

    try:
        from ingestion import ingest_url
        result = ingest_url(req.url, user_id=uid)   # Fix 2
    except Exception as exc:
        return internal_error(f"URL ingestion failed: {exc}")

    if "error" in result:
        return error_response(result["error"], code="INGESTION_ERROR")

    result = _run_post_ingest(result["document_id"], result)
    return result


@router.get("/documents")
def list_documents(
    doc_type: str | None = None,
    requires_review: bool | None = None,
    limit: int = 50,
    user=Depends(get_current_user),           # Fix 2
):
    """
    List documents for the current user, ordered newest-first.
    Supports filtering: ?doc_type=invoice  and/or  ?requires_review=true
    """
    uid = get_user_id(user)
    from db import get_all_documents

    # get_all_documents handles user scoping; apply extra filters after
    docs = get_all_documents(user_id=uid)     # Fix 2

    if doc_type:
        docs = [d for d in docs if d.get("doc_type") == doc_type]
    if requires_review is not None:
        docs = [d for d in docs if d.get("requires_review") == requires_review]

    return docs[:limit]


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Permanently delete a document and all its chunks and chats."""
    from db import delete_document_by_id
    delete_document_by_id(document_id)
    return {"status": "deleted", "document_id": document_id}


@router.get("/summary/{document_id}")
def get_doc_summary(document_id: str):
    """Return the document summary. Generates on demand if not cached."""
    import json
    from db import get_summary, save_summary
    from retrieval import generate_summary

    data = get_summary(document_id)

    if not data.get("summary"):
        try:
            summary_data = generate_summary(document_id)
            save_summary(document_id, summary_data["summary"], summary_data["summary_short"])
            data = summary_data
        except Exception as exc:
            return internal_error(f"Summary generation failed: {exc}")

    try:
        parsed = json.loads(data.get("summary", "{}"))
    except Exception:
        parsed = {}

    return {"summary_short": data.get("summary_short", ""), "details": parsed}


# ── Classification endpoints ──────────────────────────────────────────────────

@router.get("/documents/{document_id}/classification")
def get_doc_classification(document_id: str):
    """Return the stored classification for a document."""
    from db import get_classification
    classification = get_classification(document_id)
    if not classification:
        return {"doc_type": "general", "confidence": 0.0, "requires_review": False}
    return classification


@router.post("/documents/{document_id}/classification")
def override_classification(document_id: str, body: ClassificationOverrideRequest):
    """Manually override classification. Sets confidence 1.0, clears requires_review."""
    from db import save_classification, get_classification

    existing = get_classification(document_id) or {}
    existing_data = existing.get("classification_data") or {}

    updated = {
        **existing_data,
        "doc_type": body.doc_type,
        "schema_template": body.schema_template or body.doc_type,
        "confidence": 1.0,
        "manually_overridden": True,
        "requires_human_review": False,
    }
    save_classification(document_id, updated)
    return {"status": "updated", "classification": updated}