"""
routers/documents.py
Document endpoints.

Changes in this phase:
  - All endpoints switch to get_current_user_context()
  - Upload: file validation + quota check added
  - Upload: temp file cleanup guaranteed in finally block
  - Upload: org_id/team_id passed to ingest_file()
  - GET /documents: visibility-aware
  - POST /documents/{id}/visibility — new endpoint
  - DELETE /documents/{id}: audit logged
"""

import os
import shutil
import json

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from pydantic import BaseModel, validator

from core.auth import get_current_user_context, get_user_id, UserContext
from core.responses import error_response, internal_error, unsupported_file_type
from core.file_validator import validate_upload_or_raise_http
from core.quota_checker import check_upload_quota
from core.permissions import assert_document_access
from core.lineage import (
    log_deleted, log_classification_overridden, log_summarized, timed_event, LineageEvent,
)
from retrieval import generate_summary, classify_document
from db import (
    save_summary, save_classification, get_classification,
    get_summary, get_all_documents, delete_document_by_id,
    get_document_any_visibility, update_document_visibility,
)
from db_llm import invalidate_for_document
from db_audit import log_audit
from ingestion import ingest_file, ingest_url

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
    doc_type:        str
    schema_template: str | None = None


class VisibilityUpdateRequest(BaseModel):
    visibility: str
    team_id:    str | None = None
    org_id:     str | None = None

    @validator("visibility")
    def visibility_valid(cls, v):
        if v not in ("private", "team", "org"):
            raise ValueError("visibility must be 'private', 'team', or 'org'")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_post_ingest(document_id: str, result: dict, user_id: str = "system") -> dict:
    """Run summary + classification after ingestion."""
    try:
        with timed_event(
            document_id, user_id, LineageEvent.SUMMARIZED,
            event_data={"file": result.get("file", "")},
        ):
            summary_data = generate_summary(document_id, user_id=user_id)
            save_summary(document_id, summary_data["summary"], summary_data["summary_short"])

        result["summary_short"] = summary_data["summary_short"]
        log_summarized(document_id, user_id=user_id)
    except Exception as exc:
        print(f"[documents] Summary generation failed (non-blocking): {exc}")

    pre_classification = result.pop("_classification", None)

    if pre_classification and pre_classification.get("doc_type"):
        try:
            save_classification(document_id, pre_classification)
            result["classification"] = pre_classification
        except Exception as exc:
            print(f"[documents] Persisting pre-classification failed: {exc}")
            result["classification"] = pre_classification
    else:
        try:
            classification = classify_document(document_id, user_id=user_id)
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
    use_llamaparse:  str = Form("True"),
    vision_template: str = Form("general"),
    visibility:      str = Form("private"),
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    # Extension check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return unsupported_file_type(ext)

    # Quota check — before saving file to disk
    check_upload_quota(user)

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Save temp file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Magic byte + size validation
        validate_upload_or_raise_http(temp_path, file.filename)

        # Validate visibility value
        if visibility not in ("private", "team", "org"):
            visibility = "private"

        use_lp = use_llamaparse.lower() == "true"
        result = ingest_file(
            temp_path,
            use_llamaparse=use_lp,
            doc_type=vision_template,
            user_id=uid,
            org_id=str(user.org_id) if user.org_id else None,
            team_id=str(user.team_id) if user.team_id else None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        return internal_error(f"Ingestion failed: {exc}")
    finally:
        # Guaranteed cleanup — temp file always removed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if "error" in result:
        return error_response(result["error"], code="INGESTION_ERROR")

    result = _run_post_ingest(result["document_id"], result, user_id=uid)
    return result


@router.post("/ingest-url")
async def ingest_from_url(
    req: URLRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    # Quota check
    check_upload_quota(user)

    try:
        result = ingest_url(req.url, user_id=uid)
    except Exception as exc:
        return internal_error(f"URL ingestion failed: {exc}")

    if "error" in result:
        return error_response(result["error"], code="INGESTION_ERROR")

    result = _run_post_ingest(result["document_id"], result, user_id=uid)
    return result


@router.get("/documents")
def list_documents(
    doc_type:        str | None = None,
    requires_review: bool | None = None,
    visibility:      str | None = None,
    limit:           int = 50,
    user: UserContext = Depends(get_current_user_context),
):
    uid  = get_user_id(user)
    docs = get_all_documents(
        user_id=uid,
        org_id=str(user.org_id) if user.org_id else None,
        visibility_filter=visibility,
    )

    if doc_type:
        docs = [d for d in docs if d.get("doc_type") == doc_type]
    if requires_review is not None:
        docs = [d for d in docs if d.get("requires_review") == requires_review]

    return docs[:limit]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    try:
        doc      = get_document_any_visibility(document_id)
        filename = doc.get("name", "") if doc else ""
        doc_type = doc.get("doc_type", "") if doc else ""

        # Only owner can delete
        if doc and doc.get("user_id") != uid and not user.is_dev:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the document owner can delete it.",
            )
    except HTTPException:
        raise
    except Exception:
        filename = ""
        doc_type = ""

    log_deleted(document_id, user_id=uid, filename=filename, doc_type=doc_type)

    try:
        invalidate_for_document(document_id, user_id=uid)
    except Exception:
        pass

    delete_document_by_id(document_id, user_id=uid)

    log_audit(
        actor_id=uid,
        actor_role=user.org_role or "member",
        action="document_deleted",
        resource_type="document",
        resource_id=document_id,
        org_id=str(user.org_id) if user.org_id else None,
        details={"filename": filename, "doc_type": doc_type},
    )

    return {"status": "deleted", "document_id": document_id}


@router.patch("/documents/{document_id}/visibility")
def update_visibility(
    document_id: str,
    req: VisibilityUpdateRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Update document visibility. Owner only.

    Visibility rules:
      private → only owner can see
      team    → team members + org admin (if can_read_team_documents)
      org     → all active org members

    When setting to 'team', team_id is required.
    When setting to 'org', org_id is required.
    """
    uid = get_user_id(user)

    # Validate scope
    if req.visibility == "team" and not (req.team_id or user.team_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="team_id is required when setting visibility to 'team'.",
        )
    if req.visibility == "org" and not (req.org_id or user.org_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="org_id is required when setting visibility to 'org'.",
        )

    team_id = req.team_id or (str(user.team_id) if user.team_id else None)
    org_id  = req.org_id  or (str(user.org_id)  if user.org_id  else None)

    updated = update_document_visibility(
        document_id=document_id,
        user_id=uid,
        visibility=req.visibility,
        team_id=team_id if req.visibility == "team" else None,
        org_id=org_id   if req.visibility in ("team", "org") else None,
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you are not the owner.",
        )

    log_audit(
        actor_id=uid,
        actor_role=user.org_role or "member",
        action="visibility_updated",
        resource_type="document",
        resource_id=document_id,
        org_id=str(user.org_id) if user.org_id else None,
        details={"visibility": req.visibility},
    )

    return {"status": "updated", "document_id": document_id, "visibility": req.visibility}


@router.get("/summary/{document_id}")
def get_doc_summary(
    document_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    uid  = get_user_id(user)
    data = get_summary(document_id)
    if not data.get("summary"):
        try:
            summary_data = generate_summary(document_id, user_id=uid)
            save_summary(document_id, summary_data["summary"], summary_data["summary_short"])
            data = summary_data
        except Exception as exc:
            return internal_error(f"Summary generation failed: {exc}")

    try:
        parsed = json.loads(data.get("summary", "{}"))
    except Exception:
        parsed = {}

    return {"summary_short": data.get("summary_short", ""), "details": parsed}


@router.get("/documents/{document_id}/classification")
def get_doc_classification(document_id: str):
    classification = get_classification(document_id)
    if not classification:
        return {"doc_type": "general", "confidence": 0.0, "requires_review": False}
    return classification


@router.post("/documents/{document_id}/classification")
def override_classification(
    document_id: str,
    body: ClassificationOverrideRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid      = get_user_id(user)
    existing = get_classification(document_id) or {}
    old_type = existing.get("doc_type", "general")

    existing_data = existing.get("classification_data") or {}
    updated = {
        **existing_data,
        "doc_type":              body.doc_type,
        "schema_template":       body.schema_template or body.doc_type,
        "confidence":            1.0,
        "manually_overridden":   True,
        "requires_human_review": False,
    }
    save_classification(document_id, updated)

    if old_type != body.doc_type:
        log_classification_overridden(
            document_id,
            user_id=uid,
            old_type=old_type,
            new_type=body.doc_type,
        )

    return {"status": "updated", "classification": updated}