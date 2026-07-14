"""
backend/routers/comparison.py

POST /compare — diff two document versions (DOCX / non-scanned PDF for v1).

ASSUMPTIONS — adjust to match your actual code:
  - core.auth.get_current_user / get_user_id are the standard auth deps
    used by your other routers (e.g. routers/documents.py)
  - parsers.router.AutoRouter().route(file_path) -> parser instance,
    parser.parse(file_path, config) -> Document  (same as ingestion.py uses)
  - core.config.config is importable as in every other backend file

Deliberately bypasses ingestion.py — no chunking, no embeddings, no
Supabase storage. Comparison is stateless per-request for v1.
"""

import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from core.auth import get_current_user, get_user_id
from core.config import config
from core.logger import get_logger
from parsers.router import AutoRouter
from comparison import diff_documents, summarize_changes, result_to_dict

router = APIRouter()
logger = get_logger("comparison_router")

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def _save_upload_to_temp(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. v1 supports: {sorted(ALLOWED_EXTENSIONS)}",
        )
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(upload.file.read())
    return tmp_path


@router.post("/compare")
async def compare_documents(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    include_summary: bool = Form(default=False),
    user=Depends(get_current_user),
):
    """
    Compare two document versions and return a word-level, page/chunk
    -mapped diff. v1 scope: PDF and DOCX only — no scanned PDFs (no vision
    fallback), no LlamaParse routing.
    """
    user_id = get_user_id(user)
    logger.info(
        "Comparison requested",
        user_id=user_id,
        file_a=file_a.filename,
        file_b=file_b.filename,
    )

    path_a = _save_upload_to_temp(file_a)
    path_b = _save_upload_to_temp(file_b)

    try:
        auto_router = AutoRouter()

        parser_a = auto_router.route(path_a)
        parser_b = auto_router.route(path_b)

        document_a = parser_a.parse(path_a, config)
        document_b = parser_b.parse(path_b, config)

        result = diff_documents(document_a, document_b)

        if include_summary:
            result.summary = summarize_changes(result)

        return result_to_dict(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Comparison failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Comparison failed. Check server logs.")

    finally:
        for p in (path_a, path_b):
            if os.path.exists(p):
                os.remove(p)