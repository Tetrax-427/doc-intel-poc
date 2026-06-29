"""
core/file_validator.py
File validation for uploaded documents.

Validates:
  1. File extension is in the allowed list
  2. File size is within the configured limit
  3. Magic bytes match the declared file type (prevents extension spoofing)

Magic byte check:
  Reads the first 16 bytes of the file and checks against known signatures.
  Files that fail the magic byte check are rejected even if the extension
  is allowed — this prevents e.g. a .pdf file that is actually a script.

  Note: plain text formats (txt, csv, md, rtf) have no magic bytes —
  they are validated by extension only. This is expected and documented.
"""

import os
from fastapi import HTTPException, status

from core.config import config as app_config


# ---------------------------------------------------------------------------
# Magic byte signatures
# key: extension (lowercase, with dot)
# value: list of valid byte prefixes (bytes) — any match = valid
# ---------------------------------------------------------------------------

MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf":  [b"%PDF"],
    ".docx": [b"PK\x03\x04"],       # ZIP-based (Office Open XML)
    ".xlsx": [b"PK\x03\x04"],       # ZIP-based
    ".png":  [b"\x89PNG\r\n\x1a\n"],
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".webp": [b"RIFF"],             # RIFF....WEBP
    ".tiff": [b"II*\x00", b"MM\x00*"],  # little-endian, big-endian
    # Text formats — no magic bytes, extension-only validation
    ".txt":  [],
    ".csv":  [],
    ".md":   [],
    ".rtf":  [b"{\\rtf"],           # RTF has a header but not always present
}

# WEBP needs an additional check (RIFF header + WEBP marker at bytes 8-12)
_WEBP_MARKER = b"WEBP"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class FileValidationError(Exception):
    """Raised when a file fails validation."""
    def __init__(self, message: str, code: str = "FILE_VALIDATION_ERROR"):
        self.message = message
        self.code    = code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_extension(filename: str) -> str:
    """
    Check that the file extension is in the allowed list.
    Returns the extension (lowercase) if valid.
    Raises FileValidationError if not.
    """
    ext = os.path.splitext(filename)[1].lower()
    allowed = app_config.allowed_upload_extensions

    if ext not in allowed:
        raise FileValidationError(
            f"File type '{ext}' is not allowed. "
            f"Allowed types: {', '.join(sorted(allowed))}",
            code="UNSUPPORTED_FILE_TYPE",
        )
    return ext


def validate_file_size(file_path: str) -> int:
    """
    Check that the file size is within the configured limit.
    Returns file size in bytes if valid.
    Raises FileValidationError if too large.
    """
    try:
        size_bytes = os.path.getsize(file_path)
    except OSError as e:
        raise FileValidationError(
            f"Could not read file size: {e}",
            code="FILE_READ_ERROR",
        )

    max_bytes = app_config.max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileValidationError(
            f"File size ({size_bytes / 1024 / 1024:.1f} MB) exceeds the "
            f"maximum allowed size ({app_config.max_file_size_mb} MB).",
            code="FILE_TOO_LARGE",
        )

    if size_bytes == 0:
        raise FileValidationError(
            "File is empty.",
            code="FILE_EMPTY",
        )

    return size_bytes


def validate_magic_bytes(file_path: str, ext: str) -> None:
    """
    Check that the file's magic bytes match the declared extension.
    Raises FileValidationError if the content doesn't match.

    Text formats (txt, csv, md) are skipped — they have no magic bytes.
    """
    signatures = MAGIC_BYTES.get(ext)
    if signatures is None:
        # Extension not in our map — skip magic byte check
        return

    if not signatures:
        # Known text format with no magic bytes — skip
        return

    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except OSError as e:
        raise FileValidationError(
            f"Could not read file for validation: {e}",
            code="FILE_READ_ERROR",
        )

    # Special case: WEBP needs RIFF + WEBP marker
    if ext == ".webp":
        if not header.startswith(b"RIFF") or _WEBP_MARKER not in header[8:12]:
            raise FileValidationError(
                f"File content does not match declared type '{ext}'. "
                "File may be corrupted or misnamed.",
                code="FILE_MAGIC_MISMATCH",
            )
        return

    # General case: check any signature matches
    for sig in signatures:
        if header.startswith(sig):
            return

    raise FileValidationError(
        f"File content does not match declared type '{ext}'. "
        "File may be corrupted or misnamed.",
        code="FILE_MAGIC_MISMATCH",
    )


def validate_upload(file_path: str, filename: str) -> dict:
    """
    Run all validations on an uploaded file.

    Args:
        file_path: Path to the saved temp file on disk.
        filename:  Original filename from the upload (used for extension check).

    Returns dict with validation results on success.
    Raises FileValidationError on any failure.

    Call this after saving the file to disk, before passing to ingest_file().
    """
    ext       = validate_extension(filename)
    size      = validate_file_size(file_path)
    validate_magic_bytes(file_path, ext)

    return {
        "extension":  ext,
        "size_bytes": size,
        "size_mb":    round(size / 1024 / 1024, 2),
    }


def validate_upload_or_raise_http(file_path: str, filename: str) -> dict:
    """
    Wrapper that converts FileValidationError to FastAPI HTTPException.
    Use this in routers.
    """
    try:
        return validate_upload(file_path, filename)
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": e.message, "code": e.code},
        )