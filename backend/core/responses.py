"""
Consistent response shapes for all API endpoints.
Every error goes through error_response(); every success through success_response().
"""

from fastapi.responses import JSONResponse


def error_response(
    message: str,
    code: str = "UNKNOWN_ERROR",
    status_code: int = 400
) -> JSONResponse:
    """
    Return a uniform error envelope.

    Shape:
        { "error": true, "message": "...", "code": "SNAKE_CASE_CODE" }
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "message": message,
            "code": code,
        },
    )


def success_response(data: dict, message: str = "") -> dict:
    """
    Return a uniform success envelope.

    Shape:
        { "error": false, "message": "...", "data": { ... } }
    """
    return {
        "error": False,
        "message": message,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Common error factories — use these instead of repeating strings everywhere
# ---------------------------------------------------------------------------

def not_found(resource: str = "Resource") -> JSONResponse:
    return error_response(f"{resource} not found.", code="NOT_FOUND", status_code=404)


def bad_request(message: str, code: str = "BAD_REQUEST") -> JSONResponse:
    return error_response(message, code=code, status_code=400)


def internal_error(message: str = "An unexpected error occurred.") -> JSONResponse:
    return error_response(message, code="INTERNAL_ERROR", status_code=500)


def unsupported_file_type(ext: str) -> JSONResponse:
    return error_response(
        f"Unsupported file type: {ext}",
        code="UNSUPPORTED_FILE_TYPE",
        status_code=415,
    )
