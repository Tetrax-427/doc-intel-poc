"""
features/doc_comparison/compare_client.py

Calls POST /compare on the engine. Uses auth_client's auto-refresh wrapper
so a mid-session expired access token doesn't surface as a raw error to
the user — it transparently refreshes once and retries.
"""

import os

import requests
from dotenv import load_dotenv

from auth_client import auth_headers, call_with_auto_refresh, AuthError

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
COMPARE_TIMEOUT = int(os.getenv("COMPARE_TIMEOUT_SECONDS", "180"))


class CompareError(Exception):
    def __init__(self, detail: str, status_code: int = None):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def compare_documents(file_a, file_b, include_summary: bool = False) -> dict:
    """
    file_a, file_b: Streamlit UploadedFile objects.
    Returns the parsed JSON dict (segments/stats/position_type/summary)
    on success. Raises CompareError on any non-200.
    """
    files = {
        "file_a": (file_a.name, file_a.getvalue(), file_a.type),
        "file_b": (file_b.name, file_b.getvalue(), file_b.type),
    }
    data = {"include_summary": str(include_summary).lower()}

    def do_request():
        return requests.post(
            f"{BACKEND_URL}/compare",
            files=files,
            data=data,
            headers=auth_headers(),
            timeout=COMPARE_TIMEOUT,
        )

    response = call_with_auto_refresh(do_request)

    if response.status_code == 401:
        raise CompareError("Session expired. Please log in again.", 401)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text or f"HTTP {response.status_code}"
        raise CompareError(detail, response.status_code)

    return response.json()