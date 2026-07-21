"""
api_client.py
--------------
Thin HTTP wrapper around the DocIntel FastAPI backend.

CONFIRMED CONTRACTS (verified against live responses, not guessed):

POST {base_url}/upload
    multipart file upload -> {"document_id": ..., ...}

POST {base_url}/extract/nl   (schema preview mode)
    request:  {"document_id": "", "instruction": "...", "preview_only": true}
    response: {"schema": {"field_name": "field description", ...}, "extracted": null, "validation": null}
    document_id is unused when preview_only=true, so we always send "".

POST {base_url}/extract
    request:  {"document_id": "...", "fields": {"field_name": "field description", ...}}
              (no per-field type, no doc_type - the backend doesn't accept either)
    response: {
        "extracted": {"field_name": {"value": ..., "bbox": ...}, ...},
        "validation": {...},          # not used by this app
        "business_validation": {...}, # not used by this app
        "extraction_id": "..."
    }

POST {base_url}/extract/batch   (defined, not currently used by the UI - see README)
    request:  {"document_ids": [...], "fields": {"field_name": "description", ...}}

GET {base_url}/templates -> {"templates": [...]}  (list also tolerated)
GET {base_url}/health    -> 200 OK

AUTH: JWT only, via Authorization: Bearer <token>. Token comes from
config_store.get_auth_token() (env var), never entered in the UI.

All functions raise `ApiError` with a readable message on failure so the
Streamlit layer can surface it instead of crashing.
"""
import requests

# Per-operation default timeouts (seconds). "light" calls (health, templates)
# should fail fast; upload/extract can take a while for real documents.
DEFAULT_TIMEOUTS = {
    "light": 30,
    "upload": 180,
    "extract": 180,
    "batch": 300,
}


def fields_to_payload(fields: list) -> dict:
    """
    Our UI stores fields internally as a list of {name, description} (nicer
    to edit/render as rows). The backend wants a dict of
    {field_name: description}. This converts at the boundary.
    """
    return {f["name"]: f.get("description") or f["name"] for f in fields if f.get("name")}


def extract_value(field_value):
    """
    Flattens one entry from an `extracted` response into a plain value for
    a table cell. Handles the confirmed {"value": ..., "bbox": ...} shape;
    falls back to the raw value for anything else so we don't silently
    drop data if the backend returns something unexpected for a field.
    """
    if isinstance(field_value, dict) and "value" in field_value:
        return field_value["value"]
    return field_value


class ApiError(Exception):
    pass


class DocIntelClient:
    def __init__(self, base_url: str, jwt_token: str = "", timeout: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.jwt_token = jwt_token.strip()
        # If the caller passes an explicit timeout (e.g. from the sidebar),
        # it overrides the per-operation defaults for every call.
        self.timeout_override = timeout

    def _headers(self, json_mode: bool = True):
        headers = {}
        if json_mode:
            headers["Content-Type"] = "application/json"
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        return headers

    def _request(self, method: str, path: str, op: str = "light", **kwargs):
        url = f"{self.base_url}{path}"
        timeout = self.timeout_override or DEFAULT_TIMEOUTS.get(op, 60)
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout:
            raise ApiError(
                f"{method} {path} timed out after {timeout}s. The backend may still be "
                f"working (large file, slow parser, or a slow LLM call) - it's often still "
                f"running server-side even though this request gave up. Try increasing the "
                f"timeout in the sidebar's Advanced settings, or check your backend logs to "
                f"see if it actually completed."
            )
        except requests.exceptions.RequestException as e:
            raise ApiError(f"Could not reach {url}: {e}")

        if not resp.ok:
            detail = resp.text
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                pass
            raise ApiError(f"{method} {path} failed [{resp.status_code}]: {detail}")

        if resp.content:
            try:
                return resp.json()
            except ValueError:
                return {"raw": resp.text}
        return {}

    def health(self) -> bool:
        try:
            self._request("GET", "/health", op="light", headers=self._headers(json_mode=False))
            return True
        except ApiError:
            return False

    def get_templates(self):
        data = self._request("GET", "/templates", op="light", headers=self._headers(json_mode=False))
        if isinstance(data, list):
            return data
        return data.get("templates", [])

    def upload_document(self, file_bytes: bytes, filename: str) -> dict:
        files = {"file": (filename, file_bytes)}
        headers = self._headers(json_mode=False)
        return self._request("POST", "/upload", op="upload", files=files, headers=headers)

    def preview_nl_schema(self, instruction: str) -> dict:
        """
        Turns a natural-language instruction into a field schema WITHOUT
        needing any uploaded document. Returns the raw {field_name:
        description} dict from `response["schema"]`.
        """
        payload = {"document_id": "", "instruction": instruction, "preview_only": True}
        result = self._request("POST", "/extract/nl", op="light", json=payload, headers=self._headers())
        return result.get("schema", {})

    def extract_fields(self, document_id: str, fields: list) -> dict:
        payload = {"document_id": document_id, "fields": fields_to_payload(fields)}
        return self._request("POST", "/extract", op="extract", json=payload, headers=self._headers())

    def extract_batch(self, document_ids: list, fields: list) -> dict:
        payload = {"document_ids": document_ids, "fields": fields_to_payload(fields)}
        return self._request("POST", "/extract/batch", op="batch", json=payload, headers=self._headers())