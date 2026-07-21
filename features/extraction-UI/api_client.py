"""
api_client.py
--------------
HTTP wrapper around the DocIntel FastAPI backend, built on httpx (supports
both sync and async cleanly, so one library covers everything instead of
mixing `requests` + `httpx`).

CONFIRMED CONTRACTS (verified against real backend code/responses):

POST /auth/login    {"email","password"} -> {"access_token","refresh_token","token_type","user":{"id","email"}}
POST /auth/refresh  {"refresh_token"}     -> {"access_token","refresh_token","token_type"}
POST /auth/logout   (bearer token)        -> {"status":"logged_out"}
GET  /auth/me       (bearer token)        -> {"user_id","email",...}
POST /auth/reset-password  {"access_token","new_password"} -> {"status":"password_updated"}
    NOTE: there is no dedicated "change password with old password" endpoint.
    We compose it: call /auth/login with the OLD password first (this both
    verifies it's correct and yields a fresh access_token), then call
    /auth/reset-password with that access_token + the new password.

POST /upload   (multipart file, bearer token)
    SYNCHRONOUS - fully processes (parse, classify, summarize) before
    returning. No task_id/polling involved.
    -> {"document_id": ..., ...}

POST /extract/nl   (schema preview mode)
    request:  {"document_id": "", "instruction": "...", "preview_only": true}
    response: {"schema": {"field_name": "field description", ...}, "extracted": null, "validation": null}

POST /extract
    request:  {"document_id": "...", "fields": {"field_name": "description", ...}}
    response: {"extracted": {"field_name": {"value": ..., "bbox": ...}, ...}, "validation": {...}, ...}
    fields is a flat dict of {name: description} - no per-field type, no doc_type.

GET /templates -> {"templates": [...]}  (list also tolerated)
GET /health    -> 200 OK

All sync methods raise `ApiError` with a readable message on failure.
"""
import httpx

# Per-operation default timeouts (seconds), used as a floor even if the
# configured timeout is lower - health/auth calls should still fail fast,
# but never wait less than these for the heavier operations.
MIN_TIMEOUTS = {
    "light": 10,
    "upload": 60,
    "extract": 60,
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
    def __init__(self, message, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _raise_for_response(resp: httpx.Response, method: str, path: str):
    if resp.status_code < 400:
        return
    detail = resp.text
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        pass
    raise ApiError(f"{method} {path} failed [{resp.status_code}]: {detail}", status_code=resp.status_code)


def _parse_json(resp: httpx.Response) -> dict:
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


class DocIntelClient:
    """Synchronous client for auth, health, templates, and schema preview -
    all quick, one-off calls that don't need to run concurrently."""

    def __init__(self, base_url: str, access_token: str = "", timeout: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token.strip()
        self.timeout = timeout or 60

    def _headers(self, json_mode: bool = True, auth: bool = True):
        headers = {}
        if json_mode:
            headers["Content-Type"] = "application/json"
        if auth and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _request(self, method: str, path: str, op: str = "light", **kwargs):
        url = f"{self.base_url}{path}"
        timeout = max(self.timeout, MIN_TIMEOUTS.get(op, 10))
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, **kwargs)
        except httpx.TimeoutException:
            raise ApiError(
                f"{method} {path} timed out after {timeout}s. The backend may still be working "
                f"(large file, slow parser, or a slow LLM call) - increase DOCINTEL_TIMEOUT if this "
                f"keeps happening, or check your backend logs to see if it actually completed."
            )
        except httpx.HTTPError as e:
            raise ApiError(f"Could not reach {url}: {e}")

        _raise_for_response(resp, method, path)
        return _parse_json(resp)

    # ---- Auth ----
    def login(self, email: str, password: str) -> dict:
        return self._request(
            "POST", "/auth/login", op="light",
            json={"email": email, "password": password}, headers=self._headers(auth=False),
        )

    def refresh(self, refresh_token: str) -> dict:
        return self._request(
            "POST", "/auth/refresh", op="light",
            json={"refresh_token": refresh_token}, headers=self._headers(auth=False),
        )

    def logout(self) -> dict:
        return self._request("POST", "/auth/logout", op="light", headers=self._headers(json_mode=False))

    def me(self) -> dict:
        return self._request("GET", "/auth/me", op="light", headers=self._headers(json_mode=False))

    def reset_password(self, access_token: str, new_password: str) -> dict:
        return self._request(
            "POST", "/auth/reset-password", op="light",
            json={"access_token": access_token, "new_password": new_password},
            headers=self._headers(auth=False),
        )

    # ---- Health / templates / schema preview ----
    def health(self) -> bool:
        try:
            self._request("GET", "/health", op="light", headers=self._headers(json_mode=False, auth=False))
            return True
        except ApiError:
            return False

    def get_templates(self):
        data = self._request("GET", "/templates", op="light", headers=self._headers(json_mode=False))
        if isinstance(data, list):
            return data
        return data.get("templates", [])

    def preview_nl_schema(self, instruction: str) -> dict:
        payload = {"document_id": "", "instruction": instruction, "preview_only": True}
        result = self._request("POST", "/extract/nl", op="light", json=payload, headers=self._headers())
        return result.get("schema", {})


class AsyncDocIntelClient:
    """
    Async client used only by the parallel upload+extract pipeline
    (see batch_runner.py). Kept separate from DocIntelClient so the sync
    call sites (login, schema preview, etc.) stay simple.
    """

    def __init__(self, base_url: str, access_token: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token.strip()
        self.timeout = timeout

    def _headers(self, json_mode: bool = True):
        headers = {}
        if json_mode:
            headers["Content-Type"] = "application/json"
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def upload_document(self, client: httpx.AsyncClient, file_bytes: bytes, filename: str) -> dict:
        url = f"{self.base_url}/upload"
        timeout = max(self.timeout, MIN_TIMEOUTS["upload"])
        try:
            resp = await client.post(
                url, files={"file": (filename, file_bytes)},
                headers=self._headers(json_mode=False), timeout=timeout,
            )
        except httpx.TimeoutException:
            raise ApiError(f"Upload of {filename} timed out after {timeout}s.")
        except httpx.HTTPError as e:
            raise ApiError(f"Could not reach {url}: {e}")
        _raise_for_response(resp, "POST", "/upload")
        return _parse_json(resp)

    async def extract_fields(self, client: httpx.AsyncClient, document_id: str, fields: list) -> dict:
        url = f"{self.base_url}/extract"
        timeout = max(self.timeout, MIN_TIMEOUTS["extract"])
        payload = {"document_id": document_id, "fields": fields_to_payload(fields)}
        try:
            resp = await client.post(url, json=payload, headers=self._headers(), timeout=timeout)
        except httpx.TimeoutException:
            raise ApiError(f"Extraction for document {document_id} timed out after {timeout}s.")
        except httpx.HTTPError as e:
            raise ApiError(f"Could not reach {url}: {e}")
        _raise_for_response(resp, "POST", "/extract")
        return _parse_json(resp)