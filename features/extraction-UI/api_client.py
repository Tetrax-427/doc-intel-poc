"""
api_client.py
--------------
HTTP wrapper around the DocIntel FastAPI backend, built on httpx.

CONFIRMED CONTRACTS (nested schema support, verified against real backend
error response - a flat "fields" list is rejected):

POST /extract/nl   (schema preview mode)
    response: {"schema": {"schema_name": ..., "fields": [ {name,type,description,properties}, ... ]}, ...}
    NOTE: schema.fields is now a nested list (matches schema_store field shape),
    not a flat {name: description} dict.

POST /extract
    request:  {"document_id": "...", "nested_schema": {"schema_name": "...", "fields": [...]}}
    (top-level "fields" dict is still accepted for pure-scalar/legacy schemas -
    the backend validator wants EITHER "fields" (flat dict) OR "nested_schema"
    (a SchemaSpec: {schema_name, fields: [...]})  - never both.)
    Scalar field dicts in the wire format omit "properties" entirely rather
    than sending "properties": null (matches GET /schemas/example).
    response: {"extracted": {field_name: value_or_list, ...}, "validation": {...}, "business_validation": {...}}
"""
import httpx

MIN_TIMEOUTS = {
    "light": 10,
    "upload": 60,
    "extract": 60,
}


def _clean_field(field: dict) -> dict:
    """Wire format: drop 'properties' key entirely when it's None/empty,
    instead of sending 'properties': null - matches GET /schemas/example."""
    out = {"name": field["name"], "type": field.get("type") or "string", "description": field.get("description") or ""}
    props = field.get("properties")
    if props:
        out["properties"] = [_clean_field(p) for p in props]
    return out


def fields_to_payload(fields: list, schema_name: str = "extraction_schema") -> dict:
    """
    Builds the /extract request body's schema portion. The backend needs
    {"nested_schema": {"schema_name": ..., "fields": [...]}} - not a bare
    fields list - so this now returns the full wrapper dict, ready to merge
    with {"document_id": ...}.
    """
    return {"nested_schema": {"schema_name": schema_name, "fields": [_clean_field(f) for f in fields]}}


def extract_value(field_value):
    """
    Recursively unwraps one entry from an `extracted` response into a plain
    value/structure for table use:
      - scalar field -> {"value": ..., "bbox": ...} -> the value
      - list field    -> list of item-dicts (e.g. work experience entries),
                         each recursed into, with internal bookkeeping keys
                         (like "_verification") stripped out
    """
    if isinstance(field_value, dict):
        if "value" in field_value:
            return field_value["value"]
        return {k: extract_value(v) for k, v in field_value.items() if not k.startswith("_")}
    if isinstance(field_value, list):
        return [extract_value(item) for item in field_value]
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

    def preview_nl_schema(self, instruction: str) -> list:
        """Returns the nested fields list (schema['fields']), not a flat dict."""
        payload = {"document_id": "", "instruction": instruction, "preview_only": True}
        result = self._request("POST", "/extract/nl", op="light", json=payload, headers=self._headers())
        return result.get("schema", {}).get("fields", [])

    # ---- Agents ----
    def list_agents(self) -> list:
        result = self._request("GET", "/agents", op="light", headers=self._headers(json_mode=False))
        return result.get("agents", [])

    def invoke_agent(self, agent_name: str, task: str, document_ids: list, csv_data: list | None = None, extra: dict | None = None, name: str | None = None) -> dict:
        """Returns {"run_id": ..., "status": "pending"} immediately - the actual run happens in a backend worker thread."""
        payload = {"task": task, "document_ids": document_ids, "csv_data": csv_data or [], "extra": extra or {}, "name": name}
        return self._request("POST", f"/agents/{agent_name}/invoke", op="light", json=payload, headers=self._headers())

    def get_agent_run(self, run_id: str) -> dict:
        """Full agent_runs row - status, current_stage, pending_questions, result, error."""
        return self._request("GET", f"/agents/runs/{run_id}", op="light", headers=self._headers(json_mode=False))

    def resume_agent_run(self, run_id: str, answers: dict) -> dict:
        return self._request("POST", f"/agents/runs/{run_id}/resume", op="light", json={"answers": answers}, headers=self._headers())

    def list_agent_runs(self, agent_name: str | None = None, status: str | None = None, limit: int = 50) -> list:
        params = {"limit": limit}
        if agent_name:
            params["agent_name"] = agent_name
        if status:
            params["status"] = status
        return self._request("GET", "/agents/runs", op="light", params=params, headers=self._headers(json_mode=False))
    
    def send_chat_message(self, run_id: str, message: str) -> dict:
        """Returns {"role": "assistant", "content": ...}. 400s (ApiError) if the run isn't completed yet."""
        return self._request("POST", f"/agents/runs/{run_id}/chat", op="light", json={"message": message}, headers=self._headers())
 
    def get_agent_chat_history(self, run_id: str) -> list:
        """Returns the full message list, oldest first: [{"role","content","created_at"}, ...]."""
        result = self._request("GET", f"/agents/runs/{run_id}/chat", op="light", headers=self._headers(json_mode=False))
        return result.get("messages", [])


class AsyncDocIntelClient:
    """Async client used only by the parallel upload+extract pipeline (see batch_runner.py)."""

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

    async def extract_fields(self, client: httpx.AsyncClient, document_id: str, fields: list, schema_name: str = "extraction_schema") -> dict:
        url = f"{self.base_url}/extract"
        timeout = max(self.timeout, MIN_TIMEOUTS["extract"])
        payload = {"document_id": document_id, **fields_to_payload(fields, schema_name)}
        try:
            resp = await client.post(url, json=payload, headers=self._headers(), timeout=timeout)
        except httpx.TimeoutException:
            raise ApiError(f"Extraction for document {document_id} timed out after {timeout}s.")
        except httpx.HTTPError as e:
            raise ApiError(f"Could not reach {url}: {e}")
        _raise_for_response(resp, "POST", "/extract")
        return _parse_json(resp)