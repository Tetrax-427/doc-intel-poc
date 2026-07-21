# DocIntel Extraction Helper

A small standalone Streamlit app for repeatable, batch document extraction
against an existing DocIntel backend. It's separate from the main DocIntel
repo — it just talks to your FastAPI backend over HTTP.

## What it does

1. **Build a schema** — either manually (name + description per field) or by
   describing what you want in plain English and clicking **Preview
   schema**, which calls the backend's NL-preview mode to derive fields
   *without uploading any document*. Either way you review/edit the fields
   before saving. Schemas are saved locally in `data/schemas.json`.
2. **Run extraction** — pick a saved schema, select documents (browser
   upload or a local folder path), and run. Each file is uploaded to
   DocIntel then extracted using the schema.
3. **Collect results** — results land in a table you can save as a **new
   table** or **append** to an existing one. Tables are plain CSVs under
   `data/tables/`. Each extracted field is flattened to its plain value —
   no raw `{"value": ..., "bbox": ...}` objects in your cells.
4. **Export** — download any run or saved table as an `.xlsx` file.
5. **Reuse** — schemas and tables persist between sessions, so you can keep
   running the same schema over new batches and appending to the same table.

## Setup

```bash
cd extraction-helper
pip install -r requirements.txt
export DOCINTEL_JWT_TOKEN="eyJhbGciOi..."   # see Auth below
streamlit run app.py
```

## Auth — JWT only, via environment variable

There is **no auth UI** in this app on purpose. Set a long-lived JWT once
before running:

```bash
export DOCINTEL_JWT_TOKEN="eyJhbGciOi..."
```

Or, if you deploy this helper app (e.g. Streamlit Community Cloud), put the
same key in `.streamlit/secrets.toml`:

```toml
DOCINTEL_JWT_TOKEN = "eyJhbGciOi..."
```

The token is read fresh from the environment on every request — it's never
written to `data/config.json`, never shown in the sidebar, and never
round-tripped through any UI field. If it's missing, the sidebar shows a
quiet warning; if it expires, requests will start failing auth and you'll
need to generate a new one and re-export it.

**Base URL and request timeout** are still configurable in the sidebar (or
via `DOCINTEL_BASE_URL` / `DOCINTEL_TIMEOUT` env vars / secrets), same as
before — just the credential moved out of the UI.

## Confirmed API contracts

Unlike earlier drafts of this app, these are verified against real
request/response payloads, not guessed from the codebase guide:

**`POST /upload`** — multipart `file` → `{"document_id": ...}`

**`POST /extract/nl`** (schema preview mode) —
```json
// request
{"document_id": "", "instruction": "...", "preview_only": true}
// response
{"schema": {"field_name": "field description", ...}, "extracted": null, "validation": null}
```
`document_id` is unused when `preview_only` is `true`, so the app always
sends `""`.

**`POST /extract`** —
```json
// request
{"document_id": "...", "fields": {"field_name": "description", ...}}
// response (only `extracted` is used by this app)
{
  "extracted": {"field_name": {"value": ..., "bbox": ...}, ...},
  "validation": {...},
  "business_validation": {...},
  "extraction_id": "..."
}
```
Note: `fields` is a flat dict of `{name: description}` — no per-field
`type`, no `doc_type` in the request. Type detection and doc-type
classification happen automatically on the backend.

**`POST /extract/batch`** — defined in `api_client.py` (`{"document_ids":
[...], "fields": {...}}`) but not currently wired into the UI. The app
loops `upload` → `extract` per file instead, to get a live per-file
progress bar and partial results if one document fails. Swap to
`extract_batch()` if you'd rather batch instead.

**`GET /templates`**, **`GET /health`** — as before.

## File structure

```
extraction-helper/
├── app.py              # Streamlit UI (3 tabs: Schema, Run, Tables)
├── api_client.py         # DocIntel API wrapper
├── schema_store.py        # local JSON schema persistence
├── results_store.py        # local CSV table persistence + Excel export
├── config_store.py         # base_url/timeout persistence + env-only JWT read
├── requirements.txt
└── data/                 # created at runtime
    ├── config.json        # base_url + timeout ONLY (never the token)
    ├── schemas.json         # saved extraction schemas
    └── tables/*.csv           # saved result tables
```

## Notes

- Folder-path selection reads files directly off the filesystem the app is
  running on (Streamlit has no native OS folder picker in-browser).
- Supported file types for folder scanning: `.pdf .docx .csv .xlsx .txt .md
  .jpg .jpeg .png`.
- Default request timeout is 180s (300s for batch), adjustable in the
  sidebar's Advanced settings — large uploads and LLM-backed extraction can
  legitimately take a while.