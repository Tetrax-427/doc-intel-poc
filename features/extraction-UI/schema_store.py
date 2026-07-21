"""
schema_store.py
----------------
Saves/loads reusable extraction schemas to a local JSON file.

A schema is always fields-based - whether it was built manually or derived
from an NL instruction preview, both end up in the same shape (the backend
only accepts a flat {field_name: description} dict for extraction, so
there's no separate "type" or "doc_type" to track):

{
  "mode": "fields",
  "fields": [
     {"name": "invoice_number", "description": "The invoice number"},
     {"name": "total_amount", "description": "Total amount due"}
  ],
  "created_at": "2026-07-02T10:00:00"
}
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SCHEMA_FILE = DATA_DIR / "schemas.json"


def _ensure_store():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SCHEMA_FILE.exists():
        SCHEMA_FILE.write_text("{}")


def load_schemas() -> dict:
    _ensure_store()
    try:
        return json.loads(SCHEMA_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def save_schema(name: str, schema: dict):
    schemas = load_schemas()
    schema = dict(schema)
    schema["created_at"] = schema.get("created_at") or datetime.now(timezone.utc).isoformat()
    schemas[name] = schema
    _ensure_store()
    SCHEMA_FILE.write_text(json.dumps(schemas, indent=2))


def delete_schema(name: str):
    schemas = load_schemas()
    schemas.pop(name, None)
    SCHEMA_FILE.write_text(json.dumps(schemas, indent=2))


def get_schema(name: str) -> dict | None:
    return load_schemas().get(name)


def list_schema_names() -> list:
    return sorted(load_schemas().keys())