"""
schema_store.py
----------------
Saves/loads reusable extraction schemas to a local JSON file.

Fields now support nesting for list-of-object data (e.g. work experience,
education). Each field is:

{
  "name": "candidate_work_experience",
  "type": "string" | "list",
  "description": "...",
  "properties": null  # or a list of sub-fields (same shape) when type == "list"
}

Flat/scalar-only schemas (properties always null) still work unchanged -
this is additive, not a breaking change.

{
  "mode": "fields",
  "fields": [
     {"name": "candidate_name", "type": "string", "description": "...", "properties": null},
     {"name": "candidate_work_experience", "type": "list", "description": "...", "properties": [
         {"name": "company_name", "type": "string", "description": "...", "properties": null},
         ...
     ]}
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


def normalize_field(field: dict) -> dict:
    """
    Fills in defaults so every field, at every nesting level, always has
    name/type/description/properties - regardless of whether it came from
    manual entry (no "type" key yet) or an NL preview (already typed).
    """
    name = field.get("name", "")
    ftype = field.get("type") or "string"
    description = field.get("description") or ""
    properties = field.get("properties")
    if ftype == "list" and properties:
        properties = [normalize_field(p) for p in properties]
    else:
        properties = None
    return {"name": name, "type": ftype, "description": description, "properties": properties}


def normalize_fields(fields: list) -> list:
    return [normalize_field(f) for f in fields]


def load_schemas() -> dict:
    _ensure_store()
    try:
        return json.loads(SCHEMA_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def save_schema(name: str, schema: dict):
    schemas = load_schemas()
    schema = dict(schema)
    schema["fields"] = normalize_fields(schema.get("fields", []))
    schema["created_at"] = schema.get("created_at") or datetime.now(timezone.utc).isoformat()
    schemas[name] = schema
    _ensure_store()
    SCHEMA_FILE.write_text(json.dumps(schemas, indent=2))


def delete_schema(name: str):
    schemas = load_schemas()
    schemas.pop(name, None)
    SCHEMA_FILE.write_text(json.dumps(schemas, indent=2))


def get_schema(name: str) -> dict | None:
    schema = load_schemas().get(name)
    if schema is not None:
        schema = dict(schema)
        schema["fields"] = normalize_fields(schema.get("fields", []))
    return schema


def list_schema_names() -> list:
    return sorted(load_schemas().keys())


def flatten_fields_for_table(fields: list, prefix: str = "") -> list:
    """
    Produces a flat row list for st.table preview of a saved schema -
    list fields show as one summary row plus indented sub-field rows,
    so the "Saved schemas" expander stays readable without a tree widget.
    """
    rows = []
    for f in fields:
        ftype = f.get("type", "string")
        label = f"{prefix}{f.get('name', '')}" + ("  [list]" if ftype == "list" else "")
        rows.append({"name": label, "type": ftype, "description": f.get("description", "")})
        if ftype == "list" and f.get("properties"):
            rows.extend(flatten_fields_for_table(f["properties"], prefix=prefix + "  ↳ "))
    return rows