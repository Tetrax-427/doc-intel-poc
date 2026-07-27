"""
schemas/dynamic.py — Dynamic, recursive extraction schema support.

NEW FILE. Adds the meta-schema DSL (SchemaSpec / FieldSpec), a recursive
Pydantic model builder, and NL-description -> SchemaSpec generation.

Public API:
    FieldSpec, SchemaSpec        — the meta-schema DSL (recursive)
    generate_schema_spec(...)    — NL description -> SchemaSpec (via LLM)
    spec_to_model(...)           — SchemaSpec -> runtime Pydantic model (cached)
    spec_to_field_descriptions() — SchemaSpec -> human-readable field list for prompts
    clear_model_cache()          — drop the process-lifetime model cache

Used by:
    retrieval.extract_dynamic_fields() / retrieval.extract_nl()
    routers/extraction.py  (POST /extract with nested_schema, POST /extract/nl)
"""

from __future__ import annotations

import hashlib
from typing import Optional, Literal

from pydantic import BaseModel, Field, create_model

from core.logger import get_logger

logger = get_logger("schemas.dynamic")

_PRIMITIVE_TYPES = {
    "string": str,
    "number": float,
    "integer": int,
    "date": str,
    "boolean": bool,
}


# ---------------------------------------------------------------------------
# The meta-schema DSL — recursive, self-describing
# ---------------------------------------------------------------------------

class FieldSpec(BaseModel):
    name: str = Field(description="snake_case field name")
    type: Literal["string", "number", "integer", "date", "boolean", "list", "object"]
    description: str = ""
    properties: Optional[list["FieldSpec"]] = Field(
        default=None,
        description="Sub-fields — required when type is 'object', or when type is "
                    "'list' (describes the shape of each item in the list).",
    )


FieldSpec.model_rebuild()  # needed because FieldSpec references itself


class SchemaSpec(BaseModel):
    schema_name: str = Field(description="Short snake_case name for this schema")
    fields: list[FieldSpec]


# ---------------------------------------------------------------------------
# Schema generation — NL description -> SchemaSpec
# ---------------------------------------------------------------------------

def generate_schema_spec(description: str, user_id: str = "system") -> SchemaSpec:
    """
    Convert a free-text description into a SchemaSpec via LLM (Instructor-backed,
    same as every other structured call in the app — see llm/engine.call_llm()).

    Works for any domain — the caller doesn't need to know in advance whether the
    result will be flat or deeply nested; that's decided by the LLM based on the
    description and enforced by SchemaSpec's own validation.

    Raises whatever llm.engine.call_llm() raises on failure (LLMFallbackExhaustedError,
    StructuredOutputError, etc.) — callers should catch broadly, same pattern as the
    old nl_to_schema().
    """
    from llm.engine import call_llm          # local import — avoids a circular import at module load
    from prompts import SCHEMA_GEN_SYSTEM    # prompt lives in prompts.py per repo convention

    result: SchemaSpec = call_llm(
        system=SCHEMA_GEN_SYSTEM,
        user=f"Description: {description}",
        temperature=0.0,
        call_type="schema_generate",
        response_model=SchemaSpec,
        user_id=user_id,
    )
    logger.info(
        "Generated dynamic schema",
        schema_name=result.schema_name,
        field_count=len(result.fields),
    )
    return result


# ---------------------------------------------------------------------------
# Recursive model builder (SchemaSpec -> Pydantic), with process-lifetime cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, type[BaseModel]] = {}


def _fields_to_model(fields: list[FieldSpec], model_name: str) -> type[BaseModel]:
    """Recursively turn a list[FieldSpec] into a Pydantic model class."""
    kwargs: dict[str, tuple] = {}

    for f in fields:
        safe_name = f.name.strip().replace(" ", "_").replace("-", "_")

        if f.type == "list":
            item_fields = f.properties or []
            if item_fields:
                item_model = _fields_to_model(item_fields, f"{model_name}_{safe_name}_Item")
                kwargs[safe_name] = (
                    list[item_model],
                    Field(default_factory=list, description=f.description),
                )
            else:
                # No sub-fields declared -> treat as a list of plain strings
                kwargs[safe_name] = (
                    list[str],
                    Field(default_factory=list, description=f.description),
                )

        elif f.type == "object":
            nested_model = _fields_to_model(f.properties or [], f"{model_name}_{safe_name}")
            kwargs[safe_name] = (
                Optional[nested_model],
                Field(default=None, description=f.description),
            )

        else:
            py_type = _PRIMITIVE_TYPES.get(f.type, str)
            kwargs[safe_name] = (
                Optional[py_type],
                Field(default=None, description=f.description or f"Extract {safe_name} from the document"),
            )

    if not kwargs:
        # create_model requires at least one field — degenerate empty object case
        kwargs["_placeholder"] = (Optional[str], Field(default=None))

    return create_model(model_name, **kwargs)


def spec_to_model(spec: SchemaSpec) -> type[BaseModel]:
    """
    Build (or retrieve from cache) a Pydantic model from a SchemaSpec.

    Cache key is the MD5 hash of the spec's JSON — same keying pattern as
    core/cache.py's embedding cache. This means the same saved/inline schema
    reused across many extractions rebuilds the model only once per process.
    """
    cache_key = hashlib.md5(spec.model_dump_json().encode()).hexdigest()
    if cache_key not in _model_cache:
        model_name = "".join(w.capitalize() for w in spec.schema_name.split("_")) or "DynamicExtraction"
        _model_cache[cache_key] = _fields_to_model(spec.fields, model_name)
        logger.info(
            "Built dynamic extraction model",
            schema_name=spec.schema_name,
            cache_key=cache_key,
            field_count=len(spec.fields),
        )
    return _model_cache[cache_key]


def clear_model_cache() -> None:
    """Clear the dynamic model cache. In-process only — lost on restart, same as core/cache.py."""
    _model_cache.clear()


# ---------------------------------------------------------------------------
# Prompt-facing helpers
# ---------------------------------------------------------------------------

def spec_to_field_descriptions(spec: SchemaSpec) -> str:
    """
    Render a SchemaSpec as a human-readable, indented field list for the
    extraction prompt's user content — the nested equivalent of the flat
    `fields_with_desc` string built in retrieval.extract_fields().
    """
    lines: list[str] = []

    def _render(fields: list[FieldSpec], indent: int = 0):
        prefix = "  " * indent
        for f in fields:
            if f.type == "list":
                lines.append(f"{prefix}- {f.name} (list): {f.description}")
                if f.properties:
                    lines.append(f"{prefix}  each item has:")
                    _render(f.properties, indent + 2)
            elif f.type == "object":
                lines.append(f"{prefix}- {f.name} (object): {f.description}")
                if f.properties:
                    _render(f.properties, indent + 1)
            else:
                lines.append(f"{prefix}- {f.name} ({f.type}): {f.description}")

    _render(spec.fields)
    return "\n".join(lines)