"""
llm/structured.py — Instructor-powered structured output for DocIntel.

Public API:
    call_structured(raw_client, provider, model, system, user,
                    response_model, temperature, call_type, max_retries)
        -> (BaseModel, usage_dict | None)
    build_extraction_model(fields: dict[str, str]) -> type[BaseModel]

Pydantic models (used by retrieval.py call sites):
    DocumentClassification
    QueryExpansion
    ExtractionResult       (dynamic — built by build_extraction_model, flat schemas)
    DocumentSummary
    TableItem / TableList
    SchemaResult

CHANGED in this phase (dynamic/complex schema extraction):
  - call_structured() gains a `max_retries` param, passed straight through to
    Instructor's own `.create(max_retries=...)`. Instructor uses this to
    re-prompt the model with the Pydantic validation error when the response
    fails to validate — this matters far more for nested/dynamic schemas
    (schemas.dynamic.spec_to_model) than for the flat ExtractionResult models,
    since nested list-of-object shapes are easier for the LLM to get wrong on
    the first try. Default stays 0 (no behaviour change for existing callers);
    retrieval.extract_dynamic_fields() passes max_retries=2 explicitly.
  - engine.call_llm() / engine._call_single_provider() thread max_retries
    through to here (see llm/engine.py CHANGED note).

Everything else below is unchanged from the previous version.
"""

from __future__ import annotations

from typing import Any

import instructor
from pydantic import BaseModel, Field, ConfigDict, create_model

from core.errors import StructuredOutputError, LLMConfigError
from core.logger import get_logger

logger = get_logger("llm.structured")

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class DocumentClassification(BaseModel):
    """Response model for classify_document()."""
    doc_type:   str   = Field(description="Detected document type (e.g. invoice, resume, contract)")
    confidence: float = Field(description="Confidence score 0.0–1.0")
    reasoning:  str   = Field(description="One sentence explaining the classification")
    key_signals: list[str] = Field(default_factory=list,
                                   description="2–4 short phrases from the document that led to this classification")


class QueryExpansion(BaseModel):
    """Response model for expand_query()."""
    expanded_query: str = Field(
        description="Rewritten version of the query optimised for document retrieval. "
                    "If the original is already clear, return it unchanged."
    )


class DocumentSummary(BaseModel):
    """Response model for generate_summary()."""
    short:         str        = Field(description="One sentence (max 20 words) describing what this document is")
    overview:      str        = Field(description="2–3 sentence overview of the document")
    key_topics:    list[str]  = Field(default_factory=list, description="3–5 main topics covered")
    entities:      list[str]  = Field(default_factory=list, description="Important names, companies, or organisations")
    dates:         list[str]  = Field(default_factory=list, description="Important dates mentioned")
    amounts:       list[str]  = Field(default_factory=list, description="Important numbers, amounts, or figures")
    document_type: str        = Field(description="Type of document (e.g. Resume, Invoice, Contract, Report)")


class TableItem(BaseModel):
    """A single extracted table."""
    title:      str         = Field(description="Descriptive title for the table")
    headers:    list[str]   = Field(default_factory=list, description="Column names")
    rows:       list[list[str]] = Field(default_factory=list, description="Rows of values")
    chart_type: str         = Field(default="bar", description="Suggested chart type: bar, line, or pie")


class TableList(BaseModel):
    """Wrapper so Instructor can return a list of tables."""
    tables: list[TableItem] = Field(default_factory=list)


class SchemaResult(BaseModel):
    """
    Legacy response model for the old flat nl_to_schema() path.
    Superseded by schemas.dynamic.SchemaSpec for new NL extraction requests
    (see retrieval.extract_nl()), but kept here since it's still a valid,
    lightweight shape for any caller that only needs a flat field list.
    Field names are dynamic, so we allow extra fields.
    """
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Dynamic extraction model factory — FLAT schemas
# ---------------------------------------------------------------------------
# For nested/complex schemas (lists of objects, nested objects), see
# schemas.dynamic.spec_to_model() instead — this factory only ever produces
# flat Optional[str] fields and is kept for template-based / flat custom
# extraction (retrieval.extract_fields()).

def build_extraction_model(fields: dict[str, str]) -> type[BaseModel]:
    """
    Build a Pydantic model at runtime from a {field_name: description} dict.

    All fields are Optional[str] with a None default so the model never
    fails validation when the LLM omits a field.

    Args:
        fields: dict mapping snake_case field names to plain-English descriptions.

    Returns:
        A new BaseModel subclass named "ExtractionResult".
    """
    field_defs: dict[str, Any] = {}
    for name, description in fields.items():
        safe_name = name.strip().replace(" ", "_").replace("-", "_")
        field_defs[safe_name] = (
            str | None,
            Field(default=None, description=description or f"Extract {safe_name} from the document"),
        )

    if not field_defs:
        raise ValueError("build_extraction_model: fields dict must not be empty")

    return create_model("ExtractionResult", **field_defs)


# ---------------------------------------------------------------------------
# Instructor client factory
# ---------------------------------------------------------------------------

def _get_instructor_client(raw_client, provider: str):
    provider = provider.strip().lower()

    if provider == "groq":
        return instructor.from_groq(raw_client, mode=instructor.Mode.JSON)

    if provider == "openai":
        return instructor.from_openai(raw_client)

    if provider == "anthropic":
        return instructor.from_anthropic(raw_client)

    raise LLMConfigError(
        f"Instructor adapter not available for provider '{provider}'.",
        provider=provider,
    )

# ---------------------------------------------------------------------------
# Usage extraction — defensive, multi-shape
# ---------------------------------------------------------------------------

def _extract_usage(result: BaseModel, provider: str) -> dict | None:
    """
    Best-effort extraction of token usage from an Instructor-returned model.

    Instructor attaches the raw provider response to the validated model as
    `_raw_response` (this has been stable across recent instructor versions,
    but is not a guaranteed public API — hence the defensive try/except here
    rather than assuming a fixed shape).

    Provider response.usage shapes differ:
      - OpenAI / Groq (OpenAI-compatible): usage.prompt_tokens, usage.completion_tokens,
        usage.total_tokens
      - Anthropic: usage.input_tokens, usage.output_tokens (no total_tokens field —
        computed here as their sum)

    Returns:
        {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        or None if usage could not be found in any recognised shape. Returning
        None (not zeros) is deliberate — see llm/tracer.py's cost estimator,
        which treats None as "unknown" rather than "free".
    """
    try:
        raw = getattr(result, "_raw_response", None)
        if raw is None:
            return None

        usage = getattr(raw, "usage", None)
        if usage is None:
            return None

        if provider == "anthropic":
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is None or output_tokens is None:
                return None
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        # OpenAI-compatible (openai, groq)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return None
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        }

    except Exception as exc:
        # Never let usage extraction break the actual call — log once at
        # debug level and move on with usage=None.
        logger.debug("Usage extraction failed — continuing without it", provider=provider, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Core structured call
# ---------------------------------------------------------------------------

def call_structured(
    *,
    raw_client,
    provider: str,
    model: str,
    system: str,
    user: str,
    response_model: type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    call_type: str = "structured",
    max_retries: int = 0,
) -> tuple[BaseModel, dict | None]:
    """
    Make an LLM call and coerce the response into response_model via Instructor.

    Args:
        raw_client:     Raw Groq / OpenAI / Anthropic client (from fallback.build_client).
        provider:       Provider name string (for adapter selection + error context).
        model:          Model string passed to the API.
        system:         Static instruction text for this call.
        user:           Per-call content (the thing the instruction operates on).
        response_model: Pydantic BaseModel subclass to coerce into.
        temperature:    Sampling temperature (default 0.0 for structured tasks).
        max_tokens:     Max response tokens.
        call_type:      Label for tracing.
        max_retries:    Passed straight to Instructor's own `.create(max_retries=...)`.
                        Instructor re-prompts the model with the Pydantic validation
                        error on each retry — this is Instructor's internal retry
                        loop, separate from the provider-fallback loop in engine.py.
                        Default 0 (no retry) preserves prior behaviour for existing
                        callers; pass >0 for schemas where first-pass validation
                        failures are more likely (e.g. nested/dynamic schemas via
                        schemas.dynamic.spec_to_model()).

    Returns:
        (validated_instance, usage_dict_or_None)

    Raises:
        StructuredOutputError: if Instructor fails to coerce the response
                                (including after exhausting max_retries).
        LLMConfigError:        if provider has no Instructor adapter.
    """
    instructor_client = _get_instructor_client(raw_client, provider)

    if provider == "anthropic":
        # Layer 1: cache_control on the system block. Same as engine.py's
        # _call_anthropic — the system instruction is stable per call_type,
        # so marking it ephemeral gives Anthropic the chance to cache it
        # server-side and apply a token discount on repeated calls.
        # Import CACHE_ENABLED from engine here rather than duplicating the
        # env-var read, since engine.py already owns the feature-flag logic.
        try:
            from llm.engine import CACHE_ENABLED as _cache_enabled
        except ImportError:
            _cache_enabled = True  # safe default if import fails
        if _cache_enabled:
            system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            system_block = system
        kwargs: dict[str, Any] = {
            "model":          model,
            "max_tokens":     max_tokens,
            "temperature":    temperature,
            "system":         system_block,
            "messages":       [{"role": "user", "content": user}],
            "response_model": response_model,
        }
    else:
        kwargs = {
            "model":          model,
            "max_tokens":     max_tokens,
            "temperature":    temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_model": response_model,
        }

    if max_retries:
        kwargs["max_retries"] = max_retries

    try:
        result = instructor_client.messages.create(**kwargs) \
            if provider == "anthropic" \
            else instructor_client.chat.completions.create(**kwargs)

        usage = _extract_usage(result, provider)
        return result, usage

    except Exception as exc:
        logger.error(
            "Instructor structured call failed",
            provider=provider,
            model=model,
            response_model=response_model.__name__,
            max_retries=max_retries,
            error=str(exc),
        )
        raise StructuredOutputError(
            str(exc),
            response_model_name=response_model.__name__,
            provider=provider,
            model=model,
        ) from exc