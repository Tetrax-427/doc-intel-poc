"""
llm/structured.py — Instructor-powered structured output for DocIntel.

Public API:
    call_structured(raw_client, provider, model, messages,
                    response_model, temperature, call_type) -> BaseModel
    build_extraction_model(fields: dict[str, str]) -> type[BaseModel]

Pydantic models (used by retrieval.py call sites):
    DocumentClassification
    QueryExpansion
    ExtractionResult       (dynamic — built by build_extraction_model)
    DocumentSummary
    TableItem / TableList
    SchemaResult
"""

from __future__ import annotations

import time
from typing import Any

import instructor
from pydantic import BaseModel, Field, ConfigDict, create_model

from core.errors import StructuredOutputError, LLMConfigError
from core.logger import get_logger
from llm.usage import log_usage

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
    Response model for nl_to_schema().
    Field names are dynamic, so we allow extra fields.
    The model is effectively a typed dict.
    """
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Dynamic extraction model factory
# ---------------------------------------------------------------------------

def build_extraction_model(fields: dict[str, str]) -> type[BaseModel]:
    """
    Build a Pydantic model at runtime from a {field_name: description} dict.

    All fields are Optional[str] with a None default so the model never
    fails validation when the LLM omits a field.

    Example:
        fields = {"vendor_name": "Name of the vendor", "total_amount": "Total invoice amount"}
        Model = build_extraction_model(fields)
        # → class ExtractionResult(BaseModel):
        #       vendor_name: str | None = Field(default=None, description="Name of the vendor")
        #       total_amount: str | None = Field(default=None, description="Total invoice amount")

    Args:
        fields: dict mapping snake_case field names to plain-English descriptions.

    Returns:
        A new BaseModel subclass named "ExtractionResult".
    """
    field_defs: dict[str, Any] = {}
    for name, description in fields.items():
        # Sanitise field name — replace spaces/hyphens with underscores
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
    """
    Wrap a raw provider client with Instructor.

    Raises:
        LLMConfigError: if the provider has no Instructor adapter.
    """
    provider = provider.strip().lower()

    if provider in ("groq", "openai"):
        return instructor.from_openai(raw_client)

    if provider == "anthropic":
        return instructor.from_anthropic(raw_client)

    raise LLMConfigError(
        f"Instructor adapter not available for provider '{provider}'.",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Core structured call
# ---------------------------------------------------------------------------

def call_structured(
    *,
    raw_client,
    provider: str,
    model: str,
    messages: list[dict],
    response_model: type[BaseModel],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    call_type: str = "structured",
) -> BaseModel:
    """
    Make an LLM call and coerce the response into response_model via Instructor.

    Instructor handles its own internal retry logic for validation errors.
    This function does NOT retry — the outer fallback loop in engine.py
    catches StructuredOutputError and tries the next provider.

    Args:
        raw_client:     Raw Groq / OpenAI / Anthropic client (from fallback.build_client).
        provider:       Provider name string (for adapter selection + error context).
        model:          Model string passed to the API.
        messages:       Full message list including system prompt if any.
        response_model: Pydantic BaseModel subclass to coerce into.
        temperature:    Sampling temperature (default 0.0 for structured tasks).
        max_tokens:     Max response tokens.
        call_type:      Label for usage tracking.

    Returns:
        Validated instance of response_model.

    Raises:
        StructuredOutputError: if Instructor fails to coerce the response.
        LLMConfigError:        if provider has no Instructor adapter.
    """
    instructor_client = _get_instructor_client(raw_client, provider)

    # Separate system message for Anthropic (top-level param)
    if provider == "anthropic":
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model":          model,
            "max_tokens":     max_tokens,
            "temperature":    temperature,
            "messages":       user_messages,
            "response_model": response_model,
        }
        if system_msg:
            kwargs["system"] = system_msg
    else:
        kwargs = {
            "model":          model,
            "max_tokens":     max_tokens,
            "temperature":    temperature,
            "messages":       messages,
            "response_model": response_model,
        }

    start = time.time()
    try:
        result = instructor_client.messages.create(**kwargs) \
            if provider == "anthropic" \
            else instructor_client.chat.completions.create(**kwargs)

        latency = time.time() - start
        _log_structured(call_type, model, messages, result, latency)
        return result

    except Exception as exc:
        latency = time.time() - start
        logger.error(
            "Instructor structured call failed",
            provider=provider,
            model=model,
            response_model=response_model.__name__,
            latency_ms=round(latency * 1000),
            error=str(exc),
        )
        raise StructuredOutputError(
            str(exc),
            response_model_name=response_model.__name__,
            provider=provider,
            model=model,
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_structured(call_type: str, model: str, messages: list[dict],
                     result: BaseModel, latency: float):
    try:
        prompt_len   = sum(len(m.get("content", "")) for m in messages)
        response_len = len(result.model_dump_json())
        log_usage(
            call_type=call_type,
            model=model,
            prompt_len=prompt_len,
            response_len=response_len,
            latency=latency,
        )
    except Exception:
        pass  # never break a call over logging