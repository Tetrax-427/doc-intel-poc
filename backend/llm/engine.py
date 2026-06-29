from __future__ import annotations

import base64
import time
from typing import Any, Generator

import anthropic
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel

from core.config import config as app_config
from core.errors import (
    LLMError,
    LLMFallbackExhaustedError,
    LLMProviderOverrideError,
    LLMConfigError,
)
from core.logger import get_logger
from llm.fallback import get_fallback_chain, build_client, build_override_chain
from llm import tracer
from llm import cache as llm_cache

logger = get_logger("llm.engine")

# ---------------------------------------------------------------------------
# Module-level provider/model — exported for system.py /health endpoint.
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = app_config.llm_provider
LLM_MODEL:    str = app_config.llm_model

MAX_RETRIES  = 3
RETRY_DELAY  = 2

import os as _os
TRACING_ENABLED: bool = _os.getenv("LLM_TRACING_ENABLED", "true").lower() == "true"
CACHE_ENABLED:   bool = _os.getenv("CACHE_ENABLED",        "true").lower() == "true"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    json_mode: bool = False,
    call_type: str = "general",
    stream: bool = False,
    # Per-call provider override
    provider: str = None,
    model: str = None,
    # Structured output
    response_model: type[BaseModel] = None,
    # Tracing context
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
    # Org/team context — used for usage aggregation in llm_calls table
    # Optional: callers that don't have org context pass None (safe default)
    org_id: str | None = None,
    team_id: str | None = None,
) -> str | dict | BaseModel | Generator:
    """
    Central LLM caller. All LLM calls in the app go through here.

    Args:
        system:         Static instruction text. Required.
        user:           Per-call content (untrusted document content goes here,
                        sandboxed by llm/sanitizer.py before being passed in).
        temperature:    0.0 for deterministic, higher for creative.
        max_tokens:     Max response tokens.
        json_mode:      If True and response_model is None, parses response as JSON.
        call_type:      Label for tracing — becomes llm_calls.call_type.
        stream:         If True, returns a token generator.
        provider:       Override provider for this call only.
        model:          Override model for this call only.
        response_model: Pydantic BaseModel for structured output.
        user_id:        Owning user — required for correct tracing/cache scoping.
        document_id:    Document this call relates to, if any.
        session_id:     Reserved for future chat-session grouping.
        org_id:         Org context for usage aggregation. Optional.
        team_id:        Team context for usage aggregation. Optional.

    Returns:
        - Generator of str tokens  when stream=True
        - BaseModel instance       when response_model is set
        - dict                     when json_mode=True
        - str                      otherwise
    """

    # --- Streaming ---
    if stream:
        return _call_stream(
            system, user, temperature=temperature, call_type=call_type,
            provider=provider, model=model,
            user_id=user_id, document_id=document_id, session_id=session_id,
            org_id=org_id, team_id=team_id,
        )

    # --- Build chain ---
    if provider or model:
        if not (provider and model):
            raise LLMProviderOverrideError(
                provider or "", model or "",
                reason="Both provider and model must be supplied for a per-call override."
            )
        try:
            chain = build_override_chain(provider, model)
        except LLMConfigError as exc:
            raise LLMProviderOverrideError(provider, model, reason=str(exc)) from exc
        is_override = True
    else:
        chain = get_fallback_chain()
        is_override = False

    # --- Walk chain ---
    providers_tried: list[tuple[str, str]] = []
    cacheable        = CACHE_ENABLED and llm_cache.is_cacheable(call_type)
    primary_provider = None

    for (prov, mdl) in chain:
        if primary_provider is None:
            primary_provider = prov
        providers_tried.append((prov, mdl))
        used_fallback = (prov != primary_provider)

        # --- Cache lookup ---
        if cacheable:
            cache_row = llm_cache.lookup(user_id, prov, mdl, system, user)
            if cache_row is not None:
                _trace = tracer.trace(
                    call_type=call_type, provider=prov, model=mdl,
                    system=system, user=user,
                    user_id=user_id, document_id=document_id, session_id=session_id,
                    is_override=is_override, is_stream=False,
                    used_fallback=used_fallback,
                    response_model_name=response_model.__name__ if response_model else None,
                ) if TRACING_ENABLED else _noop_trace()
                with _trace as t:
                    if TRACING_ENABLED:
                        t.set_cache_hit(cache_row["response_text"])
                return _reconstruct_result(
                    cache_row["response_text"], response_model, json_mode,
                )

        try:
            _trace = tracer.trace(
                call_type=call_type, provider=prov, model=mdl,
                system=system, user=user,
                user_id=user_id, document_id=document_id, session_id=session_id,
                is_override=is_override, is_stream=False,
                used_fallback=used_fallback,
                response_model_name=response_model.__name__ if response_model else None,
            ) if TRACING_ENABLED else _noop_trace()
            with _trace as t:
                result, usage = _call_single_provider(
                    provider=prov, model=mdl,
                    system=system, user=user,
                    temperature=temperature, max_tokens=max_tokens,
                    call_type=call_type,
                    response_model=response_model,
                    json_mode=json_mode,
                )
                response_text = _stringify_result(result)
                if TRACING_ENABLED:
                    t.set_result(response_text=response_text, usage=usage)

            # --- Store in cache ---
            if cacheable:
                cost = None
                try:
                    cost = tracer.estimate_cost_usd(prov, mdl, usage)
                except Exception:
                    pass
                llm_cache.store(
                    user_id=user_id, provider=prov, model=mdl,
                    system=system, user=user, call_type=call_type,
                    response_text=response_text,
                    document_id=document_id,
                    response_model_name=response_model.__name__ if response_model else None,
                    original_cost_usd=cost,
                )

            return result

        except Exception as exc:
            logger.warning(
                "Provider failed — trying next in chain",
                provider=prov, model=mdl, error=str(exc),
                is_override=is_override,
            )
            if is_override:
                raise LLMProviderOverrideError(prov, mdl, reason=str(exc)) from exc

    raise LLMFallbackExhaustedError(providers_tried)


def call_llm_stream(
    system: str,
    user: str,
    temperature: float = 0.2,
    call_type: str = "stream",
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
    team_id: str | None = None,
) -> Generator:
    """Convenience wrapper for streaming calls."""
    return call_llm(
        system, user, temperature=temperature, call_type=call_type, stream=True,
        user_id=user_id, document_id=document_id, session_id=session_id,
        org_id=org_id, team_id=team_id,
    )


# ---------------------------------------------------------------------------
# Single-provider call
# ---------------------------------------------------------------------------

def _call_single_provider(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    call_type: str,
    response_model: type[BaseModel] | None,
    json_mode: bool,
) -> tuple[str | dict | BaseModel, dict | None]:
    client = build_client(provider)

    for attempt in range(MAX_RETRIES):
        try:
            if response_model is not None:
                from llm.structured import call_structured
                result, usage = call_structured(
                    raw_client=client,
                    provider=provider, model=model,
                    system=system, user=user,
                    response_model=response_model,
                    temperature=temperature, max_tokens=max_tokens,
                    call_type=call_type,
                )
                return result, usage

            if provider == "anthropic":
                raw, usage = _call_anthropic(client, model, system, user, temperature, max_tokens)
            else:
                raw, usage = _call_openai_compatible(client, model, system, user, temperature, max_tokens)

            if json_mode:
                return _parse_json(raw), usage
            return raw, usage

        except Exception as exc:
            error_str   = str(exc).lower()
            is_rate_lim = "rate limit" in error_str or "429" in error_str

            if is_rate_lim and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited — backing off",
                    provider=provider, model=model,
                    attempt=attempt + 1, wait_s=wait,
                )
                time.sleep(wait)
                continue

            logger.error(
                "Provider call failed",
                provider=provider, model=model,
                attempt=attempt + 1, error=str(exc),
            )
            raise LLMError(str(exc), provider=provider, model=model) from exc

    raise LLMError("Retry loop exited without result", provider=provider, model=model)


# ---------------------------------------------------------------------------
# Provider-specific raw callers
# ---------------------------------------------------------------------------

def _call_openai_compatible(client, model, system, user, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    text  = response.choices[0].message.content
    usage = _usage_from_openai_compatible_response(response)
    return text, usage


def _call_anthropic(client, model, system, user, temperature, max_tokens):
    if CACHE_ENABLED:
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_block = system

    kwargs: dict[str, Any] = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      system_block,
        "messages":    [{"role": "user", "content": user}],
    }
    response = client.messages.create(**kwargs)
    text     = response.content[0].text
    usage    = _usage_from_anthropic_response(response)
    return text, usage


def _usage_from_openai_compatible_response(response) -> dict | None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if pt is None or ct is None:
            return None
        return {
            "prompt_tokens":     pt,
            "completion_tokens": ct,
            "total_tokens":      tt if tt is not None else pt + ct,
        }
    except Exception:
        return None


def _usage_from_anthropic_response(response) -> dict | None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        it = getattr(usage, "input_tokens",  None)
        ot = getattr(usage, "output_tokens", None)
        if it is None or ot is None:
            return None
        return {
            "prompt_tokens":     it,
            "completion_tokens": ot,
            "total_tokens":      it + ot,
        }
    except Exception:
        return None


def _stringify_result(result: str | dict | BaseModel) -> str:
    try:
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, dict):
            import json
            return json.dumps(result)
        return str(result)
    except Exception:
        return "<unstringifiable result>"


def _reconstruct_result(
    response_text: str,
    response_model: type[BaseModel] | None,
    json_mode: bool,
) -> str | dict | BaseModel:
    if response_model is not None:
        import json
        try:
            data = json.loads(response_text)
            return response_model.model_validate(data)
        except Exception as exc:
            logger.warning(
                "Cached response failed to validate against response_model — "
                "returning raw parsed data instead",
                response_model=response_model.__name__, error=str(exc),
            )
            try:
                return json.loads(response_text)
            except Exception:
                return response_text

    if json_mode:
        return _parse_json(response_text)

    return response_text


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _call_stream(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    call_type: str = "stream",
    provider: str = None,
    model: str = None,
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
    team_id: str | None = None,
) -> Generator:
    if provider and model:
        chain = build_override_chain(provider, model)
    else:
        chain = get_fallback_chain()

    last_exc: Exception = None

    for (prov, mdl) in chain:
        try:
            client = build_client(prov)
            _trace = tracer.trace(
                call_type=call_type, provider=prov, model=mdl,
                system=system, user=user,
                user_id=user_id, document_id=document_id, session_id=session_id,
                is_override=bool(provider and model), is_stream=True,
            ) if TRACING_ENABLED else _noop_trace()
            with _trace as t:
                if prov == "anthropic":
                    gen = _stream_anthropic(client, mdl, system, user, temperature)
                else:
                    gen = _stream_openai_compatible(client, mdl, system, user, temperature)
                if TRACING_ENABLED:
                    t.set_result(response_text="<streamed>", usage=None)
            return gen
        except Exception as exc:
            last_exc = exc
            logger.warning("Stream provider failed — trying next",
                           provider=prov, model=mdl, error=str(exc))
            continue

    raise LLMFallbackExhaustedError([(p, m) for p, m in chain]) from last_exc


def _stream_openai_compatible(client, model, system, user, temperature) -> Generator:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=1000,
        stream=True,
    )
    def generator():
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    return generator()


def _stream_anthropic(client, model, system, user, temperature) -> Generator:
    kwargs: dict[str, Any] = {
        "model":       model,
        "max_tokens":  1000,
        "temperature": temperature,
        "system":      system,
        "messages":    [{"role": "user", "content": user}],
    }
    def generator():
        with client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield text
    return generator()


# ---------------------------------------------------------------------------
# Vision LLM
# ---------------------------------------------------------------------------

def call_vision_llm(
    image_path: str,
    prompt: str,
    call_type: str = "vision",
    user_id: str = "system",
    document_id: str | None = None,
    org_id: str | None = None,
    team_id: str | None = None,
) -> str:
    """
    Call vision LLM to describe an image.
    Returns empty string if no vision model configured.
    """
    vision_provider = app_config.vision_provider
    vision_model    = app_config.vision_model

    if not vision_provider or not vision_model:
        logger.debug("No vision model configured — skipping description")
        return ""

    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else ""
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
            "tiff": "image/tiff", "gif": "image/gif",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        if CACHE_ENABLED:
            cache_row = llm_cache.lookup_vision(user_id, vision_provider, vision_model, image_data, prompt)
            if cache_row is not None:
                _trace = tracer.trace(
                    call_type=call_type, provider=vision_provider, model=vision_model,
                    system="<vision>", user=prompt,
                    user_id=user_id, document_id=document_id,
                    is_override=False, is_stream=False,
                ) if TRACING_ENABLED else _noop_trace()
                with _trace as t:
                    if TRACING_ENABLED:
                        t.set_cache_hit(cache_row["response_text"])
                return cache_row["response_text"]

        _trace = tracer.trace(
            call_type=call_type, provider=vision_provider, model=vision_model,
            system="<vision>", user=prompt,
            user_id=user_id, document_id=document_id,
            is_override=False, is_stream=False,
        ) if TRACING_ENABLED else _noop_trace()
        with _trace as t:
            if vision_provider == "openai":
                client   = OpenAI(api_key=app_config.openai_api_key)
                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }},
                    ]}],
                    max_tokens=500,
                )
                description = response.choices[0].message.content
                usage       = _usage_from_openai_compatible_response(response)

            elif vision_provider == "anthropic":
                client   = anthropic.Anthropic(api_key=app_config.anthropic_api_key)
                response = client.messages.create(
                    model=vision_model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_data,
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                description = response.content[0].text
                usage       = _usage_from_anthropic_response(response)

            else:
                logger.warning("Vision provider not supported", provider=vision_provider)
                if TRACING_ENABLED:
                    t.set_result(response_text="", usage=None)
                return ""

            if TRACING_ENABLED:
                t.set_result(response_text=description, usage=usage)

            if CACHE_ENABLED:
                cost = None
                try:
                    cost = tracer.estimate_cost_usd(vision_provider, vision_model, usage)
                except Exception:
                    pass
                llm_cache.store_vision(
                    user_id=user_id, provider=vision_provider, model=vision_model,
                    image_data_b64=image_data, prompt=prompt,
                    response_text=description, document_id=document_id,
                    original_cost_usd=cost,
                )

            return description

    except Exception as exc:
        logger.error("Vision LLM call failed", error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    import json
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception:
        return {"error": "Could not parse JSON", "raw": text}


class _NoopTraceHandle:
    def set_result(self, *a, **kw): pass
    def set_error(self, *a, **kw): pass
    def set_cache_hit(self, *a, **kw): pass


class _NoopTraceContext:
    def __enter__(self): return _NoopTraceHandle()
    def __exit__(self, *a): return False


def _noop_trace() -> _NoopTraceContext:
    return _NoopTraceContext()