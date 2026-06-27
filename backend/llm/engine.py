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

# Retry config per provider attempt
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # seconds base; exponential backoff on rate limits

# ---------------------------------------------------------------------------
# Feature flags — read from environment, default both on.
# TODO: migrate these into core/config.py's Config dataclass + load_config()
#       alongside the other env-var-driven settings so they appear in the
#       single source of truth for config. For now they're here to avoid
#       touching config.py in this PR.
# ---------------------------------------------------------------------------
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
    # C3 — per-call override
    provider: str = None,
    model: str = None,
    # C1 — structured output
    response_model: type[BaseModel] = None,
    # Tracing context (FINAL_PLAN.md Phase B) — all optional, default to
    # safe values so callers that haven't been migrated to pass these yet
    # don't break. Every production call site SHOULD pass user_id at minimum.
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
) -> str | dict | BaseModel | Generator:
    """
    Central LLM caller. All LLM calls in the app go through here.

    Args:
        system:         Static instruction text for this call. Required —
                        every call site must separate "what to do" (system)
                        from "what to do it to" (user). See FINAL_PLAN.md §2
                        for the split used at each existing call site.
        user:           Per-call content.
        temperature:    0.0 for deterministic, higher for creative.
        max_tokens:     Max response tokens.
        json_mode:      If True and response_model is None, parses response
                        as JSON dict (legacy path — prefer response_model).
        call_type:      Label for tracing — becomes llm_calls.call_type.
        stream:         If True, returns a token generator (no fallback,
                        no structured output, no caching — streaming is
                        best-effort and excluded from Layer 2 by design).
        provider:       Override provider for this call only (C3).
                        When set, model must also be set.
        model:          Override model for this call only (C3).
        response_model: Pydantic BaseModel subclass (C1). When set,
                        routes through Instructor and returns a validated
                        model instance. Overrides json_mode.
        user_id:        Owning user — required for correct tracing/cache
                        scoping. Defaults to "system" for any not-yet-migrated
                        internal caller; production call sites should always
                        pass the real authenticated user_id.
        document_id:    Document this call relates to, if any. Used for
                        cache invalidation on document delete (Phase F).
        session_id:     Reserved for future chat-session grouping.

    Returns:
        - Generator of str tokens  when stream=True
        - BaseModel instance       when response_model is set
        - dict                     when json_mode=True
        - str                      otherwise

    Raises:
        LLMProviderOverrideError:  override provider/model failed (no fallback).
        LLMFallbackExhaustedError: all chain providers failed.
    """

    # --- Streaming: single-provider best-effort (no fallback, no structured) ---
    if stream:
        return _call_stream(
            system, user, temperature=temperature, call_type=call_type,
            provider=provider, model=model,
            user_id=user_id, document_id=document_id, session_id=session_id,
        )

    # --- Build chain: override (1 entry) or full fallback chain ---
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
    cacheable = CACHE_ENABLED and llm_cache.is_cacheable(call_type)
    primary_provider = None  # set on first iteration, used to compute used_fallback

    for (prov, mdl) in chain:
        if primary_provider is None:
            primary_provider = prov
        providers_tried.append((prov, mdl))
        used_fallback = (prov != primary_provider)

        # --- Layer 2 cache lookup ---
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
                    provider=prov,
                    model=mdl,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    call_type=call_type,
                    response_model=response_model,
                    json_mode=json_mode,
                )
                response_text = _stringify_result(result)
                if TRACING_ENABLED:
                    t.set_result(response_text=response_text, usage=usage)

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
                provider=prov,
                model=mdl,
                error=str(exc),
                is_override=is_override,
            )
            if is_override:
                raise LLMProviderOverrideError(prov, mdl, reason=str(exc)) from exc
            # else: continue to next provider

    raise LLMFallbackExhaustedError(providers_tried)


def call_llm_stream(
    system: str,
    user: str,
    temperature: float = 0.2,
    call_type: str = "stream",
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
) -> Generator:
    """Convenience wrapper for streaming calls."""
    return call_llm(
        system, user, temperature=temperature, call_type=call_type, stream=True,
        user_id=user_id, document_id=document_id, session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Single-provider call (one entry in the chain)
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
    """
    Attempt one provider with up to MAX_RETRIES on rate-limit errors.

    Returns (result, usage) on success. usage is a dict with prompt_tokens/
    completion_tokens/total_tokens, or None if the provider/SDK didn't expose
    it. Raises on non-retryable errors or after retries exhausted.
    """
    client = build_client(provider)

    for attempt in range(MAX_RETRIES):
        try:
            # --- C1: Instructor structured output ---
            if response_model is not None:
                from llm.structured import call_structured
                result, usage = call_structured(
                    raw_client=client,
                    provider=provider,
                    model=model,
                    system=system,
                    user=user,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    call_type=call_type,
                )
                return result, usage

            # --- Standard text call ---
            if provider == "anthropic":
                raw, usage = _call_anthropic(client, model, system, user, temperature, max_tokens)
            else:
                raw, usage = _call_openai_compatible(client, model, system, user, temperature, max_tokens)

            if json_mode:
                return _parse_json(raw), usage
            return raw, usage

        except Exception as exc:
            error_str = str(exc).lower()
            is_rate_limit = "rate limit" in error_str or "429" in error_str

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate limited — backing off",
                    provider=provider, model=model,
                    attempt=attempt + 1, wait_s=wait,
                )
                time.sleep(wait)
                continue

            # Non-retryable or last attempt — propagate to fallback loop
            logger.error(
                "Provider call failed",
                provider=provider, model=model,
                attempt=attempt + 1, error=str(exc),
            )
            raise LLMError(str(exc), provider=provider, model=model) from exc

    # Should never reach here — loop always raises on last attempt
    raise LLMError("Retry loop exited without result", provider=provider, model=model)


# ---------------------------------------------------------------------------
# Provider-specific raw callers — each returns (text, usage_dict_or_None)
# ---------------------------------------------------------------------------

def _call_openai_compatible(
    client: Groq | OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict | None]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    text = response.choices[0].message.content
    usage = _usage_from_openai_compatible_response(response)
    return text, usage


def _call_anthropic(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict | None]:
    """
    Call Anthropic messages API with Layer 1 cache_control on the system block.
    cache_control: {"type": "ephemeral"} tells Anthropic to cache the system
    prompt server-side for ~5 minutes. On repeated calls with the same system
    text (same call type = same instruction), Anthropic applies a ~50% token
    discount on the cached portion automatically.
    Only applied when CACHE_ENABLED=True — when caching is disabled entirely,
    we send a plain system string instead (no cache markers).
    """
    if CACHE_ENABLED:
        system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_block = system  # plain string — Anthropic accepts both forms

    kwargs: dict[str, Any] = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "system":      system_block,
        "messages":    [{"role": "user", "content": user}],
    }
    response = client.messages.create(**kwargs)
    text = response.content[0].text
    usage = _usage_from_anthropic_response(response)
    return text, usage


def _usage_from_openai_compatible_response(response) -> dict | None:
    """Extract usage from an OpenAI/Groq chat.completions response. Defensive."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
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
    except Exception:
        return None


def _usage_from_anthropic_response(response) -> dict | None:
    """Extract usage from an Anthropic messages.create response. Defensive."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None or output_tokens is None:
            return None
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    except Exception:
        return None


def _stringify_result(result: str | dict | BaseModel) -> str:
    """Normalize a call result to a string for trace/cache storage. Defensive."""
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
    """
    Reverse _stringify_result() for a cache hit. A cached response_text is
    always a plain string (that's what's stored), but the ORIGINAL call may
    have returned a BaseModel or dict — callers expect that same shape back,
    not a raw string, or every cache hit on a structured call would silently
    break downstream code expecting e.g. result.doc_type.

    Raises StructuredOutputError-shaped behavior is intentionally avoided
    here: if cached JSON no longer validates against response_model (e.g.
    the model's schema changed since this entry was cached), we log and fall
    back to returning the raw dict rather than raising — a cache-hit code
    path failing validation should degrade, not crash, since the safe
    recovery (skip the cache, call live) is exactly one cache-miss away on
    the caller's next attempt if they retry. For now this returns the raw
    parsed dict; calling code that strictly requires a model instance should
    treat this as a known edge case — see FINAL_PLAN.md open items re:
    schema-version staleness (the PROMPT_VERSION discussion, deliberately
    deferred).
    """
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
# Streaming (best-effort, first available provider)
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
) -> Generator:
    """
    Return a token generator using the first available provider.
    Falls back through the chain to find one that works, then streams.
    No structured output support in stream mode. No caching (Layer 2 never
    applies to streaming calls — see FINAL_PLAN.md §0).

    Tracing note: streaming responses don't expose usage in a way that's
    available before the generator is exhausted by the caller, and we don't
    want to hold a trace open across an arbitrarily long consumer-paced
    stream. So streaming calls are traced with usage=None and a response_text
    of "<streamed>" — latency reflects time-to-first-token-stream-start, not
    total stream duration. This matches both original plan docs' treatment
    of streaming and is a deliberate, documented limitation, not an oversight.
    """
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
            {"role": "user", "content": user},
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
        "model": model,
        "max_tokens": 1000,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
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
) -> str:
    """
    Call vision LLM to describe an image.
    Returns empty string if no vision model configured.
    Uses VISION_PROVIDER / VISION_MODEL from config (not the fallback chain —
    single-provider, no fallback, by design, matching pre-existing behavior).

    Traced separately from call_llm()'s chain loop (it isn't text-shaped and
    has no fallback chain) — see FINAL_PLAN.md §0. Cached separately too,
    via llm.cache.lookup_vision/store_vision — same llm_cache table, but the
    key hashes image bytes + prompt instead of system+user text (there's no
    meaningful system/user split for an image call).
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

        # --- Layer 2 cache lookup for vision ---
        if CACHE_ENABLED:
            cache_row = llm_cache.lookup_vision(user_id, vision_provider, vision_model, image_data, prompt)
            if cache_row is not None:
                _trace = tracer.trace(
                    call_type=call_type, provider=vision_provider, model=vision_model,
                    system="<vision call — see user field for prompt>", user=prompt,
                    user_id=user_id, document_id=document_id,
                    is_override=False, is_stream=False,
                ) if TRACING_ENABLED else _noop_trace()
                with _trace as t:
                    if TRACING_ENABLED:
                        t.set_cache_hit(cache_row["response_text"])
                return cache_row["response_text"]

        _trace = tracer.trace(
            call_type=call_type, provider=vision_provider, model=vision_model,
            system="<vision call — see user field for prompt>", user=prompt,
            user_id=user_id, document_id=document_id,
            is_override=False, is_stream=False,
        ) if TRACING_ENABLED else _noop_trace()
        with _trace as t:
            if vision_provider == "openai":
                client = OpenAI(api_key=app_config.openai_api_key)
                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }},
                        ],
                    }],
                    max_tokens=500,
                )
                description = response.choices[0].message.content
                usage = _usage_from_openai_compatible_response(response)

            elif vision_provider == "anthropic":
                client = anthropic.Anthropic(api_key=app_config.anthropic_api_key)
                response = client.messages.create(
                    model=vision_model,
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                description = response.content[0].text
                usage = _usage_from_anthropic_response(response)

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
    """Clean and parse JSON from LLM response. Legacy path for json_mode=True."""
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
    """Dropped-in replacement for _TraceHandle when TRACING_ENABLED=false."""
    def set_result(self, *a, **kw): pass
    def set_error(self, *a, **kw): pass
    def set_cache_hit(self, *a, **kw): pass


class _NoopTraceContext:
    def __enter__(self): return _NoopTraceHandle()
    def __exit__(self, *a): return False


def _noop_trace() -> _NoopTraceContext:
    """Return a no-op context manager used when TRACING_ENABLED=False."""
    return _NoopTraceContext()