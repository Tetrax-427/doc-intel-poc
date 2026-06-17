from __future__ import annotations

import base64
import time
from typing import Any, Generator, TYPE_CHECKING

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
from llm.usage import log_usage

logger = get_logger("llm.engine")

# ---------------------------------------------------------------------------
# Module-level provider/model — exported for system.py /health endpoint.
# Reflect the first entry in the resolved chain (or primary config values).
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = app_config.llm_provider
LLM_MODEL:    str = app_config.llm_model

# Retry config per provider attempt
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # seconds base; exponential backoff on rate limits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    *,
    system: str = None,
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
) -> str | dict | BaseModel | Generator:
    """
    Central LLM caller. All LLM calls in the app go through here.

    Args:
        prompt:         User message / full prompt.
        system:         Optional system prompt.
        temperature:    0.0 for deterministic, higher for creative.
        max_tokens:     Max response tokens.
        json_mode:      If True and response_model is None, parses response
                        as JSON dict (legacy path — prefer response_model).
        call_type:      Label for usage tracking.
        stream:         If True, returns a token generator (no fallback,
                        no structured output — streaming is best-effort).
        provider:       Override provider for this call only (C3).
                        When set, model must also be set.
        model:          Override model for this call only (C3).
        response_model: Pydantic BaseModel subclass (C1). When set,
                        routes through Instructor and returns a validated
                        model instance. Overrides json_mode.

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
        return _call_stream(prompt, system=system, temperature=temperature,
                            call_type=call_type, provider=provider, model=model)

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
    messages = _build_messages(prompt, system)

    for (prov, mdl) in chain:
        providers_tried.append((prov, mdl))
        try:
            result = _call_single_provider(
                provider=prov,
                model=mdl,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                call_type=call_type,
                response_model=response_model,
                json_mode=json_mode,
                prompt=prompt,
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
    prompt: str,
    system: str = None,
    temperature: float = 0.2,
    call_type: str = "stream",
) -> Generator:
    """Convenience wrapper for streaming calls."""
    return call_llm(prompt, system=system, temperature=temperature,
                    call_type=call_type, stream=True)


# ---------------------------------------------------------------------------
# Single-provider call (one entry in the chain)
# ---------------------------------------------------------------------------

def _call_single_provider(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    call_type: str,
    response_model: type[BaseModel] | None,
    json_mode: bool,
    prompt: str,
) -> str | dict | BaseModel:
    """
    Attempt one provider with up to MAX_RETRIES on rate-limit errors.
    Returns the parsed result on success.
    Raises on non-retryable errors or after retries exhausted.
    """
    client = build_client(provider)

    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()

            # --- C1: Instructor structured output ---
            if response_model is not None:
                from llm.structured import call_structured
                result = call_structured(
                    raw_client=client,
                    provider=provider,
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    temperature=temperature,
                    call_type=call_type,
                )
                latency = time.time() - start
                _log(call_type, model, prompt, str(result), latency)
                return result

            # --- Standard text call ---
            if provider == "anthropic":
                raw = _call_anthropic(client, model, messages, temperature, max_tokens)
            else:
                raw = _call_openai_compatible(client, model, messages, temperature, max_tokens)

            latency = time.time() - start
            _log(call_type, model, prompt, raw, latency)

            if json_mode:
                return _parse_json(raw)
            return raw

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
# Provider-specific raw callers
# ---------------------------------------------------------------------------

def _call_openai_compatible(
    client: Groq | OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content


def _call_anthropic(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    # Anthropic takes system as a top-level param, not in messages
    system_msg = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    kwargs: dict[str, Any] = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages":    user_messages,
    }
    if system_msg:
        kwargs["system"] = system_msg

    response = client.messages.create(**kwargs)
    return response.content[0].text


# ---------------------------------------------------------------------------
# Streaming (best-effort, first available provider)
# ---------------------------------------------------------------------------

def _call_stream(
    prompt: str,
    *,
    system: str = None,
    temperature: float = 0.2,
    call_type: str = "stream",
    provider: str = None,
    model: str = None,
) -> Generator:
    """
    Return a token generator using the first available provider.
    Falls back through the chain to find one that works, then streams.
    No structured output support in stream mode.
    """
    if provider and model:
        chain = build_override_chain(provider, model)
    else:
        chain = get_fallback_chain()

    messages = _build_messages(prompt, system)
    last_exc: Exception = None

    for (prov, mdl) in chain:
        try:
            client = build_client(prov)
            if prov == "anthropic":
                return _stream_anthropic(client, mdl, messages, temperature)
            else:
                return _stream_openai_compatible(client, mdl, messages, temperature)
        except Exception as exc:
            last_exc = exc
            logger.warning("Stream provider failed — trying next",
                           provider=prov, model=mdl, error=str(exc))
            continue

    raise LLMFallbackExhaustedError([(p, m) for p, m in chain]) from last_exc


def _stream_openai_compatible(client, model, messages, temperature) -> Generator:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
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


def _stream_anthropic(client, model, messages, temperature) -> Generator:
    system_msg = None
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            user_messages.append(m)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 1000,
        "temperature": temperature,
        "messages": user_messages,
    }
    if system_msg:
        kwargs["system"] = system_msg

    def generator():
        with client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                yield text
    return generator()


# ---------------------------------------------------------------------------
# Vision LLM (unchanged logic, uses config directly)
# ---------------------------------------------------------------------------

def call_vision_llm(
    image_path: str,
    prompt: str,
    call_type: str = "vision",
) -> str:
    """
    Call vision LLM to describe an image.
    Returns empty string if no vision model configured.
    Uses VISION_PROVIDER / VISION_MODEL from config (not the fallback chain).
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

        start = time.time()

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

        else:
            logger.warning("Vision provider not supported", provider=vision_provider)
            return ""

        latency = time.time() - start
        _log(call_type, vision_model, prompt, description, latency)
        return description

    except Exception as exc:
        logger.error("Vision LLM call failed", error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(prompt: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _log(call_type: str, model: str, prompt: str, response: str, latency: float):
    try:
        log_usage(
            call_type=call_type,
            model=model,
            prompt_len=len(prompt),
            response_len=len(response),
            latency=latency,
        )
    except Exception:
        pass  # never break a call over logging


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