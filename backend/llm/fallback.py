"""
llm/fallback.py — Fallback chain resolution and provider client factory.

Responsibilities:
- Parse config.llm_fallback_chain into (provider, model) tuples
- Build raw provider clients (Groq, OpenAI, Anthropic) on demand
- Validate API keys at chain-build time with clear errors

Nothing in this module makes LLM calls — it only builds clients and
resolves the chain. call_llm() in engine.py owns the retry/fallback loop.
"""

from __future__ import annotations

from groq import Groq
from openai import OpenAI
import anthropic

from core.config import config as app_config
from core.errors import LLMConfigError
from core.logger import get_logger

logger = get_logger("llm.fallback")

# Providers we know how to build clients for.
SUPPORTED_PROVIDERS = {"groq", "openai", "anthropic"}


# ---------------------------------------------------------------------------
# Chain resolution
# ---------------------------------------------------------------------------

def get_fallback_chain() -> list[tuple[str, str]]:
    """
    Return the ordered fallback chain as a list of (provider, model) tuples.

    Reads config.llm_fallback_chain (already parsed from LLM_FALLBACK_CHAIN
    env var by config.py). Each entry is a "provider:model" string.

    Validates:
    - Provider name is in SUPPORTED_PROVIDERS
    - API key for that provider is non-empty in config

    Invalid entries are skipped with a warning so a single bad entry doesn't
    take down the whole chain. If the result is empty, raises LLMConfigError
    so the caller fails fast at startup rather than at the first LLM call.

    Returns:
        [("groq", "llama-3.3-70b-versatile"), ("openai", "gpt-4o-mini"), ...]
    """
    raw_chain: list[str] = app_config.llm_fallback_chain
    resolved: list[tuple[str, str]] = []

    for entry in raw_chain:
        if ":" not in entry:
            logger.warning(
                "Skipping malformed fallback chain entry — no colon separator",
                entry=entry,
            )
            continue

        provider, model = entry.split(":", 1)
        provider, model = provider.strip().lower(), model.strip()

        if provider not in SUPPORTED_PROVIDERS:
            logger.warning(
                "Skipping unsupported provider in fallback chain",
                provider=provider,
                supported=", ".join(sorted(SUPPORTED_PROVIDERS)),
            )
            continue

        api_key = _get_api_key(provider)
        if not api_key:
            logger.warning(
                "Skipping provider — API key not configured",
                provider=provider,
            )
            continue

        resolved.append((provider, model))

    if not resolved:
        raise LLMConfigError(
            "LLM fallback chain is empty — no providers are configured with valid API keys. "
            "Set LLM_FALLBACK_CHAIN in .env (e.g. groq:llama-3.3-70b-versatile) and ensure "
            "the corresponding API key is present.",
        )

    logger.info(
        "Fallback chain resolved",
        chain=", ".join(f"{p}:{m}" for p, m in resolved),
    )
    return resolved


# ---------------------------------------------------------------------------
# API key lookup
# ---------------------------------------------------------------------------

def _get_api_key(provider: str) -> str:
    """Return the API key for a provider from config. Empty string if missing."""
    key_map = {
        "groq":      app_config.groq_api_key,
        "openai":    app_config.openai_api_key,
        "anthropic": app_config.anthropic_api_key,
    }
    return key_map.get(provider, "")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client(provider: str) -> Groq | OpenAI | anthropic.Anthropic:
    """
    Build and return a raw provider client for the given provider name.

    Called by engine.py once per provider attempt in the fallback loop.
    Clients are not cached here — engine.py owns caching if needed.

    Raises:
        LLMConfigError: if provider is unsupported or API key is missing.
    """
    provider = provider.strip().lower()

    api_key = _get_api_key(provider)
    if not api_key:
        raise LLMConfigError(
            f"Cannot build client for '{provider}' — API key not set in config.",
            provider=provider,
        )

    if provider == "groq":
        return Groq(api_key=api_key)

    if provider == "openai":
        return OpenAI(api_key=api_key)

    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key)

    raise LLMConfigError(
        f"Unsupported provider '{provider}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Override chain helper (C3)
# ---------------------------------------------------------------------------

def build_override_chain(provider: str, model: str) -> list[tuple[str, str]]:
    """
    Build a single-entry chain for a per-call provider/model override (C3).

    Validates the provider + key exist before returning so engine.py can
    raise LLMProviderOverrideError immediately rather than inside the loop.

    Raises:
        LLMConfigError: if provider is unsupported or key is missing.
    """
    provider = provider.strip().lower()

    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"Override provider '{provider}' is not supported. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}",
            provider=provider,
        )

    api_key = _get_api_key(provider)
    if not api_key:
        raise LLMConfigError(
            f"Override provider '{provider}' has no API key configured.",
            provider=provider,
        )

    return [(provider, model)]