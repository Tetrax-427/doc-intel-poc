import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM — primary provider (used as first entry if fallback chain is empty)
    llm_provider: str
    llm_model: str
    groq_api_key: str
    openai_api_key: str
    anthropic_api_key: str

    # LLM fallback chain — ordered list of "provider:model" strings.
    # Example: ["groq:llama-3.3-70b-versatile", "openai:gpt-4o-mini", "anthropic:claude-3-5-haiku-20241022"]
    # call_llm() walks this list in order; first success wins.
    # If empty, falls back to a single-entry chain built from llm_provider + llm_model.
    llm_fallback_chain: list[str]

    # Vision (optional)
    vision_provider: str
    vision_model: str

    # Parsers (optional — features degrade gracefully if missing)
    llama_cloud_api_key: str
    cohere_api_key: str

    # Supabase
    supabase_url: str
    supabase_key: str

    # Processing
    upload_dir: str
    max_file_size_mb: int
    chunk_size: int
    chunk_overlap: int
    compression_threshold: int       # messages before history compression
    vision_min_words: int            # pages with fewer words trigger vision
    classification_confidence_threshold: float  # below this = flag for review


def _parse_fallback_chain(raw: str, default_provider: str, default_model: str) -> list[str]:
    """
    Parse LLM_FALLBACK_CHAIN env var into a list of "provider:model" strings.

    Expected format (comma-separated):
        groq:llama-3.3-70b-versatile,openai:gpt-4o-mini,anthropic:claude-3-5-haiku-20241022

    Rules:
    - Whitespace around entries is stripped.
    - Empty entries are skipped.
    - If the env var is absent or blank, a single-entry chain is built from
      LLM_PROVIDER + LLM_MODEL so existing single-provider deployments work
      without any .env changes.
    - Invalid entries (no colon, empty provider/model) are skipped with a warning.
    """
    if not raw or not raw.strip():
        # No chain configured — build a single-entry chain from primary provider
        return [f"{default_provider}:{default_model}"]

    entries = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            logging.warning(
                f"[config] Skipping invalid LLM_FALLBACK_CHAIN entry '{part}' "
                f"— expected format 'provider:model'"
            )
            continue
        provider, model = part.split(":", 1)
        provider, model = provider.strip(), model.strip()
        if not provider or not model:
            logging.warning(
                f"[config] Skipping malformed LLM_FALLBACK_CHAIN entry '{part}' "
                f"— provider or model is empty"
            )
            continue
        entries.append(f"{provider}:{model}")

    if not entries:
        logging.warning(
            "[config] LLM_FALLBACK_CHAIN was set but produced no valid entries "
            f"— falling back to {default_provider}:{default_model}"
        )
        return [f"{default_provider}:{default_model}"]

    return entries


def load_config() -> Config:
    """
    Load all config from environment.
    Raises ValueError on missing required keys.
    Logs warnings for missing optional keys.
    """
    required = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(
            f"Missing required config keys: {missing}. Check your .env file."
        )

    # Warn on missing optional keys that degrade functionality
    optional_with_warnings = {
        "LLAMA_CLOUD_API_KEY": "LlamaParse unavailable — will fall back to pypdf for PDFs",
        "COHERE_API_KEY": "Reranking unavailable — search quality may be lower",
        "GROQ_API_KEY": "Groq unavailable — ensure another LLM provider key is set",
    }
    for key, warning_msg in optional_with_warnings.items():
        if not os.getenv(key):
            logging.warning(f"[config] {key} not set — {warning_msg}")

    default_provider = os.getenv("LLM_PROVIDER", "groq")
    default_model    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    return Config(
        llm_provider=default_provider,
        llm_model=default_model,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        llm_fallback_chain=_parse_fallback_chain(
            os.getenv("LLM_FALLBACK_CHAIN", ""),
            default_provider,
            default_model,
        ),
        vision_provider=os.getenv("VISION_PROVIDER", ""),
        vision_model=os.getenv("VISION_MODEL", ""),
        llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY"),
        upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        compression_threshold=int(os.getenv("COMPRESSION_THRESHOLD", "10")),
        vision_min_words=int(os.getenv("VISION_MIN_WORDS", "50")),
        classification_confidence_threshold=float(
            os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.7")
        ),
    )


# Singleton — import this everywhere instead of reading os.getenv() directly.
# Lazy: only loads on first access so tests can set env vars beforehand.
_config_instance: "Config | None" = None


def _get_config() -> "Config":
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


class _ConfigProxy:
    """Proxy that forwards attribute access to the lazily-loaded Config singleton."""

    def __getattr__(self, name: str):
        return getattr(_get_config(), name)

    def __repr__(self) -> str:
        return repr(_get_config())


config = _ConfigProxy()