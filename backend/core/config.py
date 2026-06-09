import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    llm_provider: str
    llm_model: str
    groq_api_key: str
    openai_api_key: str
    anthropic_api_key: str

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

    return Config(
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
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