import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Default hierarchical chunking doc types
# ---------------------------------------------------------------------------
_DEFAULT_HIERARCHICAL_DOC_TYPES = [
    "contract", "agreement", "nda", "loan_application",
    "legal_document", "court_filing", "research_paper", "report",
]


@dataclass
class Config:
    # LLM — primary provider
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

    # Parsers (optional)
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
    compression_threshold: int
    vision_min_words: int
    classification_confidence_threshold: float

    # Retrieval
    retrieval_candidate_pool: int
    retrieval_top_n: int

    # Hierarchical chunking
    hierarchical_chunking_doc_types: list[str]
    hierarchical_parent_chunk_size: int
    hierarchical_child_chunk_size: int
    hierarchical_expand_to_parent: bool

    # Two-stage classifier
    classifier_stage1_enabled: bool
    classifier_confidence_threshold: float

    # API key rotation
    api_key_rotation_grace_period_seconds: int

    # CORS
    cors_allowed_origins: str

    # ── Security Foundation ────────────────────────────────────────────────

    # Developer API key — required at startup, used to protect POST /admin/orgs
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    # Set in Railway env vars as DEVELOPER_API_KEY
    developer_api_key: str

    # Email verification enforcement
    # When True: users must verify email before any action is allowed
    # When False (dev/staging): email check is skipped
    require_email_verification: bool

    # Rate limiting (in-memory, per-process)
    # NOTE: resets on process restart; not shared across Railway instances.
    # Redis-backed rate limiting is a known gap for multi-instance deployments.
    rate_limit_login_per_minute: int       # login + signup endpoint
    rate_limit_upload_per_minute: int      # upload endpoint
    rate_limit_query_per_minute: int       # query endpoint

    # File validation
    # max_file_size_mb already exists above — reused here
    allowed_upload_extensions: list[str]   # e.g. [".pdf", ".docx", ...]

    # ── Usage & Quota Defaults ─────────────────────────────────────────────
    # Applied when no explicit quota is set for the user/team/org.
    # Org admins can override these per-user/team/org via POST /usage/quotas.

    default_max_documents: int          # 500
    default_max_uploads_per_day: int    # 20
    default_max_llm_cost_month: float   # $5.00
    default_max_queries_per_day: int    # 100

    # ---------------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------------

    def get_cors_origins(self) -> list[str]:
        raw     = self.cors_allowed_origins or ""
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins if origins else ["http://localhost:8501"]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_fallback_chain(raw: str, default_provider: str, default_model: str) -> list[str]:
    if not raw or not raw.strip():
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
                f"[config] Skipping malformed LLM_FALLBACK_CHAIN entry '{part}'"
            )
            continue
        entries.append(f"{provider}:{model}")

    if not entries:
        logging.warning(
            f"[config] LLM_FALLBACK_CHAIN produced no valid entries "
            f"— falling back to {default_provider}:{default_model}"
        )
        return [f"{default_provider}:{default_model}"]

    return entries


def _parse_hierarchical_doc_types(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return list(_DEFAULT_HIERARCHICAL_DOC_TYPES)
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def _validated_pool_and_top_n(pool: int, top_n: int) -> tuple[int, int]:
    if top_n > pool:
        logging.warning(
            f"[config] RETRIEVAL_TOP_N ({top_n}) > RETRIEVAL_CANDIDATE_POOL ({pool}) "
            f"— clamping top_n to {pool}"
        )
        top_n = pool
    return pool, top_n


def _parse_extensions(raw: str) -> list[str]:
    defaults = [
        ".pdf", ".docx", ".txt", ".csv", ".xlsx",
        ".rtf", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tiff",
    ]
    if not raw or not raw.strip():
        return defaults
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


# ---------------------------------------------------------------------------
# Required env vars — server refuses to start if any are missing
# ---------------------------------------------------------------------------

REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_KEY",
    "DEVELOPER_API_KEY",
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config() -> Config:
    # Hard stop if required vars are missing
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k, "").strip()]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}. "
            f"Server cannot start. Check your .env / Railway env vars."
        )

    optional_with_warnings = {
        "LLAMA_CLOUD_API_KEY": "LlamaParse unavailable — will fall back to pypdf",
        "COHERE_API_KEY":      "Reranking unavailable — search quality may be lower",
        "GROQ_API_KEY":        "Groq unavailable — ensure another LLM provider key is set",
    }
    for key, msg in optional_with_warnings.items():
        if not os.getenv(key):
            logging.warning(f"[config] {key} not set — {msg}")

    default_provider = os.getenv("LLM_PROVIDER", "groq")
    default_model    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    raw_pool  = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "500"))
    raw_top_n = int(os.getenv("RETRIEVAL_TOP_N", "5"))
    pool, top_n = _validated_pool_and_top_n(raw_pool, raw_top_n)

    stage1_enabled = os.getenv("CLASSIFIER_STAGE1_ENABLED", "true").lower() == "true"
    clf_threshold  = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.75"))
    grace_period   = int(os.getenv("API_KEY_ROTATION_GRACE_PERIOD_SECONDS", "86400"))

    streamlit_url = os.getenv("STREAMLIT_URL", "").strip()
    cors_raw      = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if not cors_raw and streamlit_url:
        cors_raw = f"{streamlit_url},http://localhost:8501,http://127.0.0.1:8501"
        logging.info("[config] CORS_ALLOWED_ORIGINS not set — built from STREAMLIT_URL")

    # Security
    developer_api_key          = os.getenv("DEVELOPER_API_KEY", "")
    require_email_verification = os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"

    # Rate limits
    rate_limit_login_per_minute  = int(os.getenv("RATE_LIMIT_LOGIN_PER_MINUTE",  "100"))
    rate_limit_upload_per_minute = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MINUTE", "200"))
    rate_limit_query_per_minute  = int(os.getenv("RATE_LIMIT_QUERY_PER_MINUTE",  "6000"))

    # Quota defaults
    default_max_documents      = int(os.getenv("DEFAULT_MAX_DOCUMENTS",       "5000"))
    default_max_uploads_per_day= int(os.getenv("DEFAULT_MAX_UPLOADS_PER_DAY", "2000"))
    default_max_llm_cost_month = float(os.getenv("DEFAULT_MAX_LLM_COST_MONTH","5000.00"))
    default_max_queries_per_day= int(os.getenv("DEFAULT_MAX_QUERIES_PER_DAY", "10000"))

    logging.info(f"[config] Retrieval — pool={pool}, top_n={top_n}")
    logging.info(f"[config] E1 Stage1 — enabled={stage1_enabled}, threshold={clf_threshold}")
    logging.info(f"[config] CORS — {cors_raw or '(dev wildcard)'}")
    logging.info(f"[config] Rate limits — login={rate_limit_login_per_minute}/min, "
                 f"upload={rate_limit_upload_per_minute}/min, "
                 f"query={rate_limit_query_per_minute}/min")
    logging.info(f"[config] Quota defaults — docs={default_max_documents}, "
                 f"uploads/day={default_max_uploads_per_day}, "
                 f"cost/month=${default_max_llm_cost_month}, "
                 f"queries/day={default_max_queries_per_day}")

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
        retrieval_candidate_pool=pool,
        retrieval_top_n=top_n,
        hierarchical_chunking_doc_types=_parse_hierarchical_doc_types(
            os.getenv("HIERARCHICAL_CHUNKING_DOC_TYPES", "")
        ),
        hierarchical_parent_chunk_size=int(os.getenv("HIERARCHICAL_PARENT_CHUNK_SIZE", "2000")),
        hierarchical_child_chunk_size=int(os.getenv("HIERARCHICAL_CHILD_CHUNK_SIZE", "400")),
        hierarchical_expand_to_parent=os.getenv(
            "HIERARCHICAL_EXPAND_TO_PARENT", "true"
        ).lower() == "true",
        classifier_stage1_enabled=stage1_enabled,
        classifier_confidence_threshold=clf_threshold,
        api_key_rotation_grace_period_seconds=grace_period,
        cors_allowed_origins=cors_raw,
        # Security
        developer_api_key=developer_api_key,
        require_email_verification=require_email_verification,
        rate_limit_login_per_minute=rate_limit_login_per_minute,
        rate_limit_upload_per_minute=rate_limit_upload_per_minute,
        rate_limit_query_per_minute=rate_limit_query_per_minute,
        allowed_upload_extensions=_parse_extensions(
            os.getenv("ALLOWED_UPLOAD_EXTENSIONS", "")
        ),
        # Quota defaults
        default_max_documents=default_max_documents,
        default_max_uploads_per_day=default_max_uploads_per_day,
        default_max_llm_cost_month=default_max_llm_cost_month,
        default_max_queries_per_day=default_max_queries_per_day,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uses_hierarchical_chunking(doc_type: str) -> bool:
    return doc_type.lower().strip() in config.hierarchical_chunking_doc_types


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: "Config | None" = None


def _get_config() -> "Config":
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


class _ConfigProxy:
    def __getattr__(self, name: str):
        return getattr(_get_config(), name)

    def __repr__(self) -> str:
        return repr(_get_config())


config = _ConfigProxy()