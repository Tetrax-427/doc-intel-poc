"""
backend/presidio_setup.py

Presidio analyzer setup for NER-based PII detection (names, addresses,
natural-language dates) — the entities regex can't reliably catch.

Load once at app startup (call `get_analyzer()` from main.py's startup
tasks, alongside the embedding model warmup) — NOT per-request. Model
load is slow; the analyzer instance is safe to reuse across requests.

Analysis is always restricted to PRESIDIO_ENTITIES from pii_config.py —
never run open-ended. This keeps it fast and stops Presidio re-flagging
spans regex already claimed (PAN/Aadhaar/phone/email/numeric DOB), since
those are stripped from the text before this runs (see masking.py).
"""

import logging

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pii_config import PRESIDIO_ENTITIES

logger = logging.getLogger("pii.presidio")

_analyzer: AnalyzerEngine | None = None


def _build_analyzer() -> AnalyzerEngine:
    """
    Build the AnalyzerEngine using en_core_web_lg (better recall on names
    than the default `sm` model — worth the extra load time/size for this
    use case, since PERSON is the entity we most need reliable coverage on).
    """
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


def get_analyzer() -> AnalyzerEngine:
    """
    Return the process-wide AnalyzerEngine instance, building it on first
    call. Call this once eagerly at startup so the first real request
    doesn't pay the model-load cost.
    """
    global _analyzer
    if _analyzer is None:
        logger.info("Loading Presidio analyzer (en_core_web_lg)...")
        _analyzer = _build_analyzer()
        logger.info("Presidio analyzer ready.")
    return _analyzer


def analyze_text(text: str):
    """
    Run Presidio analysis restricted to PRESIDIO_ENTITIES (from
    pii_config.py). Returns a list of RecognizerResult, each with
    .entity_type, .start, .end, .score.

    On failure (model error, etc.), logs and returns an empty list rather
    than raising — per the "if Presidio fails, keep regex-only results"
    decision, this must never block the masking pipeline.
    """
    if not text:
        return []

    try:
        analyzer = get_analyzer()
        results = analyzer.analyze(
            text=text,
            entities=PRESIDIO_ENTITIES,
            language="en",
        )
        return results
    except Exception:
        logger.exception("Presidio analysis failed — continuing with regex-only results.")
        return []