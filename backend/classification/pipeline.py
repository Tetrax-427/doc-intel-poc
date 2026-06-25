"""
E1 — Two-stage classification pipeline.

Stage 1: keyword + embedding (fast, no LLM call).
Stage 2: existing LLM classifier (only when Stage 1 confidence < threshold).

Feature flags (via core/config.py → env vars):
  CLASSIFIER_STAGE1_ENABLED          true | false  (default: true)
  CLASSIFIER_CONFIDENCE_THRESHOLD    float 0–1     (default: 0.75)

The output shape is a superset of the old classify_document() shape — all
existing keys (doc_type, schema_template, confidence, reasoning,
key_signals, requires_human_review) are preserved, plus one new additive
key: stage_used ("stage1" | "stage2").

Callers that only read doc_type / confidence continue to work unchanged.
"""

from __future__ import annotations

from core.logger import get_logger
from classification.stage1 import classify_stage1

logger = get_logger("classifier.pipeline")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify(
    full_text: str,
    get_embedding_fn,
    document_id: str | None = None,
) -> dict:
    """
    Two-stage classification pipeline.

    Args:
        full_text:        Raw document text.
        get_embedding_fn: Embedding callable (same model as chunks).
        document_id:      Optional — used only for logging.

    Returns the standard classification dict plus ``stage_used``:
        {
            "doc_type":              str,
            "schema_template":       str,
            "confidence":            float,
            "reasoning":             str,
            "key_signals":           list,
            "requires_human_review": bool,
            "stage_used":            "stage1" | "stage2",
        }
    """
    from core.config import config as app_config

    if not app_config.classifier_stage1_enabled:
        # Feature flag off → go straight to LLM (original behaviour)
        result = _call_stage2(full_text, document_id)
        result["stage_used"] = "stage2"
        return result

    # ── Stage 1 ──────────────────────────────────────────────────────────
    stage1 = classify_stage1(full_text, get_embedding_fn)

    if stage1["confidence"] >= app_config.classifier_confidence_threshold:
        logger.info(
            "Classification resolved at Stage 1",
            doc_type=stage1["doc_type"],
            confidence=stage1["confidence"],
            document_id=document_id,
        )
        return _stage1_to_full_result(stage1, app_config.classifier_confidence_threshold)

    # ── Stage 2 (LLM) ────────────────────────────────────────────────────
    logger.info(
        "Stage 1 confidence below threshold — escalating to LLM",
        stage1_doc_type=stage1["doc_type"],
        stage1_confidence=stage1["confidence"],
        threshold=app_config.classifier_confidence_threshold,
        document_id=document_id,
    )
    result = _call_stage2(full_text, document_id, stage1_hint=stage1["doc_type"])
    result["stage_used"] = "stage2"
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stage1_to_full_result(stage1: dict, threshold: float) -> dict:
    """
    Convert a Stage 1 result dict into the full classification shape
    that callers (retrieval.py, ingestion.py) expect.
    """
    from schemas.templates import get_template_for_doc_type

    doc_type = stage1["doc_type"]
    confidence = stage1["confidence"]

    return {
        "doc_type":              doc_type,
        "schema_template":       get_template_for_doc_type(doc_type),
        "confidence":            confidence,
        "reasoning":             f"Keyword/embedding match (confidence {confidence})",
        "key_signals":           [],
        "requires_human_review": confidence < threshold,
        "stage_used":            "stage1",
    }


def _call_stage2(
    full_text: str,
    document_id: str | None,
    stage1_hint: str | None = None,
) -> dict:
    """
    Call the existing LLM-based classifier (Stage 2).
    Imported here (not at module top) to avoid circular imports and to keep
    the import lazy — Stage 2 is only imported when actually needed.

    stage1_hint: if set and not "general", added to the prompt as a soft
                 nudge.  The LLM can still override it freely.
    """
    # Reuse the existing _classify_from_context from retrieval.py
    # This ensures Stage 2 uses exactly the same LLM call + cache logic
    # that existed before E1 — zero regression risk.
    from retrieval import _classify_from_context

    hint_suffix = ""
    if stage1_hint and stage1_hint != "general":
        hint_suffix = (
            f"\n\n(Preliminary keyword analysis suggests this may be a "
            f"'{stage1_hint}' — use your own judgment.)"
        )

    context = full_text[:2000] + hint_suffix
    return _classify_from_context(context, document_id=document_id or "")