"""
E1 — Stage 1 classifier.

Combines two fast, LLM-free signals:
  1. Keyword matching against the first 500 chars of document text.
  2. Embedding cosine similarity against per-doc-type exemplar passages.

Returns a confidence score (0–1) based on the spread between the top
doc type and the second-best.  Wide spread = high confidence.

If confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD (config), the caller
skips the LLM Stage 2 entirely.
"""

from __future__ import annotations

import math
from core.logger import get_logger
from classification.exemplars import KEYWORD_SIGNALS, EMBEDDING_EXEMPLARS, get_exemplar_embedding

logger = get_logger("classifier.stage1")


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def keyword_score(text_sample: str) -> dict[str, float]:
    """
    Score each doc type by how many keyword signals appear in the first
    500 characters of the document text (case-insensitive).

    Returns {doc_type: score} where score is 0.0–1.0, normalised by the
    number of signals defined for that type.
    """
    text_lower = text_sample[:500].lower()
    scores: dict[str, float] = {}
    for doc_type, signals in KEYWORD_SIGNALS.items():
        if not signals:
            scores[doc_type] = 0.0
            continue
        hits = sum(1 for s in signals if s in text_lower)
        scores[doc_type] = hits / len(signals)
    return scores


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding scoring
# ---------------------------------------------------------------------------

def embedding_score(text_sample: str, get_embedding_fn) -> dict[str, float]:
    """
    Embed the first 300 chars of the document text and compute cosine
    similarity against each doc type's exemplar embedding.

    Returns {doc_type: similarity_score} for all types in EMBEDDING_EXEMPLARS.

    Args:
        text_sample:      Raw document text (first 300 chars used).
        get_embedding_fn: Callable(text: str) -> list[float].
                          Must be the SAME model used for document chunk
                          embeddings — cross-model similarity is meaningless.
    """
    doc_vec = get_embedding_fn(text_sample[:300])
    scores: dict[str, float] = {}
    for doc_type in EMBEDDING_EXEMPLARS:
        exemplar_vec = get_exemplar_embedding(doc_type, get_embedding_fn)
        scores[doc_type] = cosine_similarity(doc_vec, exemplar_vec)
    return scores


# ---------------------------------------------------------------------------
# Combined Stage 1 classification
# ---------------------------------------------------------------------------

def classify_stage1(
    full_text: str,
    get_embedding_fn,
    keyword_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> dict:
    """
    Stage 1 classification: combine keyword + embedding scores.

    Confidence is spread-based: (best_score − second_score) / 0.3, capped at
    1.0.  This means two nearly-equal top candidates → low confidence even if
    both have high absolute scores.  This correctly escalates ambiguous docs
    to Stage 2.

    Args:
        full_text:        Full document text.
        get_embedding_fn: Embedding callable — same model as chunk embeddings.
        keyword_weight:   Weight for keyword score component (default 0.5).
        embedding_weight: Weight for embedding score component (default 0.5).

    Returns:
        {
            "doc_type":        str,
            "confidence":      float (0–1),
            "stage":           "stage1",
            "keyword_scores":  dict[str, float],
            "embedding_scores": dict[str, float],
        }
    """
    if not full_text or not full_text.strip():
        return {
            "doc_type": "general",
            "confidence": 0.0,
            "stage": "stage1",
            "keyword_scores": {},
            "embedding_scores": {},
        }

    kw_scores  = keyword_score(full_text)
    emb_scores = embedding_score(full_text, get_embedding_fn)

    # Weighted combination
    all_types = set(kw_scores) | set(emb_scores)
    combined: dict[str, float] = {}
    for doc_type in all_types:
        kw  = kw_scores.get(doc_type, 0.0)
        emb = emb_scores.get(doc_type, 0.0)
        combined[doc_type] = keyword_weight * kw + embedding_weight * emb

    if not combined:
        return {
            "doc_type": "general",
            "confidence": 0.0,
            "stage": "stage1",
            "keyword_scores": kw_scores,
            "embedding_scores": emb_scores,
        }

    sorted_types = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score   = sorted_types[0]
    second_score            = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

    # Spread-based confidence — soft cap at spread=0.3 → confidence=1.0
    spread     = best_score - second_score
    confidence = min(spread / 0.3, 1.0) if best_score > 0 else 0.0
    confidence = round(confidence, 3)

    resolved_type = best_type if confidence > 0 else "general"

    logger.info(
        "Stage 1 classification",
        doc_type=resolved_type,
        confidence=confidence,
        best_score=round(best_score, 3),
        spread=round(spread, 3),
    )

    return {
        "doc_type":        resolved_type,
        "confidence":      confidence,
        "stage":           "stage1",
        "keyword_scores":  kw_scores,
        "embedding_scores": emb_scores,
    }