# backend/vision/triggers.py

import os
from core.config import Config
from core.logger import get_logger

logger = get_logger("vision.triggers")

# File extensions treated as pure image files — always use vision
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}

# Doc types where visual content is the primary signal
# (text extraction alone misses critical information)
VISUAL_DOC_TYPES = {"construction_photo", "id_document"}


def should_use_vision(
    file_path: str,
    page_text: str,
    is_scanned: bool,
    doc_type: str,
    config: Config
) -> bool:
    """
    Decide whether to call the vision model for a given page/file.

    Vision calls are expensive — this function ensures we only trigger
    vision when it genuinely adds value over plain text extraction.

    Priority order:
      1. Vision provider not configured → always False
      2. File is a pure image → always True
      3. PDF was detected as scanned (no text layer) → always True
      4. Doc type is inherently visual → always True
      5. Page has very few words (text extraction likely missed content) → True
      6. Otherwise → False (text extraction is sufficient)

    Args:
        file_path:   Absolute path to the source file.
        page_text:   Text already extracted from this page (may be empty).
        is_scanned:  True if AutoRouter flagged this PDF as scanned.
        doc_type:    Classified doc type string (e.g. "invoice", "id_document").
        config:      App config (must have vision_provider, vision_model, vision_min_words).

    Returns:
        True if vision should run, False to skip.
    """
    # Gate 1 — vision not configured at all
    if not config.vision_provider or not config.vision_model:
        return False

    ext = os.path.splitext(file_path)[1].lower()

    # Gate 2 — pure image file (no text layer possible)
    if ext in IMAGE_EXTENSIONS:
        logger.debug("Vision triggered — image file", file=os.path.basename(file_path))
        return True

    # Gate 3 — scanned PDF (OCR text is unreliable / empty)
    if is_scanned:
        logger.debug("Vision triggered — scanned PDF", file=os.path.basename(file_path))
        return True

    # Gate 4 — doc type where visual elements carry primary information
    if doc_type in VISUAL_DOC_TYPES:
        logger.debug("Vision triggered — visual doc type", doc_type=doc_type)
        return True

    # Gate 5 — too few words on page; text extraction likely incomplete
    word_count = len(page_text.split()) if page_text else 0
    if word_count < config.vision_min_words:
        logger.debug(
            "Vision triggered — low word count",
            words=word_count,
            threshold=config.vision_min_words
        )
        return True

    return False