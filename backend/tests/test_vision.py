# tests/test_vision.py

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Helpers — build a minimal Config-like object for tests
# ---------------------------------------------------------------------------

def _make_config(
    vision_provider: str = "",
    vision_model: str = "",
    vision_min_words: int = 50,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
):
    cfg = MagicMock()
    cfg.vision_provider   = vision_provider
    cfg.vision_model      = vision_model
    cfg.vision_min_words  = vision_min_words
    cfg.openai_api_key    = openai_api_key
    cfg.anthropic_api_key = anthropic_api_key
    return cfg


# ---------------------------------------------------------------------------
# test_no_vision_returns_empty
# ---------------------------------------------------------------------------

def test_no_vision_returns_empty():
    """NoVisionModel.describe() always returns empty string, never raises."""
    from vision.null import NoVisionModel

    model = NoVisionModel()
    result = model.describe("/any/path.png", "describe this")

    assert result == "", "NoVisionModel must always return empty string"


def test_no_vision_is_always_available():
    """NoVisionModel.is_available() returns True regardless of config."""
    from vision.null import NoVisionModel

    model = NoVisionModel()
    assert model.is_available(_make_config()) is True
    assert model.is_available(_make_config(vision_provider="openai")) is True


def test_no_vision_never_raises_on_bad_path():
    """NoVisionModel must not raise even if path does not exist."""
    from vision.null import NoVisionModel

    model = NoVisionModel()
    result = model.describe("/nonexistent/file.png", "prompt")
    assert result == ""


# ---------------------------------------------------------------------------
# test_trigger_true_for_image_file
# ---------------------------------------------------------------------------

def test_trigger_true_for_image_file():
    """should_use_vision() returns True for image file extensions."""
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o")

    for ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]:
        result = should_use_vision(
            file_path=f"/uploads/photo{ext}",
            page_text="some text here with enough words to pass threshold",
            is_scanned=False,
            doc_type="general",
            config=cfg,
        )
        assert result is True, f"Expected True for image extension {ext}"


# ---------------------------------------------------------------------------
# test_trigger_false_for_text_pdf
# ---------------------------------------------------------------------------

def test_trigger_false_for_text_pdf():
    """
    should_use_vision() returns False for a text PDF with sufficient
    word count and no special doc_type.
    """
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o", vision_min_words=50)

    # 60-word page text — above the 50-word threshold
    page_text = " ".join(["word"] * 60)

    result = should_use_vision(
        file_path="/uploads/document.pdf",
        page_text=page_text,
        is_scanned=False,
        doc_type="invoice",
        config=cfg,
    )
    assert result is False, "Should not trigger vision for text PDF with enough words"


def test_trigger_false_when_no_provider_configured():
    """should_use_vision() returns False when VISION_PROVIDER is not set."""
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="", vision_model="")

    result = should_use_vision(
        file_path="/uploads/photo.png",
        page_text="",
        is_scanned=True,
        doc_type="id_document",
        config=cfg,
    )
    assert result is False, "Must return False when vision provider not configured"


# ---------------------------------------------------------------------------
# test_trigger_true_for_scanned_pdf
# ---------------------------------------------------------------------------

def test_trigger_true_for_scanned_pdf():
    """should_use_vision() returns True when is_scanned=True."""
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o")
    page_text = " ".join(["word"] * 60)  # plenty of words — scanned flag overrides

    result = should_use_vision(
        file_path="/uploads/scanned.pdf",
        page_text=page_text,
        is_scanned=True,
        doc_type="general",
        config=cfg,
    )
    assert result is True, "Should trigger vision for scanned PDFs"


def test_trigger_true_for_low_word_count():
    """should_use_vision() returns True when page has fewer words than threshold."""
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o", vision_min_words=50)
    page_text = " ".join(["word"] * 10)  # only 10 words — below threshold

    result = should_use_vision(
        file_path="/uploads/sparse.pdf",
        page_text=page_text,
        is_scanned=False,
        doc_type="general",
        config=cfg,
    )
    assert result is True, "Should trigger vision when word count is below threshold"


def test_trigger_true_for_visual_doc_type():
    """should_use_vision() returns True for inherently visual doc types."""
    from vision.triggers import should_use_vision

    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o", vision_min_words=50)
    page_text = " ".join(["word"] * 60)  # enough words — doc_type overrides

    for doc_type in ["construction_photo", "id_document"]:
        result = should_use_vision(
            file_path="/uploads/doc.pdf",
            page_text=page_text,
            is_scanned=False,
            doc_type=doc_type,
            config=cfg,
        )
        assert result is True, f"Should trigger vision for doc_type={doc_type}"


# ---------------------------------------------------------------------------
# test_vision_engine_uses_null_when_unconfigured
# ---------------------------------------------------------------------------

def test_vision_engine_uses_null_when_unconfigured():
    """
    VisionEngine returns NoVisionModel when VISION_PROVIDER is not set.
    describe_image() returns empty string in this case.
    """
    from vision.engine import get_vision_model, reset_vision_model
    from vision.null import NoVisionModel

    reset_vision_model()

    cfg = _make_config(vision_provider="", vision_model="")
    model = get_vision_model(cfg)

    assert isinstance(model, NoVisionModel), \
        "Should return NoVisionModel when vision_provider is empty"
    assert model.describe("/any/path.png", "prompt") == ""

    reset_vision_model()  # clean up singleton for other tests


def test_vision_engine_falls_back_to_null_on_missing_api_key():
    """
    VisionEngine falls back to NoVisionModel when provider is set
    but API key is missing.
    """
    from vision.engine import get_vision_model, reset_vision_model
    from vision.null import NoVisionModel

    reset_vision_model()

    # provider set, but no API key
    cfg = _make_config(vision_provider="openai", vision_model="gpt-4o", openai_api_key="")
    model = get_vision_model(cfg)

    assert isinstance(model, NoVisionModel), \
        "Should fall back to NoVisionModel when API key is missing"

    reset_vision_model()


# ---------------------------------------------------------------------------
# test_vision_cache_hit_skips_api_call
# ---------------------------------------------------------------------------

def test_vision_cache_hit_skips_api_call():
    """
    describe_image() returns cached result without calling the vision model.
    """
    from vision.engine import describe_image, reset_vision_model

    reset_vision_model()

    cached_description = "A cached invoice description"

    with patch("vision.engine.get_vision_description", return_value=cached_description) as mock_get, \
         patch("vision.engine.set_vision_description") as mock_set, \
         patch("vision.engine.get_vision_model") as mock_model:

        result = describe_image("/path/to/invoice.png", "invoice")

        assert result == cached_description, "Should return cached description"
        mock_model.assert_not_called(), "Should not initialise vision model on cache hit"
        mock_set.assert_not_called(), "Should not write to cache again on hit"

    reset_vision_model()


def test_vision_cache_miss_calls_model_and_writes_cache():
    """
    describe_image() calls the model on cache miss and writes result to cache.
    """
    from vision.engine import describe_image, reset_vision_model
    from vision.null import NoVisionModel

    reset_vision_model()

    mock_model = NoVisionModel()  # returns "" — safe, no API call

    with patch("vision.engine.get_vision_description", return_value=None), \
         patch("vision.engine.set_vision_description") as mock_set, \
         patch("vision.engine.get_vision_model", return_value=mock_model), \
         patch("vision.engine.get_vision_prompt", return_value="describe this"), \
         patch("builtins.open", mock_open(read_data=b"")):

        result = describe_image("/path/to/image.png", "general")

        # NoVisionModel returns "" so cache write is skipped (empty string not cached)
        assert result == ""
        mock_set.assert_not_called()  # empty descriptions are not cached

    reset_vision_model()