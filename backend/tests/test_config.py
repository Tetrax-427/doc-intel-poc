"""
tests/test_config.py

Tests for backend/core/config.py
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestConfigLoading:

    def test_raises_on_missing_supabase_url(self):
        """Config must raise ValueError if SUPABASE_URL is missing."""
        from core.config import load_config
        saved = os.environ.pop("SUPABASE_URL", None)
        try:
            with pytest.raises(ValueError, match="SUPABASE_URL"):
                load_config()
        finally:
            if saved:
                os.environ["SUPABASE_URL"] = saved

    def test_raises_on_missing_supabase_key(self):
        """Config must raise ValueError if SUPABASE_KEY is missing."""
        from core.config import load_config
        saved = os.environ.pop("SUPABASE_KEY", None)
        try:
            with pytest.raises(ValueError, match="SUPABASE_KEY"):
                load_config()
        finally:
            if saved:
                os.environ["SUPABASE_KEY"] = saved

    def test_raises_on_both_missing(self):
        """Config must list all missing required keys in the error."""
        from core.config import load_config
        saved_url = os.environ.pop("SUPABASE_URL", None)
        saved_key = os.environ.pop("SUPABASE_KEY", None)
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config()
            assert "SUPABASE_URL" in str(exc_info.value)
            assert "SUPABASE_KEY" in str(exc_info.value)
        finally:
            if saved_url:
                os.environ["SUPABASE_URL"] = saved_url
            if saved_key:
                os.environ["SUPABASE_KEY"] = saved_key

    def test_loads_with_required_keys(self, cfg):
        """Config loads successfully when required keys are present."""
        assert cfg.supabase_url is not None
        assert cfg.supabase_key is not None

    def test_default_llm_provider(self, cfg):
        assert cfg.llm_provider == "groq"

    def test_default_llm_model(self, cfg):
        assert cfg.llm_model == "llama-3.3-70b-versatile"

    def test_default_chunk_size(self, cfg):
        assert cfg.chunk_size == 512

    def test_default_chunk_overlap(self, cfg):
        assert cfg.chunk_overlap == 64

    def test_default_compression_threshold(self, cfg):
        assert cfg.compression_threshold == 10

    def test_default_vision_min_words(self, cfg):
        assert cfg.vision_min_words == 50

    def test_default_classification_confidence_threshold(self, cfg):
        assert cfg.classification_confidence_threshold == 0.7

    def test_default_max_file_size_mb(self, cfg):
        assert cfg.max_file_size_mb == 50

    def test_default_upload_dir(self, cfg):
        assert cfg.upload_dir == "uploads"

    def test_env_override(self):
        """Env vars should override defaults."""
        from core.config import load_config
        os.environ["CHUNK_SIZE"] = "1024"
        try:
            cfg = load_config()
            assert cfg.chunk_size == 1024
        finally:
            os.environ.pop("CHUNK_SIZE", None)

    def test_singleton_proxy(self, cfg):
        """Singleton proxy should return same values as direct load."""
        from core.config import config
        assert config.chunk_size == cfg.chunk_size
        assert config.llm_provider == cfg.llm_provider

    def test_all_18_fields_present(self, cfg):
        """All 18 required config fields must be present."""
        import dataclasses
        from core.config import Config
        fields = {f.name for f in dataclasses.fields(Config)}
        required = {
            "llm_provider", "llm_model", "groq_api_key", "openai_api_key",
            "anthropic_api_key", "vision_provider", "vision_model",
            "llama_cloud_api_key", "cohere_api_key", "supabase_url",
            "supabase_key", "upload_dir", "max_file_size_mb", "chunk_size",
            "chunk_overlap", "compression_threshold", "vision_min_words",
            "classification_confidence_threshold",
        }
        assert required.issubset(fields), f"Missing fields: {required - fields}"