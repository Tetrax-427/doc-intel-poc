"""
tests/test_llm_engine.py

Unit tests for llm/engine.py and llm/fallback.py.

All external I/O (provider clients, API calls, log_usage) is mocked.
Tests set env vars directly and reset the config singleton between tests.
"""

from __future__ import annotations

import os
import time
import pytest
from unittest.mock import MagicMock, patch, call
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers — reset config singleton between tests so env var changes take effect
# ---------------------------------------------------------------------------

def _reset_config():
    import core.config as cfg_mod
    cfg_mod._config_instance = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Provide a minimal valid env for every test."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant_test")
    monkeypatch.setenv("LLM_FALLBACK_CHAIN",
                       "groq:llama-3.3-70b-versatile,openai:gpt-4o-mini,anthropic:claude-3-5-haiku-20241022")
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# fallback.py — get_fallback_chain
# ---------------------------------------------------------------------------

class TestGetFallbackChain:

    def test_returns_ordered_tuples(self):
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        assert chain[0] == ("groq", "llama-3.3-70b-versatile")
        assert chain[1] == ("openai", "gpt-4o-mini")
        assert chain[2] == ("anthropic", "claude-3-5-haiku-20241022")

    def test_skips_provider_with_no_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "groq:llama-3.3-70b-versatile,openai:gpt-4o-mini")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        providers = [p for p, _ in chain]
        assert "groq" not in providers
        assert "openai" in providers

    def test_skips_unsupported_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "cohere:command-r,openai:gpt-4o-mini")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        providers = [p for p, _ in chain]
        assert "cohere" not in providers
        assert "openai" in providers

    def test_raises_when_chain_empty_after_filtering(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "groq:llama-3.3-70b-versatile")
        _reset_config()
        from llm.fallback import get_fallback_chain
        from core.errors import LLMConfigError
        with pytest.raises(LLMConfigError):
            get_fallback_chain()

    def test_single_entry_chain_when_no_env_var(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        assert chain == [("openai", "gpt-4o-mini")]


# ---------------------------------------------------------------------------
# fallback.py — build_client
# ---------------------------------------------------------------------------

class TestBuildClient:

    def test_builds_groq_client(self):
        from llm.fallback import build_client
        from groq import Groq
        client = build_client("groq")
        assert isinstance(client, Groq)

    def test_builds_openai_client(self):
        from llm.fallback import build_client
        from openai import OpenAI
        client = build_client("openai")
        assert isinstance(client, OpenAI)

    def test_builds_anthropic_client(self):
        from llm.fallback import build_client
        import anthropic
        client = build_client("anthropic")
        assert isinstance(client, anthropic.Anthropic)

    def test_raises_for_missing_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        _reset_config()
        from llm.fallback import build_client
        from core.errors import LLMConfigError
        with pytest.raises(LLMConfigError):
            build_client("groq")

    def test_raises_for_unknown_provider(self):
        from llm.fallback import build_client
        from core.errors import LLMConfigError
        with pytest.raises(LLMConfigError):
            build_client("cohere")


# ---------------------------------------------------------------------------
# engine.py — call_llm basic success
# ---------------------------------------------------------------------------

class TestCallLLMSuccess:

    def _mock_openai_response(self, text: str):
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @patch("llm.engine.build_client")
    @patch("llm.engine.log_usage")
    def test_returns_string_on_success(self, mock_log, mock_build):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response("hello")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm("test prompt", call_type="test")
        assert result == "hello"

    @patch("llm.engine.build_client")
    @patch("llm.engine.log_usage")
    def test_json_mode_parses_response(self, mock_log, mock_build):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            '{"key": "value"}'
        )
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm("test", json_mode=True, call_type="test")
        assert result == {"key": "value"}

    @patch("llm.engine.build_client")
    @patch("llm.engine.log_usage")
    def test_system_prompt_included_in_messages(self, mock_log, mock_build):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response("ok")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        call_llm("user prompt", system="you are helpful", call_type="test")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


# ---------------------------------------------------------------------------
# engine.py — fallback chain behaviour
# ---------------------------------------------------------------------------

class TestFallbackChain:

    def _make_failing_client(self, error_msg="some error"):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception(error_msg)
        return client

    def _make_succeeding_client(self, text="success"):
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    @patch("llm.engine.log_usage")
    @patch("llm.engine.build_client")
    def test_falls_back_to_second_provider_on_failure(self, mock_build, mock_log):
        failing = self._make_failing_client()
        succeeding = self._make_succeeding_client("fallback result")
        mock_build.side_effect = [failing, succeeding]

        from llm.engine import call_llm
        result = call_llm("prompt", call_type="test")
        assert result == "fallback result"
        assert mock_build.call_count == 2

    @patch("llm.engine.log_usage")
    @patch("llm.engine.build_client")
    def test_raises_fallback_exhausted_when_all_fail(self, mock_build, mock_log):
        mock_build.return_value = self._make_failing_client()

        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError):
            call_llm("prompt", call_type="test")

    @patch("llm.engine.log_usage")
    @patch("llm.engine.build_client")
    def test_rate_limit_triggers_retry_before_fallback(self, mock_build, mock_log):
        client = MagicMock()
        # Fail twice with rate limit, succeed on third attempt
        client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            Exception("429 too many requests"),
            MagicMock(**{
                "choices": [MagicMock(**{"message": MagicMock(**{"content": "ok"})})]
            }),
        ]
        mock_build.return_value = client

        with patch("llm.engine.time.sleep"):  # don't actually sleep
            from llm.engine import call_llm
            result = call_llm("prompt", call_type="test")
        assert result == "ok"
        assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# engine.py — C3 per-call override
# ---------------------------------------------------------------------------

class TestProviderOverride:

    def _make_succeeding_client(self, text="override result"):
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client

    @patch("llm.engine.log_usage")
    @patch("llm.engine.build_client")
    def test_override_uses_specified_provider(self, mock_build, mock_log):
        mock_build.return_value = self._make_succeeding_client()

        from llm.engine import call_llm
        result = call_llm("prompt", provider="openai", model="gpt-4o", call_type="test")
        assert result == "override result"
        # build_client should be called with the override provider
        mock_build.assert_called_with("openai")

    @patch("llm.engine.build_client")
    def test_override_raises_on_failure_no_fallback(self, mock_build):
        mock_build.return_value = MagicMock(
            **{"chat.completions.create.side_effect": Exception("fail")}
        )

        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm("prompt", provider="openai", model="gpt-4o", call_type="test")

    def test_override_requires_both_provider_and_model(self):
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm("prompt", provider="openai", call_type="test")  # model missing


# ---------------------------------------------------------------------------
# engine.py — Anthropic path
# ---------------------------------------------------------------------------

class TestAnthropicProvider:

    @patch("llm.engine.log_usage")
    @patch("llm.engine.build_client")
    def test_anthropic_call_uses_messages_create(self, mock_build, mock_log, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "anthropic:claude-3-5-haiku-20241022")
        _reset_config()

        ant_response = MagicMock()
        ant_response.content = [MagicMock(text="anthropic response")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = ant_response
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm("prompt", call_type="test")
        assert result == "anthropic response"
        mock_client.messages.create.assert_called_once()