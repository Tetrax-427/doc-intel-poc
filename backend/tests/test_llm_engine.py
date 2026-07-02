"""
tests/test_llm_engine.py

Unit tests for llm/engine.py and llm/fallback.py.

All external I/O (provider clients, API calls, DB writes) is mocked.
Tests set env vars directly and reset the config singleton between tests.

Changes from previous version:
- call_llm() now takes system= and user= (required), not a positional prompt string.
- log_usage / llm.usage removed — tracing is handled by llm/tracer.py.
  Tests that previously patched log_usage now patch tracer.trace so the DB
  write is suppressed, keeping tests fast and isolated.
- call_structured() returns (result, usage) tuple — updated accordingly.
- Provider raw callers return (text, usage) tuples — mocks updated.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_config():
    import core.config as cfg_mod
    cfg_mod._config_instance = None


def _mock_openai_response(text: str):
    """Build a minimal mock that looks like an OpenAI chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _noop_trace_ctx():
    """Return a context manager that does nothing — replaces tracer.trace in tests."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


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

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_returns_string_on_success(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("hello")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm(system="You are helpful.", user="Say hello.", call_type="test")
        assert result == "hello"

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_json_mode_parses_response(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"key": "value"}')
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm(system="Return JSON.", user="test", json_mode=True, call_type="test")
        assert result == {"key": "value"}

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_system_and_user_sent_as_separate_messages(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("ok")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        call_llm(system="You are helpful.", user="user prompt", call_type="test")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        user_content   = next(m["content"] for m in messages if m["role"] == "user")
        assert system_content == "You are helpful."
        assert user_content == "user prompt"


# ---------------------------------------------------------------------------
# engine.py — fallback chain behaviour
# ---------------------------------------------------------------------------

class TestFallbackChain:

    def _make_failing_client(self, error_msg="some error"):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception(error_msg)
        return client

    def _make_succeeding_client(self, text="success"):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response(text)
        return client

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_falls_back_to_second_provider_on_failure(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        failing   = self._make_failing_client()
        succeeding = self._make_succeeding_client("fallback result")
        mock_build.side_effect = [failing, succeeding]

        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "fallback result"
        assert mock_build.call_count == 2

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_raises_fallback_exhausted_when_all_fail(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = self._make_failing_client()

        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError):
            call_llm(system="sys", user="usr", call_type="test")

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_rate_limit_triggers_retry_before_fallback(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            Exception("429 too many requests"),
            _mock_openai_response("ok"),
        ]
        mock_build.return_value = client

        with patch("llm.engine.time.sleep"):
            from llm.engine import call_llm
            result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "ok"
        assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# engine.py — C3 per-call override
# ---------------------------------------------------------------------------

class TestProviderOverride:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_override_uses_specified_provider(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("override result")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", provider="openai", model="gpt-4o", call_type="test")
        assert result == "override result"
        mock_build.assert_called_with("openai")

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_override_raises_on_failure_no_fallback(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = MagicMock(
            **{"chat.completions.create.side_effect": Exception("fail")}
        )

        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm(system="sys", user="usr", provider="openai", model="gpt-4o", call_type="test")

    def test_override_requires_both_provider_and_model(self):
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            # model missing
            call_llm(system="sys", user="usr", provider="openai", call_type="test")


# ---------------------------------------------------------------------------
# engine.py — Anthropic path
# ---------------------------------------------------------------------------

class TestAnthropicProvider:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_anthropic_call_uses_messages_create(self, mock_build, mock_trace, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "anthropic:claude-3-5-haiku-20241022")
        _reset_config()
        mock_trace.return_value = _noop_trace_ctx()

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        ant_response = MagicMock()
        ant_response.content = [MagicMock(text="anthropic response")]
        ant_response.usage = usage
        mock_client = MagicMock()
        mock_client.messages.create.return_value = ant_response
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        result = call_llm(system="You are helpful.", user="prompt", call_type="test")
        assert result == "anthropic response"
        mock_client.messages.create.assert_called_once()

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_anthropic_system_sent_as_top_level_param(self, mock_build, mock_trace, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "anthropic:claude-3-5-haiku-20241022")
        _reset_config()
        mock_trace.return_value = _noop_trace_ctx()

        usage = MagicMock()
        usage.input_tokens = 8
        usage.output_tokens = 3
        ant_response = MagicMock()
        ant_response.content = [MagicMock(text="ok")]
        ant_response.usage = usage
        mock_client = MagicMock()
        mock_client.messages.create.return_value = ant_response
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        call_llm(system="You are a classifier.", user="Classify this document.", call_type="test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # system is a top-level param for Anthropic (may be a list for cache_control or plain string)
        assert "system" in call_kwargs
        # user content should be in messages list
        messages = call_kwargs.get("messages", [])
        assert any(m["role"] == "user" for m in messages)
        # system should NOT appear as a role inside messages
        assert all(m["role"] != "system" for m in messages)


# ---------------------------------------------------------------------------
# engine.py — tracing integration
# ---------------------------------------------------------------------------

class TestTracingIntegration:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_trace_called_for_every_successful_call(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("ok")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        call_llm(system="sys", user="usr", call_type="extract", user_id="user_1", document_id="doc_1")

        mock_trace.assert_called_once()
        trace_kwargs = mock_trace.call_args.kwargs
        assert trace_kwargs["call_type"] == "extract"
        assert trace_kwargs["user_id"] == "user_1"
        assert trace_kwargs["document_id"] == "doc_1"
        assert trace_kwargs["system"] == "sys"
        assert trace_kwargs["user"] == "usr"

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_trace_called_on_failure_too(self, mock_build, mock_trace):
        """A failed call should still produce a trace row (error row)."""
        ctx = MagicMock()
        handle = MagicMock()
        ctx.__enter__ = MagicMock(return_value=handle)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_trace.return_value = ctx

        mock_build.return_value = MagicMock(
            **{"chat.completions.create.side_effect": Exception("provider down")}
        )

        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError):
            call_llm(system="sys", user="usr", call_type="test")

        # trace was called for each provider attempt
        assert mock_trace.call_count == 3  # three providers in the chain

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_used_fallback_false_on_primary_provider(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("ok")
        mock_build.return_value = mock_client

        from llm.engine import call_llm
        call_llm(system="sys", user="usr", call_type="test")

        trace_kwargs = mock_trace.call_args.kwargs
        assert trace_kwargs["used_fallback"] is False

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_used_fallback_true_on_second_provider(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        failing   = MagicMock(**{"chat.completions.create.side_effect": Exception("down")})
        succeeding = MagicMock()
        succeeding.chat.completions.create.return_value = _mock_openai_response("ok")
        mock_build.side_effect = [failing, succeeding]

        from llm.engine import call_llm
        call_llm(system="sys", user="usr", call_type="test")

        # Second trace call (the successful one) should have used_fallback=True
        second_call_kwargs = mock_trace.call_args_list[1].kwargs
        assert second_call_kwargs["used_fallback"] is True