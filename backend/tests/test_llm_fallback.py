"""
tests/test_llm_fallback.py

Tests for C2 (fallback chain) + C3 (per-call override) + router integration.

Changes from previous version:
- call_llm() now requires system= and user= keyword args, not a positional string.
- log_usage patching replaced with tracer.trace patching.
- TestBackwardCompatibility removed (the old call_llm("prompt") positional form
  no longer exists by design — that section tested behavior that was intentionally
  replaced, not something that should remain compatible).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_config():
    import core.config as cfg_mod
    cfg_mod._config_instance = None


def _make_openai_response(text: str):
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


def _make_anthropic_response(text: str):
    usage = MagicMock()
    usage.input_tokens = 8
    usage.output_tokens = 4
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = usage
    return resp


def _make_failing_client(error: str = "provider error"):
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception(error)
    client.messages.create.side_effect = Exception(error)
    return client


def _make_succeeding_client(text: str = "success", provider: str = "openai"):
    client = MagicMock()
    if provider == "anthropic":
        client.messages.create.return_value = _make_anthropic_response(text)
    else:
        client.chat.completions.create.return_value = _make_openai_response(text)
    return client


def _noop_trace_ctx():
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=MagicMock())
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant_test")
    monkeypatch.setenv(
        "LLM_FALLBACK_CHAIN",
        "groq:llama-3.3-70b-versatile,openai:gpt-4o-mini,anthropic:claude-3-5-haiku-20241022"
    )
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# C2 — Fallback chain resolution
# ---------------------------------------------------------------------------

class TestFallbackChainResolution:

    def test_full_chain_resolved_in_order(self):
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        assert chain == [
            ("groq", "llama-3.3-70b-versatile"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-5-haiku-20241022"),
        ]

    def test_missing_key_removes_provider_from_chain(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        providers = [p for p, _ in chain]
        assert "openai" not in providers
        assert "groq" in providers
        assert "anthropic" in providers

    def test_all_keys_missing_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        _reset_config()
        from llm.fallback import get_fallback_chain
        from core.errors import LLMConfigError
        with pytest.raises(LLMConfigError) as exc_info:
            get_fallback_chain()
        assert "LLM_005" == exc_info.value.code or "empty" in str(exc_info.value).lower()

    def test_empty_chain_env_falls_back_to_primary(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        assert chain == [("openai", "gpt-4o")]

    def test_malformed_entries_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "groq:llama-3.3-70b-versatile,,bad-entry,openai:gpt-4o-mini")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        providers = [p for p, _ in chain]
        assert "groq" in providers
        assert "openai" in providers
        assert len(chain) == 2

    def test_unsupported_provider_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "cohere:command-r,openai:gpt-4o-mini")
        _reset_config()
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        providers = [p for p, _ in chain]
        assert "cohere" not in providers
        assert "openai" in providers

    def test_chain_length_matches_configured_valid_providers(self):
        from llm.fallback import get_fallback_chain
        chain = get_fallback_chain()
        assert len(chain) == 3


# ---------------------------------------------------------------------------
# C2 — Engine fallback loop
# ---------------------------------------------------------------------------

class TestEngineFallbackLoop:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_first_provider_succeeds_no_fallback(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_succeeding_client("first wins")
        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "first wins"
        assert mock_build.call_count == 1

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_falls_back_to_second_on_first_failure(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.side_effect = [
            _make_failing_client("groq down"),
            _make_succeeding_client("openai result"),
        ]
        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "openai result"
        assert mock_build.call_count == 2

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_falls_back_to_third_on_first_two_failures(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.side_effect = [
            _make_failing_client("groq down"),
            _make_failing_client("openai down"),
            _make_succeeding_client("anthropic result", provider="anthropic"),
        ]
        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "anthropic result"
        assert mock_build.call_count == 3

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_raises_fallback_exhausted_when_all_fail(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_failing_client("all down")
        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError) as exc_info:
            call_llm(system="sys", user="usr", call_type="test")
        assert "LLM_002" == exc_info.value.code
        tried = exc_info.value.context.get("providers_tried", [])
        assert len(tried) == 3

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_providers_tried_logged_in_order(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_failing_client()
        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError) as exc_info:
            call_llm(system="sys", user="usr", call_type="test")
        tried = exc_info.value.context["providers_tried"]
        assert tried[0][0] == "groq"
        assert tried[1][0] == "openai"
        assert tried[2][0] == "anthropic"

    @patch("llm.engine.time.sleep")
    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_rate_limit_retries_same_provider_before_fallback(
        self, mock_build, mock_trace, mock_sleep
    ):
        mock_trace.return_value = _noop_trace_ctx()
        client = _make_succeeding_client("ok after retry")
        client.chat.completions.create.side_effect = [
            Exception("rate limit 429"),
            Exception("rate limit 429"),
            _make_openai_response("ok after retry"),
        ]
        mock_build.return_value = client

        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "ok after retry"
        assert mock_build.call_count == 1
        assert mock_sleep.call_count == 2

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_non_rate_limit_error_goes_to_next_provider_immediately(
        self, mock_build, mock_trace
    ):
        mock_trace.return_value = _noop_trace_ctx()
        failing   = _make_failing_client("connection timeout")
        succeeding = _make_succeeding_client("second provider")
        mock_build.side_effect = [failing, succeeding]

        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", call_type="test")
        assert result == "second provider"
        assert mock_build.call_count == 2
        assert failing.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# C3 — Per-call provider/model override
# ---------------------------------------------------------------------------

class TestPerCallOverride:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_override_uses_specified_provider_and_model(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_succeeding_client("override result")
        from llm.engine import call_llm
        result = call_llm(system="sys", user="usr", provider="openai", model="gpt-4o", call_type="test")
        assert result == "override result"
        mock_build.assert_called_once_with("openai")

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_override_does_not_fall_back_on_failure(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_failing_client("override provider down")
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError) as exc_info:
            call_llm(system="sys", user="usr", provider="openai", model="gpt-4o", call_type="test")
        assert exc_info.value.code == "LLM_003"
        assert mock_build.call_count == 1

    def test_override_requires_both_provider_and_model(self):
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm(system="sys", user="usr", provider="openai", call_type="test")

    def test_override_requires_both_model_and_provider(self):
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm(system="sys", user="usr", model="gpt-4o", call_type="test")

    @patch("llm.engine.build_client")
    def test_override_with_unsupported_provider_raises(self, mock_build):
        from llm.engine import call_llm
        from core.errors import LLMProviderOverrideError
        with pytest.raises(LLMProviderOverrideError):
            call_llm(system="sys", user="usr", provider="cohere", model="command-r", call_type="test")

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_override_single_entry_chain_not_full_chain(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_succeeding_client("only once")
        from llm.engine import call_llm
        call_llm(system="sys", user="usr", provider="openai", model="gpt-4o", call_type="test")
        assert mock_build.call_count == 1

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_no_override_uses_full_fallback_chain(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_failing_client()
        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError):
            call_llm(system="sys", user="usr", call_type="test")
        assert mock_build.call_count == 3


# ---------------------------------------------------------------------------
# C2 — Streaming fallback
# ---------------------------------------------------------------------------

class TestStreamingFallback:

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_stream_falls_back_to_second_provider(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        failing = _make_failing_client("groq stream down")
        working = MagicMock()
        working.chat.completions.create.return_value = MagicMock(
            **{"__iter__": lambda s: iter([
                MagicMock(**{"choices": [MagicMock(**{"delta": MagicMock(content="hello")})]}),
            ])}
        )
        mock_build.side_effect = [failing, working]

        from llm.engine import call_llm
        call_llm(system="sys", user="usr", stream=True, call_type="test")
        assert mock_build.call_count == 2

    @patch("llm.engine.tracer.trace")
    @patch("llm.engine.build_client")
    def test_stream_raises_exhausted_when_all_providers_fail(self, mock_build, mock_trace):
        mock_trace.return_value = _noop_trace_ctx()
        mock_build.return_value = _make_failing_client("all stream down")
        from llm.engine import call_llm
        from core.errors import LLMFallbackExhaustedError
        with pytest.raises(LLMFallbackExhaustedError):
            call_llm(system="sys", user="usr", stream=True, call_type="test")


# ---------------------------------------------------------------------------
# /llm/available-models endpoint
# ---------------------------------------------------------------------------

class TestAvailableModelsEndpoint:

    def _make_test_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.system import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_returns_200(self):
        client = self._make_test_client()
        resp = client.get("/llm/available-models")
        assert resp.status_code == 200

    def test_response_shape(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        assert "providers" in data
        assert "fallback_chain" in data
        assert "primary" in data

    def test_providers_list_contains_all_supported(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        provider_names = {p["provider"] for p in data["providers"]}
        assert {"groq", "openai", "anthropic"}.issubset(provider_names)

    def test_configured_flag_true_when_key_present(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        for p in data["providers"]:
            assert p["configured"] is True

    def test_active_flag_reflects_fallback_chain(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        active = {p["provider"] for p in data["providers"] if p["active"]}
        chain_providers = {e["provider"] for e in data["fallback_chain"]}
        assert active == chain_providers

    def test_unconfigured_provider_not_active(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("LLM_FALLBACK_CHAIN", "groq:llama-3.3-70b-versatile,openai:gpt-4o-mini")
        _reset_config()
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        openai_entry = next(p for p in data["providers"] if p["provider"] == "openai")
        assert openai_entry["active"] is False
        assert openai_entry["configured"] is False

    def test_fallback_chain_ordered(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        chain = data["fallback_chain"]
        assert chain[0]["provider"] == "groq"
        assert chain[1]["provider"] == "openai"
        assert chain[2]["provider"] == "anthropic"

    def test_primary_is_first_chain_entry(self):
        client = self._make_test_client()
        data = client.get("/llm/available-models").json()
        assert data["primary"]["provider"] == "groq"
        assert data["primary"]["model"] == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# /query router — C3 passthrough
# ---------------------------------------------------------------------------

class TestQueryRouterOverride:

    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers.query import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @patch("retrieval.query_document")
    def test_provider_and_model_forwarded_to_query_document(self, mock_qd):
        mock_qd.return_value = {"answer": "ok", "sources": [], "type": "document"}
        client = self._make_app()

        resp = client.post("/query", json={
            "question": "what is this?",
            "document_id": "doc-123",
            "provider": "anthropic",
            "model": "claude-3-5-haiku-20241022",
        })

        assert resp.status_code == 200
        _, kwargs = mock_qd.call_args
        assert kwargs.get("provider") == "anthropic"
        assert kwargs.get("model") == "claude-3-5-haiku-20241022"

    @patch("retrieval.query_document")
    def test_no_override_passes_none_to_query_document(self, mock_qd):
        mock_qd.return_value = {"answer": "ok", "sources": [], "type": "document"}
        client = self._make_app()

        resp = client.post("/query", json={
            "question": "what is this?",
            "document_id": "doc-123",
        })

        assert resp.status_code == 200
        _, kwargs = mock_qd.call_args
        assert kwargs.get("provider") is None
        assert kwargs.get("model") is None

    def test_provider_without_model_returns_422(self):
        client = self._make_app()
        resp = client.post("/query", json={"question": "test", "provider": "openai"})
        assert resp.status_code == 422

    def test_model_without_provider_returns_422(self):
        client = self._make_app()
        resp = client.post("/query", json={"question": "test", "model": "gpt-4o"})
        assert resp.status_code == 422

    def test_empty_question_returns_422(self):
        client = self._make_app()
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422