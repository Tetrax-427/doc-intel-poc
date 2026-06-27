"""
llm/tracer.py — central tracing for every LLM call in the app.

Usage (inside llm/engine.py — see FINAL_PLAN.md Phase B):

    with tracer.trace(
        call_type="extract",
        provider=provider, model=model,
        system=system_text, user=user_text,
        user_id=user_id, document_id=document_id,
        is_override=is_override, is_stream=False,
    ) as t:
        result, usage = _call_single_provider(...)
        t.set_result(response_text=str(result), usage=usage)
        # on exception inside the `with` block, __exit__ calls set_error()
        # automatically and re-raises — callers don't need their own
        # try/except just for tracing.

Design rules (do not relax these without updating FINAL_PLAN.md):
- NEVER raises out of its own machinery. A tracing failure (DB write error,
  cost-table miss, bad usage shape) must never break the actual LLM call or
  mask the real exception from it.
- NEVER blocks meaningfully — DB write is a single synchronous insert, same
  pattern as the rest of db.py/db_llm_calls.py. If this becomes a latency
  concern at scale, swap to a background thread/queue — out of scope for
  this phase.
- Cost estimation is best-effort. Unknown provider/model → cost is None,
  not 0.0 — 0.0 silently understates spend in aggregates; None makes the
  gap visible (db_llm_calls.get_summary already treats None as "skip" via
  `or 0.0` at read time, which is the one place a 0-ish fallback is safe —
  it's a display concern there, not a stored-value concern here).
"""

from __future__ import annotations

import time
from typing import Any

from core.logger import get_logger
from db_llm_calls import insert_llm_call
from llm.hashing import compute_prompt_hash

logger = get_logger("llm.tracer")

# Truncate stored prompt/response text so a handful of huge document-extraction
# calls don't blow up row size / query cost. Full content was never the point
# of this table — hashing, cost, and timing are. 4000 chars is generous for
# debugging while staying well under any reasonable row-size concern.
_MAX_STORED_TEXT_CHARS = 4000


# ---------------------------------------------------------------------------
# Cost table — best-effort, deliberately incomplete.
#
# Keyed by (provider, model). Update as you change models in
# LLM_FALLBACK_CHAIN / VISION_MODEL. Prices drift — verify against the
# provider's current pricing page before trusting this for real budgeting;
# treat numbers here as "good enough for relative comparison between call
# types", not as an invoice reconciliation tool.
#
# Values are USD per 1M tokens: (input_per_million, output_per_million).
# Last checked: 2026-06-19.
# ---------------------------------------------------------------------------

_COST_PER_MILLION: dict[tuple[str, str], tuple[float, float]] = {
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("anthropic", "claude-haiku-4-5-20251001"): (1.00, 5.00),
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
}


def estimate_cost_usd(provider: str, model: str, usage: dict | None) -> float | None:
    """
    Best-effort cost estimate from real usage. Returns None (not 0.0) when
    the model isn't in the table or usage is missing — an unknown cost should
    never silently render as "free" in a dashboard.

    Public — used both internally (_TraceHandle._build_record) and externally
    by llm/engine.py's cache-store call sites, which need to compute
    original_cost_usd for a cache entry using the exact same pricing table
    the trace row itself used, so the two numbers never drift apart.
    """
    if not usage:
        return None

    rates = _COST_PER_MILLION.get((provider, model))
    if rates is None:
        return None

    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    input_rate, output_rate = rates

    cost = (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate
    return round(cost, 6)


def _truncate(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= _MAX_STORED_TEXT_CHARS:
        return text
    return text[:_MAX_STORED_TEXT_CHARS] + f"... [truncated, {len(text)} chars total]"


# ---------------------------------------------------------------------------
# Trace handle — returned by trace(), mutated by the caller mid-call
# ---------------------------------------------------------------------------

class _TraceHandle:
    """
    Mutable record built up over the lifetime of one `with tracer.trace(...)`
    block. Not meant to be constructed directly — use tracer.trace().
    """

    def __init__(
        self,
        *,
        call_type: str,
        provider: str,
        model: str,
        system: str,
        user: str,
        user_id: str,
        document_id: str | None,
        session_id: str | None,
        is_override: bool,
        is_stream: bool,
        used_fallback: bool = False,
        response_model_name: str | None = None,
    ):
        self.call_type = call_type
        self.provider = provider
        self.model = model
        self.system = system
        self.user = user
        self.user_id = user_id
        self.document_id = document_id
        self.session_id = session_id
        self.is_override = is_override
        self.is_stream = is_stream
        self.used_fallback = used_fallback
        self.response_model_name = response_model_name

        self._start = time.time()
        self._success: bool | None = None
        self._response_text: str | None = None
        self._error_message: str | None = None
        self._error_code: str | None = None
        self._usage: dict | None = None
        self._cache_hit = False
        self._cache_layer: str | None = None

    # -- called by the caller on success --
    def set_result(self, response_text: str, usage: dict | None = None) -> None:
        self._success = True
        self._response_text = response_text
        self._usage = usage

    # -- called by the caller (or __exit__) on failure --
    def set_error(self, exc: Exception) -> None:
        self._success = False
        self._error_message = str(exc)
        # core/errors.py DocIntelError subclasses carry a `.code` — use it
        # when available for clean grouping; fall back to the exception's
        # class name so unexpected non-DocIntel exceptions still get a
        # usable (if generic) bucket instead of None.
        self._error_code = getattr(exc, "code", None) or type(exc).__name__

    # -- called by Phase F's cache lookup path, before the call is even
    #    attempted — a cache hit never goes through set_result/set_error
    #    in the normal sense since no live call happened. --
    def set_cache_hit(self, response_text: str, cache_layer: str = "L2") -> None:
        self._success = True
        self._response_text = response_text
        self._cache_hit = True
        self._cache_layer = cache_layer
        self._usage = None  # no provider call happened — no real usage to report

    def _build_record(self) -> dict:
        latency_ms = round((time.time() - self._start) * 1000)
        usage = self._usage or {}

        prompt_hash = compute_prompt_hash(
            self.user_id, self.provider, self.model, self.system, self.user
        )

        cost = (
            0.0
            if self._cache_hit
            else estimate_cost_usd(self.provider, self.model, self._usage)
        )

        return {
            "user_id": self.user_id,
            "document_id": self.document_id,
            "session_id": self.session_id,
            "call_type": self.call_type,
            "provider": self.provider,
            "model": self.model,
            "is_override": self.is_override,
            "is_stream": self.is_stream,
            "used_fallback": self.used_fallback,
            "system_text": _truncate(self.system),
            "user_text": _truncate(self.user),
            "prompt_hash": prompt_hash,
            "success": bool(self._success),
            "response_text": _truncate(self._response_text),
            "response_model_name": self.response_model_name,
            "error_message": self._error_message,
            "error_code": self._error_code,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "estimated_cost_usd": cost,
            "cache_hit": self._cache_hit,
            "cache_layer": self._cache_layer,
            "latency_ms": latency_ms,
        }

    def _flush(self) -> None:
        if self._success is None:
            # Caller never called set_result/set_error and didn't raise either
            # (shouldn't happen given __exit__ handles the raise case, but
            # guard anyway rather than write a half-built row).
            logger.warning(
                "Trace block exited with no result/error set — skipping write",
                call_type=self.call_type,
            )
            return
        try:
            insert_llm_call(self._build_record())
        except Exception as exc:
            # Tracing must never break the caller. Log and move on.
            logger.warning(
                "Failed to write llm_calls row — continuing without it",
                call_type=self.call_type,
                provider=self.provider,
                model=self.model,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------

class _TraceContext:
    def __init__(self, handle: _TraceHandle):
        self._handle = handle

    def __enter__(self) -> _TraceHandle:
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None and self._handle._success is None:
            # Caller's code raised before calling set_result/set_error —
            # capture it here so the row still gets written, then let the
            # exception propagate normally (return False = don't suppress).
            self._handle.set_error(exc_val)
        self._handle._flush()
        return False  # never suppress exceptions


def trace(
    *,
    call_type: str,
    provider: str,
    model: str,
    system: str,
    user: str,
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
    is_override: bool = False,
    is_stream: bool = False,
    used_fallback: bool = False,
    response_model_name: str | None = None,
) -> _TraceContext:
    """
    Start a trace for one LLM call. See module docstring for usage.
    """
    handle = _TraceHandle(
        call_type=call_type,
        provider=provider,
        model=model,
        system=system,
        user=user,
        user_id=user_id,
        document_id=document_id,
        session_id=session_id,
        is_override=is_override,
        is_stream=is_stream,
        used_fallback=used_fallback,
        response_model_name=response_model_name,
    )
    return _TraceContext(handle)