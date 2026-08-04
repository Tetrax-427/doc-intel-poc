"""
llm/tool_orchestrator.py — custom tool-calling orchestration layer.

Sits ON TOP of llm/engine.call_llm(). Does NOT modify call_llm() or its
signature — engine.py stays exactly as-is.

Why a custom layer instead of native provider tool-calling:
  - call_llm() already has one clean structured-output path — response_model
    via Instructor (llm/structured.call_structured) — proven across every
    extraction call site (retrieval.extract_dynamic_fields, etc). Reusing
    that path here (instead of introducing OpenAI/Anthropic/Groq's three
    different native tool-calling APIs) means the orchestrator is provider-
    agnostic for free, same as call_llm() already is.
  - Trade-off accepted: this gives batched-sequential tool calls, not native
    parallel tool calls. Each LLM turn returns a *batch* of tool calls
    (ToolCallBatch.calls), the orchestrator executes all of them in this
    process (one after another, not concurrently — "sequentially" here means
    execution order, not blocking-vs-async), folds every result back into
    the next call_llm() call, and loops until the model returns zero tool
    calls (final answer). Tools within one batch must be independent of
    each other's output, since none of them have run yet when the model
    decides to call all of them.

Agent-agnostic by design: any agent (ITR helper, cv_processor, future
agents) registers its own ToolSpec list and calls run_tool_loop(). Nothing
here is ITR-specific.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from core.errors import DocIntelError
from core.logger import get_logger
from llm.engine import call_llm

logger = get_logger("llm.tool_orchestrator")

# Hard ceiling on rounds so a confused model can't loop forever burning
# tokens — mirrors engine.py's own MAX_RETRIES-style safety net.
DEFAULT_MAX_ROUNDS = 6


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ToolOrchestratorError(DocIntelError):
    """Raised for orchestrator-specific failures (unknown tool, exec failure, round cap hit)."""

    def __init__(self, message: str, code: str = "TOOL_001", context: dict | None = None):
        super().__init__(
            message,
            code=code,
            severity="ERROR",
            retryable=True,
            context=context or {},
        )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """
    One callable tool the LLM can request.

    name:        Stable identifier the LLM refers to (e.g. "recalculate_tax").
    description: Plain-English description shown to the LLM in the system
                 prompt — this is the model's only signal for when to call it.
    executor:    Callable(**kwargs) -> Any (must be JSON-serialisable, or a
                 str/dict/BaseModel — same shapes call_llm() already returns).
                 Raising inside executor is caught by the orchestrator and
                 surfaced to the LLM as a tool error string, not raised out
                 of run_tool_loop() — lets the model retry or route around a
                 single failed tool rather than aborting the whole chat turn.
    args_schema: Optional plain-English description of expected args, folded
                 into the system prompt alongside `description`. Not a strict
                 JSON-schema/Pydantic contract in v1 — kept simple since args
                 are parsed via the same Instructor/response_model path as
                 everything else in this codebase (self-correcting on
                 validation failure via structured_max_retries), not hand-
                 rolled JSON parsing.
    """
    name: str
    description: str
    executor: Callable[..., Any]
    args_schema: str = ""


class ToolRegistry:
    """Simple name -> ToolSpec lookup, built fresh per call site (per agent)."""

    def __init__(self, tools: list[ToolSpec]):
        self._by_name: dict[str, ToolSpec] = {}
        for t in tools:
            if t.name in self._by_name:
                raise ValueError(f"Duplicate tool name registered: '{t.name}'")
            self._by_name[t.name] = t

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def describe_for_prompt(self) -> str:
        if not self._by_name:
            return "(no tools available)"
        lines = []
        for t in self._by_name.values():
            line = f"- {t.name}: {t.description}"
            if t.args_schema:
                line += f"\n  args: {t.args_schema}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response model — what call_llm(response_model=...) parses each round into
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    tool_name: str = Field(description="Name of the tool to call. Must exactly match one of the available tool names.")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool, as a flat key-value object.")


class ToolCallBatch(BaseModel):
    """
    One LLM turn's decision: either a batch of tool calls to run before it
    can answer, or a final natural-language answer with no further calls.

    calls empty => `final_answer` is authoritative and the loop stops.
    calls non-empty => orchestrator executes all of them, ignores
    `final_answer` for this round (model isn't done yet), loops again.
    """
    calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tools to call this round. Leave empty if you already have enough "
                    "information to answer — in that case, put your answer in final_answer.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Your complete answer to the user. Only set this when calls is empty.",
    )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _build_round_system_prompt(base_system: str, registry: ToolRegistry) -> str:
    return (
        f"{base_system}\n\n"
        f"--- Available tools ---\n{registry.describe_for_prompt()}\n\n"
        "Each turn, decide whether you need to call one or more tools before "
        "you can answer, or whether you already have enough information. "
        "If you need tools, list them all in `calls` (you may call several "
        "tools in the same round — they will all be run before you see any "
        "results, so do not request a tool whose input depends on another "
        "tool's output in the same round; request that one in a later round "
        "instead). If you have enough information, leave `calls` empty and "
        "put your complete answer in `final_answer`."
    )


def _format_tool_results(results: list[tuple[ToolCall, Any]]) -> str:
    lines = []
    for call, result in results:
        lines.append(f"Tool: {call.tool_name}\nArgs: {call.args}\nResult: {result}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tool_loop(
    *,
    system: str,
    user: str,
    tools: list[ToolSpec],
    call_type: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    temperature: float = 0.2,
    structured_max_retries: int = 1,
    user_id: str = "system",
    document_id: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    """
    Runs the tool-calling loop to completion and returns the final answer.

    Every round is one call_llm(response_model=ToolCallBatch) call — same
    Instructor path used everywhere else in this codebase, just a different
    response_model shape. call_llm() itself is untouched.

    Args:
        system:      Base system prompt (agent/task-specific instructions).
                     Tool descriptions are appended automatically each round.
        user:        Initial user message / question for this turn.
        tools:       ToolSpec list available this turn. Agent-specific —
                     e.g. ITR chat passes [recalculate_tax, lookup_doc, ...].
        call_type:   Required, per-agent tracing label (e.g. "itr_chat",
                     "cv_processor_chat"). No shared default — each agent's
                     tool-calling chat must own its own call_type so tracing/
                     cost data doesn't blur different agents together.
        max_rounds:  Safety cap on LLM<->tool round trips.
        structured_max_retries: Instructor's re-ask-on-validation-failure
                     count for the ToolCallBatch parse (see llm/structured.py).
                     Defaults to 1 since a malformed tool_name/args on the
                     first try is plausible and cheap to self-correct.
        user_id / document_id / session_id / org_id / team_id: threaded
                     straight through to call_llm() for tracing/cache scoping,
                     same contract as every other call_llm() call site.

    Returns:
        {
            "final_answer": str,
            "rounds_used": int,
            "tool_calls_made": [{"tool_name": str, "args": dict, "result": Any}, ...],
        }

    Raises:
        ToolOrchestratorError: unknown tool name requested, or max_rounds
                                exhausted without a final_answer.
    """
    registry = ToolRegistry(tools)
    round_system = _build_round_system_prompt(system, registry)

    conversation = f"--- Task ---\n{user}"
    all_calls_made: list[dict] = []

    for round_num in range(1, max_rounds + 1):
        batch: ToolCallBatch = call_llm(
            system=round_system,
            user=conversation,
            temperature=temperature,
            call_type=call_type,
            response_model=ToolCallBatch,
            structured_max_retries=structured_max_retries,
            user_id=user_id,
            document_id=document_id,
            session_id=session_id,
            org_id=org_id,
            team_id=team_id,
        )

        if not batch.calls:
            if not batch.final_answer or not batch.final_answer.strip():
                raise ToolOrchestratorError(
                    "Model returned no tool calls and no final_answer.",
                    code="TOOL_002",
                    context={"round": round_num},
                )
            return {
                "final_answer": batch.final_answer,
                "rounds_used": round_num,
                "tool_calls_made": all_calls_made,
            }

        # Execute this round's batch — sequential execution order, all
        # independent (see module docstring), all run before the next
        # call_llm() call.
        round_results: list[tuple[ToolCall, Any]] = []
        for call in batch.calls:
            spec = registry.get(call.tool_name)
            if spec is None:
                # Don't raise out — feed the error back so the model can
                # self-correct (e.g. mis-typed tool name) instead of aborting
                # the whole chat turn on one bad call.
                result = f"ERROR: unknown tool '{call.tool_name}'. Available tools: " \
                          f"{', '.join(t.name for t in tools)}"
                logger.warning("Unknown tool requested by model", tool_name=call.tool_name, round=round_num)
            else:
                try:
                    result = spec.executor(**call.args)
                except Exception as exc:
                    result = f"ERROR: tool '{call.tool_name}' failed: {exc}"
                    logger.error("Tool execution failed", tool_name=call.tool_name, args=call.args, error=str(exc))

            round_results.append((call, result))
            all_calls_made.append({"tool_name": call.tool_name, "args": call.args, "result": result})

        conversation += (
            f"\n\n--- Round {round_num} tool results ---\n"
            f"{_format_tool_results(round_results)}"
        )

    raise ToolOrchestratorError(
        f"Tool loop exceeded max_rounds={max_rounds} without a final answer.",
        code="TOOL_003",
        context={"rounds_used": max_rounds, "tool_calls_made": all_calls_made},
    )