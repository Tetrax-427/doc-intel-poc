"""
backend/agents/chat.py
------------------------
Chat over a completed agent_runs row.

Deliberately NOT part of the base.py stage machine — this cannot mutate the
run itself (no new_stage/state persistence here); "recalculation" happens
via a chat_tool calling back into the agent's own Python functions and
returning a result, same as any other tool.

CHANGED (tool-calling chat): chat now has two modes, chosen per-call by
whether the caller passes chat_tools (agent-specific, registered in
agents/registry.py's AGENT_REGISTRY[agent_name]["chat_tools"]):

  - chat_tools is empty/None (e.g. cv_processor, which registers none):
    UNCHANGED from before — a single call_llm() text call, no tool-calling,
    AGENT_CHAT_SYSTEM's "this is read-only, I can't take action" framing
    still applies. Existing behaviour and prompt for every agent that
    doesn't opt in is untouched.

  - chat_tools is non-empty (e.g. ITR's recalc/doc-lookup/RAG tools):
    Routes through llm/tool_orchestrator.run_tool_loop() instead, with
    AGENT_CHAT_SYSTEM_WITH_TOOLS (drops the "I can't take action" framing,
    since it now can via tools). The orchestrator handles its own
    LLM<->tool round trips internally; only the final answer is persisted
    to chat history here — intermediate tool calls are not stored as chat
    messages in v1 (visible in llm_calls tracing via call_type, not in
    agent_chat_messages).

Chat is only available once agent_runs.status == "completed" — enforced
here, not in the router, so any future caller (not just the HTTP route)
gets the same guarantee. This is unchanged by the tool-calling addition —
a run must still be completed before chat (tool-enabled or not) opens up;
mid-run mutation still only happens through the resume endpoint.

Why prior turns are folded into the `user` string as plain text: llm/engine.
call_llm() takes a single system+user string pair, not a messages=[...]
array like the raw provider SDKs — there is no multi-turn primitive lower
in the stack to hand this off to. The tool orchestrator inherits the same
constraint (it's built on call_llm() too) — see tool_orchestrator.py's
own docstring for how it folds tool-round history into that same flat
string instead.

Context size: the run's `result` block is used, NOT the full internal
`state` — state can carry raw per-candidate CSV data and every sub-agent's
intermediate output, which is large enough on its own to exceed the
smallest fallback model's tokens-per-minute limit (seen in practice: a
single chat turn requested ~8100 tokens against a 6000-8000 TPM cap and
every provider in the chain rejected it with 413). `result` is the fixed,
already-compact output contract, so it's used as-is, with a hard character
cap (_truncate) as a safety net against future growth (e.g. large candidate
pools). Tools that need deeper access to `state` (e.g. ITR's recalc tool
needing the full stitched profile) read it themselves inside the tool's
executor, via run_id — they are not handed the truncated context blob.
"""
from __future__ import annotations

from core.logger import get_logger
from db import get_agent_run, save_agent_chat_message, get_agent_chat_history
from llm.engine import call_llm
from llm.tool_orchestrator import run_tool_loop, ToolSpec
from agents.chat_prompts import AGENT_CHAT_SYSTEM, AGENT_CHAT_SYSTEM_WITH_TOOLS

logger = get_logger("agents.chat")

# Older turns beyond this are dropped from the prompt (not from storage) to
# bound prompt size on long-running chats. No compaction/summarization in v1.
MAX_HISTORY_TURNS = 20

# Hard cap on the run-context block's character length — keeps the whole
# prompt (system + context + history + new message) comfortably under the
# smallest fallback model's TPM limit even if `result` itself grows large.
# ~6000 chars is roughly 1500-2000 tokens, leaving headroom for history and
# the system prompt within an 8000 TPM budget.
MAX_CONTEXT_CHARS = 6000


class AgentChatError(Exception):
    """Raised for chat-specific failure conditions (run not found / not owned / not completed / empty reply)."""


def _truncate(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated to fit within the LLM's context limit)"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no previous messages)"
    trimmed = history[-MAX_HISTORY_TURNS:]
    lines = []
    for m in trimmed:
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def _format_run_context(run: dict) -> str:
    result = run.get("result") or {}
    context = (
        f"Agent: {run['agent_name']}\n"
        f"Original task: {run['task']}\n\n"
        f"Result summary: {result.get('summary', '')}\n\n"
        f"Findings:\n" + "\n".join(f"- {f}" for f in result.get("findings", [])) + "\n\n"
        f"Data tables: {result.get('data', [])}\n\n"
        f"Extra detail (evaluation plan, full per-criterion scores, full ranking, etc.): "
        f"{result.get('extra', {})}"
    )
    return _truncate(context)


def handle_chat_message(
    run_id: str,
    message: str,
    user_id: str,
    chat_tools: list[ToolSpec] | None = None,
) -> dict:
    """
    Answers one chat message grounded in a completed agent_runs row.
    Persists both the user message and the assistant reply on success.

    Args:
        chat_tools: Agent-specific tools (from AGENT_REGISTRY[agent_name]
                    ["chat_tools"], read by the router — chat.py itself
                    doesn't import agents.registry, same circular-import
                    reasoning as agents/base.py's stage_descriptions param).
                    None/empty => today's plain call_llm() behaviour,
                    unchanged. Non-empty => routes through run_tool_loop().

    Raises AgentChatError if the run doesn't exist / isn't owned by
    user_id, isn't completed yet, or the LLM returns an empty reply —
    callers should map this to a 400.
    """
    run = get_agent_run(run_id, user_id=user_id)
    if run is None:
        raise AgentChatError(f"agent_run '{run_id}' not found for this user.")
    if run["status"] != "completed":
        raise AgentChatError(
            f"agent_run '{run_id}' is not completed yet (status={run['status']}) — "
            f"chat is only available once the run has completed."
        )

    history = get_agent_chat_history(run_id, user_id=user_id)

    user_prompt = (
        f"--- Agent run context ---\n{_format_run_context(run)}\n\n"
        f"--- Conversation so far ---\n{_format_history(history)}\n\n"
        f"--- New message ---\nUser: {message}"
    )

    if chat_tools:
        outcome = run_tool_loop(
            system=AGENT_CHAT_SYSTEM_WITH_TOOLS,
            user=user_prompt,
            tools=chat_tools,
            call_type=f"agent_chat_tools:{run['agent_name']}",
            user_id=user_id,
            session_id=run_id,
        )
        reply = outcome["final_answer"]
    else:
        reply = call_llm(
            system=AGENT_CHAT_SYSTEM,
            user=user_prompt,
            temperature=0.2,
            call_type="agent_chat",
            user_id=user_id,
            session_id=run_id,
        )

    if not reply or not reply.strip():
        logger.error("Agent chat returned an empty reply", run_id=run_id)
        raise AgentChatError("The agent didn't return a response — please try again.")

    save_agent_chat_message(run_id, "user", message, user_id)
    save_agent_chat_message(run_id, "assistant", reply, user_id)

    return {"role": "assistant", "content": reply}