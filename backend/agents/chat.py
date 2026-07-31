"""
backend/agents/chat.py
------------------------
Q&A chat over a completed agent_runs row.

Deliberately NOT part of the base.py stage machine — this is read-only Q&A
grounded in a finished run's `result` (summary, findings, data, and an
"extra" block with additional detail such as the evaluation plan and full
per-criterion scores) — not a stage that can mutate the run. No
actions/tool-calling in v1 — every reply is a single call_llm() text call,
no response_model, no task_queue (see routers/agents.py chat route
docstring for why this is synchronous unlike invoke/resume).

Chat is only available once agent_runs.status == "completed" — enforced
here, not in the router, so any future caller (not just the HTTP route)
gets the same guarantee.

Why prior turns are folded into the `user` string as plain text: llm/engine.
call_llm() takes a single system+user string pair, not a messages=[...]
array like the raw provider SDKs — there is no multi-turn primitive lower
in the stack to hand this off to.

Context size: the run's `result` block is used, NOT the full internal
`state` — state can carry raw per-candidate CSV data and every sub-agent's
intermediate output, which is large enough on its own to exceed the
smallest fallback model's tokens-per-minute limit (seen in practice: a
single chat turn requested ~8100 tokens against a 6000-8000 TPM cap and
every provider in the chain rejected it with 413). `result` is the fixed,
already-compact output contract, so it's used as-is, with a hard character
cap (_truncate) as a safety net against future growth (e.g. large candidate
pools).
"""
from __future__ import annotations

from core.logger import get_logger
from db import get_agent_run, save_agent_chat_message, get_agent_chat_history
from llm.engine import call_llm
from agents.chat_prompts import AGENT_CHAT_SYSTEM

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


def handle_chat_message(run_id: str, message: str, user_id: str) -> dict:
    """
    Answers one chat message grounded in a completed agent_runs row.
    Persists both the user message and the assistant reply on success.

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