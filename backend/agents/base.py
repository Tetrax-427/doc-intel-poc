"""
backend/agents/base.py
------------------------
Shared execution engine for all agents (CV Processor, CA Helper, ...).

An agent is defined as a `stages: dict[str, StageFn]` mapping stage name ->
plain Python function. Every stage function has the signature:

    def stage_fn(run_id: str, state: dict) -> StageOutcome

StageOutcome is a dict with:
    "state_update": dict       - merged into `state` before the next stage
                                  (or before persisting, if this is the final stage)
    "next_stage":   str | None - name of the next stage to run, or None if this
                                  was the last stage (run is now "completed")
    "questions":    list | None - if set, the run pauses here: status becomes
                                  "needs_input", pending_questions is persisted,
                                  and the stage loop stops WITHOUT advancing
                                  next_stage. Resuming re-invokes the SAME stage
                                  (current_stage unchanged) after merging the
                                  caller's answers into state["user_answers"].
    "result":       dict | None - only meaningful on the final stage (next_stage
                                  is None) - this becomes agent_runs.result and
                                  must match the fixed output contract:
                                  {"summary": str, "findings": [...],
                                   "data": [{"type","label","value"}, ...],
                                   "extra": {...}}

Design note on resume: stage functions are expected to be self-checking -
a stage that needs clarification looks for its answer under a known key in
state["user_answers"] first, and only returns `questions` if that key is
still missing. This keeps resume logic entirely in base.py (merge answers,
re-call the same stage) with no special-cased "which question was this"
bookkeeping required from callers.
"""
from __future__ import annotations

from typing import Callable, TypedDict

from core.logger import get_logger
from db import (
    create_agent_run,
    get_agent_run,
    mark_agent_run_started,
    mark_agent_run_needs_input,
    mark_agent_run_completed,
    mark_agent_run_failed,
    update_agent_run,
)

logger = get_logger("agents.base")


class StageOutcome(TypedDict, total=False):
    state_update: dict
    next_stage: str | None
    questions: list
    result: dict


StageFn = Callable[[str, dict], StageOutcome]


class AgentRunError(Exception):
    """Raised (and caught internally) when a stage function itself raises - wraps the
    original exception so mark_agent_run_failed() always gets a readable message."""


def _run_stage_loop(run_id: str, user_id: str, stages: dict[str, StageFn], start_stage: str, state: dict) -> None:
    """
    Drives the state machine forward from `start_stage` until a stage pauses
    (questions), completes (next_stage is None), or raises. Persists progress
    after every stage so a crash mid-run leaves agent_runs at the last
    successfully completed stage, not silently stuck at "running".
    """
    current_stage = start_stage

    while True:
        stage_fn = stages.get(current_stage)
        if stage_fn is None:
            mark_agent_run_failed(run_id, state, f"Unknown stage '{current_stage}' - no such stage registered.")
            return

        logger.info("Agent stage starting", run_id=run_id, stage=current_stage)
        try:
            outcome = stage_fn(run_id, state)
        except Exception as exc:
            logger.error("Agent stage raised", run_id=run_id, stage=current_stage, error=str(exc))
            mark_agent_run_failed(run_id, state, f"Stage '{current_stage}' failed: {exc}")
            return

        state = {**state, **outcome.get("state_update", {})}

        questions = outcome.get("questions")
        if questions:
            logger.info("Agent run paused for input", run_id=run_id, stage=current_stage, n_questions=len(questions))
            mark_agent_run_needs_input(run_id, current_stage, state, questions)
            return

        next_stage = outcome.get("next_stage")
        if next_stage is None:
            result = outcome.get("result")
            if result is None:
                mark_agent_run_failed(
                    run_id, state,
                    f"Stage '{current_stage}' ended the run (next_stage=None) without a 'result'.",
                )
                return
            logger.info("Agent run completed", run_id=run_id, final_stage=current_stage)
            mark_agent_run_completed(run_id, state, result)
            return

        # Persist progress between stages so a mid-run crash is recoverable/inspectable.
        update_agent_run(run_id, current_stage=next_stage, state=state)
        current_stage = next_stage


def create_and_queue_run(
    agent_name: str,
    task: str,
    input_data: dict,
    user_id: str = "anonymous",
) -> str:
    """
    Fast, synchronous step: creates the agent_runs row and returns run_id
    immediately. Does NOT execute any stages — callers (typically a router)
    are expected to hand execute_stages() to core.queue.task_queue.submit()
    right after this returns, so the HTTP response doesn't block on the run.
    """
    return create_agent_run(agent_name, task, input_data, user_id=user_id)


def execute_stages(run_id: str, stages: dict[str, StageFn], first_stage: str, user_id: str = "anonymous") -> dict:
    """
    The actual work — meant to be run in a background thread via
    core.queue.task_queue.submit(f"agent:{agent_name}", execute_stages, run_id, stages, first_stage, user_id=uid).
    Returns a dict (task_queue.Task.result expects one) - just an
    acknowledgement, since the real result lives in agent_runs.result;
    the UI polls GET /agents/runs/{run_id}, not the task_queue's own status.
    """
    mark_agent_run_started(run_id)
    initial_state = {"task": "", "input_data": {}, "user_answers": {}, "_user_id": user_id}
    run = get_agent_run(run_id, user_id=user_id)
    if run:
        initial_state["task"] = run.get("task", "")
        initial_state["input_data"] = run.get("input_data") or {}
    _run_stage_loop(run_id, user_id, stages, first_stage, initial_state)
    return {"run_id": run_id}


def start_agent_run(
    agent_name: str,
    task: str,
    input_data: dict,
    stages: dict[str, StageFn],
    first_stage: str,
    user_id: str = "anonymous",
) -> str:
    """
    Convenience wrapper that runs the full stage loop SYNCHRONOUSLY in the
    calling thread - useful for scripts/tests. HTTP callers should NOT use
    this directly (it blocks); use create_and_queue_run() + task_queue.submit
    (execute_stages) instead, as routers/agents.py does.
    """
    run_id = create_and_queue_run(agent_name, task, input_data, user_id=user_id)
    execute_stages(run_id, stages, first_stage, user_id=user_id)
    return run_id


def accept_resume_answers(run_id: str, answers: dict, user_id: str = "anonymous") -> None:
    """
    Fast, synchronous step: validates the run is actually awaiting input,
    merges `answers` into state["user_answers"], flips status back to
    "running". Callers should follow this with
    task_queue.submit(f"agent:resume", resume_stages, run_id, stages, user_id=uid)
    to actually continue the stage loop in the background.
    """
    run = get_agent_run(run_id, user_id=user_id)
    if run is None:
        raise AgentRunError(f"agent_run '{run_id}' not found for this user.")
    if run["status"] != "needs_input":
        raise AgentRunError(f"agent_run '{run_id}' is not awaiting input (status={run['status']}).")

    state = dict(run.get("state") or {})
    state["user_answers"] = {**state.get("user_answers", {}), **answers}
    update_agent_run(run_id, status="running", state=state, pending_questions=None)


def resume_stages(run_id: str, stages: dict[str, StageFn], user_id: str = "anonymous") -> dict:
    """Background-queueable: re-enters the stage loop at current_stage after accept_resume_answers()."""
    run = get_agent_run(run_id, user_id=user_id)
    if run is None:
        raise AgentRunError(f"agent_run '{run_id}' not found for this user.")
    state = run.get("state") or {}
    _run_stage_loop(run_id, user_id, stages, run["current_stage"], state)
    return {"run_id": run_id}


def resume_agent_run(
    run_id: str,
    answers: dict,
    stages: dict[str, StageFn],
    user_id: str = "anonymous",
) -> None:
    """
    Convenience wrapper that resumes SYNCHRONOUSLY in the calling thread -
    useful for scripts/tests. HTTP callers should use accept_resume_answers()
    + task_queue.submit(resume_stages) instead, as routers/agents.py does.
    """
    accept_resume_answers(run_id, answers, user_id=user_id)
    resume_stages(run_id, stages, user_id=user_id)