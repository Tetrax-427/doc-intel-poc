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

CHANGED (resume reshape): resume now handles two distinct triggers under one
entrypoint, accept_resume_input():

  1. status == "needs_input", answers given
     Unchanged from before (was accept_resume_answers). There IS a defined
     resume point here - current_stage, set when the stage originally
     returned `questions` - so no planning is needed. Answers merge into
     state["user_answers"], status flips to "running", current_stage is
     untouched, and the loop re-enters the SAME stage.

  2. status == "completed", new_input given (e.g. a new document_id added
     after the run already finished)
     There is NO defined resume point for this case - a completed run's
     current_stage is whatever stage happened to finish it (e.g.
     "summarize"), which is the wrong place to re-enter after new data
     shows up mid-way through the pipeline (e.g. a new doc affects
     "stitch" onward, not just "summarize"). So a planner
     (plan_resume_stage) decides the entry stage instead of base.py
     hardcoding one - this generalises across every agent, not just ITR.
     new_input merges into state["input_data"], the planner picks a stage
     name from the CALLER-SUPPLIED stage_descriptions (base.py does not
     import agents.registry itself, to avoid a circular import - the
     router already loads agent_def and passes stages/first_stage today,
     so passing stage_descriptions the same way is consistent), and the
     loop re-enters at that stage.

     If the agent has no stage_descriptions registered, this raises
     AgentRunError rather than silently falling back to first_stage or
     current_stage - re-running an unknown/wrong stage on financial or
     otherwise sensitive recalculated data is worse than a clear failure
     the caller can surface to the user.

  Both triggers can be present in the same call (answers AND new_input);
  they're handled independently - if both are given, needs_input takes
  precedence (case 1), since new_input in state["input_data"] simply gets
  merged in before the stage loop resumes and picked up whenever that
  stage next runs.

CHANGED (output-contract validation guardrail): when a stage ends the run
(next_stage is None), the returned `result` is now checked for the fixed
output contract's required top-level keys (summary, findings, data, extra)
before being persisted as completed. A stage that ends the run without one
of these fails the run loudly (mark_agent_run_failed) instead of silently
persisting a malformed result - this is agent-agnostic (lives here, not in
any one agent), so every agent's output is held to the same contract.
"""
from __future__ import annotations

from typing import Callable, TypedDict

from pydantic import BaseModel, Field

from core.logger import get_logger
from llm.engine import call_llm
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

# Every agent's finalize stage must return a result matching this shape -
# checked generically here so no individual agent can accidentally (or via
# a prompt-injected instruction) skip part of the fixed output contract.
REQUIRED_RESULT_KEYS = {"summary", "findings", "data", "extra"}


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

            missing_keys = REQUIRED_RESULT_KEYS - result.keys()
            if missing_keys:
                logger.error(
                    "Agent run result failed output-contract validation",
                    run_id=run_id, final_stage=current_stage, missing_keys=list(missing_keys),
                )
                mark_agent_run_failed(
                    run_id, state,
                    f"Stage '{current_stage}' returned a result missing required keys: {sorted(missing_keys)}.",
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
    name: str | None = None,
) -> str:
    """
    Fast, synchronous step: creates the agent_runs row and returns run_id
    immediately. Does NOT execute any stages — callers (typically a router)
    are expected to hand execute_stages() to core.queue.task_queue.submit()
    right after this returns, so the HTTP response doesn't block on the run.
 
    name: optional display name set at invoke time (see routers/agents.py
    InvokeAgentRequest.name). Purely cosmetic — never read by the stage
    machine itself. UI falls back to showing the run's id when this is None.
    """
    return create_agent_run(agent_name, task, input_data, user_id=user_id, name=name)

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


# ---------------------------------------------------------------------------
# Resume planning (completed + new_input case)
# ---------------------------------------------------------------------------

class _StageSelection(BaseModel):
    """Instructor response_model for plan_resume_stage() — same structured-
    output path used everywhere else in this codebase (see llm/structured.py),
    just a different shape than extraction/tool-calling response models."""
    stage_name: str = Field(
        description="The stage name to re-enter at. Must exactly match one of "
                    "the provided candidate stage names — do not invent a new one."
    )
    reasoning: str = Field(
        description="One sentence explaining why this stage is the correct "
                    "re-entry point given what changed."
    )


_RESUME_PLANNER_SYSTEM = (
    "You are deciding where a multi-stage agent pipeline should re-enter "
    "after new input was added to an already-completed run. You will be "
    "given the pipeline's stages (in order) with a description of what each "
    "does, a summary of the run's current state, and what new input was "
    "just added. Pick the EARLIEST stage whose output could change because "
    "of the new input — anything before that stage is unaffected and safe "
    "to skip re-running; anything from your chosen stage onward will be "
    "re-run in order. Respond with exactly one of the given stage names."
)


def plan_resume_stage(
    stage_descriptions: dict[str, str],
    state: dict,
    new_input: dict,
) -> str:
    """
    Picks the stage to re-enter at for a completed run that just received
    new_input, via an LLM call grounded in the agent's own stage
    descriptions. Agent-agnostic — any agent that registers
    stage_descriptions in agents/registry.py can use this.

    Raises AgentRunError if the model's answer isn't one of the known stage
    names (fed back once is out of scope for v1 — a bad pick here fails
    loudly rather than silently re-running the wrong stage on, e.g.,
    financial data).
    """
    if not stage_descriptions:
        raise AgentRunError(
            "This agent has no stage_descriptions registered — cannot plan a "
            "resume entry point for a completed run with new input."
        )

    stages_block = "\n".join(f"- {name}: {desc}" for name, desc in stage_descriptions.items())

    user_prompt = (
        f"--- Pipeline stages ---\n{stages_block}\n\n"
        f"--- Current state (summary) ---\n{state.get('input_data', {})}\n\n"
        f"--- New input just added ---\n{new_input}"
    )

    selection: _StageSelection = call_llm(
        system=_RESUME_PLANNER_SYSTEM,
        user=user_prompt,
        temperature=0.0,
        call_type="agent_resume_planner",
        response_model=_StageSelection,
        structured_max_retries=1,
        user_id=state.get("_user_id", "system"),
    )

    if selection.stage_name not in stage_descriptions:
        raise AgentRunError(
            f"Resume planner picked an unknown stage '{selection.stage_name}' — "
            f"expected one of {list(stage_descriptions.keys())}."
        )

    logger.info(
        "Resume planner picked entry stage",
        stage=selection.stage_name, reasoning=selection.reasoning,
    )
    return selection.stage_name


def accept_resume_input(
    run_id: str,
    user_id: str = "anonymous",
    answers: dict | None = None,
    new_input: dict | None = None,
    stage_descriptions: dict[str, str] | None = None,
) -> None:
    """
    Fast, synchronous step — validates the run and prepares it to continue,
    but does NOT run any stages itself. Callers should follow this with
    task_queue.submit(f"agent:resume", resume_stages, run_id, stages, user_id=uid)
    to actually continue the stage loop in the background, same as before.

    Branches on the run's current status:

      status == "needs_input" (answers given):
        Unchanged behaviour — merges answers into state["user_answers"],
        flips status to "running", current_stage untouched.

      status == "completed" (new_input given):
        Merges new_input into state["input_data"], asks plan_resume_stage()
        for the entry stage (using the caller-supplied stage_descriptions),
        sets current_stage to that stage, flips status to "running".

      Anything else (e.g. status == "running", or a mismatched
      trigger/status pairing — needs_input with no answers, or completed
      with no new_input): raises AgentRunError.
    """
    run = get_agent_run(run_id, user_id=user_id)
    if run is None:
        raise AgentRunError(f"agent_run '{run_id}' not found for this user.")

    if run["status"] == "needs_input":
        if not answers:
            raise AgentRunError(
                f"agent_run '{run_id}' is awaiting input — 'answers' is required to resume it."
            )
        state = dict(run.get("state") or {})
        state["user_answers"] = {**state.get("user_answers", {}), **answers}
        update_agent_run(run_id, status="running", state=state, pending_questions=None)
        return

    if run["status"] == "completed":
        if not new_input:
            raise AgentRunError(
                f"agent_run '{run_id}' has already completed — 'new_input' is required to resume it."
            )
        state = dict(run.get("state") or {})
        state["input_data"] = {**state.get("input_data", {}), **new_input}

        entry_stage = plan_resume_stage(stage_descriptions or {}, state, new_input)

        update_agent_run(run_id, status="running", current_stage=entry_stage, state=state)
        return

    raise AgentRunError(
        f"agent_run '{run_id}' cannot be resumed from status={run['status']!r} — "
        f"only 'needs_input' (with answers) or 'completed' (with new_input) are resumable."
    )


def resume_stages(run_id: str, stages: dict[str, StageFn], user_id: str = "anonymous") -> dict:
    """Background-queueable: re-enters the stage loop at current_stage after accept_resume_input()."""
    run = get_agent_run(run_id, user_id=user_id)
    if run is None:
        raise AgentRunError(f"agent_run '{run_id}' not found for this user.")
    state = run.get("state") or {}
    _run_stage_loop(run_id, user_id, stages, run["current_stage"], state)
    return {"run_id": run_id}


def resume_agent_run(
    run_id: str,
    stages: dict[str, StageFn],
    user_id: str = "anonymous",
    answers: dict | None = None,
    new_input: dict | None = None,
    stage_descriptions: dict[str, str] | None = None,
) -> None:
    """
    Convenience wrapper that resumes SYNCHRONOUSLY in the calling thread -
    useful for scripts/tests. HTTP callers should use accept_resume_input()
    + task_queue.submit(resume_stages) instead, as routers/agents.py does.
    """
    accept_resume_input(
        run_id, user_id=user_id, answers=answers,
        new_input=new_input, stage_descriptions=stage_descriptions,
    )
    resume_stages(run_id, stages, user_id=user_id)