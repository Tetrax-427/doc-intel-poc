# tests/test_queue.py

import time
import threading
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(max_workers: int = 2):
    """Create a fresh TaskQueue instance (not the singleton) for each test."""
    from core.queue import TaskQueue
    q = TaskQueue(max_workers=max_workers)
    q.start()
    return q


def _wait_for(queue, task_id: str, timeout: float = 3.0) -> dict:
    """Poll until task reaches a terminal state or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = queue.get_status(task_id)
        if status and status["status"] in ("done", "failed"):
            return status
        time.sleep(0.05)
    return queue.get_status(task_id)


# ---------------------------------------------------------------------------
# test_submit_returns_task_id
# ---------------------------------------------------------------------------

def test_submit_returns_task_id():
    """submit() returns a non-empty UUID string immediately."""
    import uuid

    q = _make_queue()
    task_id = q.submit("test_task", lambda: {"result": "ok"})

    assert isinstance(task_id, str), "task_id must be a string"
    assert len(task_id) > 0,         "task_id must not be empty"

    # Should be a valid UUID
    try:
        uuid.UUID(task_id)
    except ValueError:
        pytest.fail(f"task_id is not a valid UUID: {task_id}")


def test_submit_returns_unique_ids():
    """Each submit() call returns a different task_id."""
    q = _make_queue()

    ids = {q.submit(f"task_{i}", lambda: {}) for i in range(10)}
    assert len(ids) == 10, "All task IDs must be unique"


def test_submit_task_appears_in_status_immediately():
    """Submitted task is retrievable via get_status() before it completes."""
    q = _make_queue(max_workers=0)  # no workers — task stays pending

    # Use a queue with no workers so the task stays pending
    from core.queue import TaskQueue
    q2 = TaskQueue(max_workers=0)
    # Don't start — no workers consuming the queue

    task_id = q2.submit("pending_task", lambda: {})
    status  = q2.get_status(task_id)

    assert status is not None
    assert status["task_id"] == task_id
    assert status["name"]    == "pending_task"
    assert status["status"]  in ("pending", "running", "done")


# ---------------------------------------------------------------------------
# test_task_moves_to_done
# ---------------------------------------------------------------------------

def test_task_moves_to_done():
    """A successfully completed task has status 'done' and result populated."""
    q = _make_queue()

    task_id = q.submit("simple_task", lambda: {"document_id": "doc-123", "chunks": 5})
    status  = _wait_for(q, task_id)

    assert status is not None
    assert status["status"]           == "done"
    assert status["result"]           == {"document_id": "doc-123", "chunks": 5}
    assert status["error"]            == ""
    assert status["started_at"]       != ""
    assert status["completed_at"]     != ""


def test_task_timestamps_are_populated():
    """done task has created_at, started_at, completed_at all set."""
    q = _make_queue()

    task_id = q.submit("ts_task", lambda: {})
    status  = _wait_for(q, task_id)

    assert status["created_at"]   != ""
    assert status["started_at"]   != ""
    assert status["completed_at"] != ""


def test_task_result_is_empty_dict_when_func_returns_none():
    """If the task function returns None, result is stored as {}."""
    q = _make_queue()

    task_id = q.submit("none_task", lambda: None)
    status  = _wait_for(q, task_id)

    assert status["status"] == "done"
    assert status["result"] == {}


def test_multiple_tasks_all_complete():
    """All submitted tasks eventually reach 'done' status."""
    q = _make_queue(max_workers=3)

    task_ids = [
        q.submit(f"task_{i}", lambda i=i: {"index": i})
        for i in range(10)
    ]

    for task_id in task_ids:
        status = _wait_for(q, task_id, timeout=5.0)
        assert status["status"] == "done", \
            f"Task {task_id} did not complete: {status}"


# ---------------------------------------------------------------------------
# test_unknown_task_returns_none
# ---------------------------------------------------------------------------

def test_unknown_task_returns_none():
    """get_status() returns None for a task_id that was never submitted."""
    q = _make_queue()

    result = q.get_status("00000000-0000-0000-0000-000000000000")
    assert result is None


def test_get_status_returns_correct_shape():
    """get_status() result contains all expected Contract 5 keys."""
    q = _make_queue()

    task_id = q.submit("shape_task", lambda: {"ok": True})
    status  = _wait_for(q, task_id)

    required_keys = [
        "task_id", "name", "status", "result",
        "error", "created_at", "started_at", "completed_at"
    ]
    for key in required_keys:
        assert key in status, f"Missing key in status response: {key}"

    assert status["task_id"] == task_id
    assert status["name"]    == "shape_task"


# ---------------------------------------------------------------------------
# test_failed_task_captures_error
# ---------------------------------------------------------------------------

def test_failed_task_captures_error():
    """A task that raises has status 'failed' and error message populated."""
    q = _make_queue()

    def failing_func():
        raise ValueError("Something went wrong during ingestion")

    task_id = q.submit("failing_task", failing_func)
    status  = _wait_for(q, task_id)

    assert status["status"] == "failed"
    assert "Something went wrong" in status["error"]
    assert status["result"] == {}


def test_failed_task_does_not_stop_worker():
    """Worker continues processing tasks after one fails."""
    q = _make_queue(max_workers=1)

    def fail(): raise RuntimeError("boom")
    def succeed(): return {"ok": True}

    fail_id    = q.submit("fail",    fail)
    succeed_id = q.submit("succeed", succeed)

    fail_status    = _wait_for(q, fail_id,    timeout=3.0)
    succeed_status = _wait_for(q, succeed_id, timeout=3.0)

    assert fail_status["status"]    == "failed"
    assert succeed_status["status"] == "done", \
        "Worker must continue after a failed task"


def test_failed_task_has_completed_at_set():
    """Failed task still has completed_at timestamp set."""
    q = _make_queue()

    task_id = q.submit("fail_ts", lambda: 1 / 0)  # ZeroDivisionError
    status  = _wait_for(q, task_id)

    assert status["status"]       == "failed"
    assert status["completed_at"] != ""


# ---------------------------------------------------------------------------
# test_task_queue_start_is_idempotent
# ---------------------------------------------------------------------------

def test_start_is_idempotent():
    """Calling start() multiple times does not spawn extra workers or raise."""
    from core.queue import TaskQueue

    q = TaskQueue(max_workers=2)
    q.start()
    q.start()  # second call must be a no-op
    q.start()  # third call must also be a no-op

    task_id = q.submit("idempotent_test", lambda: {"ok": True})
    status  = _wait_for(q, task_id)

    assert status["status"] == "done"


# ---------------------------------------------------------------------------
# test_pending_count
# ---------------------------------------------------------------------------

def test_pending_count_reflects_queue_depth():
    """pending_count() returns number of tasks waiting to be picked up."""
    from core.queue import TaskQueue

    # Queue with no workers — tasks pile up
    q = TaskQueue(max_workers=0)

    assert q.pending_count() == 0

    q.submit("t1", lambda: {})
    q.submit("t2", lambda: {})
    q.submit("t3", lambda: {})

    assert q.pending_count() == 3


# ---------------------------------------------------------------------------
# Concurrency smoke test
# ---------------------------------------------------------------------------

def test_concurrent_submits_all_complete():
    """
    Tasks submitted from multiple threads all complete successfully.
    No deadlocks, no lost tasks.
    """
    q      = _make_queue(max_workers=4)
    results = {}
    lock   = threading.Lock()

    def submit_and_store(i):
        task_id = q.submit(f"concurrent_{i}", lambda i=i: {"index": i})
        status  = _wait_for(q, task_id, timeout=5.0)
        with lock:
            results[i] = status["status"]

    threads = [threading.Thread(target=submit_and_store, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert all(s == "done" for s in results.values()), \
        f"Some tasks did not complete: {results}"


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

def test_singleton_is_importable():
    """The module-level task_queue singleton imports without error."""
    from core.queue import task_queue
    assert task_queue is not None


def test_singleton_start_and_submit():
    """The singleton task_queue can be started and accepts tasks."""
    from core.queue import task_queue

    task_queue.start()  # idempotent — safe even if already started

    task_id = task_queue.submit("singleton_test", lambda: {"singleton": True})
    status  = _wait_for(task_queue, task_id, timeout=3.0)

    assert status["status"] == "done"
    assert status["result"] == {"singleton": True}