# backend/core/queue.py

import threading
import uuid
from queue import Queue, Empty
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any

from core.logger import get_logger

logger = get_logger("queue")


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    Represents a single async unit of work.

    lifecycle: pending → running → done | failed
    """
    id:           str
    name:         str
    func:         Callable
    args:         tuple
    kwargs:       dict
    status:       str = "pending"               # pending | running | done | failed
    result:       dict = field(default_factory=dict)
    error:        str = ""
    created_at:   str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at:   str = ""
    completed_at: str = ""


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------

class TaskQueue:
    """
    In-process async task queue backed by Python threads.

    Callers submit a function + args and receive a task_id immediately.
    They poll GET /tasks/{task_id} for status and result.


    """

    def __init__(self, max_workers: int = 3):
        self._queue:   Queue          = Queue()
        self._tasks:   dict[str, Task] = {}
        self._lock:    threading.RLock = threading.RLock()
        self.max_workers = max_workers
        self._started = False

    def start(self):
        """
        Start background worker threads.
        Call once on server startup (in main.py lifespan or startup event).
        Calling more than once is safe — subsequent calls are no-ops.
        """
        if self._started:
            return
        for i in range(self.max_workers):
            t = threading.Thread(
                target=self._worker,
                daemon=True,
                name=f"docintel-worker-{i}",
            )
            t.start()
        self._started = True
        logger.info("Task queue started", workers=self.max_workers)

    def submit(self, name: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for async execution.

        Args:
            name:   Human-readable task name for logging and status responses.
            func:   Callable to execute in a worker thread.
            *args:  Positional args forwarded to func.
            **kwargs: Keyword args forwarded to func.

        Returns:
            task_id (UUID string) — use with get_status() to poll.
        """
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
        )
        with self._lock:
            self._tasks[task_id] = task
        self._queue.put(task_id)
        logger.info("Task submitted", task_id=task_id, name=name)
        return task_id

    def get_status(self, task_id: str) -> dict | None:
        """
        Return current status of a task, or None if task_id not found.

        Returns a Contract 5 compatible dict:
        {
            "task_id":      "...",
            "name":         "ingest_file",
            "status":       "done",        # pending | running | done | failed
            "result":       {...},          # populated when done
            "error":        "",            # populated when failed
            "created_at":   "...",
            "started_at":   "...",
            "completed_at": "..."
        }
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id":      task.id,
            "name":         task.name,
            "status":       task.status,
            "result":       task.result,
            "error":        task.error,
            "created_at":   task.created_at,
            "started_at":   task.started_at,
            "completed_at": task.completed_at,
        }

    def pending_count(self) -> int:
        """Return number of tasks currently waiting in the queue."""
        return self._queue.qsize()

    def all_statuses(self) -> list[dict]:
        """
        Return status of all tracked tasks.
        Useful for an admin /tasks endpoint.
        """
        with self._lock:
            task_ids = list(self._tasks.keys())
        return [self.get_status(tid) for tid in task_ids]

    # -----------------------------------------------------------------------
    # Internal worker loop
    # -----------------------------------------------------------------------

    def _worker(self):
        """
        Worker thread — runs forever, pulling task IDs from the queue.
        Each task is executed synchronously within the worker.
        Failures are caught and recorded — the worker never dies on error.
        """
        while True:
            try:
                task_id = self._queue.get(timeout=1)
            except Empty:
                continue  # no work — loop back

            with self._lock:
                task = self._tasks.get(task_id)

            if not task:
                self._queue.task_done()
                continue

            # Mark running
            task.status     = "running"
            task.started_at = datetime.utcnow().isoformat()
            logger.info("Task started", task_id=task_id, name=task.name)

            try:
                result = task.func(*task.args, **task.kwargs)
                task.status       = "done"
                task.result       = result if isinstance(result, dict) else {}
                task.completed_at = datetime.utcnow().isoformat()
                logger.info(
                    "Task completed",
                    task_id=task_id,
                    name=task.name,
                    duration_s=round(
                        (datetime.fromisoformat(task.completed_at) -
                         datetime.fromisoformat(task.started_at)).total_seconds(), 2
                    )
                )
            except Exception as e:
                task.status       = "failed"
                task.error        = str(e)
                task.completed_at = datetime.utcnow().isoformat()
                logger.error(
                    "Task failed",
                    task_id=task_id,
                    name=task.name,
                    error=str(e),
                )
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

task_queue = TaskQueue(max_workers=3)