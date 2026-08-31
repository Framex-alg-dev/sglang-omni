"""Bounded asynchronous scheduling for session-memory extraction."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Literal


class SessionMemoryScheduler:
    """Fair scheduler with one active extraction per session key.

    A key occurs at most once in the queued/running set. A worker returns
    ``True`` when more turns arrived while it ran; that key is then placed at
    the fair tail without creating an unbounded task backlog.
    """

    def __init__(
        self,
        *,
        max_queued_sessions: int = 256,
        max_concurrent_extractions: int = 1,
    ) -> None:
        self.max_queued_sessions = max(1, max_queued_sessions)
        self.max_concurrent_extractions = max(1, max_concurrent_extractions)
        self._queue: deque[tuple[str, Callable[[], Awaitable[bool]]]] = deque()
        self._active_keys: set[str] = set()
        self._dirty_keys: set[str] = set()
        self._cancelled_keys: set[str] = set()
        self._running_tasks: dict[str, asyncio.Task[bool]] = {}
        self._running_factories: dict[str, Callable[[], Awaitable[bool]]] = {}
        self._drainer: asyncio.Task[None] | None = None
        self._rejected_submission_count = 0

    def submit(self, key: str, factory: Callable[[], Awaitable[bool]]) -> bool:
        return self.submit_status(key, factory) == "submitted"

    def submit_status(
        self, key: str, factory: Callable[[], Awaitable[bool]]
    ) -> Literal["submitted", "coalesced", "rejected"]:
        if key in self._cancelled_keys:
            self._rejected_submission_count += 1
            return "rejected"
        if key in self._active_keys:
            self._dirty_keys.add(key)
            return "coalesced"
        if len(self._queue) >= self.max_queued_sessions:
            self._rejected_submission_count += 1
            return "rejected"
        self._active_keys.add(key)
        self._queue.append((key, factory))
        if self._drainer is None or self._drainer.done():
            self._drainer = asyncio.create_task(
                self._drain(), name="session-memory-scheduler"
            )
        return "submitted"

    async def cancel(self, key: str) -> None:
        self._cancelled_keys.add(key)
        if self._queue:
            self._queue = deque(
                (queued_key, factory)
                for queued_key, factory in self._queue
                if queued_key != key
            )
        task = self._running_tasks.get(key)
        was_running = task is not None and not task.done()
        if was_running:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._dirty_keys.discard(key)
        if self._queue:
            self._queue = deque(
                (queued_key, factory)
                for queued_key, factory in self._queue
                if queued_key != key
            )
        if not was_running:
            self._running_tasks.pop(key, None)
            self._running_factories.pop(key, None)
            self._active_keys.discard(key)
            self._cancelled_keys.discard(key)

    async def wait_idle(self) -> None:
        drainer = self._drainer
        if drainer is not None:
            await asyncio.shield(drainer)

    def snapshot(self) -> dict[str, int]:
        return {
            "queued_session_count": len(self._queue),
            "running_job_count": len(self._running_tasks),
            "dirty_session_count": len(self._dirty_keys),
            "rejected_submission_count": self._rejected_submission_count,
            "max_queued_sessions": self.max_queued_sessions,
            "max_concurrent_extractions": self.max_concurrent_extractions,
        }

    def prioritize(self, key: str) -> bool:
        """Move one queued session to the front for bounded R1 catch-up."""

        for index, (queued_key, factory) in enumerate(self._queue):
            if queued_key != key:
                continue
            del self._queue[index]
            self._queue.appendleft((queued_key, factory))
            return True
        return key in self._running_tasks

    async def _drain(self) -> None:
        while self._queue or self._running_tasks:
            self._start_available_jobs()
            if not self._running_tasks:
                continue
            done, _ = await asyncio.wait(
                tuple(self._running_tasks.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                self._complete_job(task)

    def _start_available_jobs(self) -> None:
        while (
            self._queue
            and len(self._running_tasks) < self.max_concurrent_extractions
        ):
            key, factory = self._queue.popleft()
            task = asyncio.create_task(factory(), name=f"session-memory:{key}")
            self._running_tasks[key] = task
            self._running_factories[key] = factory

    def _complete_job(self, task: asyncio.Task[bool]) -> None:
        key = next(
            (
                running_key
                for running_key, running_task in self._running_tasks.items()
                if running_task is task
            ),
            None,
        )
        if key is None:
            return
        factory = self._running_factories.pop(key)
        reschedule = False
        try:
            reschedule = task.result()
        except (asyncio.CancelledError, Exception):
            # The owning session logs extraction details. One worker must not
            # terminate scheduling for other sessions.
            pass
        self._running_tasks.pop(key, None)
        notified_while_active = key in self._dirty_keys
        self._dirty_keys.discard(key)
        self._active_keys.discard(key)
        if (
            (reschedule or notified_while_active)
            and key not in self._cancelled_keys
        ):
            self._active_keys.add(key)
            self._queue.append((key, factory))
        self._cancelled_keys.discard(key)
