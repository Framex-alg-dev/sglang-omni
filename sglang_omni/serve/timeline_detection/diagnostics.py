"""Bounded, metadata-only diagnostics for the private timeline service."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from .contracts import MediaKind, TimelineMediaChunk, TimelineSessionStart


logger = logging.getLogger(__name__)


def log_diagnostic(stage: str, **fields: Any) -> None:
    """Observability must never change media admission or model decisions."""
    try:
        logger.info(
            "timeline diagnostic %s",
            json.dumps({"stage": stage, **fields}, separators=(",", ":")),
        )
    except Exception:
        # Includes logging handlers failing; diagnostics are strictly best-effort.
        pass


class TimelineIngressDiagnostics:
    """Compare transport acceptance with assembler delivery without retaining media."""

    def __init__(
        self, start: TimelineSessionStart, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._identity = {
            "session_id": start.session_id,
            "observer_epoch": start.observer_epoch,
        }
        self._clock = clock
        self.reset(start.stream_epoch)

    def reset(self, stream_epoch: int) -> None:
        self._stream_epoch = stream_epoch
        self._counts = {"received_audio": 0, "received_video": 0,
                        "processed_audio": 0, "processed_video": 0}
        self._previous_counts = dict(self._counts)
        self._last: dict[str, dict[str, int | float]] = {}
        self._last_report_at = self._clock()

    def received(self, chunk: TimelineMediaChunk) -> None:
        self._mark("received", chunk)

    def processed(self, chunk: TimelineMediaChunk) -> None:
        self._mark("processed", chunk)

    def _mark(self, phase: str, chunk: TimelineMediaChunk) -> None:
        kind = "video" if chunk.kind is MediaKind.VIDEO else "audio"
        key = f"{phase}_{kind}"
        self._counts[key] += 1
        self._last[key] = {
            "sequence": chunk.sequence,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "at": self._clock(),
        }

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        record: dict[str, Any] = {
            **self._identity,
            "stream_epoch": self._stream_epoch,
            "sampled_at_monotonic_ms": round(now * 1000, 1),
            "interval_ms": round((now - self._last_report_at) * 1000, 1),
            **self._counts,
            "interval_counts": {key: value - self._previous_counts[key]
                                for key, value in self._counts.items()},
        }
        for key in self._counts:
            last = self._last.get(key)
            record[f"last_{key}"] = (
                {"sequence": last["sequence"], "start_ms": last["start_ms"],
                 "end_ms": last["end_ms"],
                 "age_ms": round(max(0, now - last["at"]) * 1000, 1)}
                if last is not None else None
            )
        self._previous_counts = dict(self._counts)
        self._last_report_at = now
        return record

    def report(self, *, reason: str, pending_chunks: int, pending_bytes: int) -> None:
        log_diagnostic(
            "ingress_progress", **self.snapshot(), reason=reason,
            pending_chunks=pending_chunks, pending_bytes=pending_bytes,
        )

    async def run(
        self, pending: Callable[[], tuple[int, int]], *, interval_seconds: float = 1.0
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("diagnostic interval must be positive")
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                chunks, size = pending()
                self.report(reason="interval", pending_chunks=chunks, pending_bytes=size)
            except Exception:
                # A failed diagnostic sample must not terminate the heartbeat.
                pass
