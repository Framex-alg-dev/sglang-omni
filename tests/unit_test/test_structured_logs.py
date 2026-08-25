from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from sglang_omni.utils.structured_logs import PartitionedJSONLWriter


def test_structured_logs_partition_by_hour_type_pid_and_size(tmp_path) -> None:
    writer = PartitionedJSONLWriter(
        tmp_path,
        max_queue_size=32,
        max_file_bytes=450,
        timezone=ZoneInfo("UTC"),
    )
    try:
        assert writer.emit(
            "lifecycle",
            "turn_started",
            session_id="session-1",
            turn_id="turn-1",
            trace_id="trace-1",
            detail="x" * 200,
        )
        assert writer.emit(
            "lifecycle",
            "turn_completed",
            session_id="session-1",
            turn_id="turn-1",
            trace_id="trace-1",
            detail="y" * 200,
        )
        assert writer.emit(
            "performance",
            "turn_timing",
            session_id="session-1",
            turn_id="turn-1",
            total_after_commit_ms=123.4,
        )
        assert writer.emit(
            "resource",
            "resource_sample",
            application={"active_session_count": 1},
            gpus=[{"gpu_utilization_percent": 75}],
        )
        writer.flush()

        hour = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d/%H")
        directory = tmp_path / hour
        lifecycle_files = sorted(directory.glob("lifecycle_api_*_*.jsonl"))
        performance_files = sorted(directory.glob("performance_api_*_*.jsonl"))
        resource_files = sorted(directory.glob("resource_api_*_*.jsonl"))
        assert len(lifecycle_files) == 2
        assert len(performance_files) == 1
        assert len(resource_files) == 1

        records = []
        for path in [*lifecycle_files, *performance_files, *resource_files]:
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        assert {record["event"] for record in records} == {
            "turn_started",
            "turn_completed",
            "turn_timing",
            "resource_sample",
        }
        assert all(record["schema_version"] == "1.0" for record in records)
        assert all(record["pid"] == writer.pid for record in records)
        assert writer.health()["written_records"] == 4
        assert writer.health()["dropped_records"] == 0
    finally:
        writer.close()


def test_structured_log_rejects_unknown_type(tmp_path) -> None:
    writer = PartitionedJSONLWriter(tmp_path)
    try:
        try:
            writer.emit("unknown", "event")
        except ValueError as exc:
            assert "unsupported structured log type" in str(exc)
        else:
            raise AssertionError("unknown log type was accepted")
    finally:
        writer.close()
