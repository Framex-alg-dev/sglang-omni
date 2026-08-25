from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from sglang_omni.serve import openai_api
from sglang_omni.utils import resource_monitor
from sglang_omni.utils.resource_monitor import (
    PeriodicResourceMonitor,
    collect_resource_snapshot,
    resource_log_interval_s,
)


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def __init__(self) -> None:
        self.requested_indices: list[int] = []
        self.shutdown_count = 0

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        self.shutdown_count += 1

    def nvmlDeviceGetCount(self) -> int:
        return 8

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        self.requested_indices.append(index)
        return f"gpu-{index}"

    def nvmlDeviceGetUUID(self, handle: str) -> str:
        return "GPU-test"

    def nvmlDeviceGetName(self, handle: str) -> str:
        return "Test GPU"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        return SimpleNamespace(total=1000, used=600, free=400)

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> SimpleNamespace:
        return SimpleNamespace(gpu=75, memory=30)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: str) -> list[SimpleNamespace]:
        return [SimpleNamespace(pid=999999, usedGpuMemory=250)]

    def nvmlDeviceGetTemperature(self, handle: str, sensor: int) -> int:
        return 61

    def nvmlDeviceGetPowerUsage(self, handle: str) -> int:
        return 123000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle: str) -> int:
        return 300000

    def nvmlDeviceGetClockInfo(self, handle: str, clock: int) -> int:
        return 2100 if clock == self.NVML_CLOCK_SM else 3000


def test_resource_log_interval() -> None:
    assert resource_log_interval_s({}) == 0.0
    assert resource_log_interval_s({"SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S": "1.5"}) == 1.5
    assert resource_log_interval_s({"SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S": "-1"}) == 0.0
    assert resource_log_interval_s({"SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S": "bad"}) == 0.0


def test_collect_resource_snapshot_maps_visible_physical_gpu(tmp_path) -> None:
    nvml = FakeNVML()
    snapshot = collect_resource_snapshot(
        env={
            "CUDA_VISIBLE_DEVICES": "5",
            "SGLANG_OMNI_REALTIME_LOG_DIR": str(tmp_path),
        },
        pynvml_module=nvml,
    )

    assert nvml.requested_indices == [5]
    assert nvml.shutdown_count == 1
    assert snapshot["collection_errors"] == []
    gpu = snapshot["gpus"][0]
    assert gpu["logical_index"] == 0
    assert gpu["physical_device"] == "5"
    assert gpu["gpu_utilization_percent"] == 75
    assert gpu["memory_used_bytes"] == 600
    assert gpu["power_usage_watts"] == 123.0
    assert gpu["compute_process_memory_bytes"] == 250
    assert snapshot["api_process"]["rss_bytes"] is not None
    assert snapshot["disk"]["free_bytes"] > 0


def test_collect_resource_snapshot_without_nvml(tmp_path) -> None:
    snapshot = collect_resource_snapshot(
        env={"SGLANG_OMNI_REALTIME_LOG_DIR": str(tmp_path)},
        pynvml_module=None,
    )
    assert snapshot["gpus"] == []
    assert snapshot["collection_errors"] == ["pynvml is unavailable"]


def test_register_resource_monitor_uses_router_lifecycle(monkeypatch) -> None:
    app = FastAPI()
    app.state.client = SimpleNamespace()
    app.state.model_name = "test-model"
    manager = SimpleNamespace(
        resource_sample_requester=None,
        set_resource_sample_requester=lambda requester: setattr(
            manager, "resource_sample_requester", requester
        ),
    )
    app.state.multimodal_realtime_manager = manager
    monkeypatch.setattr(openai_api, "resource_log_interval_s", lambda: 1.0)

    openai_api._register_resource_monitor(app)

    assert app.state.resource_monitor is not None
    assert manager.resource_sample_requester == app.state.resource_monitor.request_sample
    assert len(app.router.on_startup) == 1
    assert len(app.router.on_shutdown) == 1


def test_register_resource_monitor_handles_nested_fastapi_router(monkeypatch) -> None:
    app = FastAPI()
    app.state.client = SimpleNamespace()
    app.state.model_name = "test-model"
    app.router = FastAPI()
    monkeypatch.setattr(openai_api, "resource_log_interval_s", lambda: 1.0)

    openai_api._register_resource_monitor(app)

    assert len(app.router.router.on_startup) == 1
    assert len(app.router.router.on_shutdown) == 1


@pytest.mark.asyncio
async def test_periodic_resource_monitor_emits_samples(monkeypatch) -> None:
    records: list[tuple[str, str, dict]] = []

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    def fake_emit(log_type: str, event: str, **fields) -> bool:
        records.append((log_type, event, fields))
        return True

    monkeypatch.setattr(resource_monitor.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(resource_monitor, "emit_structured_log", fake_emit)
    monitor = PeriodicResourceMonitor(
        0.01,
        application_snapshot=lambda: {"active_session_count": 2},
        collector=lambda: {
            "host": {},
            "api_process": {},
            "gpus": [],
            "disk": {},
            "collection_errors": [],
        },
    )
    monitor.start()
    await asyncio.sleep(0.035)
    await monitor.stop()

    events = [event for _, event, _ in records]
    assert events[0] == "resource_monitor_started"
    assert "resource_sample" in events
    assert events[-1] == "resource_monitor_stopped"
    sample = next(fields for _, event, fields in records if event == "resource_sample")
    assert sample["sample_trigger"] == "periodic"
    assert isinstance(sample["sample_requested_unix_ms"], int)
    assert sample["sample_queue_wait_ms"] >= 0
    assert sample["application"] == {"active_session_count": 2}
    assert "cpu_percent" in sample["api_process"]


@pytest.mark.asyncio
async def test_immediate_turn_samples_are_nonblocking_serial_and_paired(
    monkeypatch,
) -> None:
    records: list[tuple[str, str, dict]] = []
    active_collectors = 0
    max_active_collectors = 0

    async def fake_to_thread(function, *args, **kwargs):
        nonlocal active_collectors, max_active_collectors
        active_collectors += 1
        max_active_collectors = max(max_active_collectors, active_collectors)
        await asyncio.sleep(0.005)
        try:
            return function(*args, **kwargs)
        finally:
            active_collectors -= 1

    def fake_emit(log_type: str, event: str, **fields) -> bool:
        records.append((log_type, event, fields))
        return True

    monkeypatch.setattr(resource_monitor.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(resource_monitor, "emit_structured_log", fake_emit)
    monitor = PeriodicResourceMonitor(
        60.0,
        application_snapshot=lambda: {"active_turn_count": 1},
        collector=lambda: {
            "host": {},
            "api_process": {},
            "gpus": [],
            "disk": {},
            "collection_errors": [],
        },
    )

    assert monitor.request_sample("turn_before_inference") is False
    monitor.start()
    assert monitor.request_sample(
        "turn_before_inference",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
    )
    assert monitor.request_sample(
        "turn_after_terminal",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
        turn_outcome="completed",
    )
    await monitor.stop()

    samples = [
        fields for _, event, fields in records if event == "resource_sample"
    ]
    turn_samples = [
        fields for fields in samples if fields["sample_trigger"].startswith("turn_")
    ]
    assert [fields["sample_trigger"] for fields in turn_samples] == [
        "turn_before_inference",
        "turn_after_terminal",
    ]
    assert max_active_collectors == 1
    assert turn_samples[0]["session_id"] == "session-1"
    assert turn_samples[0]["turn_id"] == "turn-1"
    assert turn_samples[0]["api_process"]["cpu_percent"] is None
    assert turn_samples[1]["api_process"]["cpu_percent"] is not None
    assert turn_samples[1]["turn_outcome"] == "completed"
    assert records[-1][1] == "resource_monitor_stopped"
    assert monitor.request_sample("turn_before_inference") is False
