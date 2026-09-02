# SPDX-License-Identifier: Apache-2.0
"""Low-overhead periodic resource telemetry for the API process.

The sampler is intentionally best-effort. Missing NVML support or individual
driver-query failures are recorded in the sample and never affect inference.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sglang_omni.utils.gpu_memory import (
    _decode_nvml_string,
    _get_device_handle,
    _shutdown_nvml,
    _try_import_pynvml,
    parse_cuda_visible_devices,
    resolve_visible_device_id,
)
from sglang_omni.utils.structured_logs import emit_structured_log


logger = logging.getLogger(__name__)

_AUTO_NVML = object()


def resource_log_interval_s(
    env: Mapping[str, str] | None = None,
) -> float:
    """Return the configured sampling interval; zero disables telemetry."""

    source = os.environ if env is None else env
    raw = source.get("SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S", "0").strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "invalid SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S=%r; disabling resource logs",
            raw,
        )
        return 0.0
    if value < 0:
        logger.warning(
            "SGLANG_OMNI_RESOURCE_LOG_INTERVAL_S must be non-negative; disabling"
        )
        return 0.0
    return value


def _read_key_value_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def _kib_value(value: str | None) -> int | None:
    if not value:
        return None
    pieces = value.split()
    try:
        amount = int(pieces[0])
    except (IndexError, ValueError):
        return None
    multiplier = 1024 if len(pieces) == 1 or pieces[1].lower() == "kb" else 1
    return amount * multiplier


def _process_snapshot(pid: int) -> dict[str, Any]:
    status = _read_key_value_file(Path(f"/proc/{pid}/status"))
    snapshot: dict[str, Any] = {
        "pid": pid,
        "rss_bytes": _kib_value(status.get("VmRSS")),
        "peak_rss_bytes": _kib_value(status.get("VmHWM")),
        "virtual_memory_bytes": _kib_value(status.get("VmSize")),
        "thread_count": int(status["Threads"]) if status.get("Threads", "").isdigit() else None,
    }
    return snapshot


def _host_snapshot() -> dict[str, Any]:
    memory = _read_key_value_file(Path("/proc/meminfo"))
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m = load_5m = load_15m = None
    return {
        "cpu_count": os.cpu_count(),
        "load_average_1m": load_1m,
        "load_average_5m": load_5m,
        "load_average_15m": load_15m,
        "memory_total_bytes": _kib_value(memory.get("MemTotal")),
        "memory_available_bytes": _kib_value(memory.get("MemAvailable")),
        "swap_total_bytes": _kib_value(memory.get("SwapTotal")),
        "swap_free_bytes": _kib_value(memory.get("SwapFree")),
    }


def _optional_nvml_value(
    target: dict[str, Any],
    key: str,
    query: Callable[[], Any],
    *,
    transform: Callable[[Any], Any] | None = None,
) -> None:
    try:
        value = query()
        target[key] = transform(value) if transform is not None else value
    except Exception:
        # Driver versions expose different optional metrics. Their absence is
        # expected and should not make every sample noisy.
        target[key] = None


def _valid_used_gpu_memory(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    # NVML_VALUE_NOT_AVAILABLE is represented by an unsigned sentinel.
    return number if 0 <= number < (1 << 60) else None


def _gpu_processes(pynvml: Any, handle: Any) -> list[dict[str, Any]]:
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception:
        return []
    result: list[dict[str, Any]] = []
    for process in processes:
        pid = int(process.pid)
        try:
            name = Path(f"/proc/{pid}/comm").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            name = None
        result.append(
            {
                "pid": pid,
                "process_name": name,
                "used_gpu_memory_bytes": _valid_used_gpu_memory(
                    getattr(process, "usedGpuMemory", None)
                ),
            }
        )
    return sorted(result, key=lambda item: item["pid"])


def _gpu_snapshots(
    *,
    env: Mapping[str, str],
    pynvml_module: Any | None | object = _AUTO_NVML,
) -> tuple[list[dict[str, Any]], list[str]]:
    pynvml = _try_import_pynvml() if pynvml_module is _AUTO_NVML else pynvml_module
    if pynvml is None:
        return [], ["pynvml is unavailable"]

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return [], [f"NVML initialization failed: {type(exc).__name__}: {exc}"]

    snapshots: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        visible_devices = parse_cuda_visible_devices(
            env.get("CUDA_VISIBLE_DEVICES")
        )
        if visible_devices:
            targets: list[int | str] = list(visible_devices)
        else:
            targets = list(range(int(pynvml.nvmlDeviceGetCount())))

        for logical_index, device_id in enumerate(targets):
            try:
                resolved = resolve_visible_device_id(logical_index, visible_devices)
                handle = _get_device_handle(pynvml, resolved)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                processes = _gpu_processes(pynvml, handle)
                snapshot: dict[str, Any] = {
                    "logical_index": logical_index,
                    "physical_device": str(device_id),
                    "uuid": _decode_nvml_string(pynvml.nvmlDeviceGetUUID(handle)),
                    "name": _decode_nvml_string(pynvml.nvmlDeviceGetName(handle)),
                    "gpu_utilization_percent": int(utilization.gpu),
                    "memory_utilization_percent": int(utilization.memory),
                    "memory_total_bytes": int(memory.total),
                    "memory_used_bytes": int(memory.used),
                    "memory_free_bytes": int(memory.free),
                    "compute_process_count": len(processes),
                    "compute_process_memory_bytes": sum(
                        int(item["used_gpu_memory_bytes"] or 0)
                        for item in processes
                    ),
                    "compute_processes": processes,
                }
                _optional_nvml_value(
                    snapshot,
                    "temperature_c",
                    lambda: pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    transform=int,
                )
                _optional_nvml_value(
                    snapshot,
                    "power_usage_watts",
                    lambda: pynvml.nvmlDeviceGetPowerUsage(handle),
                    transform=lambda value: round(float(value) / 1000.0, 3),
                )
                _optional_nvml_value(
                    snapshot,
                    "power_limit_watts",
                    lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle),
                    transform=lambda value: round(float(value) / 1000.0, 3),
                )
                _optional_nvml_value(
                    snapshot,
                    "sm_clock_mhz",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_SM
                    ),
                    transform=int,
                )
                _optional_nvml_value(
                    snapshot,
                    "memory_clock_mhz",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    ),
                    transform=int,
                )
                snapshots.append(snapshot)
            except Exception as exc:
                errors.append(
                    f"GPU query failed for logical_index={logical_index} "
                    f"device={device_id!r}: {type(exc).__name__}: {exc}"
                )
    finally:
        _shutdown_nvml(pynvml)
    return snapshots, errors


def collect_resource_snapshot(
    *,
    env: Mapping[str, str] | None = None,
    pynvml_module: Any | None | object = _AUTO_NVML,
) -> dict[str, Any]:
    """Collect one host/process/GPU snapshot without raising."""

    source_env = os.environ if env is None else env
    gpus, errors = _gpu_snapshots(
        env=source_env,
        pynvml_module=pynvml_module,
    )
    log_root = Path(
        source_env.get(
            "SGLANG_OMNI_REALTIME_LOG_DIR",
            "/tmp/sglang-omni-realtime-logs",
        )
    )
    disk_target = log_root if log_root.exists() else log_root.parent
    disk: dict[str, Any]
    try:
        usage = shutil.disk_usage(disk_target)
        disk = {
            "path": str(disk_target),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    except OSError as exc:
        disk = {"path": str(disk_target), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "host": _host_snapshot(),
        "api_process": _process_snapshot(os.getpid()),
        "gpus": gpus,
        "disk": disk,
        "collection_errors": errors,
    }


class PeriodicResourceMonitor:
    """Emit resource samples without blocking the API event loop."""

    def __init__(
        self,
        interval_s: float,
        *,
        application_snapshot: Callable[[], dict[str, Any]],
        collector: Callable[[], dict[str, Any]] = collect_resource_snapshot,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("resource monitor interval must be positive")
        self.interval_s = interval_s
        self.application_snapshot = application_snapshot
        self.collector = collector
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._sample_lock = asyncio.Lock()
        self._sample_tasks: set[asyncio.Task[None]] = set()
        self._cpu_baselines: dict[str, tuple[float, float]] = {}
        self._accept_immediate_samples = False

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._accept_immediate_samples = True
        self._task = asyncio.create_task(
            self._run(), name="sglang-omni-resource-monitor"
        )

    async def stop(self) -> None:
        self._accept_immediate_samples = False
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            await task
        while self._sample_tasks:
            await asyncio.gather(*tuple(self._sample_tasks), return_exceptions=True)

    def request_sample(self, sample_trigger: str, **context: Any) -> bool:
        """Schedule one best-effort sample without blocking the caller."""

        if not self._accept_immediate_samples:
            return False
        requested_at = time.perf_counter()
        requested_unix_ms = int(time.time() * 1000)
        task = asyncio.create_task(
            self._emit_sample(
                sample_trigger=sample_trigger,
                requested_at=requested_at,
                requested_unix_ms=requested_unix_ms,
                context=context,
            ),
            name=f"sglang-omni-resource-sample-{sample_trigger}",
        )
        self._sample_tasks.add(task)
        task.add_done_callback(self._sample_tasks.discard)
        return True

    def _process_cpu_percent(
        self,
        baseline_key: str,
        *,
        clear_baseline: bool = False,
    ) -> float | None:
        wall_time = time.monotonic()
        process_time = time.process_time()
        result = None
        previous = self._cpu_baselines.get(baseline_key)
        if previous is not None:
            last_wall_time, last_process_time = previous
            wall_delta = wall_time - last_wall_time
            if wall_delta > 0:
                result = round(
                    100.0 * (process_time - last_process_time) / wall_delta,
                    3,
                )
        if clear_baseline:
            self._cpu_baselines.pop(baseline_key, None)
        else:
            self._cpu_baselines[baseline_key] = (wall_time, process_time)
        return result

    @staticmethod
    def _cpu_baseline_key(
        sample_trigger: str,
        context: Mapping[str, Any],
    ) -> str:
        if sample_trigger == "periodic":
            return "periodic"
        session_id = context.get("session_id")
        turn_id = context.get("turn_id")
        if session_id is not None and turn_id is not None:
            return f"turn:{session_id}:{turn_id}"
        return sample_trigger

    async def _emit_sample(
        self,
        *,
        sample_trigger: str,
        requested_at: float,
        requested_unix_ms: int,
        context: Mapping[str, Any],
    ) -> None:
        async with self._sample_lock:
            sample_started = time.perf_counter()
            queue_wait_ms = round(
                max(0.0, sample_started - requested_at) * 1000.0,
                3,
            )
            common_fields = {
                "sample_trigger": sample_trigger,
                "sample_requested_unix_ms": requested_unix_ms,
                "sample_queue_wait_ms": queue_wait_ms,
                **dict(context),
            }
            try:
                resource = await asyncio.to_thread(self.collector)
                baseline_key = self._cpu_baseline_key(sample_trigger, context)
                resource["api_process"]["cpu_percent"] = (
                    self._process_cpu_percent(
                        baseline_key,
                        clear_baseline=(sample_trigger == "turn_after_terminal"),
                    )
                )
                application = self.application_snapshot()
                emit_structured_log(
                    "resource",
                    "resource_sample",
                    sample_duration_ms=round(
                        (time.perf_counter() - sample_started) * 1000.0,
                        3,
                    ),
                    application=application,
                    **common_fields,
                    **resource,
                )
            except Exception as exc:
                emit_structured_log(
                    "resource",
                    "resource_sample_failed",
                    level="warning",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    **common_fields,
                )

    async def _run(self) -> None:
        emit_structured_log(
            "resource",
            "resource_monitor_started",
            interval_s=self.interval_s,
        )
        try:
            while not self._stop_event.is_set():
                sample_started = time.perf_counter()
                await self._emit_sample(
                    sample_trigger="periodic",
                    requested_at=sample_started,
                    requested_unix_ms=int(time.time() * 1000),
                    context={},
                )
                remaining = max(
                    self.interval_s - (time.perf_counter() - sample_started), 0.01
                )
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
        finally:
            while self._sample_tasks:
                await asyncio.gather(
                    *tuple(self._sample_tasks),
                    return_exceptions=True,
                )
            emit_structured_log("resource", "resource_monitor_stopped")


__all__ = [
    "PeriodicResourceMonitor",
    "collect_resource_snapshot",
    "resource_log_interval_s",
]
