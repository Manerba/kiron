"""Eigenstaendige 250-ms-Livehistorie fuer den Verlauf-Tab.

Dieser Pfad liest lokale Host-Counter und NVML direkt. Er verwendet weder den
zentralen Metrics-Cache noch die SQLite-Historie. Dadurch koennen die 1m- und
10m-Ansicht hochaufgeloest laufen, ohne den umfangreichen Service-Collector zu
beschleunigen oder die langfristige Datenmenge zu vergroessern.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import psutil

try:
    import pynvml
except ImportError:  # Tests und Hosts ohne installierte optionale Runtime-Abhaengigkeit
    pynvml = None


log = logging.getLogger("history.high_res")

SAMPLE_INTERVAL_SECONDS = 0.25
HISTORY_WINDOWS_SECONDS = {
    "1m": 60.0,
    "10m": 600.0,
}
RETENTION_SECONDS = max(HISTORY_WINDOWS_SECONDS.values())
# 2.400 regulaere Ticks plus vier Sekunden Reserve fuer Timer-Jitter.
BUFFER_CAPACITY = 2416
PROTOCOL_VERSION = 1
NVML_RETRY_SECONDS = 30.0

METRIC_KEYS = (
    "cpu_usage",
    "cpu_load_1m",
    "memory_usage",
    "memory_used_gb",
    "gpu_util",
    "vram_usage",
    "disk_read_mb_s",
    "disk_write_mb_s",
)

_CPU_TIME_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rounded(value: object, digits: int = 2) -> float | None:
    parsed = _finite_float(value)
    return round(parsed, digits) if parsed is not None else None


def _cpu_totals(cpu_times: object) -> tuple[float, float] | None:
    """Liefert (gesamt, idle) ohne guest-Doppelzaehlung."""
    values: dict[str, float] = {}
    for field in _CPU_TIME_FIELDS:
        parsed = _finite_float(getattr(cpu_times, field, 0.0))
        if parsed is None or parsed < 0:
            return None
        values[field] = parsed
    total = sum(values.values())
    idle = values["idle"] + values["iowait"]
    return total, idle


def _cpu_usage_percent(
    previous: tuple[float, float] | None,
    current: tuple[float, float] | None,
) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        return None
    return round(max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100.0)), 2)


def _next_deadline(deadline: float, now: float, interval: float) -> tuple[float, int]:
    """Rueckt eine ueberholte Deadline vor, ohne Catch-up-Ticks zu erzeugen."""
    candidate = deadline + interval
    if candidate >= now:
        return candidate, 0
    skipped = int((now - candidate) // interval) + 1
    return candidate + skipped * interval, skipped


class NvmlMetricsSource:
    """Langlebiger NVML-Reader mit begrenzten Reinitialisierungsversuchen."""

    def __init__(
        self,
        api: object | None = None,
        *,
        device_index: int = 0,
        retry_seconds: float = NVML_RETRY_SECONDS,
    ) -> None:
        self._api = pynvml if api is None else api
        self._device_index = device_index
        self._retry_seconds = retry_seconds
        self._handle: object | None = None
        self._initialized = False
        self._next_retry = 0.0
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _initialize(self, now: float) -> bool:
        if self._initialized:
            return True
        if self._api is None:
            self._last_error = "nvidia-ml-py ist nicht installiert"
            self._next_retry = now + self._retry_seconds
            return False
        if now < self._next_retry:
            return False
        try:
            self._api.nvmlInit()
            self._handle = self._api.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._next_retry = now + self._retry_seconds
            with contextlib.suppress(Exception):
                self._api.nvmlShutdown()
            self._handle = None
            self._initialized = False
            return False
        self._initialized = True
        self._last_error = None
        self._next_retry = 0.0
        return True

    def read(self, now: float) -> tuple[float | None, float | None, str]:
        if not self._initialize(now):
            return None, None, "unavailable"
        try:
            utilization = self._api.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._api.nvmlDeviceGetMemoryInfo(self._handle)
            gpu_util = _rounded(getattr(utilization, "gpu", None))
            total = _finite_float(getattr(memory, "total", None))
            used = _finite_float(getattr(memory, "used", None))
            if total is None or used is None or total <= 0:
                vram_usage = None
            else:
                vram_usage = round(max(0.0, min(100.0, used / total * 100.0)), 2)
            state = "ok" if gpu_util is not None and vram_usage is not None else "partial"
            self._last_error = None
            return gpu_util, vram_usage, state
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.close()
            self._next_retry = now + self._retry_seconds
            return None, None, "error"

    def close(self) -> None:
        if self._initialized and self._api is not None:
            with contextlib.suppress(Exception):
                self._api.nvmlShutdown()
        self._handle = None
        self._initialized = False


class HighResMetricCollector:
    """Sammelt nur die acht Kurvenwerte mit unabhaengigen Counter-Baselines."""

    def __init__(
        self,
        *,
        psutil_module: object = psutil,
        loadavg: Callable[[], tuple[float, float, float]] = os.getloadavg,
        gpu_source: NvmlMetricsSource | None = None,
    ) -> None:
        self._psutil = psutil_module
        self._loadavg = loadavg
        self._gpu = gpu_source if gpu_source is not None else NvmlMetricsSource()
        self._previous_cpu: tuple[float, float] | None = None
        self._previous_disk: tuple[float, float, float] | None = None

    @property
    def gpu_error(self) -> str | None:
        return self._gpu.last_error

    def start(self) -> None:
        self._previous_cpu = None
        self._previous_disk = None

    def close(self) -> None:
        self._gpu.close()

    def _sample_cpu(self) -> tuple[float | None, float | None, str]:
        usage: float | None = None
        load_1m: float | None = None
        state = "ok"
        try:
            current = _cpu_totals(self._psutil.cpu_times())
            previous = self._previous_cpu
            usage = _cpu_usage_percent(previous, current)
            self._previous_cpu = current
            if usage is None:
                if current is None:
                    state = "error"
                elif previous is None:
                    state = "warming_up"
                else:
                    state = "reset"
        except Exception:
            self._previous_cpu = None
            state = "error"
        try:
            load_1m = _rounded(self._loadavg()[0])
            if load_1m is None and state == "ok":
                state = "partial"
        except Exception:
            if state == "ok":
                state = "partial"
        return usage, load_1m, state

    def _sample_memory(self) -> tuple[float | None, float | None, str]:
        try:
            memory = self._psutil.virtual_memory()
            usage = _rounded(getattr(memory, "percent", None))
            used = _finite_float(getattr(memory, "used", None))
            used_gb = round(used / (1024 ** 3), 2) if used is not None else None
            state = "ok" if usage is not None and used_gb is not None else "partial"
            return usage, used_gb, state
        except Exception:
            return None, None, "error"

    def _sample_disk(self, monotonic_now: float) -> tuple[float | None, float | None, str]:
        try:
            counters = self._psutil.disk_io_counters()
            read_bytes = _finite_float(getattr(counters, "read_bytes", None))
            write_bytes = _finite_float(getattr(counters, "write_bytes", None))
            if read_bytes is None or write_bytes is None:
                raise ValueError("disk counters unavailable")
        except Exception:
            self._previous_disk = None
            return None, None, "error"

        current = (read_bytes, write_bytes, monotonic_now)
        previous = self._previous_disk
        self._previous_disk = current
        if previous is None:
            return None, None, "warming_up"

        elapsed = monotonic_now - previous[2]
        read_delta = read_bytes - previous[0]
        write_delta = write_bytes - previous[1]
        if elapsed <= 0 or read_delta < 0 or write_delta < 0:
            return None, None, "reset"
        divisor = (1024 ** 2) * elapsed
        return round(read_delta / divisor, 2), round(write_delta / divisor, 2), "ok"

    def sample(self, *, timestamp: float, monotonic_now: float) -> dict[str, object]:
        values = {key: None for key in METRIC_KEYS}
        states: dict[str, str] = {}

        cpu_usage, load_1m, states["cpu"] = self._sample_cpu()
        values["cpu_usage"] = cpu_usage
        values["cpu_load_1m"] = load_1m

        memory_usage, memory_used_gb, states["memory"] = self._sample_memory()
        values["memory_usage"] = memory_usage
        values["memory_used_gb"] = memory_used_gb

        disk_read, disk_write, states["disk"] = self._sample_disk(monotonic_now)
        values["disk_read_mb_s"] = disk_read
        values["disk_write_mb_s"] = disk_write

        gpu_util, vram_usage, states["gpu"] = self._gpu.read(monotonic_now)
        values["gpu_util"] = gpu_util
        values["vram_usage"] = vram_usage

        return {
            "timestamp": timestamp,
            "values": values,
            "states": states,
        }


class HighResHistoryBuffer:
    """Thread-sicherer, zeit- und kapazitaetsbegrenzter Sample-Ring."""

    def __init__(
        self,
        *,
        window_seconds: float = RETENTION_SECONDS,
        capacity: int = BUFFER_CAPACITY,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.window_seconds = float(window_seconds)
        self.capacity = int(capacity)
        self._samples: deque[dict[str, object]] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self._next_sequence = 1

    @staticmethod
    def _public(sample: Mapping[str, object]) -> dict[str, object]:
        return {
            "sequence": sample["sequence"],
            "timestamp": sample["timestamp"],
            "values": dict(sample["values"]),
            "states": dict(sample["states"]),
        }

    def append(self, sample: Mapping[str, object], *, monotonic_now: float) -> dict[str, object]:
        timestamp = _finite_float(sample.get("timestamp"))
        if timestamp is None:
            raise ValueError("sample timestamp must be finite")
        incoming_values = sample.get("values")
        incoming_states = sample.get("states")
        values = incoming_values if isinstance(incoming_values, Mapping) else {}
        states = incoming_states if isinstance(incoming_states, Mapping) else {}

        with self._lock:
            stored = {
                "sequence": self._next_sequence,
                "timestamp": timestamp,
                "values": {key: _rounded(values.get(key)) for key in METRIC_KEYS},
                "states": {
                    key: str(value)
                    for key, value in states.items()
                    if key in {"cpu", "memory", "disk", "gpu"}
                },
                "_monotonic": float(monotonic_now),
            }
            self._next_sequence += 1
            self._samples.append(stored)
            cutoff = monotonic_now - self.window_seconds
            while self._samples and self._samples[0]["_monotonic"] < cutoff:
                self._samples.popleft()
            return self._public(stored)

    def snapshot(self, *, window_seconds: float | None = None) -> list[dict[str, object]]:
        with self._lock:
            if window_seconds is None:
                selected = self._samples
            else:
                requested_window = _finite_float(window_seconds)
                if (
                    requested_window is None
                    or requested_window <= 0
                    or requested_window > self.window_seconds
                ):
                    raise ValueError("window_seconds exceeds retained history")
                if not self._samples:
                    return []
                cutoff = float(self._samples[-1]["_monotonic"]) - requested_window
                selected = (
                    sample
                    for sample in self._samples
                    if float(sample["_monotonic"]) > cutoff
                )
            return [self._public(sample) for sample in selected]

    def latest_after(self, sequence: int) -> dict[str, object] | None:
        with self._lock:
            if not self._samples or int(self._samples[-1]["sequence"]) <= sequence:
                return None
            # Absichtlich nur das neueste Sample: langsame Clients coalescen und
            # bauen dadurch niemals eine unbeschraenkte Warteschlange auf.
            return self._public(self._samples[-1])

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


class HighResHistoryClosed(RuntimeError):
    pass


class HighResHistoryService:
    """Lifecycle, Sampling und fan-out ohne per-Client-Queues."""

    def __init__(
        self,
        *,
        collector: HighResMetricCollector | None = None,
        buffer: HighResHistoryBuffer | None = None,
        interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self.collector = collector if collector is not None else HighResMetricCollector()
        self.buffer = buffer if buffer is not None else HighResHistoryBuffer()
        self.interval_seconds = float(interval_seconds)
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._condition = asyncio.Condition()
        self._task: asyncio.Task | None = None
        self._running = False
        self._closed = True
        self._missed_ticks = 0
        self._last_states: dict[str, str] = {}
        self._last_internal_error: str | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        self.collector.start()
        self._closed = False
        self._running = True
        self._task = asyncio.create_task(self._run(), name="history-high-res-sampler")
        return self._task

    async def stop(self) -> None:
        task = self._task
        if task is None and self._closed:
            return
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not self._closed:
            # Ein Task kann vor seinem allerersten Event-Loop-Schritt gecancelt
            # werden; dann erreicht die Coroutine ihren finally-Block nicht.
            self.collector.close()
            self._running = False
            self._closed = True
            async with self._condition:
                self._condition.notify_all()
        self._task = None

    async def _publish(self, sample: Mapping[str, object], monotonic_now: float) -> None:
        public = self.buffer.append(sample, monotonic_now=monotonic_now)
        states = public.get("states")
        self._last_states = dict(states) if isinstance(states, Mapping) else {}
        async with self._condition:
            self._condition.notify_all()

    async def _run(self) -> None:
        self._running = True
        mono_origin = self._monotonic()
        wall_origin = self._wall_clock()
        deadline = mono_origin
        try:
            while True:
                captured_mono = self._monotonic()
                timestamp = wall_origin + (captured_mono - mono_origin)
                try:
                    sample = self.collector.sample(
                        timestamp=timestamp,
                        monotonic_now=captured_mono,
                    )
                    self._last_internal_error = None
                except Exception as exc:
                    self._last_internal_error = f"{type(exc).__name__}: {exc}"
                    sample = {
                        "timestamp": timestamp,
                        "values": {key: None for key in METRIC_KEYS},
                        "states": {
                            "cpu": "error",
                            "memory": "error",
                            "disk": "error",
                            "gpu": "error",
                        },
                    }
                await self._publish(sample, captured_mono)

                after_sample = self._monotonic()
                deadline, skipped = _next_deadline(
                    deadline,
                    after_sample,
                    self.interval_seconds,
                )
                self._missed_ticks += skipped
                await self._sleep(max(0.0, deadline - after_sample))
        except asyncio.CancelledError:
            raise
        finally:
            self.collector.close()
            self._running = False
            self._closed = True
            async with self._condition:
                self._condition.notify_all()

    async def wait_for_next(self, after_sequence: int) -> dict[str, object]:
        async with self._condition:
            while True:
                sample = self.buffer.latest_after(after_sequence)
                if sample is not None:
                    return sample
                if self._closed:
                    raise HighResHistoryClosed("high-resolution history stopped")
                await self._condition.wait()

    def _status_for_snapshot(
        self,
        snapshot: list[dict[str, object]],
        *,
        window_seconds: float,
    ) -> dict[str, object]:
        return {
            "running": self._running,
            "interval_ms": round(self.interval_seconds * 1000),
            "window_seconds": window_seconds,
            "retention_seconds": self.buffer.window_seconds,
            "buffered_points": len(snapshot),
            "oldest_timestamp": snapshot[0]["timestamp"] if snapshot else None,
            "newest_timestamp": snapshot[-1]["timestamp"] if snapshot else None,
            "missed_ticks": self._missed_ticks,
            "sources": dict(self._last_states),
            "gpu_error": self.collector.gpu_error,
            "internal_error": self._last_internal_error,
        }

    def status(self) -> dict[str, object]:
        snapshot = self.buffer.snapshot()
        return self._status_for_snapshot(
            snapshot,
            window_seconds=self.buffer.window_seconds,
        )

    def snapshot_frame(self, range_key: str) -> dict[str, object]:
        try:
            window_seconds = HISTORY_WINDOWS_SECONDS[range_key]
        except KeyError as exc:
            raise ValueError(f"unsupported history range: {range_key}") from exc
        samples = self.buffer.snapshot(window_seconds=window_seconds)
        return {
            "type": f"history_{range_key}_snapshot",
            "version": PROTOCOL_VERSION,
            "interval_ms": round(self.interval_seconds * 1000),
            "window_seconds": window_seconds,
            "samples": samples,
            "status": self._status_for_snapshot(
                samples,
                window_seconds=window_seconds,
            ),
        }

    @staticmethod
    def point_frame(sample: Mapping[str, object], range_key: str) -> dict[str, object]:
        if range_key not in HISTORY_WINDOWS_SECONDS:
            raise ValueError(f"unsupported history range: {range_key}")
        return {
            "type": f"history_{range_key}_point",
            "version": PROTOCOL_VERSION,
            "sample": dict(sample),
        }
