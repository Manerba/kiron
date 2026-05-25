"""Stdlib-Unittest fuer services/kiron-proxy/metrics.get_all_metrics Timeout-Guard.

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_metrics_timeout

Deckt Issue #589 ab:
- Gesamtbudget TOTAL_TIMEOUT_S fuer get_all_metrics.
- Dedizierter Metrics-Executor; kein asyncio.to_thread im Default-Pool.
- Per-Probe-In-Flight-Guard gegen Pool-Saturation.
- 3-Strike-Backoff plus Post-Backoff-Retry-Marker.
- Partial Results, normalisierte Timeout-/Backoff-/In-Flight-Payloads.
- gpu_processes meldet Timeout/Backoff/In-Flight als Status-Payload.
- State-Change-Logging auf "metrics.timeout_guard".

Tests isolieren pro Test sowohl _probe_health als auch _metrics_executor.
Hanger werden mit threading.Event gebaut; Release vor Executor-Shutdown.
"""

import asyncio
import concurrent.futures
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import history_db  # noqa: E402


def _make_hanger_sync(return_value, event):
    """Sync-Hanger: wartet auf Event, liefert dann return_value."""
    def _hang():
        event.wait()
        return return_value
    return _hang


async def _make_hanger_async(event_awaitable_factory, return_value):
    """Async-Hanger: wartet auf Event (asyncio) oder sleep(long)."""
    ev = event_awaitable_factory()
    await ev
    return return_value


def _stub_sync(value):
    def _s():
        return value
    return _s


def _stub_async(value):
    async def _s():
        return value
    return _s


def _stub_exc_sync(exc):
    def _s():
        raise exc
    return _s


def _stub_exc_async(exc):
    async def _s():
        raise exc
    return _s


class _BaseTimeoutTest(unittest.IsolatedAsyncioTestCase):
    """Basis mit Per-Test Executor- und Health-State-Isolation."""

    def setUp(self):
        self._release_events: list[threading.Event] = []

        # Fresh executor per test, verhindert Leichen zwischen Tests.
        self._test_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="metrics-probe-test",
        )
        self._original_executor = metrics._metrics_executor
        metrics._metrics_executor = self._test_executor

        # Health-State isoliert neu aufbauen.
        self._original_probe_health = metrics._probe_health
        metrics._probe_health = {}

        # Kurze Timeouts damit Tests schnell laufen.
        self._patch_total = mock.patch.object(metrics, "TOTAL_TIMEOUT_S", 0.3)
        self._patch_drain = mock.patch.object(metrics, "DRAIN_TIMEOUT_S", 0.1)
        self._patch_backoff = mock.patch.object(metrics, "_BACKOFF_S", 0.2)
        self._patch_total.start()
        self._patch_drain.start()
        self._patch_backoff.start()

    def tearDown(self):
        # Erst alle Hanger freigeben, bevor der Executor shuttet.
        for ev in self._release_events:
            try:
                ev.set()
            except Exception:
                pass
        self._release_events.clear()

        # Patches zurueckdrehen.
        self._patch_total.stop()
        self._patch_drain.stop()
        self._patch_backoff.stop()

        # Test-Executor sauber schliessen; cancel_futures zieht queued Arbeit
        # raus, wait=True laesst laufende Worker fertiglaufen.
        try:
            self._test_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        metrics._metrics_executor = self._original_executor
        metrics._probe_health = self._original_probe_health

    def _hang_event(self) -> threading.Event:
        ev = threading.Event()
        self._release_events.append(ev)
        return ev

    def _install_probes(self, blocking=None, async_probes=None):
        """Ersetzt metrics._BLOCKING_PROBES / _ASYNC_PROBES fuer diesen Test.

        blocking: list of (name, callable, is_list) oder None fuer stub-Defaults.
        async_probes: list of (name, factory) oder None fuer stub-Defaults.
        """
        default_blocking = [
            ("cpu", _stub_sync({"usage_percent": 12.3}), False),
            ("memory", _stub_sync({"usage_percent": 45.6}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.1}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 50.0}), False),
            ("gpu_processes", _stub_sync([
                {"pid": 1, "process": "x", "label": "other",
                 "vram_mb": 100.0, "vram_state": "known"},
            ]), False),
            ("docling", _stub_sync({"running": True, "device": "cuda",
                                    "container": "docling-serve"}), False),
        ]
        default_async = [
            ("ollama", _stub_async({"version": "0.1.0",
                                    "models_available": 1})),
            ("embedding", _stub_async({"running": True, "model": "m"})),
            ("deberta", _stub_async({"running": True, "model": "d"})),
        ]

        bp = tuple(blocking) if blocking is not None else tuple(default_blocking)
        ap = tuple(async_probes) if async_probes is not None else tuple(default_async)

        self._patch_bp = mock.patch.object(metrics, "_BLOCKING_PROBES", bp)
        self._patch_ap = mock.patch.object(metrics, "_ASYNC_PROBES", ap)
        self._patch_bp.start()
        self._patch_ap.start()
        self.addCleanup(self._patch_bp.stop)
        self.addCleanup(self._patch_ap.stop)


class NormalPathTest(_BaseTimeoutTest):
    async def test_all_probes_return_real_values(self):
        self._install_probes()
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        # Alle Result-Keys muessen vorhanden sein
        for key in ("cpu", "memory", "disk_io", "gpu", "ollama",
                    "gpu_processes", "gpu_process_vram_unknown_count",
                    "docling", "embedding", "deberta", "timestamp"):
            self.assertIn(key, result)

        # Im Normalpfad liefern Dicts keine "code"-Felder
        for probe in ("cpu", "memory", "disk_io", "gpu", "ollama",
                       "docling", "embedding", "deberta"):
            self.assertNotIn("code", result[probe],
                             f"{probe} sollte kein code-Feld haben")

        self.assertEqual(result["gpu_processes"]["state"], "ok")
        self.assertIsInstance(result["gpu_processes"]["data"], list)
        self.assertIn("ok", result["gpu_processes"]["states"])
        self.assertIn("vram_mb", result["gpu_processes"]["fields"])
        # Unknown-Count korrekt (0, weil Stub vram_state=known)
        self.assertEqual(result["gpu_process_vram_unknown_count"], 0)

    async def test_unknown_vram_count_is_summed(self):
        unknown_list = [
            {"pid": 1, "vram_state": "unknown"},
            {"pid": 2, "vram_state": "known"},
            {"pid": 3, "vram_state": "unknown"},
        ]
        blocking = [
            ("cpu", _stub_sync({"usage_percent": 1.0}), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.1}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync(unknown_list), False),
            ("docling", _stub_sync({"running": False, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result["gpu_process_vram_unknown_count"], 2)


class BlockingHangerTest(_BaseTimeoutTest):
    async def test_single_blocking_hanger_times_out_others_ok(self):
        ev = self._hang_event()
        blocking = [
            ("cpu", _make_hanger_sync({"never": True}, ev), False),
            ("memory", _stub_sync({"usage_percent": 50.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 10.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        t0 = time.monotonic()
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        elapsed = time.monotonic() - t0

        # Unter Budget fertig (TOTAL_TIMEOUT_S=0.3 + Overhead < 2s)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(result["cpu"]["code"], "collection_timeout")
        self.assertEqual(result["cpu"]["probe"], "cpu")
        # Andere liefern echte Werte
        self.assertEqual(result["memory"]["usage_percent"], 50.0)
        self.assertEqual(result["gpu"]["gpu_util_percent"], 10.0)


class AsyncHangerTest(_BaseTimeoutTest):
    async def test_single_async_hanger_strike_counter(self):
        async def _hang():
            await asyncio.sleep(10.0)
            return {"never": True}

        blocking = [
            ("cpu", _stub_sync({"usage_percent": 1.0}), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        async_probes = [
            ("ollama", _hang),
            ("embedding", _stub_async({"running": True, "model": "m"})),
            ("deberta", _stub_async({"running": True, "model": "d"})),
        ]
        self._install_probes(blocking=blocking, async_probes=async_probes)

        # Dreimal aufrufen; nach drei Timeouts geht ollama in Backoff.
        for i in range(3):
            result = await asyncio.wait_for(
                metrics.get_all_metrics(), timeout=2.0)
            self.assertEqual(result["ollama"]["code"], "collection_timeout")

        # Nach drittem Strike: Backoff aktiv
        entry = metrics._probe_health["ollama"]
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)

        # Backoff-Fenster ist in diesem Test kurz (_BACKOFF_S=0.2); wir
        # forcieren die Dauer hier, damit der naechste Aufruf zuverlaessig
        # in den Backoff-Payload-Pfad faellt.
        entry["backoff_until"] = time.monotonic() + 10.0

        # Naechster Aufruf liefert Backoff-Payload (nicht Timeout)
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result["ollama"]["code"], "probe_backoff")


class DefaultPoolIsolationTest(_BaseTimeoutTest):
    async def test_default_pool_unaffected_by_metrics_pool_hang(self):
        # Saturiere den Metrics-Pool mit 4 blockierten Workern
        ev = self._hang_event()
        saturating = [self._test_executor.submit(_make_hanger_sync("x", ev))
                      for _ in range(4)]

        # Default-Pool-Aufruf muss trotzdem schnell fertig werden
        t0 = time.monotonic()
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: "ok"), timeout=2.0)
        elapsed = time.monotonic() - t0

        self.assertEqual(result, "ok")
        self.assertLess(elapsed, 1.0)

        ev.set()
        for f in saturating:
            f.result(timeout=2.0)


class InFlightGuardTest(_BaseTimeoutTest):
    async def test_in_flight_skip_does_not_increment_strike(self):
        ev = self._hang_event()
        blocking = [
            ("cpu", _make_hanger_sync({"cpu": True}, ev), False),
            ("memory", _stub_sync({"usage_percent": 50.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 10.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Tick 1: cpu wird gestartet, hangs, timeout. Strike=1.
        result1 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result1["cpu"]["code"], "collection_timeout")
        self.assertEqual(metrics._probe_health["cpu"]["timeouts"], 1)

        # Tick 2: cpu future ist noch in-flight (haengt weiter).
        result2 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result2["cpu"]["code"], "probe_in_flight")
        # In-Flight-Skips erhoehen den Strike-Counter NICHT
        self.assertEqual(metrics._probe_health["cpu"]["timeouts"], 1)

        # Weitere in-flight Skips auch nicht
        result3 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result3["cpu"]["code"], "probe_in_flight")
        self.assertEqual(metrics._probe_health["cpu"]["timeouts"], 1)


class TwoHangersSaturationTest(_BaseTimeoutTest):
    async def test_gpu_and_gpu_processes_hang_cpu_mem_still_work(self):
        ev_gpu = self._hang_event()
        ev_gp = self._hang_event()
        blocking = [
            ("cpu", _stub_sync({"usage_percent": 42.0}), False),
            ("memory", _stub_sync({"usage_percent": 64.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _make_hanger_sync({"gpu": "hang"}, ev_gpu), False),
            ("gpu_processes",
             _make_hanger_sync([{"pid": 1, "vram_state": "unknown"}], ev_gp),
             False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        t0 = time.monotonic()
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 2.0)
        # CPU und Memory muessen echte Werte liefern
        self.assertEqual(result["cpu"]["usage_percent"], 42.0)
        self.assertEqual(result["memory"]["usage_percent"], 64.0)
        # gpu/gpu_processes timed out
        self.assertEqual(result["gpu"]["code"], "collection_timeout")
        self.assertEqual(result["gpu_processes"]["state"], "timeout")
        self.assertIsNone(result["gpu_process_vram_unknown_count"])


class BackoffRecoveryTest(_BaseTimeoutTest):
    async def test_recovery_after_backoff_resets_state(self):
        # Direkt state in "Backoff abgelaufen" versetzen und Stub liefert
        # schnell. Damit testen wir den Pfad: Backoff-Ende -> Post-Backoff-
        # Retry -> fertig mit Wert -> state ok, Counter 0.
        blocking = [
            ("cpu", _stub_sync({"usage_percent": 99.0}), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        metrics._probe_health["cpu"] = {
            "timeouts": 0,
            "backoff_until": time.monotonic() - 1.0,  # abgelaufen
            "last_state": metrics._STATE_BACKOFF,
            "in_flight": None,
            "post_backoff_retry": False,
        }

        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        # Post-Backoff-Retry erfolgreich -> state ok, counter 0
        self.assertEqual(result["cpu"]["usage_percent"], 99.0)
        entry = metrics._probe_health["cpu"]
        self.assertEqual(entry["last_state"], metrics._STATE_OK)
        self.assertEqual(entry["timeouts"], 0)
        self.assertFalse(entry["post_backoff_retry"])

    async def test_post_backoff_retry_timeout_triggers_immediate_backoff(self):
        ev = self._hang_event()
        blocking = [
            ("cpu", _make_hanger_sync({"n": True}, ev), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Direkt state in Backoff-Ende versetzen, ohne 3 Strikes zu laufen.
        # Dadurch ist der naechste Tick ein Post-Backoff-Retry.
        metrics._probe_health["cpu"] = {
            "timeouts": 0,
            "backoff_until": time.monotonic() - 1.0,  # abgelaufen
            "last_state": metrics._STATE_BACKOFF,
            "in_flight": None,
            "post_backoff_retry": False,
        }

        t_before = time.monotonic()
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(result["cpu"]["code"], "collection_timeout")

        entry = metrics._probe_health["cpu"]
        # Sofort wieder Backoff gesetzt, ohne erneute 3 Strikes.
        # backoff_until wurde mit einem "now" aus der aktuellen Ausfuehrung
        # gesetzt, also mindestens nach t_before + 0 (monoton steigend).
        self.assertGreaterEqual(entry["backoff_until"], t_before)
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertFalse(entry["post_backoff_retry"])


class InFlightHangRecoveryTest(_BaseTimeoutTest):
    """Issue #671/#779: stale in-flight Futures starten Backoff.

    Das Future bleibt als Outstanding-Limit referenziert, damit ein permanenter
    Haenger den Metrics-Executor nicht pro Backoff-Zyklus mit Duplikaten fuellt.
    """

    def _hanger_blocking(self, ev):
        return [
            ("cpu", _make_hanger_sync({"never": True}, ev), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]

    async def test_in_flight_skips_escalate_to_backoff(self):
        ev = self._hang_event()
        self._install_probes(blocking=self._hanger_blocking(ev))

        # Tick 1: erster Timeout setzt in_flight, last_state=timeout.
        r1 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r1["cpu"]["code"], "collection_timeout")
        self.assertEqual(metrics._probe_health["cpu"]["in_flight_skips"], 0)

        # Ticks 2,3: in_flight skips zaehlen 1,2 und liefern in_flight payload.
        for expected_skips in (1, 2):
            r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            self.assertEqual(r["cpu"]["code"], "probe_in_flight")
            self.assertEqual(
                metrics._probe_health["cpu"]["in_flight_skips"],
                expected_skips,
            )
            # In-Flight-Skips erhoehen den timeout-Strike-Counter NICHT.
            self.assertEqual(metrics._probe_health["cpu"]["timeouts"], 1)

        # Tick 4: dritter Skip eskaliert in Backoff, in_flight bleibt als
        # Outstanding-Limit erhalten.
        r4 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r4["cpu"]["code"], "probe_backoff")
        entry = metrics._probe_health["cpu"]
        self.assertIsNotNone(entry["in_flight"])
        self.assertFalse(entry["in_flight"].done())
        self.assertEqual(entry["in_flight_skips"], 0)
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertGreater(entry["backoff_until"], time.monotonic())

    async def test_backoff_payload_during_backoff_after_escalation(self):
        ev = self._hang_event()
        self._install_probes(blocking=self._hanger_blocking(ev))

        # 1 timeout + 3 in-flight skips -> Eskalation
        for _ in range(4):
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        # _BACKOFF_S ist im Test 0.2s; Fenster verlaengern, damit der
        # Folge-Tick zuverlaessig im Backoff-Pfad landet.
        metrics._probe_health["cpu"]["backoff_until"] = (
            time.monotonic() + 10.0
        )

        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["cpu"]["code"], "probe_backoff")

    async def test_backoff_expiry_does_not_submit_duplicate_while_future_running(self):
        ev = self._hang_event()
        state = {"mode": "hang"}
        calls = {"count": 0}

        def _func():
            calls["count"] += 1
            if state["mode"] == "hang":
                ev.wait()
                return {"never": True}
            return {"usage_percent": 77.0}

        blocking = [
            ("cpu", _func, False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Eskalation: 1 timeout + 3 in-flight skips
        for _ in range(4):
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        entry = metrics._probe_health["cpu"]
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertEqual(calls["count"], 1)

        # Backoff manuell ablaufen lassen; solange das alte Future noch laeuft,
        # darf trotzdem kein zweites executor.submit() fuer cpu passieren.
        entry["backoff_until"] = time.monotonic() - 1.0
        state["mode"] = "ok"

        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["cpu"]["code"], "probe_in_flight")
        self.assertEqual(calls["count"], 1)

        # Weitere Skips duerfen hoechstens wieder Backoff starten, aber kein
        # Duplikat in den Executor einreihen.
        await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["cpu"]["code"], "probe_backoff")
        self.assertEqual(calls["count"], 1)

        # Erst wenn das alte Future wirklich fertig ist, darf nach Backoff-Ende
        # ein frischer Retry starten und recovern.
        ev.set()
        await asyncio.sleep(0.1)
        metrics._probe_health["cpu"]["backoff_until"] = time.monotonic() - 1.0
        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        self.assertEqual(r["cpu"]["usage_percent"], 77.0)
        entry = metrics._probe_health["cpu"]
        self.assertEqual(entry["last_state"], metrics._STATE_OK)
        self.assertEqual(entry["in_flight_skips"], 0)
        self.assertIsNone(entry["in_flight"])
        self.assertEqual(entry["timeouts"], 0)
        self.assertFalse(entry["post_backoff_retry"])
        self.assertEqual(calls["count"], 2)

    async def test_in_flight_logging_no_spam_visible_escalation(self):
        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _CaptureHandler()
        logger = logging.getLogger("metrics.timeout_guard")
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            ev = self._hang_event()
            self._install_probes(blocking=self._hanger_blocking(ev))

            def _cpu_warns():
                return [r for r in records
                        if r.levelno == logging.WARNING
                        and "cpu" in r.getMessage()]

            # Tick 1: ok->timeout transition logged ("probe timeout").
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            after_t1 = _cpu_warns()
            self.assertEqual(len(after_t1), 1)
            self.assertIn("probe timeout", after_t1[0].getMessage())

            # Tick 2: erster in-flight Skip wird einmalig sichtbar geloggt.
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            after_t2 = _cpu_warns()
            self.assertEqual(len(after_t2), 2)
            self.assertIn("in-flight", after_t2[1].getMessage())

            # Tick 3: zweiter Skip - KEIN zusaetzlicher WARN (kein Spam).
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            after_t3 = _cpu_warns()
            self.assertEqual(len(after_t3), 2)

            # Tick 4: Eskalation -> backoff started WARN.
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            after_t4 = _cpu_warns()
            self.assertEqual(len(after_t4), 3)
            self.assertIn("backoff started", after_t4[2].getMessage())
            self.assertIn(
                f"{metrics._IN_FLIGHT_SKIP_STRIKES} in-flight skips",
                after_t4[2].getMessage(),
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    async def test_done_in_flight_resets_skip_counter(self):
        # Wenn das alte Future doch noch fertig wird, soll der
        # Skip-Counter zurueckgesetzt werden und ein frisches Submit
        # losgehen.
        ev = self._hang_event()
        state = {"mode": "hang"}

        def _func():
            if state["mode"] == "hang":
                ev.wait()
                return {"old": True}
            return {"usage_percent": 33.0}

        blocking = [
            ("cpu", _func, False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Tick 1: Timeout, in_flight gesetzt.
        await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        # Tick 2: erster in-flight Skip.
        await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(metrics._probe_health["cpu"]["in_flight_skips"], 1)

        # Hanger freigeben, Stub auf ok-Mode umschalten und kurz warten,
        # damit der Worker fertig wird.
        state["mode"] = "ok"
        ev.set()
        await asyncio.sleep(0.1)

        # Tick 3: in_flight ist done(); wird verworfen, Skip-Counter
        # zurueckgesetzt, frisches Submit liefert 33.0.
        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["cpu"]["usage_percent"], 33.0)
        entry = metrics._probe_health["cpu"]
        self.assertEqual(entry["in_flight_skips"], 0)
        self.assertEqual(entry["last_state"], metrics._STATE_OK)


class ExceptionAfterTimeoutTest(_BaseTimeoutTest):
    async def test_exception_probe_resets_counter(self):
        # Zwei Strikes aufbauen, dann Probe wirft Exception.
        state = {"mode": "hang"}
        ev = self._hang_event()

        def _func():
            if state["mode"] == "hang":
                ev.wait()
                return {"never": True}
            raise RuntimeError("boom")

        blocking = [
            ("cpu", _func, False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Zwei Timeouts
        await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        # Warten bis in_flight released und der Worker fertig wird, damit
        # der naechste Tick frisch submittet.
        ev.set()
        await asyncio.sleep(0.1)

        # Jetzt wirft der Probe
        state["mode"] = "raise"
        result = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)

        # Exception wurde zu error-Dict konvertiert
        self.assertIn("error", result["cpu"])
        self.assertIn("RuntimeError", result["cpu"]["error"])

        # Counter zurueckgesetzt
        entry = metrics._probe_health["cpu"]
        self.assertEqual(entry["timeouts"], 0)
        self.assertEqual(entry["last_state"], metrics._STATE_OK)


class OuterCancellationTest(_BaseTimeoutTest):
    async def test_cancel_during_collection_cleans_up(self):
        ev = self._hang_event()
        blocking = [
            ("cpu", _make_hanger_sync({"n": True}, ev), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]

        async def _async_hang():
            await asyncio.sleep(100.0)
            return {"never": True}

        async_probes = [
            ("ollama", _async_hang),
            ("embedding", _stub_async({"running": True, "model": "m"})),
            ("deberta", _stub_async({"running": True, "model": "d"})),
        ]
        self._install_probes(blocking=blocking, async_probes=async_probes)

        task = asyncio.create_task(metrics.get_all_metrics())
        # Warte einen Moment damit asyncio.wait laeuft
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # In-Flight-Guard: der naechste Aufruf darf nicht doppelt submitten.
        # Noch ist cpu in-flight (Event nicht gesetzt).
        # Wir muessen den asyncio-Event-Loop kurz weiterlaufen lassen damit
        # pending async Tasks drainen.
        await asyncio.sleep(0.1)

        # Counter wurden NICHT erhoeht durch den Cancel-Pfad.
        # (ollama wurde gecancelt waehrend Setup, kein Strike.)
        # cpu ist ggf. noch in-flight.
        entry_cpu = metrics._probe_health.get("cpu", {})
        # Entweder in_flight ist gesetzt (running wurde nicht cancellt),
        # oder None (war noch queued und wurde gecancelt).
        # Beide Faelle sind erlaubt.
        self.assertIn(entry_cpu.get("last_state", metrics._STATE_OK),
                      (metrics._STATE_OK, metrics._STATE_TIMEOUT,
                       metrics._STATE_BACKOFF))


class DoclingDefaultPoolProtectionTest(_BaseTimeoutTest):
    async def test_docling_does_not_use_default_pool_to_thread(self):
        # Patch asyncio.to_thread so es explodiert. get_all_metrics und
        # der Docling-Pfad duerfen es nicht aufrufen.
        def _boom(*args, **kwargs):
            raise AssertionError("asyncio.to_thread must not be used in get_all_metrics")

        self._install_probes()  # default stubs
        with mock.patch.object(asyncio, "to_thread", _boom):
            result = await asyncio.wait_for(
                metrics.get_all_metrics(), timeout=2.0)
        # Docling liefert korrekte Shape
        self.assertEqual(result["docling"]["running"], True)
        self.assertEqual(result["docling"]["device"], "cuda")
        self.assertEqual(result["docling"]["container"], "docling-serve")


class GpuProcessesShapeTest(_BaseTimeoutTest):
    async def test_gpu_processes_status_payload_on_all_error_states(self):
        ev = self._hang_event()
        blocking = [
            ("cpu", _stub_sync({"usage_percent": 1.0}), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes", _make_hanger_sync([], ev), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]
        self._install_probes(blocking=blocking)

        # Timeout
        r1 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r1["gpu_processes"]["state"], "timeout")
        self.assertIsNone(r1["gpu_process_vram_unknown_count"])

        # In-Flight (naechster Tick)
        r2 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r2["gpu_processes"]["state"], "in_flight")

        # Backoff via direktem State-Set forcieren
        metrics._probe_health["gpu_processes"]["backoff_until"] = \
            time.monotonic() + 10.0
        metrics._probe_health["gpu_processes"]["in_flight"] = None

        r3 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r3["gpu_processes"]["state"], "backoff")
        self.assertIsNone(r3["gpu_process_vram_unknown_count"])


class ProbeInternalTimeoutTest(_BaseTimeoutTest):
    """Issue #776: Probes mit eigenem Sub-Timeout (z.B. nvidia-smi -q -x mit
    timeout=5s) liefern ein Error-Payload bevor TOTAL_TIMEOUT_S erreicht ist
    und landen daher in `done`. Ohne explizite Erkennung wuerde der Done-Pfad
    Strikes/State auf OK zuruecksetzen und Backoff komplett umgehen.
    """

    @staticmethod
    def _gpu_timeout_payload():
        return {
            "error": "nvidia-smi timed out after 5 seconds.",
            "code": "probe_timeout",
            "probe": "gpu",
        }

    @staticmethod
    def _gpu_processes_timeout_payload():
        return metrics._gpu_processes_error_payload(
            "timeout", "nvidia-smi process query timed out"
        )

    def _blocking_with(self, gpu_func=None, gpu_processes_func=None):
        return [
            ("cpu", _stub_sync({"usage_percent": 1.0}), False),
            ("memory", _stub_sync({"usage_percent": 1.0}), False),
            ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
            ("gpu", gpu_func or _stub_sync({"gpu_util_percent": 0.0}), False),
            ("gpu_processes",
             gpu_processes_func or _stub_sync([]), False),
            ("docling", _stub_sync({"running": True, "device": "cpu",
                                    "container": "docling-serve"}), False),
        ]

    def test_helper_detects_gpu_timeout_payload(self):
        self.assertTrue(
            metrics._is_probe_internal_timeout(self._gpu_timeout_payload())
        )

    def test_helper_detects_gpu_processes_timeout_payload(self):
        self.assertTrue(
            metrics._is_probe_internal_timeout(self._gpu_processes_timeout_payload())
        )

    def test_helper_rejects_success_payloads(self):
        # gpu success
        self.assertFalse(metrics._is_probe_internal_timeout(
            {"name": "RTX 3060", "gpu_util_percent": 50.0}
        ))
        # gpu_processes success
        self.assertFalse(metrics._is_probe_internal_timeout(
            metrics._gpu_processes_payload("ok", [{"pid": 1}])
        ))
        self.assertFalse(metrics._is_probe_internal_timeout(
            metrics._gpu_processes_payload("empty", [])
        ))
        # generic error without timeout marker
        self.assertFalse(metrics._is_probe_internal_timeout(
            {"error": "nvidia-smi not found"}
        ))
        # nicht-dict / list-Probes
        self.assertFalse(metrics._is_probe_internal_timeout([]))
        self.assertFalse(metrics._is_probe_internal_timeout(None))

    async def test_gpu_probe_internal_timeout_increments_strike(self):
        self._install_probes(
            blocking=self._blocking_with(
                gpu_func=_stub_sync(self._gpu_timeout_payload())
            )
        )

        # Erster Tick: Strike=1, last_state=timeout, Payload bleibt sichtbar.
        r1 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r1["gpu"]["code"], "probe_timeout")
        entry = metrics._probe_health["gpu"]
        self.assertEqual(entry["timeouts"], 1)
        self.assertEqual(entry["last_state"], metrics._STATE_TIMEOUT)

        # Zweiter Tick: Strike=2, weiterhin nicht im Backoff.
        r2 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r2["gpu"]["code"], "probe_timeout")
        self.assertEqual(metrics._probe_health["gpu"]["timeouts"], 2)

        # Dritter Tick: dritter Strike triggert Backoff (Counter zurueck auf 0).
        r3 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r3["gpu"]["code"], "probe_timeout")
        entry = metrics._probe_health["gpu"]
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertEqual(entry["timeouts"], 0)
        self.assertGreater(entry["backoff_until"], time.monotonic())

        # Vierter Tick: Backoff aktiv -> Probe wird ueberhaupt nicht aufgerufen,
        # Backoff-Payload statt probe_timeout-Payload.
        r4 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r4["gpu"]["code"], "probe_backoff")

    async def test_gpu_processes_probe_internal_timeout_triggers_backoff(self):
        self._install_probes(
            blocking=self._blocking_with(
                gpu_processes_func=_stub_sync(self._gpu_processes_timeout_payload())
            )
        )

        # Drei Strikes in Folge -> Backoff.
        for expected_strikes in (1, 2):
            r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            self.assertEqual(r["gpu_processes"]["state"], "timeout")
            entry = metrics._probe_health["gpu_processes"]
            self.assertEqual(entry["timeouts"], expected_strikes)
            self.assertEqual(entry["last_state"], metrics._STATE_TIMEOUT)

        # Dritter Strike -> Backoff.
        r3 = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r3["gpu_processes"]["state"], "timeout")
        entry = metrics._probe_health["gpu_processes"]
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertEqual(entry["timeouts"], 0)

    async def test_post_backoff_retry_with_internal_timeout_re_enters_backoff(self):
        self._install_probes(
            blocking=self._blocking_with(
                gpu_func=_stub_sync(self._gpu_timeout_payload())
            )
        )

        # Direkt state in Backoff-Ende versetzen, ohne 3 Strikes zu laufen.
        # Naechster Tick ist Post-Backoff-Retry und liefert Probe-Internal-Timeout.
        metrics._probe_health["gpu"] = {
            "timeouts": 0,
            "backoff_until": time.monotonic() - 1.0,  # abgelaufen
            "last_state": metrics._STATE_BACKOFF,
            "in_flight": None,
            "in_flight_skips": 0,
            "post_backoff_retry": False,
        }

        t_before = time.monotonic()
        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["gpu"]["code"], "probe_timeout")

        entry = metrics._probe_health["gpu"]
        # Sofort wieder Backoff, ohne erneute 3 Strikes abzuwarten.
        self.assertGreaterEqual(entry["backoff_until"], t_before)
        self.assertEqual(entry["last_state"], metrics._STATE_BACKOFF)
        self.assertFalse(entry["post_backoff_retry"])
        self.assertEqual(entry["timeouts"], 0)

    async def test_recovery_resets_strike_counter_after_internal_timeout(self):
        # Erste 2 Ticks liefern probe-internal timeout, dann recovery mit echtem Wert.
        state = {"mode": "timeout"}

        def _gpu_func():
            if state["mode"] == "timeout":
                return self._gpu_timeout_payload()
            return {"gpu_util_percent": 42.0}

        self._install_probes(
            blocking=self._blocking_with(gpu_func=_gpu_func)
        )

        # 2 Strikes
        for _ in range(2):
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(metrics._probe_health["gpu"]["timeouts"], 2)
        self.assertEqual(metrics._probe_health["gpu"]["last_state"], metrics._STATE_TIMEOUT)

        # Recovery: Probe liefert echten Wert.
        state["mode"] = "ok"
        r = await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
        self.assertEqual(r["gpu"]["gpu_util_percent"], 42.0)
        entry = metrics._probe_health["gpu"]
        self.assertEqual(entry["timeouts"], 0)
        self.assertEqual(entry["last_state"], metrics._STATE_OK)


class MetricsDBCompatTest(unittest.TestCase):
    """Integriert mit MetricsDB.insert_metrics(), keine Async-Isolation noetig."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = history_db.MetricsDB(self.path)
        self.db.init_db()

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suffix)
            except OSError:
                pass

    def test_insert_accepts_timeout_backoff_inflight_dicts(self):
        # CPU/Memory/Disk/GPU jeweils Error-Dict; insert darf nicht crashen
        # und muss die abhaengigen Spalten als NULL persistieren.
        error_dict = {"error": "collection timeout",
                      "code": "collection_timeout", "probe": "x"}
        metrics_dict = {
            "cpu": dict(error_dict, probe="cpu"),
            "memory": dict(error_dict, probe="memory"),
            "disk_io": dict(error_dict, probe="disk_io"),
            "gpu": dict(error_dict, probe="gpu"),
            "timestamp": time.time(),
        }
        self.db.insert_metrics(metrics_dict)

        conn = self.db._get_conn()
        row = conn.execute(
            "SELECT cpu_usage, memory_usage, disk_read_mb_s, "
            "disk_write_mb_s, gpu_util, vram_usage FROM metrics_history"
        ).fetchone()
        self.assertIsNone(row[0])  # cpu_usage
        self.assertIsNone(row[1])  # memory_usage
        self.assertIsNone(row[2])  # disk_read
        self.assertIsNone(row[3])  # disk_write
        self.assertIsNone(row[4])  # gpu_util
        self.assertIsNone(row[5])  # vram_usage

    def test_insert_accepts_backoff_and_inflight_variants(self):
        # Backoff-Payload
        backoff = {"error": "probe stuck, in backoff",
                   "code": "probe_backoff", "probe": "gpu"}
        in_flight = {"error": "probe still running",
                     "code": "probe_in_flight", "probe": "cpu"}
        metrics_dict = {
            "cpu": in_flight,
            "memory": {"usage_percent": 50.0, "used_gb": 8.0},
            "gpu": backoff,
            "disk_io": {"read_mb_s": 1.0, "write_mb_s": 2.0},
            "timestamp": time.time(),
        }
        self.db.insert_metrics(metrics_dict)

        conn = self.db._get_conn()
        row = conn.execute(
            "SELECT cpu_usage, memory_usage, gpu_util, disk_read_mb_s "
            "FROM metrics_history"
        ).fetchone()
        self.assertIsNone(row[0])
        self.assertEqual(row[1], 50.0)
        self.assertIsNone(row[2])
        self.assertEqual(row[3], 1.0)


class RestPayloadSmokeTest(_BaseTimeoutTest):
    async def test_rest_endpoint_attaches_maintenance_and_preserves_keys(self):
        # REST liest ab #585 aus dem zentralen Metrics-Cache.
        try:
            import app as app_module  # noqa
        except Exception as e:
            self.skipTest(f"app import failed: {e}")

        stub_metrics = {
            "cpu": {"usage_percent": 1.0},
            "memory": {"usage_percent": 2.0},
            "disk_io": {"read_mb_s": 0.0},
            "gpu": {"gpu_util_percent": 0.0},
            "ollama": {"version": "0.1.0"},
            "gpu_processes": metrics._gpu_processes_payload("empty", []),
            "gpu_process_vram_unknown_count": 0,
            "docling": {"running": True, "device": "cpu",
                        "container": "docling-serve"},
            "embedding": {"running": True, "model": "m"},
            "deberta": {"running": True, "model": "d"},
            "timestamp": time.time(),
        }

        payload = {
            "system": dict(stub_metrics),
            "summary": {"requests_today": 0},
            "produced_at_monotonic": time.monotonic(),
        }
        with mock.patch.object(app_module, "get_cached_payload", return_value=payload):
            result = await app_module.get_system_metrics()

        self.assertIn("maintenance", result)
        for key in stub_metrics.keys():
            self.assertIn(key, result)


class LoggingStateChangeTest(_BaseTimeoutTest):
    async def test_state_changes_log_once_not_every_tick(self):
        # Setzte unseren Test-Logger auf Capture.
        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _CaptureHandler()
        logger = logging.getLogger("metrics.timeout_guard")
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            ev = self._hang_event()
            blocking = [
                ("cpu", _make_hanger_sync({"n": True}, ev), False),
                ("memory", _stub_sync({"usage_percent": 1.0}), False),
                ("disk_io", _stub_sync({"read_mb_s": 0.0}), False),
                ("gpu", _stub_sync({"gpu_util_percent": 0.0}), False),
                ("gpu_processes", _stub_sync([]), False),
                ("docling", _stub_sync({"running": True, "device": "cpu",
                                        "container": "docling-serve"}), False),
            ]
            self._install_probes(blocking=blocking)

            # Tick 1: cpu timeout -> WARN "probe timeout"
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            timeout_warnings_1 = [r for r in records
                                  if r.levelno == logging.WARNING
                                  and "cpu" in r.getMessage()
                                  and "timeout" in r.getMessage()]
            self.assertEqual(len(timeout_warnings_1), 1)

            # Tick 2: cpu ist in-flight, KEIN zusaetzlicher WARN
            await asyncio.wait_for(metrics.get_all_metrics(), timeout=2.0)
            timeout_warnings_2 = [r for r in records
                                  if r.levelno == logging.WARNING
                                  and "cpu" in r.getMessage()
                                  and "timeout" in r.getMessage()]
            self.assertEqual(len(timeout_warnings_2), 1)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)


if __name__ == "__main__":
    unittest.main()
