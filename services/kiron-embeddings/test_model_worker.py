"""Tests fuer SerialModelWorker und ModelManager.

Ohne echte SentenceTransformer-/CUDA-Loads: alles laeuft gegen FakeManager
(bzw. eine instrumentierte Variante des realen Managers fuer Atomicity-Tests).
"""

import asyncio
import os
import sys
import threading
import time
import unittest
from unittest import mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import model_worker as mw  # noqa: E402
from model_worker import (  # noqa: E402
    EncodeResult,
    LateEmbedResult,
    ModelJob,
    NonFiniteEmbeddingError,
    SerialModelWorker,
    WorkerQueueFullError,
    WorkerStoppedError,
    _complete_future_with_exception,
    _complete_future_with_result,
    derive_health_status,
)


class FakeManager:
    """Fake-Manager mit Test-Hooks (Blocker, Counter, Crash-Modi)."""

    def __init__(self):
        self.state_lock = threading.RLock()
        self.owner_thread_id: int | None = None
        self.current_model_name: str | None = None
        self.loading_model: str | None = None
        self.device: str = "cpu"
        self.model = None

        # Test-Counter
        self.load_count = 0
        self.encode_count = 0
        self.drop_count = 0
        self.late_embed_count = 0

        # Concurrency-Tracking fuer Serial-Garantie
        self._conc_lock = threading.Lock()
        self.active_loads = 0
        self.peak_concurrent_loads = 0
        self.active_encodes = 0
        self.peak_concurrent_encodes = 0
        self.active_late_embeds = 0
        self.peak_concurrent_late_embeds = 0

        # Blocker / Hooks
        self.load_blocker: threading.Event | None = None
        self.encode_blocker: threading.Event | None = None
        self.late_embed_blocker: threading.Event | None = None
        self.load_started: threading.Event = threading.Event()
        self.encode_started: threading.Event = threading.Event()
        self.late_embed_started: threading.Event = threading.Event()

        # Crash-Modi
        self.load_raises: BaseException | None = None
        self.drop_raises: BaseException | None = None

        # Encode-Result-Override
        self.encode_result_override: EncodeResult | None = None
        self.encode_raises: BaseException | None = None

        # Late-Embed-Result-Override / Call-Recording
        self.late_embed_result_override: LateEmbedResult | None = None
        self.late_embed_raises: BaseException | None = None
        self.late_embed_calls: list[tuple] = []

    def set_worker_thread(self) -> None:
        with self.state_lock:
            self.owner_thread_id = threading.get_ident()

    def assert_worker_thread(self) -> None:
        if (
            self.owner_thread_id is None
            or self.owner_thread_id != threading.get_ident()
        ):
            raise RuntimeError("not in worker thread")

    def snapshot(self) -> dict[str, object]:
        with self.state_lock:
            return {
                "current_model": self.current_model_name,
                "loading_model": self.loading_model,
                "device": self.device,
            }

    def clear_loading_for_crash(self) -> None:
        with self.state_lock:
            self.loading_model = None

    def _ensure_model_sync(self, name: str):
        self.assert_worker_thread()
        with self.state_lock:
            if self.current_model_name == name and self.model is not None:
                return (self.model, 0)

        with self._conc_lock:
            self.active_loads += 1
            self.peak_concurrent_loads = max(
                self.peak_concurrent_loads, self.active_loads
            )
        self.load_started.set()
        try:
            with self.state_lock:
                self.loading_model = name
            try:
                if self.load_blocker is not None:
                    self.load_blocker.wait()
                if self.load_raises is not None:
                    raise self.load_raises
                self.load_count += 1
                model_obj = object()
                with self.state_lock:
                    self.model = model_obj
                    self.current_model_name = name
                return (model_obj, 1_000_000)
            finally:
                with self.state_lock:
                    self.loading_model = None
        finally:
            with self._conc_lock:
                self.active_loads -= 1

    def _encode_sync(self, name: str, texts: list[str], input_type):
        self.assert_worker_thread()
        with self._conc_lock:
            self.active_encodes += 1
            self.peak_concurrent_encodes = max(
                self.peak_concurrent_encodes, self.active_encodes
            )
        self.encode_started.set()
        try:
            if self.encode_blocker is not None:
                self.encode_blocker.wait()
            if self.encode_raises is not None:
                raise self.encode_raises
            self.encode_count += 1
            if self.encode_result_override is not None:
                return self.encode_result_override
            return EncodeResult(
                embeddings=[[1.0, 0.0]] * len(texts),
                prompt_eval_count=len(texts),
                load_duration_ns=0,
            )
        finally:
            with self._conc_lock:
                self.active_encodes -= 1

    def _late_embed_sync(self, name: str, document: str, chunks: list, input_type):
        self.assert_worker_thread()
        with self._conc_lock:
            self.active_late_embeds += 1
            self.peak_concurrent_late_embeds = max(
                self.peak_concurrent_late_embeds, self.active_late_embeds
            )
        self.late_embed_started.set()
        try:
            if self.late_embed_blocker is not None:
                self.late_embed_blocker.wait()
            if self.late_embed_raises is not None:
                raise self.late_embed_raises
            self.late_embed_count += 1
            self.late_embed_calls.append((name, document, list(chunks), input_type))
            if self.late_embed_result_override is not None:
                return self.late_embed_result_override
            return LateEmbedResult(
                embeddings=[[1.0, 0.0]] * len(chunks),
                prompt_eval_count=len(chunks),
                load_duration_ns=0,
                fallback_count=0,
            )
        finally:
            with self._conc_lock:
                self.active_late_embeds -= 1

    def _drop_model_sync(self) -> None:
        self.assert_worker_thread()
        if self.drop_raises is not None:
            raise self.drop_raises
        self.drop_count += 1
        with self.state_lock:
            self.model = None
            self.current_model_name = None


# --- Helper-Tests (synchron) -------------------------------------------------


class CompletionHelperTests(unittest.IsolatedAsyncioTestCase):
    """Tests fuer _complete_future_with_*."""

    async def test_set_result_when_pending(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = self._make_job(future, loop)
        _complete_future_with_result(future, job, "value")
        self.assertEqual(future.result(), "value")

    async def test_skips_result_when_already_done(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result("first")
        job = self._make_job(future, loop)
        _complete_future_with_result(future, job, "second")
        self.assertEqual(future.result(), "first")

    async def test_skips_result_when_future_cancelled(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.cancel()
        job = self._make_job(future, loop)
        _complete_future_with_result(future, job, "value")
        self.assertTrue(future.cancelled())

    async def test_skips_when_only_job_cancelled(self):
        # Awaiter cancelt; ohne future.cancel() ist future.cancelled() False,
        # aber job.cancelled.is_set() True -> Helper ueberspringt.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = self._make_job(future, loop)
        job.cancelled.set()

        _complete_future_with_result(future, job, "value")
        self.assertFalse(future.done())

        _complete_future_with_exception(future, job, RuntimeError("err"))
        self.assertFalse(future.done())

    def _make_job(self, future, loop) -> ModelJob:
        return ModelJob(
            kind="load",
            payload={},
            future=future,
            loop=loop,
            cancelled=threading.Event(),
            job_id=1,
        )


# --- derive_health_status ---------------------------------------------------


class DeriveHealthStatusTests(unittest.TestCase):
    def test_stopping_when_thread_dead(self):
        snap = {
            "worker_thread_alive": False,
            "worker_accepting": True,
            "current_model": "x",
            "loading_model": None,
            "current_job": None,
        }
        self.assertEqual(derive_health_status(snap), "stopping")

    def test_stopping_when_not_accepting(self):
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": False,
            "current_model": "x",
        }
        self.assertEqual(derive_health_status(snap), "stopping")

    def test_loading_takes_precedence_over_ok(self):
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": True,
            "current_model": "x",
            "loading_model": "y",
        }
        self.assertEqual(derive_health_status(snap), "loading")

    def test_current_job_load_without_loading_model_is_not_loading(self):
        # #906: status="loading" erfordert atomar gesetztes loading_model.
        # current_job=="load" allein (Same-Model-Cache-Hit oder Race vor
        # _load_model_sync) darf nicht "loading" melden, sonst antwortet
        # /health mit status=loading ohne loading_model-Feld.
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": True,
            "current_model": None,
            "loading_model": None,
            "current_job": "load",
        }
        self.assertEqual(derive_health_status(snap), "no_model")

    def test_current_job_load_with_cached_model_reports_ok(self):
        # Cache-Hit-Pfad: _ensure_model_sync kehrt zurueck ohne loading_model
        # zu setzen, weil das Modell bereits geladen ist. current_job=="load"
        # ist hier noch gesetzt, aber Status muss "ok" sein.
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": True,
            "current_model": "x",
            "loading_model": None,
            "current_job": "load",
        }
        self.assertEqual(derive_health_status(snap), "ok")

    def test_ok_during_encode(self):
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": True,
            "current_model": "x",
            "current_job": "encode",
        }
        self.assertEqual(derive_health_status(snap), "ok")

    def test_no_model_when_idle(self):
        snap = {
            "worker_thread_alive": True,
            "worker_accepting": True,
            "current_model": None,
            "loading_model": None,
            "current_job": None,
        }
        self.assertEqual(derive_health_status(snap), "no_model")


# --- SerialModelWorker (gross integriert mit FakeManager) -------------------


class SerialModelWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = FakeManager()
        self.worker = SerialModelWorker(self.manager, queue_max=4)

    async def asyncTearDown(self):
        # Sicherstellen, dass alle Blocker freigegeben werden
        if self.manager.load_blocker is not None:
            self.manager.load_blocker.set()
        if self.manager.encode_blocker is not None:
            self.manager.encode_blocker.set()
        if self.manager.late_embed_blocker is not None:
            self.manager.late_embed_blocker.set()
        if self.worker._started:
            try:
                await self.worker.stop(timeout=2.0)
            except Exception:
                pass

    # ---- Serial-Garantie

    async def test_two_parallel_loads_run_serially(self):
        self.manager.load_blocker = threading.Event()
        self.worker.start()

        f1 = asyncio.create_task(self.worker.load("model-a"))
        f2 = asyncio.create_task(self.worker.load("model-b"))

        # Lass beide Loads pollen — first one started, second waits in queue.
        await asyncio.sleep(0.1)
        self.assertEqual(self.manager.peak_concurrent_loads, 1)

        self.manager.load_blocker.set()
        await asyncio.wait_for(asyncio.gather(f1, f2), timeout=3.0)

        self.assertEqual(self.manager.peak_concurrent_loads, 1)
        self.assertEqual(self.manager.load_count, 2)

    # ---- Cancellation: waiting

    async def test_cancellation_of_waiting_job_prevents_execution(self):
        self.manager.load_blocker = threading.Event()
        self.worker.start()

        # Erster Load blockt im Worker; zweiter wartet in Queue.
        f1 = asyncio.create_task(self.worker.load("model-a"))
        f2 = asyncio.create_task(self.worker.load("model-b"))
        await asyncio.sleep(0.1)
        self.assertTrue(self.manager.load_started.is_set())
        self.assertEqual(self.manager.load_count, 0)  # blockiert

        # Cancel den wartenden Job
        f2.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await f2

        # Release ersten Load
        self.manager.load_blocker.set()
        await asyncio.wait_for(f1, timeout=2.0)

        # Worker sollte den gecancelten Job verworfen haben — load_count == 1.
        await asyncio.sleep(0.2)
        self.assertEqual(self.manager.load_count, 1)

    # ---- Cancellation: running

    async def test_cancellation_of_running_job_releases_awaiter_but_worker_completes(self):
        self.manager.load_blocker = threading.Event()
        self.worker.start()

        f1 = asyncio.create_task(self.worker.load("model-a"))
        await asyncio.sleep(0.1)
        self.assertTrue(self.manager.load_started.is_set())

        # Cancel waehrend Job laeuft
        f1.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await f1

        # Worker steckt noch im _ensure_model_sync
        self.assertEqual(self.manager.load_count, 0)

        # Submit nachfolgenden Job
        f2 = asyncio.create_task(self.worker.load("model-b"))
        await asyncio.sleep(0.1)
        # f2 darf nicht starten, solange f1's Sync-Methode noch laeuft
        self.assertEqual(self.manager.peak_concurrent_loads, 1)

        # Release ersten Load — f1's Sync-Methode beendet sich
        self.manager.load_blocker.set()
        await asyncio.wait_for(f2, timeout=2.0)

        self.assertEqual(self.manager.peak_concurrent_loads, 1)
        self.assertEqual(self.manager.load_count, 2)

    # ---- Queue-Full

    async def test_queue_full_raises(self):
        worker = SerialModelWorker(self.manager, queue_max=2)
        self.worker = worker  # ueberschreiben fuer tearDown
        self.manager.load_blocker = threading.Event()
        worker.start()

        # 1 laeuft, 2 in Queue (size 2 erreicht)
        f1 = asyncio.create_task(worker.load("a"))
        f2 = asyncio.create_task(worker.load("b"))
        f3 = asyncio.create_task(worker.load("c"))
        await asyncio.sleep(0.1)

        with self.assertRaises(WorkerQueueFullError):
            await worker.load("d")

        self.manager.load_blocker.set()
        await asyncio.wait_for(asyncio.gather(f1, f2, f3), timeout=3.0)

    # ---- Stop

    async def test_stop_drains_pending_with_worker_stopped_error(self):
        self.manager.load_blocker = threading.Event()
        self.worker.start()

        f1 = asyncio.create_task(self.worker.load("a"))
        f2 = asyncio.create_task(self.worker.load("b"))
        f3 = asyncio.create_task(self.worker.load("c"))
        await asyncio.sleep(0.1)

        # f1 blockiert im Worker; f2, f3 in der Queue.
        # Stop kann erst durchlaufen, wenn f1 fertig ist — Blocker freigeben,
        # damit Worker zur Iterations-Spitze zurueckkehrt.
        stop_task = asyncio.create_task(self.worker.stop(timeout=3.0))
        await asyncio.sleep(0.1)

        # Submit nach stop liefert WorkerStoppedError
        with self.assertRaises(WorkerStoppedError):
            await self.worker.load("rejected")

        self.manager.load_blocker.set()
        await asyncio.wait_for(stop_task, timeout=3.0)

        # f1 ist erfolgreich fertig (Blocker wurde gesetzt, Worker hatte schon dispatched)
        await f1

        # f2, f3 muessen WorkerStoppedError haben
        with self.assertRaises(WorkerStoppedError):
            await f2
        with self.assertRaises(WorkerStoppedError):
            await f3

        # Drop wurde gerufen
        self.assertEqual(self.manager.drop_count, 1)
        self.assertFalse(self.worker._thread.is_alive())

    async def test_stop_on_empty_queue_exits_and_runs_drop(self):
        self.worker.start()

        # Idle: gar keine Jobs queued
        start = time.monotonic()
        await self.worker.stop(timeout=3.0)
        elapsed = time.monotonic() - start

        self.assertLessEqual(elapsed, 0.5, f"Stop dauerte {elapsed:.3f}s, Limit 0.5s")
        self.assertEqual(self.manager.drop_count, 1)
        self.assertFalse(self.worker._thread.is_alive())

    # ---- Exception-Recovery

    async def test_exception_in_load_keeps_worker_usable(self):
        self.worker.start()

        self.manager.load_raises = OSError("model file missing")
        with self.assertRaises(OSError):
            await self.worker.load("a")

        # Worker noch lebendig
        snap = self.worker.snapshot()
        self.assertTrue(snap["worker_thread_alive"])

        # Naechster Job muss durchlaufen
        self.manager.load_raises = None
        result = await self.worker.load("b")
        self.assertEqual(result, "b")

    # ---- Cancellation-Future-Schutz: kein Result/Exception auf cancelled

    async def test_cancelled_future_never_gets_result_or_exception(self):
        # Setup: load blockiert, awaiter cancelt, worker laeuft trotzdem zu Ende.
        # Die Helper-Funktionen muessen das set_result/set_exception auf
        # cancelled future ueberspringen.
        self.manager.load_blocker = threading.Event()
        self.worker.start()

        f1 = asyncio.create_task(self.worker.load("a"))
        await asyncio.sleep(0.1)

        # Future ueber den Job zugaenglich machen — wir holen aus dem Worker
        # die Future. Weil _submit das Future zurueckgibt, fangen wir hier
        # ueber den hooks: einfach den awaiter cancellen.
        f1.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await f1

        # Worker laeuft noch in _ensure_model_sync; release jetzt.
        self.manager.load_blocker.set()
        await asyncio.sleep(0.1)
        # Hier wuerde InvalidStateError fliegen, wenn unser Helper falsch
        # waere — wir testen indirekt via "kein Crash, naechster Job klappt".

        result = await self.worker.load("b")
        self.assertEqual(result, "b")

    # ---- Snapshot waehrend laufendem Job

    async def test_snapshot_consistent_during_running_job(self):
        self.manager.encode_blocker = threading.Event()
        # Pre-load damit encode loslaufen kann
        self.worker.start()
        await self.worker.load("a")

        encode_task = asyncio.create_task(self.worker.encode("a", ["text"], None))
        await asyncio.sleep(0.05)

        snap = self.worker.snapshot()
        self.assertEqual(snap["current_job"], "encode")
        self.assertTrue(snap["worker_busy"] if "worker_busy" in snap else snap["current_job"] is not None)
        self.assertEqual(snap["current_model"], "a")

        self.manager.encode_blocker.set()
        await asyncio.wait_for(encode_task, timeout=2.0)

        snap_after = self.worker.snapshot()
        self.assertIsNone(snap_after["current_job"])

    # ---- current_job-Tracking

    async def test_current_job_tracked_during_dispatch_and_cleared_after(self):
        self.manager.encode_blocker = threading.Event()
        self.worker.start()
        await self.worker.load("a")

        # current_job sollte None sein zwischen Jobs
        self.assertIsNone(self.worker.snapshot()["current_job"])

        encode_task = asyncio.create_task(self.worker.encode("a", ["x"], None))
        await asyncio.sleep(0.05)
        self.assertEqual(self.worker.snapshot()["current_job"], "encode")

        self.manager.encode_blocker.set()
        await asyncio.wait_for(encode_task, timeout=2.0)

        # Nach Dispatch wieder None
        await asyncio.sleep(0.05)
        self.assertIsNone(self.worker.snapshot()["current_job"])

    # ---- Status-Praezedenz: thread_alive=False

    def test_status_precedence_when_thread_alive_false(self):
        worker = SerialModelWorker(FakeManager())
        # Manuell setzen, OHNE den Thread zu killen oder os._exit auszuloesen.
        worker._started = True
        worker._accepting = True
        worker._thread_alive = False

        snap = worker.snapshot()
        self.assertFalse(snap["worker_thread_alive"])
        self.assertEqual(derive_health_status(snap), "stopping")

    # ---- Snapshot ohne Lock-Nesting

    def test_snapshot_releases_worker_lock_before_manager_snapshot(self):
        manager_snapshot_called = threading.Event()
        block_snapshot = threading.Event()
        lock_was_free = threading.Event()

        class BlockingManager(FakeManager):
            def snapshot(_self):
                manager_snapshot_called.set()
                block_snapshot.wait(timeout=2.0)
                return {
                    "current_model": None,
                    "loading_model": None,
                    "device": "cpu",
                }

        manager = BlockingManager()
        worker = SerialModelWorker(manager)

        def take_snapshot_in_thread():
            worker.snapshot()

        snap_thread = threading.Thread(target=take_snapshot_in_thread)
        snap_thread.start()
        try:
            self.assertTrue(
                manager_snapshot_called.wait(timeout=2.0),
                "manager.snapshot() wurde nie gerufen",
            )
            # Worker-Lock muss frei sein, waehrend manager.snapshot() blockiert.
            acquired = worker._lock.acquire(timeout=0.5)
            try:
                if acquired:
                    lock_was_free.set()
            finally:
                if acquired:
                    worker._lock.release()
        finally:
            block_snapshot.set()
            snap_thread.join(timeout=2.0)

        self.assertTrue(
            lock_was_free.is_set(),
            "Worker-Lock wurde waehrend manager.snapshot() gehalten — Lock-Nesting!",
        )

    # ---- Drop wirft im Stop-Pfad

    async def test_stop_handles_drop_raising(self):
        self.manager.drop_raises = RuntimeError("drop crashed")
        self.worker.start()

        # Stop darf nicht raisen, auch wenn drop crasht
        await asyncio.wait_for(self.worker.stop(timeout=2.0), timeout=3.0)
        self.assertFalse(self.worker._thread.is_alive())

    # ---- Concurrent Submit + Stop: keine pending futures

    async def test_concurrent_submit_and_stop_no_unresolved_future(self):
        worker = SerialModelWorker(self.manager, queue_max=64)
        self.worker = worker
        worker.start()

        results: list = []
        N = 20

        async def submit_one(i):
            try:
                value = await worker.load(f"m-{i}")
                results.append(("ok", i, value))
            except WorkerStoppedError:
                results.append(("stopped-await", i, None))
            except asyncio.CancelledError:
                results.append(("cancelled", i, None))

        # Setup: kein blocker; loads gehen schnell durch
        async def do_stop():
            await asyncio.sleep(0.01)
            try:
                await worker.stop(timeout=3.0)
            except Exception:
                pass

        submit_tasks = [asyncio.create_task(submit_one(i)) for i in range(N)]
        stop_task = asyncio.create_task(do_stop())

        # Wait for everything; some submits may raise WorkerStoppedError
        # synchronously inside submit_one — verzeichnen wir auch.
        # Wir sammeln die submit-task-Ergebnisse.
        for t in submit_tasks:
            try:
                await t
            except WorkerStoppedError:
                results.append(("stopped-submit-raise", -1, None))

        await stop_task

        # Keine pending Futures
        for t in submit_tasks:
            self.assertTrue(t.done(), f"task {t} blieb pending")

        # Queue darf nichts mehr enthalten
        self.assertEqual(worker._queue.qsize(), 0)

    # ---- Atomarer Modell-/Name-Set (gegen den realen ModelManager)

    def test_atomic_model_name_set_via_real_manager(self):
        import main as main_mod  # lokal importieren, Pfad ist bereits gesetzt

        manager = main_mod.ModelManager()
        manager.set_worker_thread()
        manager.device = "cpu"

        # Initialer Zustand: vorhandenes Modell, damit der Wechsel-Pfad genommen wird
        initial_obj = object()
        with manager.state_lock:
            manager.model = initial_obj
            manager.current_model_name = "mxbai-embed-large"

        new_obj = object()
        snapshots: list = []
        stop_reader = threading.Event()

        def reader():
            while not stop_reader.is_set():
                with manager.state_lock:
                    snapshots.append((manager.model, manager.current_model_name))

        with mock.patch.object(main_mod, "SentenceTransformer", return_value=new_obj), \
             mock.patch.object(manager, "_detect_device", return_value="cpu"):
            t = threading.Thread(target=reader, daemon=True)
            t.start()
            try:
                manager._load_model_sync("snowflake-arctic-embed")
            finally:
                stop_reader.set()
                t.join(timeout=2.0)

        valid_pairs = {
            (initial_obj, "mxbai-embed-large"),
            (None, None),
            (new_obj, "snowflake-arctic-embed"),
        }
        for m, n in snapshots:
            self.assertIn(
                (m, n),
                valid_pairs,
                f"inkonsistente Beobachtung: model={m!r}, name={n!r}",
            )

    # ---- thread-Target-finally setzt loading_model auf None

    async def test_worker_finally_clears_loading_model_after_crash(self):
        # Simuliere fatalen Crash mitten im Load: loading_model gesetzt,
        # dann eine BaseException die durch die per-Job-try (catches Exception)
        # bubblet -> outer try -> fatal -> finally -> clear_loading_for_crash.

        class CrashingLoad(FakeManager):
            def _ensure_model_sync(_self, name):
                _self.assert_worker_thread()
                with _self.state_lock:
                    _self.loading_model = name
                # BaseException-Subclass — entweicht der per-Job-Exception-Catch.
                raise SystemExit("crash mitten im Load")

        manager = CrashingLoad()
        worker = SerialModelWorker(manager)
        self.worker = worker

        with mock.patch("os._exit") as exit_spy:
            worker.start()

            with self.assertRaises(BaseException):
                # Wir koennen nicht sicher catch — SystemExit entweicht den
                # per-Job-catch. Der Worker verwirklicht das und triggert os._exit.
                try:
                    await asyncio.wait_for(worker.load("x"), timeout=2.0)
                except asyncio.TimeoutError:
                    raise AssertionError("Worker hat den Future nicht resolved")

            # Warte kurz, bis der Worker tatsaechlich exit gerufen hat
            for _ in range(50):
                if exit_spy.called:
                    break
                await asyncio.sleep(0.05)

        self.assertTrue(exit_spy.called, "os._exit wurde nicht gerufen")
        self.assertIsNone(manager.loading_model)

    # ---- Fatal Worker-Loop: os._exit + State-Cleanup

    async def test_worker_loop_fatal_exit_terminates_process(self):
        captured: dict = {}

        class CrashingManager(FakeManager):
            def _ensure_model_sync(_self, name):
                _self.assert_worker_thread()
                with _self.state_lock:
                    _self.loading_model = name
                raise SystemExit("simulated fatal")

        manager = CrashingManager()
        worker = SerialModelWorker(manager)
        self.worker = worker

        def spy_exit(code):
            with worker._lock:
                captured["thread_alive"] = worker._thread_alive
                captured["current_job"] = worker._current_job
            captured["loading_model"] = manager.loading_model
            captured["code"] = code

        with mock.patch("os._exit", side_effect=spy_exit) as exit_mock:
            worker.start()

            # Trigger Crash: submit ein load
            try:
                await asyncio.wait_for(worker.load("x"), timeout=2.0)
            except BaseException:
                pass

            # Warte bis spy gerufen wurde
            for _ in range(50):
                if exit_mock.called:
                    break
                await asyncio.sleep(0.05)

        self.assertTrue(exit_mock.called, "os._exit wurde nicht gerufen")
        self.assertEqual(captured.get("code"), 1)
        self.assertFalse(captured.get("thread_alive"), "thread_alive sollte False sein")
        self.assertIsNone(captured.get("current_job"), "current_job sollte None sein")
        self.assertIsNone(
            captured.get("loading_model"),
            "loading_model sollte None sein vor os._exit",
        )

    # ---- assert_worker_thread aus Event-Loop-Thread

    def test_assert_worker_thread_raises_from_other_thread(self):
        import main as main_mod

        manager = main_mod.ModelManager()
        # owner_thread_id is None — sollte sofort raisen
        with self.assertRaises(RuntimeError):
            manager.assert_worker_thread()

        # Setze owner auf einen ANDEREN Thread
        other_thread_started = threading.Event()
        other_thread_done = threading.Event()

        def other():
            manager.set_worker_thread()
            other_thread_started.set()
            other_thread_done.wait(timeout=2.0)

        t = threading.Thread(target=other, daemon=True)
        t.start()
        try:
            other_thread_started.wait(timeout=2.0)
            with self.assertRaises(RuntimeError):
                manager.assert_worker_thread()
        finally:
            other_thread_done.set()
            t.join(timeout=2.0)


class SerialModelWorkerLateEmbedTests(unittest.IsolatedAsyncioTestCase):
    """Tests fuer SerialModelWorker.late_embed (D10/D11/D15-Worker-Layer)."""

    async def asyncSetUp(self):
        self.manager = FakeManager()
        self.worker = SerialModelWorker(self.manager, queue_max=4)

    async def asyncTearDown(self):
        if self.manager.load_blocker is not None:
            self.manager.load_blocker.set()
        if self.manager.encode_blocker is not None:
            self.manager.encode_blocker.set()
        if self.manager.late_embed_blocker is not None:
            self.manager.late_embed_blocker.set()
        if self.worker._started:
            try:
                await self.worker.stop(timeout=2.0)
            except Exception:
                pass

    async def test_late_embed_serial_garantie(self):
        # Zwei parallele Late-Embeds laufen seriell, Reihenfolge erhalten.
        self.manager.late_embed_blocker = threading.Event()
        self.worker.start()

        f1 = asyncio.create_task(
            self.worker.late_embed("a", "doc1", [{"text": "x", "char_start": 0, "char_end": 1}], None)
        )
        f2 = asyncio.create_task(
            self.worker.late_embed("b", "doc2", [{"text": "y", "char_start": 0, "char_end": 1}], None)
        )

        await asyncio.sleep(0.1)
        self.assertEqual(self.manager.peak_concurrent_late_embeds, 1)

        self.manager.late_embed_blocker.set()
        await asyncio.wait_for(asyncio.gather(f1, f2), timeout=3.0)

        self.assertEqual(self.manager.peak_concurrent_late_embeds, 1)
        self.assertEqual(self.manager.late_embed_count, 2)
        # Reihenfolge erhalten: erster Call mit doc1, zweiter mit doc2.
        self.assertEqual(self.manager.late_embed_calls[0][1], "doc1")
        self.assertEqual(self.manager.late_embed_calls[1][1], "doc2")

    async def test_late_embed_cancellation_pre_dispatch(self):
        # Job wartet in Queue, awaiter cancelt -> Worker verwirft ohne Modell-Touch.
        self.manager.late_embed_blocker = threading.Event()
        self.worker.start()

        # Erster Job blockt im Worker; zweiter wartet in Queue.
        f1 = asyncio.create_task(
            self.worker.late_embed("a", "d", [{"text": "x", "char_start": 0, "char_end": 1}], None)
        )
        f2 = asyncio.create_task(
            self.worker.late_embed("b", "d", [{"text": "x", "char_start": 0, "char_end": 1}], None)
        )
        await asyncio.sleep(0.1)
        self.assertTrue(self.manager.late_embed_started.is_set())
        self.assertEqual(self.manager.late_embed_count, 0)  # blockiert

        # Cancel den wartenden Job
        f2.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await f2

        self.manager.late_embed_blocker.set()
        await asyncio.wait_for(f1, timeout=2.0)

        await asyncio.sleep(0.2)
        # f2 wurde pre-dispatch verworfen — nur f1 hat das Modell beruehrt.
        self.assertEqual(self.manager.late_embed_count, 1)

    async def test_late_embed_queue_full(self):
        worker = SerialModelWorker(self.manager, queue_max=2)
        self.worker = worker
        self.manager.late_embed_blocker = threading.Event()
        worker.start()

        chunk = [{"text": "x", "char_start": 0, "char_end": 1}]
        f1 = asyncio.create_task(worker.late_embed("a", "d", chunk, None))
        f2 = asyncio.create_task(worker.late_embed("b", "d", chunk, None))
        f3 = asyncio.create_task(worker.late_embed("c", "d", chunk, None))
        await asyncio.sleep(0.1)

        with self.assertRaises(WorkerQueueFullError):
            await worker.late_embed("d", "d", chunk, None)

        self.manager.late_embed_blocker.set()
        await asyncio.wait_for(asyncio.gather(f1, f2, f3), timeout=3.0)

    async def test_late_embed_stop_drain(self):
        # worker.stop() waehrend Late-Embed-Job pending -> WorkerStoppedError an Future.
        self.manager.late_embed_blocker = threading.Event()
        self.worker.start()

        chunk = [{"text": "x", "char_start": 0, "char_end": 1}]
        f1 = asyncio.create_task(self.worker.late_embed("a", "d", chunk, None))
        f2 = asyncio.create_task(self.worker.late_embed("b", "d", chunk, None))
        f3 = asyncio.create_task(self.worker.late_embed("c", "d", chunk, None))
        await asyncio.sleep(0.1)

        stop_task = asyncio.create_task(self.worker.stop(timeout=3.0))
        await asyncio.sleep(0.1)

        with self.assertRaises(WorkerStoppedError):
            await self.worker.late_embed("rejected", "d", chunk, None)

        self.manager.late_embed_blocker.set()
        await asyncio.wait_for(stop_task, timeout=3.0)

        # f1 lief schon im Worker, kommt durch
        await f1
        # f2, f3 wurden gedraint
        with self.assertRaises(WorkerStoppedError):
            await f2
        with self.assertRaises(WorkerStoppedError):
            await f3

        self.assertEqual(self.manager.drop_count, 1)
        self.assertFalse(self.worker._thread.is_alive())

    async def test_late_embed_dispatch_payload(self):
        # Job mit Payload landet bei _late_embed_sync mit korrekten Argumenten.
        self.worker.start()

        chunk = [{"text": "abc", "char_start": 0, "char_end": 3}]
        result = await self.worker.late_embed("nomic-embed-text", "abc def", chunk, "search_query")

        self.assertIsInstance(result, LateEmbedResult)
        self.assertEqual(self.manager.late_embed_count, 1)
        recorded_name, recorded_doc, recorded_chunks, recorded_input = (
            self.manager.late_embed_calls[0]
        )
        self.assertEqual(recorded_name, "nomic-embed-text")
        self.assertEqual(recorded_doc, "abc def")
        self.assertEqual(recorded_chunks, chunk)
        self.assertEqual(recorded_input, "search_query")

    async def test_late_embed_atomicity(self):
        # Stop und Submit konkurrent -> entweder WorkerStoppedError oder ordentliche
        # Verarbeitung; kein InvalidStateError.
        worker = SerialModelWorker(self.manager, queue_max=64)
        self.worker = worker
        worker.start()

        chunk = [{"text": "x", "char_start": 0, "char_end": 1}]
        results: list = []
        N = 20

        async def submit_one(i):
            try:
                r = await worker.late_embed(f"m-{i}", "d", chunk, None)
                results.append(("ok", i, r))
            except WorkerStoppedError:
                results.append(("stopped-await", i, None))
            except asyncio.CancelledError:
                results.append(("cancelled", i, None))

        async def do_stop():
            await asyncio.sleep(0.01)
            try:
                await worker.stop(timeout=3.0)
            except Exception:
                pass

        submit_tasks = [asyncio.create_task(submit_one(i)) for i in range(N)]
        stop_task = asyncio.create_task(do_stop())

        for t in submit_tasks:
            try:
                await t
            except WorkerStoppedError:
                results.append(("stopped-submit-raise", -1, None))

        await stop_task

        for t in submit_tasks:
            self.assertTrue(t.done(), f"task {t} blieb pending")
        self.assertEqual(worker._queue.qsize(), 0)


class ModelManagerOomCleanupTests(unittest.TestCase):
    def test_encode_oom_drops_loaded_model(self):
        import main as main_mod

        class OomModel:
            max_seq_length = 512

            def tokenizer(self, texts, **kwargs):
                return {"input_ids": [[1, 2, 3] for _ in texts]}

            def encode(self, texts, batch_size, convert_to_numpy):
                raise main_mod.torch.cuda.OutOfMemoryError("simulated oom")

        manager = main_mod.ModelManager()
        manager.set_worker_thread()
        manager.device = "cpu"
        model = OomModel()
        with manager.state_lock:
            manager.model = model
            manager.current_model_name = "mxbai-embed-large"

        with mock.patch.object(
            manager, "_ensure_model_sync", return_value=(model, 0)
        ):
            with self.assertRaises(main_mod.torch.cuda.OutOfMemoryError):
                manager._encode_sync("mxbai-embed-large", ["hello"], None)

        self.assertIsNone(manager.model)
        self.assertIsNone(manager.current_model_name)


# --- #757 Regression --------------------------------------------------------


class IssueSeven57RegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_during_load_does_not_spawn_parallel_load(self):
        manager = FakeManager()
        manager.load_blocker = threading.Event()
        worker = SerialModelWorker(manager, queue_max=8)
        worker.start()
        try:
            # Load A startet und blockt
            f_a = asyncio.create_task(worker.load("a"))
            await asyncio.sleep(0.1)
            self.assertTrue(manager.load_started.is_set())
            self.assertEqual(manager.peak_concurrent_loads, 1)

            # Awaiter A wird gecancelt
            f_a.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await f_a

            # Load B wird submitted — darf NICHT starten, solange A noch laeuft
            f_b = asyncio.create_task(worker.load("b"))
            await asyncio.sleep(0.1)
            self.assertEqual(manager.peak_concurrent_loads, 1)

            # Release A — Worker kann zu B
            manager.load_blocker.set()
            await asyncio.wait_for(f_b, timeout=2.0)

            # Beweise: nie mehr als ein aktiver Sync-Load
            self.assertEqual(manager.peak_concurrent_loads, 1)
            self.assertEqual(manager.load_count, 2)
        finally:
            manager.load_blocker.set()
            await worker.stop(timeout=2.0)


# --- #880 Regression --------------------------------------------------------


class IssueEightEightyRegressionTests(unittest.TestCase):
    """#880: _ensure_model_sync probt das Device vor dem Short-Circuit, sodass
    CPU-Fallback nach CUDA-Recovery bzw. stales CUDA-Modell nach CUDA-Tod
    beim naechsten Encode erkannt und neu geladen wird."""

    def _make_manager(self):
        import main as main_mod

        manager = main_mod.ModelManager()
        manager.set_worker_thread()
        return manager

    def test_short_circuit_when_device_unchanged(self):
        manager = self._make_manager()
        manager.device = "cuda"
        cached_obj = object()
        with manager.state_lock:
            manager.model = cached_obj
            manager.current_model_name = "mxbai-embed-large"

        with mock.patch.object(manager, "_detect_device", return_value="cuda") as probe, \
             mock.patch.object(manager, "_load_model_sync") as loader:
            model, load_ns = manager._ensure_model_sync("mxbai-embed-large")

        self.assertIs(model, cached_obj)
        self.assertEqual(load_ns, 0)
        probe.assert_called_once()
        loader.assert_not_called()

    def test_forces_reload_when_cuda_died(self):
        manager = self._make_manager()
        manager.device = "cuda"
        cached_obj = object()
        with manager.state_lock:
            manager.model = cached_obj
            manager.current_model_name = "mxbai-embed-large"

        new_obj = object()
        with mock.patch.object(manager, "_detect_device", return_value="cpu"), \
             mock.patch.object(
                 manager, "_load_model_sync", return_value=new_obj
             ) as loader:
            model, _ = manager._ensure_model_sync("mxbai-embed-large")

        self.assertIs(model, new_obj)
        loader.assert_called_once_with("mxbai-embed-large")

    def test_forces_reload_when_cuda_recovered(self):
        manager = self._make_manager()
        manager.device = "cpu"
        cached_obj = object()
        with manager.state_lock:
            manager.model = cached_obj
            manager.current_model_name = "mxbai-embed-large"

        new_obj = object()
        with mock.patch.object(manager, "_detect_device", return_value="cuda"), \
             mock.patch.object(
                 manager, "_load_model_sync", return_value=new_obj
             ) as loader:
            model, _ = manager._ensure_model_sync("mxbai-embed-large")

        self.assertIs(model, new_obj)
        loader.assert_called_once_with("mxbai-embed-large")


if __name__ == "__main__":
    unittest.main()
