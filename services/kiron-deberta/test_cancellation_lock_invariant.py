"""Regression: Lock muss waehrend laufendem Predict-Thread gehalten bleiben (#835)."""
import asyncio
import threading
import time
import unittest

import main


class ShieldedToThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_waits_for_thread_completion(self):
        """CancelledError darf erst propagieren, wenn der Worker-Thread fertig ist."""
        thread_done = threading.Event()
        worker_finished = []

        def slow_work():
            time.sleep(0.2)
            worker_finished.append(time.monotonic())
            thread_done.set()
            return "done"

        async def runner():
            return await main._shielded_to_thread(slow_work)

        task = asyncio.create_task(runner())
        await asyncio.sleep(0.05)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        cancel_observed = time.monotonic()
        self.assertTrue(thread_done.is_set(),
                        "Worker-Thread muss vor CancelledError-Propagation abgeschlossen sein")
        self.assertGreaterEqual(cancel_observed, worker_finished[0])

    async def test_normal_completion_returns_value(self):
        async def runner():
            return await main._shielded_to_thread(lambda: 42)

        self.assertEqual(await runner(), 42)

    async def test_exception_in_worker_propagates(self):
        def boom():
            raise RuntimeError("worker boom")

        with self.assertRaises(RuntimeError) as ctx:
            await main._shielded_to_thread(boom)
        self.assertIn("worker boom", str(ctx.exception))

    async def test_lock_held_until_thread_completes_on_cancel(self):
        """Lock-Invariante: Lock wird erst nach Thread-Ende freigegeben."""
        lock = asyncio.Lock()
        worker_done = threading.Event()

        def slow_work():
            time.sleep(0.15)
            worker_done.set()

        async def holder():
            async with lock:
                await main._shielded_to_thread(slow_work)

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.03)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(worker_done.is_set())
        self.assertFalse(lock.locked(),
                         "Lock muss nach Thread-Ende freigegeben sein")

    async def test_worker_exception_after_cancel_signaled_via_callback(self):
        """#855: Worker-Exception nach Cancellation muss an on_cancel_error gemeldet werden."""
        observed: list[BaseException] = []

        def boom_after_delay():
            time.sleep(0.15)
            raise RuntimeError("CUDA out of memory")

        async def runner():
            return await main._shielded_to_thread(
                boom_after_delay,
                on_cancel_error=observed.append,
            )

        task = asyncio.create_task(runner())
        await asyncio.sleep(0.03)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(len(observed), 1, "Callback muss genau einmal aufgerufen werden")
        self.assertIsInstance(observed[0], RuntimeError)
        self.assertIn("CUDA out of memory", str(observed[0]))

    async def test_no_callback_does_not_break_cancel_flow(self):
        """Backwards-Kompatibilitaet: ohne on_cancel_error verhaelt sich Helper wie zuvor."""
        def boom_after_delay():
            time.sleep(0.1)
            raise RuntimeError("worker boom")

        async def runner():
            return await main._shielded_to_thread(boom_after_delay)

        task = asyncio.create_task(runner())
        await asyncio.sleep(0.02)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_callback_exception_does_not_break_cancel_flow(self):
        """Falls Callback selbst raised, CancelledError muss trotzdem propagieren."""
        def boom_after_delay():
            time.sleep(0.1)
            raise RuntimeError("worker CUDA boom")

        def bad_callback(exc):
            raise ValueError("callback broken")

        async def runner():
            return await main._shielded_to_thread(
                boom_after_delay, on_cancel_error=bad_callback,
            )

        task = asyncio.create_task(runner())
        await asyncio.sleep(0.02)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_double_cancel_lock_held_until_thread_completes(self):
        """#1032: Auch bei Doppel-Cancel waehrend Recovery-Wait bleibt Lock bis Thread-Ende."""
        lock = asyncio.Lock()
        worker_done = threading.Event()

        def slow_work():
            time.sleep(0.2)
            worker_done.set()

        async def holder():
            async with lock:
                await main._shielded_to_thread(slow_work)

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.03)
        task.cancel()
        await asyncio.sleep(0.03)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(worker_done.is_set(),
                        "Worker-Thread muss vor CancelledError-Propagation abgeschlossen sein")
        self.assertFalse(lock.locked(),
                         "Lock muss nach Thread-Ende freigegeben sein, auch bei Doppel-Cancel")


if __name__ == "__main__":
    unittest.main()
