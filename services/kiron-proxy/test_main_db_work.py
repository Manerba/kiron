"""Stdlib-Unittest fuer main.DBWorkTracker (#644).

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_main_db_work
"""

import asyncio
import json
import os
import pathlib
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402


class DBWorkTrackerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_does_not_cancel_worker_thread(self):
        tracker = main.DBWorkTracker()
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def slow_work():
            started.set()
            release.wait()
            finished.set()
            return "ok"

        task = asyncio.create_task(tracker.to_thread(slow_work))
        self.assertTrue(await asyncio.to_thread(started.wait, 1.0))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertFalse(finished.is_set())
        release.set()
        drained = await tracker.drain(timeout=1.0)

        self.assertTrue(drained)
        self.assertTrue(finished.is_set())


class PrebindTcpSocketsTests(unittest.TestCase):
    def _free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
        finally:
            sock.close()

    def _listening_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock

    def test_prebind_tcp_sockets_reserves_listening_port(self):
        sockets = main._prebind_tcp_sockets(
            (("test", "127.0.0.1", 0),),
            backlog=1,
        )
        try:
            bound = sockets["test"]
            port = bound.getsockname()[1]
            self.assertFalse(bound.getblocking())

            rival = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                rival.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with self.assertRaises(OSError):
                    rival.bind(("127.0.0.1", port))
            finally:
                rival.close()
        finally:
            main._close_sockets(sockets.values())

    def test_prebind_tcp_sockets_closes_previous_socket_on_later_conflict(self):
        first_port = self._free_port()
        conflict = self._listening_socket()
        try:
            conflict_port = conflict.getsockname()[1]
            with self.assertRaises(OSError):
                main._prebind_tcp_sockets(
                    (
                        ("first", "127.0.0.1", first_port),
                        ("conflict", "127.0.0.1", conflict_port),
                    ),
                    backlog=1,
                )

            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", first_port))
                probe.listen(1)
            finally:
                probe.close()
        finally:
            conflict.close()


class StartupPortPrebindOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_prebind_failure_happens_before_store_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            fake_main = root / "services" / "kiron-proxy" / "main.py"
            data_dir = root / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "db_config.json").write_text(json.dumps({
                "host": "127.0.0.1",
                "user": "kiron",
                "password": "secret",
                "database": "kiron",
            }))

            real_path = pathlib.Path

            def fake_path(*args, **kwargs):
                if args == (main.__file__,) and not kwargs:
                    return fake_main
                return real_path(*args, **kwargs)

            with mock.patch.object(main, "Path", new=fake_path), \
                    mock.patch.object(main, "_prebind_tcp_sockets", side_effect=OSError("port busy")) as prebind, \
                    mock.patch.object(main, "RequestStore") as request_store:
                with self.assertRaisesRegex(OSError, "port busy"):
                    await main.main()

            prebind.assert_called_once_with()
            request_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
