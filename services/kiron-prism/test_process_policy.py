"""Bounded offline process/permission checks; no model or systemd unit is started."""
import asyncio
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from main import control_socket
from kiron_common.prism_runtime_policy import PolicyError, checked_open
from process import NativeChild, group_running, require_free_port


class PolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "artifact.gguf"
        self.path.write_bytes(b"fixture")
        self.sha = hashlib.sha256(b"fixture").hexdigest()

    def test_hash_and_size_validation_close_failed_descriptors(self):
        fd = checked_open(self.path, self.sha, 7, anchor=self.root)
        self.assertEqual(os.read(fd, 7), b"fixture")
        os.close(fd)
        for sha, size in (("f" * 64, 7), (self.sha, 8)):
            with self.assertRaises(PolicyError):
                checked_open(self.path, sha, size, anchor=self.root)

    def test_symlink_hardlink_fifo_and_writable_paths_are_rejected(self):
        linked = self.root / "linked"
        linked.symlink_to(self.path)
        with self.assertRaises(PolicyError):
            checked_open(linked, self.sha, anchor=self.root)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        with self.assertRaises(PolicyError):
            checked_open(fifo, self.sha, anchor=self.root)
        self.path.chmod(0o666)
        with self.assertRaises(PolicyError):
            checked_open(self.path, self.sha, anchor=self.root)
        self.path.chmod(0o644)
        os.link(self.path, self.root / "hardlink")
        with self.assertRaises(PolicyError):
            checked_open(self.path, self.sha, anchor=self.root)

    def test_file_swap_between_path_validation_and_open_is_rejected(self):
        real_open = os.open
        def swapped(path, flags):
            replacement = self.root / "replacement"
            replacement.write_bytes(b"fixture")
            replacement.replace(self.path)
            return real_open(path, flags)
        with mock.patch("kiron_common.prism_runtime_policy.os.open", side_effect=swapped):
            with self.assertRaisesRegex(PolicyError, "changed"):
                checked_open(self.path, self.sha, anchor=self.root)

    def test_root_native_launch_and_control_socket_are_rejected_before_spawn(self):
        with mock.patch("process.os.geteuid", return_value=0), mock.patch("process.subprocess.Popen") as spawn:
            with self.assertRaises(PermissionError):
                NativeChild([], {}, [], None)
            spawn.assert_not_called()
        with mock.patch("main.os.geteuid", return_value=0):
            with self.assertRaises(PermissionError), control_socket(self.root / "control.sock", control_gid=0):
                pass

    def test_foreign_loopback_listener_is_rejected_without_contact(self):
        for reuse in (0, 1):
            with self.subTest(reuse=reuse), socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuse)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                with self.assertRaises(OSError):
                    require_free_port(listener.getsockname()[1])
                listener.settimeout(0.05)
                with self.assertRaises(TimeoutError):
                    listener.accept()

    def test_ended_connection_time_wait_does_not_block_native_restart(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            client.settimeout(2)
            client.connect(("127.0.0.1", port))
            connection, _ = listener.accept()
            connection.close()  # The server side becomes the active closer.
            self.assertEqual(client.recv(1), b'')
        # Reproduce the old guard's failure against an actually ended socket.
        with socket.socket() as strict:
            with self.assertRaises(OSError):
                strict.bind(("127.0.0.1", port))
        require_free_port(port)

    @unittest.skipUnless(os.geteuid() == 0, "real unprivileged socket fixture needs root to drop UID")
    def test_actual_non_root_socket_permissions_and_singleton(self):
        self.root.chmod(0o755)
        directory = self.root / "control"
        directory.mkdir()
        os.chown(directory, 65534, 65534)
        directory.chmod(0o750)
        # A tiny Python socket fixture under nobody, never a model/HTTP server.
        script = """import os, stat, sys
from pathlib import Path
from main import control_socket
path = Path(sys.argv[1])
assert os.geteuid() == 65534
with control_socket(path, control_gid=65534):
    info = path.stat()
    assert stat.S_IMODE(info.st_mode) == 0o660
    assert info.st_uid == info.st_gid == 65534
    try:
        with control_socket(path, control_gid=65534):
            raise AssertionError('duplicate controller')
    except BlockingIOError:
        pass
assert not path.exists()
"""
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent),
                                             str(Path(__file__).parents[1] / "kiron-common")))}
        result = subprocess.run([sys.executable, "-c", script, str(directory / "control.sock")],
                                env=env, user=65534, group=65534, extra_groups=[],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_pid_cannot_claim_successful_health(self):
        child = NativeChild.__new__(NativeChild)
        child.policy = SimpleNamespace(health_timeout=0.1)
        child.alive = mock.Mock(return_value=True)
        child.owns_port = mock.Mock(return_value=False)
        child._json = mock.AsyncMock()
        self.assertFalse(await child.ready("expected"))
        child._json.assert_not_awaited()

    async def test_readiness_requires_model_alias_and_total_deadline(self):
        child = NativeChild.__new__(NativeChild)
        child.policy = SimpleNamespace(health_timeout=0.02)
        child.alive = mock.Mock(return_value=True)
        child.owns_port = mock.Mock(return_value=True)
        child._json = mock.AsyncMock(side_effect=[{"status": "ok"}, {"data": [{"id": "foreign"}]}])
        self.assertFalse(await child.ready("expected"))
        async def slow(path):
            await asyncio.sleep(1)
        child._json.side_effect = slow
        self.assertFalse(await child.ready("expected"))

    async def test_slot_snapshot_is_strict_and_does_not_expose_prompt_metadata(self):
        child = NativeChild.__new__(NativeChild)
        child.policy = SimpleNamespace(health_timeout=0.1)
        child.alive = mock.Mock(return_value=True)
        child.owns_port = mock.Mock(return_value=True)
        child._json = mock.AsyncMock(return_value=[{"id": 0, "id_task": 8, "is_processing": True,
                                                   "prompt": "fixture must not escape"}])
        result = await child.slots()
        self.assertEqual(result["active_requests"], 1)
        self.assertEqual(result["slot_task_id"], 8)
        self.assertNotIn("prompt", result)
        for value in ([], [{"id": 0, "is_processing": "false"}],
                      [{"id": 0, "is_processing": True}], [{"id": 0, "is_processing": False, "id_task": True}]):
            child._json.return_value = value
            self.assertIsNone(await child.slots())

    async def test_group_cleanup_retains_leader_until_term_resistant_descendant_is_killed(self):
        script = """import os, signal, time
r,w=os.pipe()
if os.fork():
    os.close(w)
    os.read(r,1)
    os._exit(0)
os.close(r)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
os.write(w,b'x')
os.close(w)
time.sleep(10)
"""
        process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child = NativeChild.__new__(NativeChild)
        child.process = process
        child.policy = SimpleNamespace(term_timeout=0.1, kill_timeout=0.5)
        child.client = SimpleNamespace(aclose=mock.AsyncMock())
        try:
            for _ in range(100):
                if not child.alive():
                    break
                await asyncio.sleep(0.005)
            self.assertFalse(child.alive())
            self.assertIsNone(process.returncode, "leader must remain reserved until final group signal")
            self.assertTrue(await child.stop())
            self.assertFalse(group_running(process.pid))
            self.assertIsNotNone(process.returncode)
            with mock.patch("process.os.killpg") as kill:
                self.assertTrue(await child.stop())
                kill.assert_not_called()
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)

    def test_unit_keeps_controller_and_children_in_one_unprivileged_cgroup(self):
        unit = (Path(__file__).parents[2] / "systemd/kiron-prism.service").read_text()
        for setting in ("User=kiron-prism", "Group=kiron-prism", "NoNewPrivileges=true",
                        "ProtectSystem=strict", "ProtectHome=true", "KillMode=mixed",
                        "TimeoutStopSec=65", "SendSIGKILL=yes", "CapabilityBoundingSet=",
                        "ReadWritePaths=/run/kiron/prism /run/kiron/vram"):
            self.assertIn(setting, unit)
        self.assertNotIn("test-venvs", unit)
        self.assertNotIn("test-runtimes", unit)


if __name__ == "__main__":
    unittest.main()
