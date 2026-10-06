"""Offline tests: no binary, GPU command or HTTP connection is started."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prism_supervisor", Path(__file__).with_name("supervise-probe.py"))
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)
SAFE = {"ram_available_bytes": 8 * 1024**3, "gpu_used_bytes": 6 * 1024**3,
        "gpus": [{"used_bytes": 6 * 1024**3, "total_bytes": 12 * 1024**3}], "process_rss_bytes": 1024}


class SupervisorTests(unittest.TestCase):
    def test_health_total_deadline_interrupts_a_trickling_body(self):
        connection = mock.Mock()
        released = threading.Event()
        connection.sock.shutdown.side_effect = lambda how: released.set()
        reply = connection.getresponse.return_value
        reply.status = 200
        # Model read() receiving bytes forever: only socket shutdown releases it.
        def slow_body(limit):
            self.assertTrue(released.wait(0.5), "health did not close the socket at its deadline")
            return b'{"status":"ok"}'
        reply.read.side_effect = slow_body
        with mock.patch.object(supervisor.http.client, "HTTPConnection", return_value=connection):
            result = supervisor.health(18089, timeout=0.02)
        self.assertEqual(result, {"ready": False, "error": "health_total_timeout"})
        connection.sock.shutdown.assert_called_once_with(supervisor.socket.SHUT_RDWR)
        connection.close.assert_called_once()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.binary = self.base / "llama-server"
        self.binary.write_bytes(b"not a real binary")
        self.binary.chmod(0o755)
        self.model = self.base / "model.gguf"
        self.model.write_bytes(b"fixture")
        self.args = argparse.Namespace(binary=self.binary, model=self.model, mmproj=None,
                                       output_dir=self.base / "report", port=18089, gpu_layers=0, threads=2,
                                       library_path=None, startup_timeout=2, max_lifetime=5)
        self.clock = 0.0
        self.process = mock.Mock(pid=123456, returncode=None)
        self.process.poll.return_value = None

    def sleep(self, seconds):
        self.clock += seconds

    def run_probe(self, *, measured=None, ready=True, sleep=None, stopped=None):
        with mock.patch.object(supervisor, "LOCK_PATH", self.base / "probe.lock"), \
                mock.patch.object(supervisor.socket, "socket"), \
                mock.patch.object(supervisor, "resources", side_effect=measured or [SAFE] * 10), \
                mock.patch.object(supervisor, "health", return_value={"ready": ready}), \
                mock.patch.object(supervisor, "stop_process", side_effect=stopped) as stop, \
                mock.patch.object(supervisor.time, "monotonic", side_effect=lambda: self.clock), \
                mock.patch.object(supervisor.time, "sleep", side_effect=sleep or self.sleep), \
                mock.patch.object(supervisor.subprocess, "Popen", return_value=self.process) as popen:
            result = supervisor.probe(self.args)
        return result, popen, stop

    def test_fixed_argv_cpu_and_vision_disable_native_execution(self):
        argv = supervisor.command(self.args)
        for flag in ("--jinja", "--no-warmup", "--no-context-shift", "--no-agent", "--no-webui", "--no-ui-mcp-proxy", "--no-mmproj"):
            self.assertIn(flag, argv)
        for name, value in (("--alias", "bonsai-probe"), ("--log-verbosity", "4"), ("--ctx-size", "1024"), ("--parallel", "1"), ("--batch-size", "128"),
                            ("--threads", "2"), ("--threads-batch", "2"), ("--fit", "off"), ("--cache-ram", "0"),
                            ("--reasoning", "off"), ("--reasoning-format", "deepseek"), ("--device", "none")):
            self.assertEqual(argv[argv.index(name) + 1], value)
        self.args.mmproj, self.args.gpu_layers = self.model, 4
        argv = supervisor.command(self.args)
        self.assertIn("--no-mmproj-offload", argv)
        self.assertNotIn("--device", argv)
        self.assertNotIn("--no-mmproj", argv)
        self.assertEqual(argv[argv.index("--mmproj") + 1], str(self.model))

    def test_threads_share_one_bounded_inference_and_batch_setting(self):
        for threads in (1, 4):
            with self.subTest(threads=threads):
                self.args.threads = threads
                supervisor.validate(self.args)
                argv = supervisor.command(self.args)
                for flag in ("--threads", "--threads-batch"):
                    self.assertEqual(argv[argv.index(flag) + 1], str(threads))

    def test_environment_never_inherits_agent_cloud_or_library_overrides(self):
        with mock.patch.dict(supervisor.os.environ, {"LLAMA_ARG_AGENT": "1", "LLAMA_ARG_MCP_SERVERS": "evil", "LD_LIBRARY_PATH": "/evil", "API_KEY": "secret"}):
            self.assertEqual(supervisor.environment(self.args), {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
            self.args.library_path = str(self.base)
            self.assertEqual(supervisor.environment(self.args)["LD_LIBRARY_PATH"], str(self.base))

    def test_bad_paths_ports_and_limits_are_rejected_before_spawn(self):
        for field, value in (("port", 11442), ("port", 8505), ("port", 80), ("gpu_layers", 66),
                             ("gpu_layers", -1), ("threads", 0), ("threads", 5),
                             ("max_lifetime", 1801), ("startup_timeout", 181),
                             ("binary", Path("relative")), ("library_path", ":/tmp")):
            with self.subTest(field=field, value=value):
                changed = argparse.Namespace(**vars(self.args))
                setattr(changed, field, value)
                with self.assertRaises(ValueError):
                    supervisor.validate(changed)
        link = self.base / "linked-model"
        link.symlink_to(self.model)
        self.args.model = link
        with self.assertRaisesRegex(ValueError, "symlinks"):
            supervisor.validate(self.args)

    def test_ready_then_stop_file_cleans_up_and_records_exact_launch(self):
        def stop_after_first_tick(seconds):
            self.sleep(seconds)
            (self.args.output_dir / "STOP").touch()
        result, popen, stop = self.run_probe(sleep=stop_after_first_tick)
        self.assertEqual(result, 0)
        stop.assert_called_once_with(self.process)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["cwd"], "/")
        if supervisor.os.geteuid() == 0:
            self.assertEqual(popen.call_args.kwargs["user"], supervisor.pwd.getpwnam("nobody").pw_uid)
            self.assertEqual(popen.call_args.kwargs["group"], supervisor.grp.getgrnam("kiron-common").gr_gid)
            self.assertEqual(popen.call_args.kwargs["extra_groups"], [])
        record = json.loads((self.args.output_dir / "result.json").read_text())
        self.assertEqual(record["reason"], "stop_file")
        self.assertEqual(record["samples"], 1)
        self.assertTrue((self.args.output_dir / "ready.json").exists())
        self.assertEqual(json.loads((self.args.output_dir / "launch.json").read_text())["argv"], supervisor.command(self.args))

    def test_lifetime_and_startup_timeout_are_finite_and_cleanup(self):
        result, _, stop = self.run_probe(ready=False)
        self.assertEqual(result, 1)
        self.assertLessEqual(self.clock, self.args.startup_timeout)
        self.assertEqual(json.loads((self.args.output_dir / "result.json").read_text())["reason"], "startup_timeout")
        stop.assert_called_once_with(self.process)

    def test_healthy_probe_stops_at_lifetime(self):
        result, _, stop = self.run_probe()
        self.assertEqual(result, 0)
        self.assertEqual(self.clock, self.args.max_lifetime)
        self.assertEqual(json.loads((self.args.output_dir / "result.json").read_text())["reason"], "max_lifetime")
        stop.assert_called_once_with(self.process)

    def test_ram_gpu_thresholds_and_runtime_abort(self):
        self.assertIsNone(supervisor.resource_failure({**SAFE, "gpu_used_bytes": supervisor.MAX_GPU_BYTES, "ram_available_bytes": supervisor.MIN_RAM_BYTES}))
        self.assertEqual(supervisor.resource_failure({**SAFE, "ram_available_bytes": supervisor.MIN_RAM_BYTES - 1}), "ram_limit_exceeded")
        danger = {**SAFE, "gpu_used_bytes": supervisor.MAX_GPU_BYTES + 1}
        result, _, stop = self.run_probe(measured=[SAFE, danger])
        self.assertEqual(result, 1)
        self.assertEqual(json.loads((self.args.output_dir / "result.json").read_text())["reason"], "gpu_limit_exceeded")
        stop.assert_called_once_with(self.process)

    def test_bad_preflight_never_starts_binary(self):
        with mock.patch.object(supervisor, "LOCK_PATH", self.base / "lock"), \
                mock.patch.object(supervisor.socket, "socket"), \
                mock.patch.object(supervisor, "resources", return_value={**SAFE, "ram_available_bytes": 0}), \
                mock.patch.object(supervisor.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "preflight"):
                supervisor.probe(self.args)
            popen.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_unreadable_runtime_telemetry_still_stops_and_records_failure(self):
        def failed_measurement(pid=None):
            if pid is not None:
                raise subprocess.TimeoutExpired("nvidia-smi", 2)
            return SAFE
        with mock.patch.object(supervisor, "LOCK_PATH", self.base / "lock"), \
                mock.patch.object(supervisor.socket, "socket"), \
                mock.patch.object(supervisor, "resources", side_effect=failed_measurement), \
                mock.patch.object(supervisor.subprocess, "Popen", return_value=self.process), \
                mock.patch.object(supervisor, "stop_process") as stop:
            with self.assertRaises(subprocess.TimeoutExpired):
                supervisor.probe(self.args)
            stop.assert_called_once_with(self.process)
        self.assertEqual(json.loads((self.args.output_dir / "result.json").read_text())["reason"], "supervisor_error:TimeoutExpired")

    def test_gpu_measurement_subprocess_has_two_second_timeout(self):
        with mock.patch.object(Path, "read_text", return_value="MemAvailable: 8388608 kB\n"), \
                mock.patch.object(supervisor.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 2)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                supervisor.resources()
            self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_group_cleanup_escalates_and_reaps_without_host_signals(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired("child", 10), -9]
        with mock.patch.object(supervisor.os, "killpg") as kill:
            supervisor.stop_process(self.process)
        self.assertEqual(kill.call_args_list, [mock.call(self.process.pid, signal.SIGTERM), mock.call(self.process.pid, signal.SIGKILL)])
        self.assertEqual(self.process.wait.call_args_list, [mock.call(timeout=10), mock.call(timeout=5)])

    def test_successful_term_does_not_signal_reaped_process_group_again(self):
        self.process.wait.return_value = -15
        with mock.patch.object(supervisor.os, "killpg") as kill:
            supervisor.stop_process(self.process)
        kill.assert_called_once_with(self.process.pid, signal.SIGTERM)
        self.process.wait.assert_called_once_with(timeout=10)

    def test_already_reaped_child_never_receives_a_process_group_signal(self):
        self.process.returncode = 0
        with mock.patch.object(supervisor.os, "killpg") as kill:
            supervisor.stop_process(self.process)
        kill.assert_not_called()
        self.process.wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
