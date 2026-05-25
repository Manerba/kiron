#!/usr/bin/env python3
"""Fast unit tests for scripts/check-ollama-compat.py.

These tests do not start Docker or call a real Ollama instance.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).with_name("check-ollama-compat.py")


def load_module():
    spec = importlib.util.spec_from_file_location("check_ollama_compat", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ValidatorUnitTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_host_ollama_missing_is_diagnostic_only(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        with mock.patch.object(subprocess, "run", side_effect=FileNotFoundError("ollama")):
            self.assertIsNone(validator._host_ollama_version())

    def test_gpu_visible_check_rejects_empty_nvidia_smi(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._docker = lambda args, check=True: subprocess.CompletedProcess(args, 0, "", "")
        result = validator._gpu_visible_check()
        self.assertFalse(result.ok)
        self.assertIn("no GPU lines", result.message)

    def test_gpu_visible_check_rejects_nonzero_returncode(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._docker = lambda args, check=True: subprocess.CompletedProcess(
            args, 127, "", "nvidia-smi: not found"
        )
        result = validator._gpu_visible_check()
        self.assertFalse(result.ok)
        self.assertIn("nvidia-smi -L failed", result.message)

    def test_gpu_visible_check_accepts_gpu_listing(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._docker = lambda args, check=True: subprocess.CompletedProcess(
            args, 0, "GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-abc)\n", ""
        )
        result = validator._gpu_visible_check()
        self.assertTrue(result.ok)
        self.assertEqual(result.data["gpu_count"], 1)

    def test_baseline_gpu_resident_rejects_zero_size_vram(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:135m", "size_vram": 0}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._baseline_gpu_resident_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertEqual(result.data, {"size_vram": 0})

    def test_baseline_gpu_resident_accepts_positive_size_vram(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:135m", "size_vram": 270000000}]}
            return {}

        validator._json = fake_json
        result = validator._baseline_gpu_resident_check("smollm2:135m")
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"size_vram": 270000000})

    def test_baseline_gpu_resident_rejects_bool_size_vram(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:135m", "size_vram": True}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._baseline_gpu_resident_check("smollm2:135m")
        self.assertFalse(result.ok)

    def test_num_gpu_zero_rejects_bool_size_vram(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:135m", "size_vram": False}]}
            return {}

        validator._json = fake_json
        result = validator._num_gpu_zero_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertEqual(result.data, {"num_gpu_zero_effective": False})

    def test_unload_check_verifies_when_model_absent_from_valid_ps(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "other:1b"}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None):
            result = validator._unload_check("smollm2:135m")
        self.assertTrue(result.ok)
        self.assertIn("unload verified", result.message)

    def test_unload_check_canonicalizes_tagless_model_still_loaded(self):
        # /api/ps liefert kanonische Namen (z.B. 'smollm2:latest'); ein
        # tag-loser Caller-Name muss vor dem Vergleich kanonisiert werden,
        # sonst meldet der Check faelschlich "unload verified" obwohl das
        # Modell noch geladen ist.
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:latest"}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2")
        self.assertFalse(result.ok)
        self.assertIn("still listed", result.message)

    def test_baseline_gpu_resident_canonicalizes_tagless_model(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:latest", "size_vram": 270000000}]}
            return {}

        validator._json = fake_json
        result = validator._baseline_gpu_resident_check("smollm2")
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"size_vram": 270000000})

    def test_num_gpu_zero_path_canonicalizes_tagless_model(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:latest", "size_vram": 0}]}
            return {}

        validator._json = fake_json
        self.assertTrue(
            validator._num_gpu_zero_path_effective(
                "smollm2", "/api/generate", {"prompt": "hi"}
            )
        )

    def test_unload_for_path_canonicalizes_tagless_model(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "other:latest"}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None):
            self.assertTrue(validator._unload_for_path_check("smollm2"))

    def test_unload_check_rejects_malformed_models_string(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": "smollm2:135m"}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertIn("never valid", result.message)

    def test_unload_check_rejects_models_dict_shape(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": {"smollm2:135m": {}}}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertIn("never valid", result.message)

    def test_unload_check_rejects_entries_without_string_name(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": 123, "size_vram": 1}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertIn("never valid", result.message)

    def test_unload_check_rejects_entries_missing_name(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"size_vram": 1}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertIn("never valid", result.message)

    def test_unload_check_reports_still_listed_when_model_remains(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        def fake_json(method, path, payload=None, timeout=120.0):
            if method == "GET" and path == "/api/ps":
                return {"models": [{"name": "smollm2:135m"}]}
            return {}

        validator._json = fake_json
        with mock.patch.object(self.mod.time, "sleep", lambda _s: None), \
             mock.patch.object(self.mod.time, "monotonic", side_effect=[0, 1, 31]):
            result = validator._unload_check("smollm2:135m")
        self.assertFalse(result.ok)
        self.assertIn("still listed", result.message)

    def test_cleanup_stale_removes_only_compat_containers(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        calls = []

        def fake_docker(args, check=True):
            calls.append(args)
            if args[:3] == ["ps", "-a", "--format"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    "kiron-ollama-compat-old|exited\nkiron-ollama|running\n",
                    "",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        validator._docker = fake_docker
        validator._cleanup_stale()
        self.assertIn(["rm", "-f", "kiron-ollama-compat-old"], calls)
        self.assertNotIn(["rm", "-f", "kiron-ollama"], calls)

    def test_cleanup_stale_preserves_sibling_validator_states(self):
        # Parallel laufende Sibling-Validatoren koennen Container in created
        # (waehrend docker-run-Anlauf), paused (Operator-Debug) oder removing
        # (waehrend laufendem rm) haben. Die duerfen nicht mid-flight gekillt
        # werden — nur exited/dead sind wirklich tot.
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        calls = []

        def fake_docker(args, check=True):
            calls.append(args)
            if args[:3] == ["ps", "-a", "--format"]:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    "kiron-ollama-compat-a|created\n"
                    "kiron-ollama-compat-b|paused\n"
                    "kiron-ollama-compat-c|removing\n"
                    "kiron-ollama-compat-d|restarting\n"
                    "kiron-ollama-compat-e|running\n"
                    "kiron-ollama-compat-f|exited\n"
                    "kiron-ollama-compat-g|dead\n",
                    "",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        validator._docker = fake_docker
        validator._cleanup_stale()
        rm_calls = [c for c in calls if c[:2] == ["rm", "-f"]]
        self.assertEqual(
            sorted(c[2] for c in rm_calls),
            ["kiron-ollama-compat-f", "kiron-ollama-compat-g"],
        )

    def test_write_reports_contains_runtime_handoff_fields(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator.digest = "ollama/ollama@sha256:abc"
        validator.capabilities = {
            "num_gpu_zero_chat_generate": self.mod.Check(
                True,
                "fail",
                "num_gpu=0 effective",
                {"num_gpu_zero_effective": True},
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            self.mod.REPORT_DIR = pathlib.Path(tmpdir)
            validator._host_ollama_version = lambda: None
            validator._prod_container_field = lambda field: None
            validator._write_reports("passed")

            reports = sorted(pathlib.Path(tmpdir).glob("*.json"))
            self.assertEqual(len(reports), 1)
            data = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(data["image"], "ollama/ollama:0.21.1")
            self.assertEqual(data["image_digest"], "ollama/ollama@sha256:abc")
            self.assertTrue(data["upgrade_allowed"])
            self.assertTrue(
                data["capabilities"]["num_gpu_zero_chat_generate"]["data"]["num_gpu_zero_effective"]
            )
            self.assertEqual(len(list(pathlib.Path(tmpdir).glob("*.md"))), 1)

    def test_fixture_repull_failure_is_reported_and_followup_checks_continue(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)

        ok = self.mod.Check(True, "fail", "ok")
        validator._gpu_visible_check = lambda: ok
        validator._version_check = lambda: ok
        validator._tags_check = lambda: ok
        validator._ps_check = lambda require_size_vram: ok
        validator._generate_check = lambda model: ok
        validator._chat_check = lambda model: ok
        validator._baseline_gpu_resident_check = lambda model: ok
        validator._stream_check = lambda model: ok
        validator._think_false_check = lambda model: ok
        validator._unload_check = lambda model: ok
        validator._num_gpu_zero_check = lambda model: ok
        validator._error_shape_check = lambda: ok

        def pull_boom(model):
            raise RuntimeError("pull failed")

        validator._pull_fixture = pull_boom
        validator._run_standard_checks("smollm2:135m")

        self.assertIn("fixture_repull", validator.capabilities)
        self.assertFalse(validator.capabilities["fixture_repull"].ok)
        self.assertIn("fixture_repull: pull failed", validator.failures)
        self.assertIn("num_gpu_zero_chat_generate", validator.capabilities)

    def test_pull_fixture_raises_on_error_field_in_200_body(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._json = lambda method, path, payload=None, timeout=120.0: {
            "error": "pull model manifest: registry returned 502",
        }
        with self.assertRaises(RuntimeError) as ctx:
            validator._pull_fixture("smollm2:135m")
        self.assertIn("registry returned 502", str(ctx.exception))

    def test_pull_fixture_raises_when_status_not_success(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._json = lambda method, path, payload=None, timeout=120.0: {
            "status": "pulling manifest",
        }
        with self.assertRaises(RuntimeError):
            validator._pull_fixture("smollm2:135m")

    def test_pull_fixture_accepts_status_success(self):
        validator = self.mod.Validator("ollama/ollama:0.21.1", "standard", False)
        validator._json = lambda method, path, payload=None, timeout=120.0: {
            "status": "success",
        }
        validator._pull_fixture("smollm2:135m")

    def test_find_matching_report_accepts_nested_num_gpu_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = pathlib.Path(tmpdir)
            path = report_dir / "ok.json"
            path.write_text(json.dumps({
                "image": "ollama/ollama:0.21.1",
                "image_digest": "ollama/ollama@sha256:abc",
                "validator_version": "v1",
                "report_status": "passed",
                "upgrade_allowed": True,
                "capabilities": {
                    "num_gpu_zero_chat_generate": {
                        "ok": True,
                        "data": {"num_gpu_zero_effective": True},
                    },
                },
            }), encoding="utf-8")

            found = self.mod.find_matching_report(
                report_dir,
                "ollama/ollama:0.21.1",
                "ollama/ollama@sha256:abc",
            )
            self.assertIsNotNone(found)
            found_path, safe = found
            self.assertEqual(found_path, path)
            self.assertTrue(safe)

    def test_find_matching_report_prefers_tested_at_over_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = pathlib.Path(tmpdir)
            base = {
                "image": "ollama/ollama:0.21.1",
                "image_digest": "ollama/ollama@sha256:abc",
                "validator_version": "v1",
                "report_status": "passed",
                "upgrade_allowed": True,
                "capabilities": {
                    "num_gpu_zero_chat_generate": {
                        "ok": True,
                        "data": {"num_gpu_zero_effective": True},
                    },
                },
            }
            old_path = report_dir / "20260101-100000-ollama-0.21.1.json"
            old_path.write_text(json.dumps({**base, "tested_at": "2026-01-01T10:00:00Z"}), encoding="utf-8")
            new_path = report_dir / "20260201-100000-ollama-0.21.1.json"
            new_path.write_text(json.dumps({**base, "tested_at": "2026-02-01T10:00:00Z"}), encoding="utf-8")

            # Simulate `git checkout` / `rsync` / `touch` flipping mtime so the older
            # report looks newer to the filesystem.
            old_mtime = new_path.stat().st_mtime + 1000
            new_mtime = new_path.stat().st_mtime - 1000
            os.utime(old_path, (old_mtime, old_mtime))
            os.utime(new_path, (new_mtime, new_mtime))

            found = self.mod.find_matching_report(
                report_dir,
                "ollama/ollama:0.21.1",
                "ollama/ollama@sha256:abc",
            )
            self.assertIsNotNone(found)
            found_path, _ = found
            self.assertEqual(found_path, new_path)

    def test_find_matching_report_rejects_ok_without_nested_num_gpu_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = pathlib.Path(tmpdir)
            path = report_dir / "ok_without_data.json"
            path.write_text(json.dumps({
                "image": "ollama/ollama:0.21.1",
                "image_digest": "ollama/ollama@sha256:abc",
                "validator_version": "v1",
                "report_status": "passed",
                "upgrade_allowed": True,
                "capabilities": {
                    "num_gpu_zero_chat_generate": {
                        "ok": True,
                    },
                },
            }), encoding="utf-8")

            found = self.mod.find_matching_report(
                report_dir,
                "ollama/ollama:0.21.1",
                "ollama/ollama@sha256:abc",
            )
            self.assertIsNotNone(found)
            found_path, safe = found
            self.assertEqual(found_path, path)
            self.assertFalse(safe)

    def test_find_matching_report_raises_on_missing_dir(self):
        # Misconfigured --report-dir darf nicht als "kein Match" durchgereicht
        # werden — sonst diagnostiziert deploy-local.sh den falschen Pfad als
        # fehlenden Validator-Lauf.
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = pathlib.Path(tmpdir) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                self.mod.find_matching_report(
                    missing,
                    "ollama/ollama:0.21.1",
                    "ollama/ollama@sha256:abc",
                )

    def test_find_matching_report_raises_when_path_is_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            not_a_dir = pathlib.Path(tmpdir) / "report_path_is_file"
            not_a_dir.write_text("", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                self.mod.find_matching_report(
                    not_a_dir,
                    "ollama/ollama:0.21.1",
                    "ollama/ollama@sha256:abc",
                )

    def test_find_matching_report_rejects_mismatched_validator_version(self):
        # Schema-Drift-Schutz: ein zukuenftiger Validator-Bump muss alte
        # Reports automatisch invalidieren, sonst clearen v1-Reports einen
        # v2-Gate mit strikteren capability-Definitionen.
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = pathlib.Path(tmpdir)
            old = report_dir / "old_version.json"
            old.write_text(json.dumps({
                "image": "ollama/ollama:0.21.1",
                "image_digest": "ollama/ollama@sha256:abc",
                "validator_version": "v0",
                "report_status": "passed",
                "upgrade_allowed": True,
                "capabilities": {
                    "num_gpu_zero_chat_generate": {
                        "ok": True,
                        "data": {"num_gpu_zero_effective": True},
                    },
                },
            }), encoding="utf-8")
            missing = report_dir / "missing_version.json"
            missing.write_text(json.dumps({
                "image": "ollama/ollama:0.21.1",
                "image_digest": "ollama/ollama@sha256:abc",
                "report_status": "passed",
                "upgrade_allowed": True,
                "capabilities": {
                    "num_gpu_zero_chat_generate": {
                        "ok": True,
                        "data": {"num_gpu_zero_effective": True},
                    },
                },
            }), encoding="utf-8")

            found = self.mod.find_matching_report(
                report_dir,
                "ollama/ollama:0.21.1",
                "ollama/ollama@sha256:abc",
            )
            self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
