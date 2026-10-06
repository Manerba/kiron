import asyncio
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from kiron_common.local_inference import LocalInferenceError, RequestContext
from kiron_common.local_model_registry import RuntimeModelRegistry
from kiron_common.model_catalog import BackendType
from kiron_common.ollama_compat import OllamaCapabilities

from provider_transport import with_context
from runtime_composition import build_runtime_service, load_ollama_compatibility


class CompatibilityCompositionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        checks = {field.name: {"ok": True, "severity": "info", "data": None} for field in fields(OllamaCapabilities)}
        checks["version_endpoint"]["data"] = {"version": "0.18.0"}
        checks["num_gpu_zero_chat_generate"]["data"] = {"num_gpu_zero_effective": True}
        self.report = {"image_digest": "ollama@sha256:" + "a" * 64, "report_status": "passed",
                       "upgrade_allowed": True, "failures": [], "capabilities": checks}
        self.report_path = self.root / "report.json"
        self.runtime_path = self.root / "ollama_compat_runtime.json"

    def write(self, *, enabled=True):
        raw = json.dumps(self.report).encode()
        self.report_path.write_bytes(raw)
        self.report_path.chmod(0o640)
        value = dict(image_digest=self.report["image_digest"], report_path=str(self.report_path),
                     report_sha256=hashlib.sha256(raw).hexdigest(), num_gpu_zero_effective=enabled)
        self.runtime_path.write_text(json.dumps(value))
        self.runtime_path.chmod(0o640)

    async def test_complete_gate_uses_measured_version_and_explicit_closed_handoff(self):
        self.write(enabled=False)
        compatibility, version, digest = load_ollama_compatibility(self.runtime_path, data_root=self.root)
        self.assertEqual(version, "0.18.0")
        self.assertTrue(compatibility.chat_nonstream.ok)
        self.assertFalse(compatibility.num_gpu_zero_chat_generate.ok)

    async def test_report_tampering_does_not_get_positive_capabilities(self):
        self.write()
        self.report_path.write_text(json.dumps({**self.report, "extra": True}))
        with self.assertRaisesRegex(ValueError, "digest"):
            load_ollama_compatibility(self.runtime_path, data_root=self.root)

    async def test_missing_check_is_explicitly_unverified(self):
        del self.report["capabilities"]["think_false"]
        self.write()
        compatibility, _, _ = load_ollama_compatibility(self.runtime_path, data_root=self.root)
        self.assertFalse(compatibility.think_false.ok)
        self.assertEqual(compatibility.think_false.code, "evidence_missing")

    async def test_external_report_path_and_symlink_are_rejected(self):
        self.write()
        value = json.loads(self.runtime_path.read_text())
        value["report_path"] = "/opt/kiron/report.json"
        self.runtime_path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            load_ollama_compatibility(self.runtime_path, data_root=self.root)
        self.write()
        self.report_path.rename(self.root / "original.json")
        self.report_path.symlink_to(self.root / "original.json")
        with self.assertRaises(ValueError):
            load_ollama_compatibility(self.runtime_path, data_root=self.root)

    async def test_factory_constructs_no_positive_compatibility_on_missing_report(self):
        service = build_runtime_service(data_root=self.root, registry=RuntimeModelRegistry(self.root / "registry.json"),
                                        measure=lambda: None)
        self.addAsyncCleanup(service.aclose)
        self.assertEqual(set(service.providers), {BackendType.OLLAMA, BackendType.KIRON_EMBEDDINGS})
        self.assertIsNone(service.providers[BackendType.OLLAMA].compatibility)

    async def test_deadline_is_bounded_for_cancellation_resistant_transport(self):
        release = asyncio.Event()
        stopped = asyncio.Event()
        async def resistant():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            finally:
                stopped.set()
        context = RequestContext("bounded", time.monotonic() + .01, asyncio.Event())
        started = time.monotonic()
        try:
            with self.assertRaises(LocalInferenceError):
                await asyncio.wait_for(with_context(resistant(), context), 1)
            self.assertLess(time.monotonic() - started, .8)
            self.assertFalse(stopped.is_set())
        finally:
            release.set()
            await asyncio.wait_for(stopped.wait(), 1)


if __name__ == "__main__":
    unittest.main()
