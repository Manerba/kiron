import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet,
    CapabilityStatus, ParameterConstraint, ResolvedDeployment, ResourceProfile, RuntimeImplementation,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from runtime_capabilities import decode_provider_evidence, encode_provider_evidence
from runtime_composition import adapter_revision, load_capability_evidence


class CapabilityReportTests(unittest.TestCase):
    def setUp(self):
        self.deployment = ResolvedDeployment("registered", BackendType.PRISM, "/local/model.gguf",
            ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100),
            LoaderType.PRISM_GGUF, ResourceProfile("measured", 1024, 128, 128, 1, 4, 40, False, 40, 20), "b" * 64)
        self.implementation = RuntimeImplementation("binary", None, "adapter")
        self.evidence = CapabilityEvidence("binary", self.deployment.artifact_identity.fingerprint,
            "a" * 64, None, None, "adapter", "b" * 64, "isolated-test/result.json",
            datetime(2026, 9, 22, tzinfo=timezone.utc))
        self.capabilities = CapabilitySet({CapabilityName.CHAT: Capability(CapabilityStatus.SUPPORTED,
            {"max_output_tokens": ParameterConstraint(minimum=1, maximum=128),
             "roles": ParameterConstraint(allowed_values=("user", "assistant"))}, (self.evidence,))})
        self.value = encode_provider_evidence(BackendType.PRISM, {"registered": self.capabilities})

    def test_roundtrip_keeps_evidence_bound_and_missing_capabilities_unverified(self):
        decoded = decode_provider_evidence(json.loads(json.dumps(self.value)), BackendType.PRISM)
        self.assertEqual(decoded["registered"], self.capabilities)
        self.assertTrue(decoded["registered"].supports(CapabilityName.CHAT, self.deployment, self.implementation))
        self.assertFalse(decoded["registered"].supports(CapabilityName.VISION, self.deployment, self.implementation))
        self.assertFalse(decoded["registered"].supports(CapabilityName.CHAT, self.deployment,
            replace(self.implementation, parser_revision="changed")))
        self.assertFalse(decoded["registered"].supports(CapabilityName.CHAT,
            replace(self.deployment, configuration_fingerprint="c" * 64), self.implementation))

    def test_closed_schema_and_typed_evidence_reject_malformed_reports(self):
        mutations = [
            lambda v: v.update(extra=True),
            lambda v: v.update(version=True),
            lambda v: v.update(provider="ollama"),
            lambda v: v["deployments"]["registered"].update(invented={}),
            lambda v: v["deployments"]["registered"]["chat"].update(evidence=[]),
            lambda v: v["deployments"]["registered"]["chat"]["evidence"][0].update(observed_at="2026-09-22"),
            lambda v: v["deployments"]["registered"]["chat"]["evidence"][0].update(parser_revision={}),
            lambda v: v["deployments"]["registered"]["chat"]["evidence"][0].update(artifact_sha256="unverified"),
            lambda v: v["deployments"]["registered"]["chat"]["constraints"]["roles"].update(allowed_values=[{}]),
            lambda v: v["deployments"]["registered"]["chat"]["constraints"]["max_output_tokens"].update(minimum=float("nan")),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(self.value)
                mutation(value)
                with self.assertRaises((ValueError, TypeError)):
                    decode_provider_evidence(value, BackendType.PRISM)

    def test_independent_provider_files_permissions_and_absence_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(load_capability_evidence(root), {})
            directory = root / "local-inference-capabilities"
            directory.mkdir(mode=0o750)
            valid = directory / "prism.json"
            valid.write_text(json.dumps(self.value))
            valid.chmod(0o640)
            bad = directory / "ollama.json"
            bad.write_text('{"version":1}')
            bad.chmod(0o640)
            self.assertEqual(set(load_capability_evidence(root)), {BackendType.PRISM})
            bad.write_text('{"extra":' + '[' * 12000 + '0' + ']' * 12000 + '}')
            self.assertEqual(set(load_capability_evidence(root)), {BackendType.PRISM})
            valid.chmod(0o660)
            self.assertEqual(load_capability_evidence(root), {})
            valid.chmod(0o640)
            valid.write_text(json.dumps(self.value).replace('"version": 1', '"version": 2, "version": 1'))
            self.assertEqual(load_capability_evidence(root), {})

    def test_adapter_revision_is_stable_and_provider_specific(self):
        prism = adapter_revision("prism_provider.py")
        self.assertRegex(prism, r"^sha256:[a-f0-9]{64}$")
        self.assertEqual(prism, adapter_revision("prism_provider.py"))
        self.assertNotEqual(prism, adapter_revision("ollama_provider.py"))

    def test_adapter_revision_covers_helpers_common_semantics_and_decoder(self):
        baseline = adapter_revision("prism_provider.py")
        original = Path.read_bytes
        changed = {"json_schema.py", "content.py", "capabilities.py", "provider.py", "runtime_capabilities.py",
                   "prism_reasoning.py", "prism_reasoning_profile.py", "provider_features.py",
                   "openai_generation.py", "prism_structured.py", "prism_tools.py", "prism_vision.py",
                   "prism_runtime_policy.py", "controller.py", "process.py", "composition.py",
                   "admission.py", "main.py", "openai_embeddings.py", "openai_responses.py",
                   "openai_responses_stream.py"}
        for filename in changed:
            with self.subTest(filename=filename):
                def read(path):
                    return original(path) + (b"\n# changed\n" if path.name == filename else b"")
                with mock.patch.object(Path, "read_bytes", read):
                    self.assertNotEqual(baseline, adapter_revision("prism_provider.py"))
        with self.assertRaises(ValueError):
            adapter_revision("../unrelated.py")


if __name__ == "__main__":
    unittest.main()
