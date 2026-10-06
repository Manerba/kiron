"""Offline archive/export contract tests; never start a runtime or use host reports."""
import asyncio
import base64
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("export_capabilities", HERE / "export-capabilities.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
exporter.initialize()
import capability_cases as cases
from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.local_inference import (
    CapabilityName as N, CapabilitySet, CapabilityStatus, DiscoveredModel, DiscoverySnapshot,
    ModelLifecycleOperation, ProviderHealth, ProviderObservation, RequestContext, ResourceProfile,
    RuntimeGeneration, RuntimeImplementation, RuntimeTimeouts, TokenUsage, build_resolver_snapshot,
)
from kiron_common.local_model_registry.models import RegistryEntry, LocalArtifactFile
from kiron_common.local_model_registry.codec import encode_registry
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from runtime_capabilities import decode_provider_evidence
from runtime_service import RuntimeService


def fixture(kind):
    value = json.loads((HERE / "fixtures/capability-cases" / (kind + ".json")).read_text())
    for call in value["calls"]:
        call["usage"] = TokenUsage(**call["usage"])
    return value


class ExportTests(unittest.TestCase):
    def setUp(self):
        # A root-owned isolated directory outside production; /tmp itself is
        # intentionally disallowed as a publication ancestor by the exporter.
        self.temp = tempfile.TemporaryDirectory(prefix=".capability-test-", dir=Path.home())
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.report = self.base / "controller-fixture"
        self.report.mkdir()
        self.patcher = mock.patch.object(exporter, "REPORT_ROOT", self.base)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.profile = ResourceProfile("fixture", 1024, 128, 128, 1, 4, 40, False, 40, 20)
        self.entry = RegistryEntry.create(runtime_provider=BackendType.PRISM, artifact_origin=ArtifactType.LOCAL,
            artifact_format=ArtifactFormat.GGUF, reference=str(self.base / "model.gguf"), display_name="Fixture",
            loader=LoaderType.PRISM_GGUF, sha256="a" * 64, size_bytes=10,
            projector=LocalArtifactFile(str(self.base / "projector.gguf"), "b" * 64, 20), runtime_profile="fixture")
        self.snapshot = build_resolver_snapshot(MODEL_CATALOG, (self.entry,), resource_profiles={"fixture": self.profile})
        self.model = self.snapshot.resolve(self.entry.id)
        self.implementation = RuntimeImplementation("runtime", None, "parser")

    def write(self, name, value):
        path = self.report / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if type(value) is bytes else exporter.canonical_json(value))
        return path

    def archive(self):
        data = fixture("public-responses-budget")
        call = data["calls"][0]
        def rebind(value):
            if type(value) is dict:
                return {k: self.model.api_model_id if k == "model" else rebind(v) for k, v in value.items()}
            return [rebind(v) for v in value] if type(value) is list else value
        results = rebind(data["results"])
        for name, value in results.items():
            self.write("results/sdk-responses-" + name + ".json", value)
        generation = {"boot_id": "boot", "process_id": call["body"]["model"].removeprefix("kiron-prism-")}
        for index in range(1, 5):
            if index == 1:
                request = {"method": "POST", "path": "/v1/chat/completions", "body": call["body"]}
                response = call["response"]
                choice = response["choices"][0]
                frame = {**response, "choices": [{"index": 0, "delta": choice["message"], "finish_reason": choice["finish_reason"]}]}
                raw = b":\n\ndata: " + exporter.canonical_json(frame) + b"\n\ndata: [DONE]\n\n"
            elif index == 2:
                request = {"method": "POST", "path": "/tokenize", "body": {
                    "content": call["response"]["__verbose"]["tokens"], "with_pieces": True, "add_special": False, "parse_special": False}}
                raw = exporter.canonical_json({"tokens": call["token_pieces"]})
            else:
                request = {"method": "POST", "path": "/v1/chat/completions", "body": {
                    "model": call["body"]["model"], "max_tokens": 1, "stream": False}}
                raw = b'{"error":"unauthorized"}'
            prefix = f"results/native-{index:03d}"
            self.write(prefix + ".request.json", request)
            self.write(prefix + ".status.json", {"status": 200 if index <= 2 else 401})
            self.write(prefix + ".response.raw", raw)
            self.write(prefix + ".eof", b"")
        self.write("registry/models.json", encode_registry((self.entry,)))
        self.source = {"fixture.py": "c" * 64}
        self.harness = SimpleNamespace(BUNDLE=self.base / "bundle", BUNDLE_MANIFEST_SHA="d" * 64,
            RUNTIME=self.base / "bundle" / "runtime",
            PROFILE_ID="fixture", MODEL_SHA=self.entry.sha256, MODEL_BYTES=10, MODEL=Path(self.entry.reference),
            PROJECTOR_SHA=self.entry.projector.sha256, PROJECTOR_BYTES=20, PROJECTOR=Path(self.entry.projector.reference),
            PORT=18089, verify_snapshot=mock.Mock())
        self.policy = SimpleNamespace(runtime_revision="runtime", resource_profiles=lambda: {"fixture": self.profile})
        self.plan = {"probe": "public-responses-budget", "source_sha256": self.source,
            "source_root": str(self.report / "source"), "bundle": str(self.harness.BUNDLE),
            "artifact_layout": "test", "runtime_root": str(self.harness.RUNTIME),
            "projector_path": str(self.harness.PROJECTOR),
            "bundle_manifest_sha256": self.harness.BUNDLE_MANIFEST_SHA, "profile": "fixture",
            "model_sha256": self.entry.sha256, "projector_sha256": self.entry.projector.sha256,
            "uid": 65534, "gid": 982, "port": 18089, "max_seconds": 600,
            "registry_sha256": exporter.digest((self.report / "registry/models.json").read_bytes()), "model_id": self.entry.id}
        self.write("plan.json", self.plan)
        self.result = {"status": "passed", "controller_cleanup_confirmed": True,
            "loaded_health": {"provider": "prism", "health": "available", "generation": generation,
                "models": {self.entry.id: {"state": "loaded", "generation": generation,
                    "configuration_fingerprint": self.model.deployment.configuration_fingerprint}}},
            "after": {"models": {}}, "unloaded": {"observation": {"models": {}}},
            "foreign_bearer": {"statuses": [401, 401], "active_requests": 0},
            "started_at": "2026-09-22T00:00:00+00:00", "finished_at": "2026-09-22T00:01:00+00:00",
            "public_features": {"kind": "public-responses-budget", "sdk_version": "2.29.0",
                "public_transport": "httpx.ASGITransport", "results": results}}
        self.write("results/result.json", self.result)
        self.write("admission/admission.json", {"tickets": []})
        self.production = {"gpu": "5922, 5989, 0\n", "units": "MainPID=123", "gpu_processes": "123, ollama, 5528\n",
                           "production_registry_sha256": "f" * 64, "markers": {}}
        self.write("production-before.json", {**self.production, "observed_at": "2026-09-21T23:59:59+00:00", "ollama": {"models": [{"expires_at": "earlier", "size": 10}]}})
        self.write("production-after.json", {**self.production, "observed_at": "2026-09-22T00:02:00+00:00", "ollama": {"models": [{"expires_at": "later", "size": 10}]}})

    def validate(self):
        sha = exporter.digest(exporter.canonical_json(exporter.archive_inventory(self.report)))
        return exporter.validate_archive(self.report, sha, self.harness, self.source, self.policy, "parser")

    def reports(self):
        identity = {"deployment_id": self.model.deployment.id,
            "artifact_fingerprint": self.model.deployment.artifact_identity.fingerprint,
            "configuration_fingerprint": self.model.deployment.configuration_fingerprint,
            "implementation": asdict(self.implementation), "resource_profile": asdict(self.profile)}
        return [{"kind": kind, "identity": identity, "model": self.model, "implementation": self.implementation,
                 "sha256": str(index) * 64, "finished": datetime.now(timezone.utc),
                 "measured": cases.validate_cases(kind, fixture(kind)["results"], fixture(kind)["calls"])}
                for index, kind in enumerate(sorted(exporter.KINDS), 1)]

    def test_budget_archive_checks_full_native_id_proof_and_individual_registry(self):
        self.archive()
        value = self.validate()
        self.assertEqual(value["measured"], {"budgets": [48]})
        self.assertEqual(value["snapshot_revision"], self.snapshot.revision)
        observations = value["production_observations"]
        self.assertEqual([observations[key]["ollama_expires_at"] for key in ("before", "after")], [["earlier"], ["later"]])
        self.harness.verify_snapshot.assert_called_once_with(self.report, self.source)

    def test_artifact_layout_and_both_paths_are_bound_in_export(self):
        self.archive()
        for key, foreign in (("artifact_layout", "production"), ("runtime_root", "/tmp/foreign"),
                             ("projector_path", "/tmp/foreign.gguf")):
            with self.subTest(key=key):
                self.write("plan.json", {**self.plan, key: foreign})
                with self.assertRaisesRegex(ValueError, 'pinned runtime identity'):
                    self.validate()
        self.write("plan.json", {**self.plan, "artifact_layout": "production"})
        sha = exporter.digest(exporter.canonical_json(exporter.archive_inventory(self.report)))
        self.assertEqual(exporter.validate_archive(self.report, sha, self.harness, self.source,
            self.policy, "parser", 'production')["model"], self.model)

    def test_only_transient_gpu_utilization_changes_and_settled_evidence_is_preserved(self):
        self.archive()
        before = exporter.json_file(self.report / "production-before.json")
        after = exporter.json_file(self.report / "production-after.json")
        after["gpu"] = "5922, 5989, 4\n"
        self.write("production-after.json", after)
        settled = {**after, "observed_at": "2026-09-22T00:02:01.123456Z", "gpu": "5922, 5989, 0\n"}
        self.write("production-after-settled.json", settled)
        observed = self.validate()["production_observations"]
        self.assertEqual(list(observed), ["before", "after", "after_settled"])
        self.assertEqual([entry["gpu"]["utilization_percent"] for entry in observed.values()], [0, 4, 0])
        self.assertEqual([entry["gpu"]["raw"] for entry in observed.values()], [before["gpu"], after["gpu"], settled["gpu"]])
        self.assertEqual([entry["observed_at"] for entry in observed.values()],
                         [before["observed_at"], after["observed_at"], settled["observed_at"]])
        self.assertEqual(exporter.json_file(self.report / "production-before.json"), before)
        self.assertEqual(exporter.json_file(self.report / "production-after.json"), after)
        self.assertEqual(exporter.json_file(self.report / "production-after-settled.json"), settled)
        settled["gpu"] = "5921, 5990, 0\n"
        self.write("production-after-settled.json", settled)
        with self.assertRaisesRegex(ValueError, "production invariants"):
            self.validate()

    def test_observation_timestamps_reject_legacy_naive_non_utc_and_invalid_dates(self):
        self.archive()
        original = {label: exporter.json_file(self.report / ("production-" + label + ".json"))
                    for label in ("before", "after")}
        for invalid in (None, 1, True, "", "2026-09-22", "2026-09-22T00:02:00",
                        "2026-09-22T00:02:00+01:00", "2026-09-22T00:02:00-00:00",
                        "2026-09-22 00:02:00+00:00", "2026-02-30T00:02:00Z",
                        "2026-09-22T24:02:00Z", "2026-09-22T00:02:00.1234567Z"):
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(original)
                changed["after"]["observed_at"] = invalid
                with self.assertRaisesRegex(ValueError, "observed_at"):
                    exporter.validate_production_observations(changed)
        for mutation in ("missing", "legacy_only", "both"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(original)
                if mutation != "both":
                    changed["after"].pop("observed_at")
                if mutation != "missing":
                    changed["after"]["monotonic"] = 2
                with self.assertRaisesRegex(ValueError, "observed_at"):
                    exporter.validate_production_observations(changed)

    def test_observation_labels_and_chronology_are_explicit(self):
        self.archive()
        original = {label: exporter.json_file(self.report / ("production-" + label + ".json"))
                    for label in ("before", "after")}
        for timestamps in (("2026-09-22T00:02:00Z", "2026-09-22T00:02:00+00:00"),
                           ("2026-09-22T00:02:01Z", "2026-09-22T00:02:00Z")):
            with self.subTest(timestamps=timestamps):
                changed = copy.deepcopy(original)
                for label, timestamp in zip(("before", "after"), timestamps):
                    changed[label]["observed_at"] = timestamp
                with self.assertRaisesRegex(ValueError, "chronology"):
                    exporter.validate_production_observations(changed)
        for timestamp in (original["after"]["observed_at"], original["before"]["observed_at"]):
            with self.subTest(settled=timestamp):
                changed = {**original, "after_settled": {**original["after"], "observed_at": timestamp}}
                with self.assertRaisesRegex(ValueError, "chronology"):
                    exporter.validate_production_observations(changed)
        for labels in ({}, {"before": original["before"]}, {**original, "unknown": original["after"]}):
            with self.subTest(labels=list(labels)):
                with self.assertRaisesRegex(ValueError, "require before/after"):
                    exporter.validate_production_observations(labels)
        reversed_input = {"after": original["after"], "before": original["before"]}
        self.assertEqual(list(exporter.validate_production_observations(reversed_input)), ["before", "after"])

    def test_gpu_memory_process_registry_and_other_production_drift_still_fail(self):
        self.archive()
        original = exporter.json_file(self.report / "production-after.json")
        for key, value in (("gpu", "5921, 5989, 4\n"), ("gpu", "5922, 5990, 4\n"),
                           ("units", "MainPID=124"), ("gpu_processes", "124, ollama, 5528\n"),
                           ("production_registry_sha256", "e" * 64), ("markers", {"training": True}),
                           ("ollama", {"models": [{"expires_at": "later", "size": 11}]})):
            with self.subTest(key=key, value=value):
                self.write("production-after.json", {**original, key: value})
                with self.assertRaisesRegex(ValueError, "production invariants"):
                    self.validate()

    def test_gpu_triplet_is_closed_and_utilization_range_is_checked(self):
        self.archive()
        original = exporter.json_file(self.report / "production-after.json")
        for value in (None, [5922, 5989, 4], "5922, 5989", "5922, 5989, 4, 0", "5922, 5989, -1",
                      "5922, 5989, 101", "-1, 5989, 4", "5922, 5989, 4.5", "5922, 5989, 0\n5922, 5989, 4\n"):
            with self.subTest(value=value):
                self.write("production-after.json", {**original, "gpu": value})
                with self.assertRaisesRegex(ValueError, "production GPU"):
                    self.validate()

    def test_failed_unknown_old_source_and_drift_are_rejected_even_with_new_inventory(self):
        self.archive()
        for field, value in (("status", "failed"), ("controller_cleanup_confirmed", False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.write("results/result.json", {**self.result, field: value}); self.validate()
        self.write("results/result.json", self.result)
        for field, value in (("probe", "public-api"), ("source_sha256", {}), ("registry_sha256", "e" * 64),
                             ("model_sha256", "e" * 64), ("bundle_manifest_sha256", "e" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.write("plan.json", {**self.plan, field: value}); self.validate()
        self.write("plan.json", self.plan)
        self.write("production-after.json", {**self.production, "observed_at": "2026-09-22T00:02:00+00:00", "ollama": {"models": [{"expires_at": "later", "size": 11}]}})
        with self.assertRaisesRegex(ValueError, "production"):
            self.validate()

    def test_missing_eof_wrong_lookup_cleanup_or_public_counter_fail(self):
        self.archive()
        target = self.report / "results/native-002.eof"
        target.unlink()
        with self.assertRaises(FileNotFoundError): self.validate()
        target.write_bytes(b"")
        lookup = self.report / "results/native-002.response.raw"
        original = lookup.read_bytes(); changed = json.loads(original); changed["tokens"][0]["id"] += 1
        lookup.write_bytes(exporter.canonical_json(changed))
        with self.assertRaisesRegex(ValueError, "lookup"): self.validate()
        lookup.write_bytes(original)
        self.write("admission/admission.json", {"tickets": ["unknown"]})
        with self.assertRaisesRegex(ValueError, "cleanup"): self.validate()
        self.write("admission/admission.json", {"tickets": []})
        self.result["public_features"]["results"]["budget-terminal"]["usage"]["total_tokens"] += 1
        self.write("results/result.json", self.result)
        with self.assertRaisesRegex(ValueError, "standalone"): self.validate()

    def test_prior_inventory_pin_rejects_mutation(self):
        self.archive()
        sha = exporter.digest(exporter.canonical_json(exporter.archive_inventory(self.report)))
        self.write("unreviewed.json", {})
        with self.assertRaisesRegex(ValueError, "reviewed inventory"):
            exporter.validate_archive(self.report, sha, self.harness, self.source, self.policy, "parser")

    def test_all_six_real_case_shapes_and_damaged_results(self):
        for kind in exporter.KINDS:
            with self.subTest(kind=kind):
                value = fixture(kind)
                self.assertTrue(cases.validate_cases(kind, value["results"], value["calls"])["budgets"])
                value["results"]["invented-case"] = {}
                with self.assertRaises(ValueError): cases.validate_cases(kind, value["results"], value["calls"])
        value = fixture("public-tools")
        value["results"]["tool-named"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"city":"Tokyo"}'
        with self.assertRaises(ValueError): cases.validate_cases("public-tools", value["results"], value["calls"])
        value = fixture("public-responses")
        replay = value["results"]["tools-replay-request"]["input"]
        outputs = [item for item in replay if item.get("type") == "function_call_output"]
        outputs[0]["output"], outputs[1]["output"] = outputs[1]["output"], outputs[0]["output"]
        with self.assertRaisesRegex(ValueError, "reversed"): cases.validate_cases("public-responses", value["results"], value["calls"])

    def test_constructed_ordered_replay_requires_exact_history_native_order_and_scope(self):
        for change in ('missing_scope', 'claimed_generated', 'input_order', 'native_order', 'native_text', 'native_answer', 'native_stream'):
            value = fixture('public-responses')
            results, native = value['results'], value['calls'][4]['body']
            if change == 'missing_scope':
                results.pop('tools-ordered-replay-scope')
            elif change == 'claimed_generated':
                results['tools-ordered-replay-scope']['history'] = 'generated'
            elif change == 'input_order':
                items = results['tools-ordered-replay-request']['input']
                items[2], items[3] = items[3], items[2]
            elif change == 'native_order':
                native['messages'][1], native['messages'][2] = native['messages'][2], native['messages'][1]
            elif change == 'native_text':
                native['messages'][2]['content'] = ''
            elif change == 'native_answer':
                native['messages'][3]['content'], native['messages'][4]['content'] = native['messages'][4]['content'], native['messages'][3]['content']
            else:
                native['stream'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                cases.validate_cases('public-responses', results, value['calls'])

    def test_native_sampling_value_must_match_exported_zero_policy(self):
        self.archive()
        path = self.report / "results/native-001.request.json"
        original = json.loads(path.read_text())
        for value in (1, False, None):
            with self.subTest(value=value):
                altered = copy.deepcopy(original); altered["body"]["temperature"] = value
                self.write("results/native-001.request.json", altered)
                with self.assertRaisesRegex(ValueError, "sampling"):
                    self.validate()

    def test_structured_results_alone_never_prove_native_grammar_selection(self):
        for index in (0, 1):
            for mutation in ("missing", "looser_schema", "strict_flag"):
                with self.subTest(index=index, mutation=mutation):
                    value = fixture("public-structured")
                    body = value["calls"][index]["body"]
                    if mutation == "missing":
                        body.pop("response_format")
                    elif mutation == "looser_schema":
                        body["response_format"]["json_schema"]["schema"] = {}
                    else:
                        form = body["response_format"]["json_schema"]
                        form["strict"] = not form["strict"]
                    with self.assertRaisesRegex(ValueError, "structured grammar"):
                        cases.validate_cases("public-structured", value["results"], value["calls"])

    def test_profile_is_narrow_defaults_valid_and_no_embedding_grant(self):
        reports = self.reports()
        encoded = exporter.profile(reports)
        capabilities = decode_provider_evidence(encoded, BackendType.PRISM)[self.model.deployment.id]
        self.assertEqual(capabilities.by_name[N.EMBEDDINGS].status, CapabilityStatus.UNVERIFIED)
        self.assertFalse(capabilities.supports(N.EMBEDDINGS, self.model.deployment, self.implementation))
        for name in set(N) - {N.EMBEDDINGS}:
            self.assertTrue(capabilities.supports(name, self.model.deployment, self.implementation))
        chat = capabilities.by_name[N.CHAT].constraints
        self.assertTrue(chat["max_output_tokens"].accepts(64))
        self.assertFalse(chat["roles"].accepts("system"))
        self.assertFalse(chat["token_budget"].accepts("max_tokens"))
        tools = capabilities.by_name[N.FUNCTION_TOOLS].constraints
        self.assertFalse(tools["tool_choice"].accepts("auto"))
        self.assertFalse(tools["strict"].accepts(False))
        self.assertFalse(capabilities.by_name[N.VISION].constraints["width"].accepts(256))

    def test_profile_rejects_missing_duplicate_identity_and_constraint_conflicts(self):
        reports = self.reports()
        for bad in (reports[:-1], [reports[0]] * 6):
            with self.assertRaisesRegex(ValueError, "six"): exporter.profile(bad)
        bad = copy.copy(reports); bad[0] = {**bad[0], "identity": {**bad[0]["identity"], "configuration_fingerprint": "e" * 64}}
        with self.assertRaisesRegex(ValueError, "identities"): exporter.profile(bad)
        real = exporter.current_module
        module = real("probe-openai-responses.py")
        original = module.capabilities
        def conflicting(kind, evidence):
            values = dict(original(kind, evidence).by_name)
            values[N.CHAT] = replace(values[N.CHAT], constraints={})
            return CapabilitySet(values)
        module.capabilities = conflicting
        with mock.patch.object(exporter, "current_module", side_effect=lambda name: module if "responses" in name else real(name)):
            with self.assertRaisesRegex(ValueError, "constraints conflict"): exporter.profile(reports)

    def test_no_archived_code_and_input_file_guards(self):
        with self.assertRaises(ValueError): exporter.current_module(str(self.report / "payload.py"))
        self.write("payload.py", b"raise AssertionError('must never execute')")
        self.assertIn("payload.py", exporter.archive_inventory(self.report))
        target = self.write("value.json", b'{"duplicate":1,"duplicate":2}')
        with self.assertRaises(ValueError): exporter.json_file(target)
        with self.assertRaises(ValueError): exporter.regular_bytes(target, 1)
        link = self.report / "link"; link.symlink_to(target)
        with self.assertRaises(ValueError): exporter.archive_inventory(self.report)
        link.unlink(); os.link(target, link)
        with self.assertRaises(ValueError): exporter.archive_inventory(self.report)

    def test_traversal_counts_empty_directories_and_bounds_depth_before_materialization(self):
        for index in range(4):
            (self.report / str(index)).mkdir()
        with mock.patch.object(exporter, "MAX_ENTRIES", 3):
            with self.assertRaisesRegex(ValueError, "entry count"):
                exporter.archive_inventory(self.report)
        (self.report / "0" / "nested").mkdir()
        with mock.patch.object(exporter, "MAX_DEPTH", 1):
            with self.assertRaisesRegex(ValueError, "depth"):
                exporter.archive_inventory(self.report)

    def test_atomic_candidate_no_overwrite_and_reverification(self):
        if os.geteuid() != 0: self.skipTest("root-owned export policy")
        root = self.base / "candidates"
        candidate, provenance = {"candidate": 1}, {"reports": [], "candidate_sha256": "e" * 64,
                                                 "artifact_layout": "production"}
        with mock.patch.object(exporter, "OUTPUT_ROOT", root):
            path = root / "review"
            exporter.publish(path, candidate, provenance)
            with self.assertRaises(OSError): exporter.publish(path, {"candidate": 2}, provenance)
            self.assertEqual(json.loads((path / "prism.json").read_text()), candidate)
            self.assertEqual(list(root.iterdir()), [path])
            with mock.patch.object(exporter, "assemble", return_value=(candidate, provenance)) as assemble:
                self.assertEqual(exporter.verify_candidate(path), "e" * 64)
                assemble.assert_called_once_with([], "production")
                (path / "prism.json").write_text('{"candidate":2}\n')
                with self.assertRaisesRegex(ValueError, "differs"): exporter.verify_candidate(path)
            with self.assertRaises(ValueError): exporter.output_path(Path("/usr/lib/kiron/data/capabilities/review"))


class RuntimeAddressabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_exported_profile_lists_and_validates_without_provider_mutation(self):
        helper = ExportTests("test_profile_is_narrow_defaults_valid_and_no_embedding_grant")
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        caps = decode_provider_evidence(exporter.profile(helper.reports()), BackendType.PRISM)[helper.model.deployment.id]
        now = datetime.now(timezone.utc)
        provider = SimpleNamespace(implementation=helper.implementation,
            model_lifecycle_operations=frozenset((ModelLifecycleOperation.LOAD, ModelLifecycleOperation.UNLOAD)),
            health=mock.AsyncMock(return_value=ProviderObservation(BackendType.PRISM, None, now, ProviderHealth.STARTABLE)),
            discover=mock.AsyncMock(return_value=DiscoverySnapshot(BackendType.PRISM, "fixture", now,
                (DiscoveredModel(helper.model.deployment.reference, helper.model.deployment.artifact_identity, True),))),
            capabilities=mock.AsyncMock(return_value=caps), validate_request=mock.Mock(), load=mock.AsyncMock(), aclose=mock.AsyncMock())
        service = RuntimeService(resolver=SimpleNamespace(snapshot=mock.AsyncMock(return_value=helper.snapshot)),
            providers={BackendType.PRISM: provider}, admission=mock.Mock(), measure=mock.Mock(), timeouts=RuntimeTimeouts(1,1,1,1,1,1,1))
        self.addAsyncCleanup(service.aclose)
        context = RequestContext("review", time.monotonic() + 2, asyncio.Event())
        self.assertEqual((await service.public_models(context))[0]["id"], helper.model.api_model_id)
        from openai_wire import parse_chat
        parsed = parse_chat({"model": helper.model.api_model_id, "messages": [{"role": "user", "content": "OK"}], "temperature": 0})
        request = await service.validate_chat(parsed, helper.model, context)
        self.assertEqual(request.options.max_output_tokens, 64)
        features = exporter.current_module("probe-openai-features.py")
        schema = {"type": "object", "properties": {"city": {"type": "string", "enum": ["Berlin", "Paris"]}},
                  "required": ["city"], "additionalProperties": False}
        tool = {"type": "function", "function": {"name": "weather", "parameters": schema, "strict": True}}
        basic = {"model": helper.model.api_model_id, "messages": [{"role": "user", "content": "fixture"}], "temperature": 0}
        for options in (
            {"tools": [tool], "tool_choice": "required", "parallel_tool_calls": True, "max_completion_tokens": 128},
            {"tools": [tool], "tool_choice": {"type": "function", "function": {"name": "weather"}}, "parallel_tool_calls": False},
            {"response_format": {"type": "json_schema", "json_schema": {"name": "report", "strict": True, "schema": features.SCHEMA}}},
            {"reasoning_effort": "low", "max_completion_tokens": 4, "stream": True},
        ):
            with self.subTest(options=tuple(options)):
                await service.validate_chat(parse_chat({**basic, **options}), helper.model, context)
        from kiron_common.local_inference import ImagePart
        vision = fixture("public-vision")["calls"][0]["body"]["messages"]
        image_parts = {}
        for index in (1, 3):
            uri = vision[0]["content"][index]["image_url"]["url"]
            mime = uri.split(";", 1)[0][5:]
            raw = base64.b64decode(uri.split(",", 1)[1])
            image_parts[(0, index)] = ImagePart(mime, raw, exporter.digest(raw), 96, 96)
        await service.validate_chat(parse_chat({**basic, "messages": vision, "max_completion_tokens": 16},
                                              image_parts=image_parts), helper.model, context)
        from openai_responses import prepare_response, parse_response
        for name, value in fixture("public-responses")["results"].items():
            if name.endswith("-request"):
                with self.subTest(response=name):
                    parsed = parse_response(prepare_response({**value, "model": helper.model.api_model_id}))
                    await service.validate_response(parsed, helper.model, context)
        provider.load.assert_not_called()
        service.admission.reserve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
