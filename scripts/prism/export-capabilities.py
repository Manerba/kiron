#!/usr/bin/env python3
"""Review-only export for one measured Prism profile. Never install or run models."""
from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = Path("/usr/lib/kiron/test-runtimes/prism/reports")
OUTPUT_ROOT = Path("/usr/lib/kiron/test-runtimes/prism/capability-candidates")
PROFILE = "prism-bonsai27b-measured-review-v1"
KINDS = frozenset(("public-tools", "public-vision", "public-structured",
                   "public-reasoning", "public-responses", "public-responses-budget"))
MAX_FILE = 16 * 1024**2
MAX_ARCHIVE = 192 * 1024**2
MAX_FILES = 1024
MAX_ENTRIES = 2048
MAX_DEPTH = 16


def current_module(name):
    # Only a fixed filename below this checkout; never execute archived Python.
    if name not in {"smoke-controller.py", "probe-openai-features.py", "probe-openai-responses.py",
                    "capability_cases.py"}:
        raise ValueError("unknown trusted exporter helper")
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("export_" + name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path, limit=MAX_FILE):
    if path.resolve() != path:
        raise ValueError("noncanonical input path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ValueError("input must be a bounded regular single-link file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid", "st_gid")
        if len(data) > limit or any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError("input changed while reading")
        return data
    finally:
        os.close(fd)


def json_file(path):
    from provider_transport import decode_provider_json
    # Reuse the bounded strict decoder while allowing top-level event arrays.
    return decode_provider_json(b'{"value":' + regular_bytes(path) + b'}')["value"]


def archive_inventory(path):
    if (not path.is_absolute() or path.parent != REPORT_ROOT or path.resolve() != path
            or not re.fullmatch(r"controller-[A-Za-z0-9_-]+", path.name)):
        raise ValueError("only canonical isolated controller report directories are accepted")
    entries = 0

    def files(directory, depth):
        nonlocal entries
        if depth > MAX_DEPTH:
            raise ValueError("archive directory depth exceeded")
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > MAX_ENTRIES:
                    raise ValueError("archive entry count exceeded")
                if child.is_dir(follow_symlinks=False):
                    yield from files(Path(child.path), depth + 1)
                else:
                    yield Path(child.path)

    result, total = {}, 0
    # Both discovery and file reads are bounded before any materialization.
    for item in files(path, 0):
        info = item.lstat()
        if len(result) >= MAX_FILES:
            raise ValueError("archive file count exceeded")
        data = regular_bytes(item)
        total += len(data)
        if total > MAX_ARCHIVE:
            raise ValueError("archive byte limit exceeded")
        result[str(item.relative_to(path))] = {
            "sha256": digest(data), "size": len(data), "uid": info.st_uid,
            "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode),
        }
    if not result:
        raise ValueError("empty archive")
    return dict(sorted(result.items()))


def initialize():
    for directory in ("kiron-common", "kiron-proxy"):
        sys.path.insert(0, str(ROOT / "services" / directory))
    sys.path.insert(0, str(Path(__file__).parent))
    return current_module("smoke-controller.py")


def validate_production_observations(observations):
    """Compare identities and allocated memory; preserve transient utilization."""
    if type(observations) is not dict or set(observations) not in (
            {"before", "after"}, {"before", "after", "after_settled"}):
        raise ValueError("production observations require before/after and optional after_settled")
    normalized, evidence = [], {}
    previous = None
    for label in ("before", "after", "after_settled"):
        if label not in observations:
            continue
        original = observations[label]
        if type(original) is not dict or "monotonic" in original:
            raise ValueError("production observation must use observed_at without monotonic")
        value = copy.deepcopy(original)
        observed_at = value.pop("observed_at", None)
        if (type(observed_at) is not str or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|\+00:00)",
                observed_at) is None):
            raise ValueError("production observed_at must be a UTC datetime")
        try:
            observed = datetime.fromisoformat(observed_at)
        except ValueError as exc:
            raise ValueError("production observed_at contains an invalid datetime") from exc
        if previous is not None and observed <= previous:
            raise ValueError("production observation chronology is invalid")
        previous = observed
        raw = value.get("gpu")
        match = re.fullmatch(r"[ \t]*([0-9]+)[ \t]*,[ \t]*([0-9]+)[ \t]*,[ \t]*([0-9]+)[ \t]*(?:\r?\n)?", raw) if type(raw) is str else None
        if match is None:
            raise ValueError("production GPU observation is invalid")
        free, used, utilization = map(int, match.groups())
        if not 0 <= utilization <= 100:
            raise ValueError("production GPU utilization is invalid")
        evidence[label] = {
            "observed_at": observed_at,
            "gpu": {"raw": raw, "free_mib": free, "used_mib": used,
                    "utilization_percent": utilization},
            "ollama_expires_at": [row.pop("expires_at", None) for row in value["ollama"]["models"]],
        }
        value["gpu"] = {"free_mib": free, "used_mib": used}
        normalized.append(value)
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError("production invariants changed")
    return evidence


def validate_archive(path, expected_digest, harness, source, policy, parser_revision, layout='test'):
    from kiron_common.embedding_registry import MODEL_CATALOG
    from kiron_common.local_inference import build_resolver_snapshot, RuntimeImplementation
    from kiron_common.local_model_registry.codec import decode_registry
    inventory = archive_inventory(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or digest(canonical_json(inventory)) != expected_digest:
        raise ValueError("archive differs from the independently reviewed inventory")
    plan, result = json_file(path / "plan.json"), json_file(path / "results/result.json")
    kind = plan.get("probe")
    if kind not in KINDS or result.get("status") != "passed" or result.get("controller_cleanup_confirmed") is not True:
        raise ValueError("unrecognized or unsuccessful feature archive")
    if plan.get("source_sha256") != source or plan.get("source_root") != str(path / "source"):
        raise ValueError("historical or different source revision cannot be promoted")
    harness.verify_snapshot(path, source)
    for key, value in {"bundle": str(harness.BUNDLE), "bundle_manifest_sha256": harness.BUNDLE_MANIFEST_SHA,
                       "artifact_layout": layout, "runtime_root": str(harness.RUNTIME),
                       "projector_path": str(harness.PROJECTOR),
                       "profile": harness.PROFILE_ID, "model_sha256": harness.MODEL_SHA,
                       "projector_sha256": harness.PROJECTOR_SHA, "uid": 65534, "gid": 982,
                       "port": harness.PORT, "max_seconds": 600}.items():
        if plan.get(key) != value:
            raise ValueError("unexpected pinned runtime identity: " + key)
    registry = regular_bytes(path / "registry/models.json", 1024**2)
    if digest(registry) != plan.get("registry_sha256"):
        raise ValueError("registry pin differs")
    entries = decode_registry(registry)
    if len(entries) != 1:
        raise ValueError("isolated profile requires exactly one registered deployment")
    entry = entries[0]
    if (entry.reference != str(harness.MODEL) or entry.sha256 != harness.MODEL_SHA
            or entry.size_bytes != harness.MODEL_BYTES or entry.projector is None
            or entry.projector.reference != str(harness.PROJECTOR)
            or entry.projector.sha256 != harness.PROJECTOR_SHA or entry.projector.size_bytes != harness.PROJECTOR_BYTES):
        raise ValueError("registered artifacts differ from measured files")
    snapshot = build_resolver_snapshot(MODEL_CATALOG, entries, resource_profiles=policy.resource_profiles())
    model = snapshot.resolve(plan["model_id"])
    implementation = RuntimeImplementation(policy.runtime_revision, None, parser_revision)
    loaded = result["loaded_health"]
    resident = loaded["models"][model.deployment.id]
    generation = loaded["generation"]
    if (loaded["provider"] != "prism" or loaded["health"] != "available" or resident["state"] != "loaded"
            or not generation["process_id"] or resident["generation"] != generation
            or resident["configuration_fingerprint"] != model.deployment.configuration_fingerprint
            or result["after"]["models"] or result["unloaded"]["observation"]["models"]):
        raise ValueError("runtime identity or load/unload proof differs")
    if json_file(path / "admission/admission.json")["tickets"] or list((path / "uds").glob("*.sock")):
        raise ValueError("archive has unconfirmed cleanup")
    if result["foreign_bearer"]["statuses"] != [401, 401] or result["foreign_bearer"]["active_requests"] != 0:
        raise ValueError("generation guard proof differs")
    started, finished = (datetime.fromisoformat(result[key]) for key in ("started_at", "finished_at"))
    if started.tzinfo is None or finished.tzinfo is None or not 0 < (finished - started).total_seconds() < 730:
        raise ValueError("invalid bounded run timestamps")
    payload = result["public_features"]
    if payload["kind"] != kind or payload["sdk_version"] != "2.29.0" or payload["public_transport"] != "httpx.ASGITransport":
        raise ValueError("unexpected public API proof")
    prefix = "sdk-responses-" if kind.startswith("public-responses") else "sdk-"
    for name, value in payload["results"].items():
        if json_file(path / "results" / (prefix + name + ".json")) != value:
            raise ValueError("standalone SDK result differs")
        if type(value) is dict and "model" in value and value["model"] != model.api_model_id:
            raise ValueError("public request/result model differs")
    from capability_cases import native_calls, validate_cases
    calls = native_calls(path / "results", generation["process_id"], json_file, regular_bytes,
                         model_sha256=harness.MODEL_SHA)
    measured = validate_cases(kind, payload["results"], calls)
    observations = {name: json_file(path / ("production-" + name + ".json")) for name in ("before", "after")}
    if (path / "production-after-settled.json").exists():
        observations["after_settled"] = json_file(path / "production-after-settled.json")
    production = validate_production_observations(observations)
    identity = {"deployment_id": model.deployment.id, "artifact_fingerprint": model.deployment.artifact_identity.fingerprint,
                "configuration_fingerprint": model.deployment.configuration_fingerprint,
                "implementation": asdict(implementation), "resource_profile": asdict(model.deployment.resource_profile)}
    return {"path": path, "sha256": expected_digest, "inventory": inventory, "kind": kind,
            "identity": identity, "model": model, "implementation": implementation, "finished": finished,
            "measured": measured, "production_observations": production,
            "registry_sha256": digest(registry), "snapshot_revision": snapshot.revision}


def profile(reports):
    from kiron_common.local_inference import Capability, CapabilityEvidence, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    from runtime_capabilities import encode_provider_evidence
    from runtime_service import chat_profile
    from kiron_common.model_catalog import BackendType
    if len(reports) != len(KINDS) or {r["kind"] for r in reports} != KINDS:
        raise ValueError("exactly six different successful feature kinds are required")
    first = reports[0]
    if any(r["identity"] != first["identity"] for r in reports):
        raise ValueError("feature parts have different evidence identities")
    features, responses = current_module("probe-openai-features.py"), current_module("probe-openai-responses.py")
    maps, evidence, base = {}, {}, None
    for report in reports:
        identity, impl = report["identity"], report["implementation"]
        model = report["model"]
        item = CapabilityEvidence(impl.provider_revision, identity["artifact_fingerprint"],
            model.deployment.artifact_identity.sha256, model.deployment.artifact_identity.projector.sha256,
            impl.template_revision, impl.parser_revision, identity["configuration_fingerprint"],
            "archive-sha256:" + report["sha256"], report["finished"])
        evidence[report["kind"]] = item
        module = responses if report["kind"].startswith("public-responses") else features
        candidate = module.capabilities(report["kind"], item)
        shared = candidate.by_name[N.CHAT].constraints
        if base is not None and dict(base) != dict(shared):
            raise ValueError("shared chat constraints conflict; no weakening merge is permitted")
        base = shared
        maps[report["kind"]] = candidate
    # Explicit review policy; bounds never come from arbitrary archive maps.
    budgets = (4, 8, 16, 48, 64, 96, 128)
    observed = set().union(*(set(r["measured"]["budgets"]) for r in reports))
    if not set(budgets) <= observed:
        raise ValueError("review profile budget values were not all exercised")
    chat = dict(base)
    chat.update(roles=P(allowed_values=("user", "assistant", "tool")),
                max_output_tokens=P(allowed_values=budgets),
                default_max_output_tokens=P(allowed_values=(64,)),
                token_budget=P(allowed_values=("max_completion_tokens",)))
    common_evidence = tuple(evidence[kind] for kind in sorted(KINDS))
    values = {N.CHAT: Capability(CapabilityStatus.SUPPORTED, chat, common_evidence),
              N.STREAMING: Capability(CapabilityStatus.SUPPORTED, chat, common_evidence)}
    tools = dict(maps["public-tools"].by_name[N.FUNCTION_TOOLS].constraints)
    tools.update(tool_choice=P(allowed_values=("none", "required", "named")), strict=P(allowed_values=(True,)),
                 max_tools=P(allowed_values=(1, 2)))
    tool_evidence = (evidence["public-tools"], evidence["public-responses"])
    values[N.FUNCTION_TOOLS] = Capability(CapabilityStatus.SUPPORTED, tools, tool_evidence)
    values[N.PARALLEL_TOOLS] = Capability(CapabilityStatus.SUPPORTED, {}, tool_evidence)
    vision = dict(maps["public-vision"].by_name[N.VISION].constraints)
    size = next(r["measured"]["vision_bytes"] for r in reports if r["kind"] == "public-vision")
    vision.update(images=P(allowed_values=(2,)), width=P(allowed_values=(96,)), height=P(allowed_values=(96,)),
                  image_pixels=P(allowed_values=(96 * 96,)), total_pixels=P(allowed_values=(2 * 96 * 96,)),
                  normalized_bytes=P(minimum=1, maximum=size))
    values[N.VISION] = Capability(CapabilityStatus.SUPPORTED, vision, (evidence["public-vision"],))
    values[N.STRUCTURED_OUTPUT] = maps["public-structured"].by_name[N.STRUCTURED_OUTPUT]
    values[N.REASONING] = replace(maps["public-reasoning"].by_name[N.REASONING],
        evidence=(evidence["public-reasoning"], evidence["public-responses"], evidence["public-responses-budget"]))
    capabilities = CapabilitySet(values)
    chat_profile(capabilities.by_name[N.CHAT], first["model"].deployment)
    if any(not capabilities.supports(name, first["model"].deployment, first["implementation"]) for name in values):
        raise ValueError("exported evidence does not match the executable deployment")
    return encode_provider_evidence(BackendType.PRISM, {first["model"].deployment.id: capabilities})


def assemble(specifications, layout='test'):
    from runtime_composition import adapter_revision
    harness = current_module("smoke-controller.py")
    harness.RUNTIME, harness.PROJECTOR = harness.artifact_layout(layout)
    source = harness.code_inventory()
    policy = harness.policy()
    policy.verify_bundle()
    revision = adapter_revision("prism_provider.py")
    reports = [validate_archive(path, expected, harness, source, policy, revision, layout) for path, expected in specifications]
    candidate = profile(reports)
    # Verify actual immutable bytes once, not once per archive. No inference.
    for file, sha, size in ((harness.MODEL, harness.MODEL_SHA, harness.MODEL_BYTES),
                             (harness.PROJECTOR, harness.PROJECTOR_SHA, harness.PROJECTOR_BYTES)):
        fd = policy.open_artifact(str(file), sha, size)
        os.close(fd)
    if harness.code_inventory() != source or adapter_revision("prism_provider.py") != revision:
        raise ValueError("workspace changed during export")
    for report in reports:
        if archive_inventory(report["path"]) != report["inventory"]:
            raise ValueError("archive changed during export")
    provenance = {"version": 1, "profile": PROFILE, "artifact_layout": layout,
        "scope": "isolated deployment review only; not a production grant",
        "identity": reports[0]["identity"], "candidate_sha256": digest(canonical_json(candidate)),
        "exporter_sha256": {name: digest(regular_bytes(Path(__file__).with_name(name)))
                             for name in ("export-capabilities.py", "capability_cases.py")},
        "reports": [{"path": str(r["path"]), "sha256": r["sha256"], "kind": r["kind"],
                     "files": r["inventory"], "measured": r["measured"],
                     "registry_sha256": r["registry_sha256"], "snapshot_revision": r["snapshot_revision"],
                     "production_observations": r["production_observations"]} for r in sorted(reports, key=lambda r: r["kind"])],
        "limitations": ["Projector path is part of artifact and configuration identity.",
                       "Production provisioning must match the measured paths, bytes and policy; live service acceptance is separate.",
                       "No system/developer role, automatic tools, nonstrict tool definitions, arbitrary image dimensions, or other reasoning efforts.",
                       "Independent grants also admit unmeasured vision+tools/structured/reasoning combinations; tools+structured and active reasoning+tools/structured/stop remain rejected.",
                       "Strict constraints apply to every declared tool definition; mixed lists cannot bypass the measured strict-only profile.",
                       "Default budget 64 is explicit local policy chosen from exercised values.",
                       "Ollama and embeddings are not covered by this exporter."]}
    return candidate, provenance


def output_path(path):
    if (not path.is_absolute() or path.parent != OUTPUT_ROOT or path.resolve() != path
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", path.name)):
        raise ValueError("output must be a new isolated capability-candidates directory")
    return path


def publish(path, candidate, provenance):
    output_path(path)
    if os.geteuid() != 0:
        raise ValueError("candidate publication requires root-owned review files")
    for parent in OUTPUT_ROOT.parents:
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("unsafe candidate parent")
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o750)
        os.chown(OUTPUT_ROOT, 0, 982)
    info = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError("unsafe candidate parent")
    stage = Path(tempfile.mkdtemp(prefix=".candidate-", dir=OUTPUT_ROOT))
    try:
        for name, value in (("prism.json", candidate), ("provenance.json", provenance)):
            target = stage / name
            with target.open("xb") as stream:
                stream.write(canonical_json(value) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chown(target, 0, 982)
            target.chmod(0o640)
        os.chown(stage, 0, 982)
        stage.chmod(0o750)
        libc = ctypes.CDLL(None, use_errno=True)
        # Linux renameat2 RENAME_NOREPLACE: never overwrite an existing review.
        if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(path), 1) != 0:
            raise OSError(ctypes.get_errno(), "candidate atomic publication failed")
        directory_fd = os.open(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def specification(value):
    path, separator, expected = value.rpartition("=")
    if not separator or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise argparse.ArgumentTypeError("report requires absolute-path=reviewed-inventory-sha256")
    return Path(path), expected


def verify_candidate(path):
    path = output_path(path)
    if {f.name for f in path.iterdir()} != {"prism.json", "provenance.json"}:
        raise ValueError("candidate file set differs")
    for item in (path, path / "prism.json", path / "provenance.json"):
        info = item.lstat()
        expected_mode = 0o750 if item == path else 0o640
        if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, 982, expected_mode):
            raise ValueError("candidate ownership or mode differs")
    prior = json_file(path / "provenance.json")
    specifications = [(Path(r["path"]), r["sha256"]) for r in prior["reports"]]
    candidate, provenance = assemble(specifications, prior['artifact_layout'])
    if (regular_bytes(path / "provenance.json") != canonical_json(provenance) + b"\n"
            or regular_bytes(path / "prism.json") != canonical_json(candidate) + b"\n"):
        raise ValueError("candidate or provenance differs from current verified evidence")
    return provenance["candidate_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--report", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--report", type=specification, action="append", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--artifact-layout", choices=('test', 'production'), default='test')
    verify = commands.add_parser("verify")
    verify.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    initialize()
    if args.command == "inspect":
        inventory = archive_inventory(args.report)
        print(json.dumps({"path": str(args.report), "sha256": digest(canonical_json(inventory)),
                          "files": len(inventory), "scope": "inventory only; no capability approval"}))
    elif args.command == "export":
        output_path(args.output)
        candidate, provenance = assemble(args.report, args.artifact_layout)
        publish(args.output, candidate, provenance)
        print(json.dumps({"candidate": str(args.output), "sha256": provenance["candidate_sha256"]}))
    else:
        print(json.dumps({"status": "verified", "sha256": verify_candidate(args.candidate)}))


if __name__ == "__main__":
    main()
