"""Common contracts: identity, immutable snapshots, evidence and event semantics."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    DeploymentObservation, EmbeddingRequest, EmbeddingRole, ErrorCode, EventKind, FinishReason,
    GenerateRequest, GenerationOptions, ImagePart, InferenceEvent, InferenceRequest, InferenceResult,
    LocalInferenceError, Message, MessageRole, ParameterConstraint, ProviderHealth,
    ProviderObservation, RequestContext, ResolverConfigurationError, ResourceProfile,
    RuntimeGeneration, RuntimeImplementation, RuntimeTimeouts, TextPart, TokenUsage,
    ToolCall, ToolChoice, ToolChoiceKind, ToolDefinition, build_resolver_snapshot,
    validate_event_sequence,
)
from kiron_common.local_model_registry.models import LocalArtifactFile, RegistryEntry
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType, ModelCatalog, load_catalog
from kiron_common.model_state import RuntimeState


NOW = datetime(2026, 9, 21, tzinfo=timezone.utc)
IMPLEMENTATION = RuntimeImplementation("prism-pin", "template-pin", "parser-pin")
PROFILE = ResourceProfile("fixture-cuda40", 1024, 128, 128, 1, 4, 40, False,
                          5 * 1024**3, 8 * 1024**3, 256 * 1024**2)
PROFILES = {PROFILE.id: PROFILE}


def local_model(**changes):
    values = dict(runtime_provider=BackendType.PRISM, artifact_origin=ArtifactType.LOCAL,
                  artifact_format=ArtifactFormat.GGUF, reference="/models/fixture.gguf",
                  display_name="Fixture", loader=LoaderType.PRISM_GGUF,
                  sha256="a" * 64, size_bytes=100, runtime_profile=PROFILE.id)
    values.update(changes)
    return replace(RegistryEntry.create(**values), registered_at=NOW)


def catalog_manifest(**changes):
    value = {
        "schema_version": 2, "canonical_model_id": "fixture", "aliases": ["fixture-alias"],
        "deployments": [{"id": "fixture.deployment", "backend": {"type": "prism", "parameters": {"model_name": "fixture"}},
                         "artifact": {"type": "local", "format": "gguf", "repository": None, "revision": None,
                                      "manifest_digest": None, "trust_remote_code": False,
                                      "weights": [{"path": "/models/fixture.gguf", "sha256": "a" * 64, "size_bytes": 100}],
                                      "auxiliary": [], "projector": None, "metadata": {}},
                         "routes": [{"task": "chat", "endpoint": "/v1/chat/completions"}],
                         "loader": {"type": "prism_gguf", "parameters": {}},
                         "runtime_profile": PROFILE.id, "metadata": {}}],
        "profiles": [{"id": "fixture.chat", "deployment_id": "fixture.deployment", "task": "chat",
                      "endpoint": "/v1/chat/completions", "default_for_endpoint": True,
                      "metadata": {"created": 1758412800}}],
        "request_defaults": [], "metadata": {},
    }
    value.update(changes)
    return value


def snapshot(entry=None, *, profile=PROFILE):
    return build_resolver_snapshot(ModelCatalog(), [entry or local_model()], resource_profiles={profile.id: profile})


def resolved():
    entry = local_model()
    return snapshot(entry).resolve(entry.id)


def evidence(deployment):
    artifact = deployment.artifact_identity
    return CapabilityEvidence(IMPLEMENTATION.provider_revision, artifact.fingerprint, artifact.sha256,
                              artifact.projector.sha256 if artifact.projector else None,
                              IMPLEMENTATION.template_revision, IMPLEMENTATION.parser_revision,
                              deployment.configuration_fingerprint, "fixture:verified", NOW)


def event(kind, **kwargs):
    return InferenceEvent(kind, "request-1", **kwargs)


def test_snapshot_merges_only_exact_content_and_preserves_catalog_configuration():
    catalog = ModelCatalog.from_manifests([catalog_manifest()])
    entry = local_model(reference="/models/another-copy.gguf")
    snap = build_resolver_snapshot(catalog, [entry], resource_profiles=PROFILES)
    assert len(snap.deployments) == 1
    model = snap.resolve("fixture-alias")
    assert model.deployment.registry_ids == (entry.id,)
    assert model.deployment.reference == "/models/fixture.gguf"
    assert snap.resolve(entry.id).deployment is model.deployment
    assert snap.resolve_deployment(model.deployment.id, snap.revision) is model.deployment
    assert snap.resolve_deployment(entry.id, snap.revision) is model.deployment
    with pytest.raises(LocalInferenceError) as caught:
        snap.resolve_deployment(model.deployment.id, "0" * 64)
    assert caught.value.failure.code is ErrorCode.CONFLICT


def test_catalog_api_aliases_share_canonical_id_and_explicit_created():
    catalog = ModelCatalog.from_manifests([catalog_manifest()])
    snap = build_resolver_snapshot(catalog, (), resource_profiles=PROFILES)
    canonical = snap.resolve("fixture.chat")
    assert canonical.api_model_id == "fixture.chat"
    assert canonical.created == 1758412800
    for name in ("fixture", "fixture-alias"):
        alias = snap.resolve(name)
        assert alias.api_model_id == canonical.public_model_id
        assert replace(alias, public_model_id=canonical.public_model_id) == canonical


def test_registered_record_retains_timestamp_and_profile_independence_after_exact_merge():
    from kiron_common.local_model_registry.codec import decode_registry, encode_registry
    entry = local_model(reference="/models/local-copy.gguf")
    local = snapshot(entry).resolve(entry.id)
    merged = build_resolver_snapshot(ModelCatalog.from_manifests([catalog_manifest()]),
        decode_registry(encode_registry((entry,))), resource_profiles=PROFILES).resolve(entry.id)
    assert local.created == merged.created == int(NOW.timestamp())
    assert local.api_model_id == merged.api_model_id == entry.id
    assert local.profile_id is merged.profile_id is None
    assert local.profile_metadata == merged.profile_metadata == {}
    assert merged.deployment.id == "fixture.deployment"


def test_registration_timestamp_is_not_artifact_or_runtime_configuration_identity():
    entry = local_model()
    older = snapshot(entry).resolve(entry.id)
    newer = snapshot(replace(entry, registered_at=NOW + timedelta(seconds=5))).resolve(entry.id)
    assert older.api_model_id == newer.api_model_id
    assert newer.created == older.created + 5
    assert older.deployment == newer.deployment
    assert older.snapshot_revision != newer.snapshot_revision


@pytest.mark.parametrize("created", [None, False, -1, 1.5, "1758412800"])
def test_chat_api_profiles_require_explicit_integer_created(created):
    manifest = catalog_manifest()
    manifest["profiles"][0]["metadata"] = {} if created is None else {"created": created}
    catalog = ModelCatalog.from_manifests([manifest])
    with pytest.raises(ResolverConfigurationError, match="explicit created"):
        build_resolver_snapshot(catalog, (), resource_profiles=PROFILES)


def test_non_api_profiles_have_no_invented_creation_time():
    snap = build_resolver_snapshot(load_catalog(), ())
    assert snap.models
    assert all(model.created is None for model in snap.models.values()
               if "created" not in model.profile_metadata)
    assert all(model.api_model_id == model.public_model_id for model in snap.models.values())


def test_resolved_model_api_metadata_validation_and_canonical_snapshot_binding():
    from kiron_common.local_inference import ResolverSnapshot
    entry = local_model()
    snap = snapshot(entry)
    model = snap.resolve(entry.id)
    for created in (False, -1, "1"):
        with pytest.raises(ValueError, match="creation timestamp"):
            replace(model, created=created)
    with pytest.raises(ResolverConfigurationError, match="canonical API record"):
        ResolverSnapshot(snap.revision, snap.deployments,
                         {entry.id: replace(model, canonical_model_id="missing")})


def test_same_path_and_size_with_different_digest_do_not_merge():
    snap = build_resolver_snapshot(ModelCatalog.from_manifests([catalog_manifest()]),
                                   [local_model(sha256="b" * 64)], resource_profiles=PROFILES)
    assert len(snap.deployments) == 2


def test_conflicting_measured_size_or_loader_parameters_do_not_merge():
    catalog = ModelCatalog.from_manifests([catalog_manifest()])
    snap = build_resolver_snapshot(catalog, [local_model(size_bytes=101)], resource_profiles=PROFILES)
    assert len(snap.deployments) == 2
    manifest = catalog_manifest()
    manifest["deployments"][0]["loader"]["parameters"] = {"chat_template": "different"}
    snap = build_resolver_snapshot(ModelCatalog.from_manifests([manifest]), [local_model()], resource_profiles=PROFILES)
    assert len(snap.deployments) == 2


def test_unknown_local_digest_never_merges_with_known_catalog_tag():
    catalog = load_catalog()
    entry = RegistryEntry.create(runtime_provider=BackendType.OLLAMA, artifact_origin=ArtifactType.OLLAMA,
                                 artifact_format=ArtifactFormat.OLLAMA_MANIFEST, reference="nomic-embed-text:latest",
                                 display_name="same name", loader=LoaderType.OLLAMA)
    snap = build_resolver_snapshot(catalog, [entry])
    assert snap.resolve(entry.id).deployment.id == entry.id
    assert snap.resolve(entry.id).deployment.artifact_identity.content_key is None


def test_projector_and_profile_changes_do_not_merge():
    catalog = ModelCatalog.from_manifests([catalog_manifest()])
    projector = LocalArtifactFile("/models/projector.gguf", "b" * 64, 25)
    snap = build_resolver_snapshot(catalog, [local_model(projector=projector)], resource_profiles=PROFILES)
    assert len(snap.deployments) == 2
    other = replace(PROFILE, id="other-profile", context_tokens=2048)
    snap = build_resolver_snapshot(catalog, [local_model(runtime_profile=other.id)],
                                   resource_profiles={PROFILE.id: PROFILE, other.id: other})
    assert len(snap.deployments) == 2


def test_snapshot_retains_old_configuration_after_atomic_replacement():
    entry = local_model()
    old = snapshot(entry)
    request_model = old.resolve(entry.id)
    new = snapshot(local_model(sha256="c" * 64))
    assert old.revision != new.revision
    assert request_model.deployment.artifact_identity.sha256 == "a" * 64
    assert new.resolve(entry.id).deployment.artifact_identity.sha256 == "c" * 64
    empty = build_resolver_snapshot(ModelCatalog(), ())
    with pytest.raises(LocalInferenceError):
        empty.resolve(entry.id)
    assert request_model is old.resolve(entry.id)
    with pytest.raises(TypeError):
        old.models["bad"] = request_model


def test_same_profile_id_with_changed_values_invalidates_configuration():
    entry = local_model()
    first = snapshot(entry).resolve(entry.id)
    second = snapshot(entry, profile=replace(PROFILE, threads=2)).resolve(entry.id)
    assert first.deployment.id == second.deployment.id
    assert first.deployment.configuration_fingerprint != second.deployment.configuration_fingerprint
    assert first.snapshot_revision != second.snapshot_revision


def test_duplicate_registry_and_public_alias_collisions_fail_closed():
    entry = local_model()
    with pytest.raises(ResolverConfigurationError, match="Duplicate"):
        build_resolver_snapshot(ModelCatalog(), [entry, entry], resource_profiles=PROFILES)
    catalog = ModelCatalog.from_manifests([catalog_manifest(aliases=[entry.id])])
    with pytest.raises(ResolverConfigurationError, match="collision"):
        build_resolver_snapshot(catalog, [local_model(sha256="b" * 64)], resource_profiles=PROFILES)
    with pytest.raises(ResolverConfigurationError, match="Unknown runtime profile"):
        build_resolver_snapshot(ModelCatalog(), [entry])


def test_multiple_exact_catalog_targets_are_an_explicit_configuration_error():
    manifest = catalog_manifest()
    deployment = copy.deepcopy(manifest["deployments"][0])
    deployment["id"] = "fixture.other"
    manifest["deployments"].append(deployment)
    profile = copy.deepcopy(manifest["profiles"][0])
    profile.update(id="fixture.other.chat", deployment_id=deployment["id"], default_for_endpoint=False)
    manifest["profiles"].append(profile)
    catalog = ModelCatalog.from_manifests([manifest])
    with pytest.raises(ResolverConfigurationError, match="Ambiguous exact artifact"):
        build_resolver_snapshot(catalog, [local_model()], resource_profiles=PROFILES)


def test_registry_order_does_not_change_revision_or_rebind_aliases():
    a, b = local_model(), local_model(reference="/models/b.gguf", sha256="b" * 64)
    first = build_resolver_snapshot(ModelCatalog(), [a, b], resource_profiles=PROFILES)
    second = build_resolver_snapshot(ModelCatalog(), [b, a], resource_profiles=dict(PROFILES))
    assert first.revision == second.revision
    with pytest.raises(LocalInferenceError):
        first.resolve(a.display_name)


def test_embedding_roles_are_explicit_and_capture_pipeline_metadata():
    snap = build_resolver_snapshot(load_catalog(), ())
    query = snap.resolve("kiron-nomic-dense-v1.query")
    document = snap.resolve("kiron-nomic-dense-v1.document")
    assert query.embedding_role.value == "search_query"
    assert document.embedding_role.value == "search_document"
    assert query.deployment is document.deployment
    assert query.profile_metadata["pipeline"]
    with pytest.raises(LocalInferenceError):
        snap.resolve("nomic-embed-text")
    with pytest.raises(ValueError, match="profile"):
        EmbeddingRequest(resolved(), ("text",), RequestContext("r", 10, threading.Event()))


def test_dense_api_roles_creation_and_neutral_profile_survive_snapshot_rebuild():
    first = build_resolver_snapshot(load_catalog(), ())
    second = build_resolver_snapshot(load_catalog(), ())
    context = RequestContext("embedding", 10, threading.Event())
    neutral = first.resolve("kiron-bge-m3-dense-v1")
    assert neutral.embedding_role is None
    assert EmbeddingRequest(neutral, (" original ",), context).inputs == (" original ",)
    for profile in ("kiron-bge-m3-dense-v1", "kiron-nomic-dense-v1", "kiron-mankei-dense-v1"):
        for suffix, role in ((".query", EmbeddingRole.QUERY), (".document", EmbeddingRole.DOCUMENT)):
            model = first.resolve(profile + suffix)
            assert model == second.resolve(profile + suffix)
            assert model.created == 1790035200
            assert model.api_model_id == profile + suffix
            assert model.embedding_role is role
    with pytest.raises(LocalInferenceError):
        first.resolve("kiron-nomic-dense-v1")
    with pytest.raises(ValueError):
        EmbeddingRequest(replace(neutral, profile_metadata={**neutral.profile_metadata,
            "verification": {"status": "unverified"}}), ("text",), context)


def test_capability_evidence_requires_exact_configuration_and_implementation():
    deployment = resolved().deployment
    caps = CapabilitySet({CapabilityName.CHAT: Capability(CapabilityStatus.SUPPORTED,
                        {"temperature": ParameterConstraint(minimum=0, maximum=1)}, (evidence(deployment),))})
    assert caps.supports(CapabilityName.CHAT, deployment, IMPLEMENTATION)
    for changed in (replace(IMPLEMENTATION, provider_revision="new-binary"),
                    replace(IMPLEMENTATION, parser_revision="new-parser"),
                    replace(IMPLEMENTATION, template_revision="new-template")):
        assert not caps.supports(CapabilityName.CHAT, deployment, changed)
    changed = replace(deployment, configuration_fingerprint="d" * 64)
    assert not caps.supports(CapabilityName.CHAT, changed, IMPLEMENTATION)
    assert caps.for_deployment(changed, IMPLEMENTATION).by_name[CapabilityName.CHAT].status is CapabilityStatus.UNVERIFIED
    assert not caps.supports(CapabilityName.VISION, deployment, IMPLEMENTATION)
    with pytest.raises(ValueError, match="evidence"):
        Capability(CapabilityStatus.SUPPORTED)


def test_projector_identity_invalidates_prior_evidence_without_changing_model_id():
    original = snapshot(local_model()).resolve(local_model().id).deployment
    verified = CapabilitySet({CapabilityName.CHAT: Capability(CapabilityStatus.SUPPORTED, evidence=(evidence(original),))})
    replacement = snapshot(local_model(projector=LocalArtifactFile("/models/mmproj.gguf", "f" * 64, 20)))
    updated = replacement.resolve(original.id).deployment
    assert updated.id == original.id
    assert not verified.supports(CapabilityName.CHAT, updated, IMPLEMENTATION)


def test_constraint_bounds_distinguish_bool_and_integer():
    assert not ParameterConstraint(minimum=0, maximum=1).accepts(True)
    assert not ParameterConstraint(allowed_values=(1,)).accepts(True)
    assert ParameterConstraint(allowed_values=("auto", "required")).accepts("required")
    assert not ParameterConstraint(minimum=0, maximum=1).accepts(float("nan"))


def test_ordered_multimodal_and_nested_tool_schemas_are_immutable():
    image = ImagePart("image/png", b"validated-image", hashlib.sha256(b"validated-image").hexdigest(), 1, 1)
    parts = [TextPart("before"), image, TextPart("after")]
    schema = {"type": "object", "properties": {"city": {"enum": ["Berlin"]}}}
    tools = [ToolDefinition("weather", schema, strict=True)]
    request = InferenceRequest(resolved(), [Message(MessageRole.USER, parts)], GenerationOptions(32),
                               RequestContext("r", 10, threading.Event()), tools, ToolChoice(ToolChoiceKind.REQUIRED))
    parts.reverse()
    schema["properties"]["city"]["enum"].append("Paris")
    tools.clear()
    assert [type(p) for p in request.messages[0].content] == [TextPart, ImagePart, TextPart]
    assert request.messages[0].content[0].text == "before"
    assert request.tools[0].parameters["properties"]["city"]["enum"] == ("Berlin",)
    with pytest.raises(FrozenInstanceError):
        image.width = 2


def test_usage_preserves_unknown_details_and_never_guesses_zero():
    usage = TokenUsage(12, 3)
    assert usage.total_tokens == 15
    assert usage.reasoning_output_tokens is None and usage.cached_input_tokens is None
    assert TokenUsage(12, 3, 0, 0).reasoning_output_tokens == 0
    for values in ((True, 3), (3, -1), (3, 1, 4), (3, 1, None, 2)):
        with pytest.raises(ValueError):
            TokenUsage(*values)


def test_normalized_image_keeps_source_identity_and_detail_without_ambiguity():
    data = b"validated-normalized-png"
    image = ImagePart("image/png", data, hashlib.sha256(data).hexdigest(), 2, 3,
                      detail="auto", source_media_type="image/webp", source_sha256="a" * 64)
    assert image.source_media_type == "image/webp" and image.media_type == "image/png"
    for changed in ({"detail": "original"}, {"source_sha256": None}, {"source_media_type": "image/svg+xml"},
                    {"source_sha256": "unverified"}):
        with pytest.raises(ValueError):
            replace(image, **changed)


def test_execution_generation_is_bound_after_normalization_and_cannot_mutate():
    model = resolved()
    context = RequestContext("r", 10, threading.Event())
    requests = [InferenceRequest(model, (Message(MessageRole.USER, (TextPart("hi"),)),), GenerationOptions(8), context),
                GenerateRequest(model, "hi", GenerationOptions(8), context),
                EmbeddingRequest(replace(model, profile_id="embedding", embedding_role=EmbeddingRole.QUERY), ("hi",), context)]
    generation = RuntimeGeneration("boot", "spawn")
    for request in requests:
        assert request.execution_generation is None
        bound = replace(request, execution_generation=generation)
        assert bound.execution_generation == generation
        assert request.execution_generation is None
        with pytest.raises(FrozenInstanceError):
            bound.execution_generation = None
        with pytest.raises(TypeError, match="execution generation"):
            replace(request, execution_generation="spawn")


def test_interleaved_provider_calls_keep_identity_and_exact_argument_text():
    a, b = ToolCall(0, "a", "weather", '{"city":"Berlin"}'), ToolCall(1, "b", "weather", '{"city":"Paris"}')
    stream = [event(EventKind.STARTED),
              event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="weather", output_item_index=0),
              event(EventKind.TOOL_CALL_STARTED, call_index=1, call_id="b", name="weather", output_item_index=1),
              event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text='{"city":', output_item_index=0),
              event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=1, text=b.arguments, output_item_index=1),
              event(EventKind.TOOL_CALL_COMPLETED, tool_call=b, output_item_index=(b).index),
              event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text='"Berlin"}', output_item_index=0),
              event(EventKind.TOOL_CALL_COMPLETED, tool_call=a, output_item_index=(a).index),
              event(EventKind.USAGE, usage=TokenUsage(5, 20)),
              event(EventKind.COMPLETED, finish_reason=FinishReason.TOOL_CALLS)]
    validate_event_sequence(stream)
    with pytest.raises(ValueError, match="terminal"):
        validate_event_sequence([*stream, event(EventKind.CANCELLED)])
    with pytest.raises(ValueError, match="accumulated"):
        validate_event_sequence([*stream[:4], event(EventKind.TOOL_CALL_COMPLETED, tool_call=a, output_item_index=(a).index)])


def test_cancellation_and_length_keep_partial_arguments_without_claiming_completion():
    partial = [event(EventKind.STARTED), event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="fn", output_item_index=0),
               event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text='{"x":', output_item_index=0)]
    validate_event_sequence([*partial, event(EventKind.CANCELLED)])
    validate_event_sequence([*partial, event(EventKind.COMPLETED, finish_reason=FinishReason.LENGTH)])
    with pytest.raises(ValueError, match="unfinished"):
        validate_event_sequence([*partial, event(EventKind.COMPLETED, finish_reason=FinishReason.STOP)])
    call = ToolCall(0, "a", "fn", '{"x":', complete=False)
    InferenceResult("r", (), (), (call,), None, FinishReason.LENGTH)
    with pytest.raises(ValueError, match="complete"):
        event(EventKind.TOOL_CALL_COMPLETED, tool_call=call, output_item_index=(call).index)
    with pytest.raises(ValueError, match="payload"):
        event(EventKind.TEXT_DELTA, text="x", usage=TokenUsage(1, 1), output_item_index=0, part_index=0)


def test_loaded_state_needs_matching_healthy_generation_and_configuration():
    generation = RuntimeGeneration("boot", "spawn")
    model = DeploymentObservation("fixture", RuntimeState.LOADED, generation, "a" * 64)
    ProviderObservation(BackendType.PRISM, generation, NOW, ProviderHealth.AVAILABLE, {"fixture": model})
    for health, current in ((ProviderHealth.UNKNOWN, generation), (ProviderHealth.AVAILABLE, RuntimeGeneration("new-boot", "spawn"))):
        with pytest.raises(ValueError, match="generation"):
            ProviderObservation(BackendType.PRISM, current, NOW, health, {"fixture": model})
    with pytest.raises(ValueError, match="evidence"):
        DeploymentObservation("fixture", RuntimeState.LOADED, None, None)
    with pytest.raises(ValueError, match="positive"):
        RuntimeTimeouts(1, 1, 1, 1, 0, 1, 1)


def test_import_has_no_network_or_process_side_effects():
    code = '''
import sys
def audit(event, args):
    if event in ("socket.connect", "subprocess.Popen", "os.system"):
        raise AssertionError(event)
sys.addaudithook(audit)
import kiron_common.local_inference
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                 "PYTHONPATH": str(Path(__file__).resolve().parents[1])}, timeout=10)
    assert result.returncode == 0, result.stderr
