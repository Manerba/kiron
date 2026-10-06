"""Pure Catalog + local Registry projection, with immutable request snapshots."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType

from kiron_common.model_catalog import ArtifactFormat, BackendType, ModelCatalog, ModelEndpoint, ModelTask
from kiron_common.local_model_registry.models import RegistryEntry

from ._values import fingerprint, sha256
from .identity import (
    ArtifactFileReference, ArtifactIdentity, EmbeddingRole, ModelSource,
    ResolvedDeployment, ResolvedModel, ResourceProfile,
)
from .lifecycle import ErrorCode, LocalInferenceError, ResolverConfigurationError, RuntimeFailure
from .embedding import neutral_profile


@dataclass(frozen=True, slots=True)
class ResolverSnapshot:
    revision: str
    deployments: Mapping[str, ResolvedDeployment]
    models: Mapping[str, ResolvedModel]
    deployment_aliases: Mapping[str, str] = field(init=False)

    def __post_init__(self):
        sha256(self.revision, "snapshot revision")
        deployments, models = dict(self.deployments), dict(self.models)
        for key, value in deployments.items():
            if not isinstance(value, ResolvedDeployment) or key != value.id:
                raise ResolverConfigurationError("invalid deployment index")
        for key, value in models.items():
            if (not isinstance(value, ResolvedModel) or key != value.public_model_id
                    or value.snapshot_revision != self.revision
                    or deployments.get(value.deployment.id) != value.deployment):
                raise ResolverConfigurationError("model is not bound to this snapshot")
            canonical = models.get(value.api_model_id)
            if canonical != replace(value, public_model_id=value.api_model_id):
                raise ResolverConfigurationError("model alias differs from its canonical API record")
        aliases = {}
        for deployment in deployments.values():
            for alias in deployment.registry_ids:
                if alias == deployment.id:
                    continue
                if alias in deployments or alias in aliases:
                    raise ResolverConfigurationError(f"Deployment alias collision: {alias}")
                aliases[alias] = deployment.id
        object.__setattr__(self, "deployments", MappingProxyType(deployments))
        object.__setattr__(self, "models", MappingProxyType(models))
        object.__setattr__(self, "deployment_aliases", MappingProxyType(aliases))

    def resolve(self, public_model_id: str) -> ResolvedModel:
        if type(public_model_id) is not str or public_model_id not in self.models:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.MODEL_NOT_FOUND, "Unknown model", "model"))
        return self.models[public_model_id]

    def resolve_deployment(self, deployment_id: str, expected_revision: str | None = None) -> ResolvedDeployment:
        if expected_revision is not None and expected_revision != self.revision:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.CONFLICT, "Resolver revision changed"))
        if type(deployment_id) is not str:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.MODEL_NOT_FOUND, "Unknown deployment"))
        canonical = self.deployment_aliases.get(deployment_id, deployment_id)
        if canonical not in self.deployments:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.MODEL_NOT_FOUND, "Unknown deployment"))
        return self.deployments[canonical]


def _profile(profile_id: str | None, profiles: Mapping[str, ResourceProfile]) -> ResourceProfile | None:
    if profile_id is None:
        return None
    if profile_id not in profiles:
        raise ResolverConfigurationError(f"Unknown runtime profile: {profile_id}")
    return profiles[profile_id]


def _projector(value) -> ArtifactFileReference | None:
    return None if value is None else ArtifactFileReference(value.path, value.sha256, value.size_bytes)


def _catalog_deployment(value, profiles) -> ResolvedDeployment:
    artifact = value.artifact
    primary_hash, primary_size = None, None
    if artifact.format is ArtifactFormat.GGUF:
        if len(artifact.weights) != 1:
            raise ResolverConfigurationError("GGUF needs exactly one primary weight file")
        primary = artifact.weights[0]
        reference, primary_hash, primary_size = primary.path, primary.sha256, primary.size_bytes
    else:
        reference = value.backend.parameters.get("model_name")
        if not isinstance(reference, str) or not reference:
            raise ResolverConfigurationError(f"Missing canonical model_name for {value.id}")
        if artifact.format is ArtifactFormat.OLLAMA_MANIFEST and artifact.manifest_digest:
            primary_hash = artifact.manifest_digest.removeprefix("sha256:")
    identity = ArtifactIdentity(
        origin=artifact.type, format=artifact.format, sha256=primary_hash, size_bytes=primary_size,
        projector=_projector(artifact.projector), repository=artifact.repository,
        revision=artifact.revision, manifest_digest=artifact.manifest_digest,
        files=(*artifact.weights, *artifact.auxiliary),
    )
    profile = _profile(value.runtime_profile, profiles)
    config = fingerprint("kiron.local-inference.deployment.v1", {
        "definition": value.to_dict(), "resource_profile": profile,
    })
    return ResolvedDeployment(
        id=value.id, provider=value.backend.type, reference=reference, artifact_identity=identity,
        loader=value.loader.type, resource_profile=profile, configuration_fingerprint=config,
        backend_parameters=value.backend.parameters, loader_parameters=value.loader.parameters,
        metadata=value.metadata,
    )


def _local_deployment(entry: RegistryEntry, profiles) -> ResolvedDeployment:
    projector = entry.projector
    identity = ArtifactIdentity(
        origin=entry.artifact_origin, format=entry.artifact_format, sha256=entry.sha256,
        size_bytes=entry.size_bytes,
        projector=(ArtifactFileReference(projector.reference, projector.sha256, projector.size_bytes)
                   if projector else None),
    )
    profile = _profile(entry.runtime_profile, profiles)
    config = fingerprint("kiron.local-inference.deployment.v1", {
        "registry_configuration": entry.configuration_fingerprint, "resource_profile": profile,
    })
    return ResolvedDeployment(
        id=entry.id, provider=entry.runtime_provider, reference=entry.reference,
        artifact_identity=identity, loader=entry.loader, resource_profile=profile,
        configuration_fingerprint=config, source=ModelSource.LOCAL_REGISTRY, registry_ids=(entry.id,),
    )


def _merge_key(deployment: ResolvedDeployment):
    content = deployment.artifact_identity.content_key
    if content is None or deployment.loader_parameters:
        return None
    return (deployment.provider, deployment.loader, deployment.resource_profile, content)


def build_resolver_snapshot(
    catalog: ModelCatalog,
    registry_entries: Iterable[RegistryEntry],
    *,
    resource_profiles: Mapping[str, ResourceProfile] | None = None,
) -> ResolverSnapshot:
    """Read already-validated values only. Caller publishes the result atomically.

    Unknown local artifact identity never merges by tag, name, path or size.
    Registration adds status to an exact deployment without overriding catalog
    configuration. Missing profiles and ambiguous aliases fail the whole build.
    """
    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be ModelCatalog")
    profiles = dict(resource_profiles or {})
    if any(not isinstance(value, ResourceProfile) or key != value.id for key, value in profiles.items()):
        raise ResolverConfigurationError("invalid resource profile map")
    deployments, exact, addresses = {}, {}, {}

    def address(name, deployment_id, profile=None, role=None, *, created=None, canonical_model_id=None):
        if profile is not None and profile.task is ModelTask.CHAT:
            created = profile.metadata.get("created")
            if type(created) is not int or created < 0:
                raise ResolverConfigurationError(f"Chat API profile needs explicit created metadata: {profile.id}")
            canonical_model_id = profile.id
        elif profile is not None and profile.task is ModelTask.EMBEDDING and "created" in profile.metadata:
            created = profile.metadata["created"]
            if type(created) is not int or created < 0:
                raise ResolverConfigurationError(f"Embedding API profile needs valid created metadata: {profile.id}")
            canonical_model_id = name
        value = (deployment_id, profile.id if profile else None, role,
                 profile.task if profile else None, profile.endpoint if profile else None,
                 profile.metadata if profile else {}, created, canonical_model_id)
        if name in addresses and addresses[name] != value:
            raise ResolverConfigurationError(f"Public model ID collision: {name}")
        addresses[name] = value

    for group in catalog.groups:
        for definition in group.deployments:
            deployment = _catalog_deployment(definition, profiles)
            if deployment.id in deployments:
                raise ResolverConfigurationError(f"Duplicate deployment ID: {deployment.id}")
            deployments[deployment.id] = deployment
            key = _merge_key(deployment)
            if key is not None:
                exact.setdefault(key, []).append(deployment.id)
        chat_profiles = []
        for profile in group.profiles:
            if profile.task is ModelTask.EMBEDDING:
                if profile.endpoint is ModelEndpoint.EMBED and neutral_profile(profile.metadata):
                    address(profile.id, profile.deployment_id, profile)
                roles = profile.metadata.get("input_type", {}).get("supported", ())
                for suffix, role in (("query", EmbeddingRole.QUERY), ("document", EmbeddingRole.DOCUMENT)):
                    if role.value in roles:
                        address(f"{profile.id}.{suffix}", profile.deployment_id, profile, role)
            else:
                address(profile.id, profile.deployment_id, profile)
                if profile.task is ModelTask.CHAT:
                    chat_profiles.append(profile)
        if chat_profiles:
            defaults = [p for p in chat_profiles if p.default_for_endpoint
                        and p.endpoint is ModelEndpoint.CHAT_COMPLETIONS]
            selected = defaults[0] if len(defaults) == 1 else chat_profiles[0] if len(chat_profiles) == 1 else None
            if selected is None:
                raise ResolverConfigurationError(f"Ambiguous chat aliases for {group.canonical_model_id}")
            for name in (group.canonical_model_id, *group.aliases):
                address(name, selected.deployment_id, selected)

    entries = tuple(registry_entries)
    seen = set()
    for entry in sorted(entries, key=lambda value: value.id if isinstance(value, RegistryEntry) else ""):
        if not isinstance(entry, RegistryEntry):
            raise ResolverConfigurationError("registry contains invalid entries")
        if entry.id in seen or entry.id in deployments:
            raise ResolverConfigurationError(f"Duplicate registry/deployment ID: {entry.id}")
        seen.add(entry.id)
        local = _local_deployment(entry, profiles)
        registered_age = entry.registered_at - datetime(1970, 1, 1, tzinfo=timezone.utc)
        created = registered_age.days * 86400 + registered_age.seconds
        matches = exact.get(_merge_key(local), [])
        if len(matches) > 1:
            raise ResolverConfigurationError(f"Ambiguous exact artifact match: {entry.id}")
        if matches:
            target = deployments[matches[0]]
            deployments[target.id] = replace(target, registry_ids=(*target.registry_ids, entry.id))
            # Content merges runtime ownership, never the independently registered
            # public record or a catalog profile's API/presentation contract.
            address(entry.id, target.id, created=created, canonical_model_id=entry.id)
        else:
            deployments[local.id] = local
            address(entry.id, local.id, created=created, canonical_model_id=entry.id)

    revision = fingerprint("kiron.local-inference.resolver.v1", {
        "catalog_digest": catalog.catalog_digest, "deployments": deployments, "addresses": addresses,
    })
    models = {name: ResolvedModel(name, deployments[value[0]], value[1], value[2], revision,
                                  task=value[3], endpoint=value[4], profile_metadata=value[5],
                                  created=value[6], canonical_model_id=value[7])
              for name, value in sorted(addresses.items())}
    return ResolverSnapshot(revision, deployments, models)
