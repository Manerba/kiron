"""Immutable catalog facade and deterministic derived views."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TypeVar

from kiron_common.embedding_contract import (
    JCSCanonicalizationError,
    canonical_json,
)

from .errors import CatalogLookupError, CatalogValidationError
from .models import (
    BackendType,
    ModelEndpoint,
    ModelGroup,
    ModelTask,
    REQUEST_DEFAULT_ENDPOINTS,
    WireProfile,
)
from .parser import SCHEMA_VERSION, parse_manifest


EnumT = TypeVar("EnumT")


def _exact_enum(value: object, enum_type: type[EnumT]) -> EnumT | None:
    if isinstance(value, enum_type):
        return value
    if type(value) is not str:
        return None
    try:
        return enum_type(value)
    except ValueError:
        return None


class ModelCatalog:
    """A fully validated, deeply immutable catalog and its exact indexes."""

    __slots__ = (
        "_catalog_digest",
        "_defaults",
        "_groups",
        "_names",
        "_request_defaults",
        "_wire_by_backend",
        "_wire_by_endpoint",
        "_wire_by_profile_id",
        "_wire_by_task",
        "_wire_profiles",
        "_sealed",
    )

    def __init__(self, groups: Iterable[ModelGroup] = ()) -> None:
        validated = tuple(
            parse_manifest(
                group.to_manifest_dict(schema_version=SCHEMA_VERSION),
                source=f"catalog:{group.canonical_model_id}",
            )
            for group in groups
        )
        ordered = tuple(
            sorted(validated, key=lambda group: group.canonical_model_id)
        )

        names: dict[str, ModelGroup] = {}
        deployment_ids: dict[str, str] = {}
        profile_ids: dict[str, str] = {}
        wires: list[WireProfile] = []
        defaults: dict[tuple[str, ModelEndpoint], WireProfile] = {}
        request_defaults: dict[ModelEndpoint, WireProfile] = {}

        for group_index, group in enumerate(ordered):
            path = f"/models/{group_index}"
            for name in (group.canonical_model_id, *group.aliases):
                previous = names.get(name)
                if previous is not None:
                    raise CatalogValidationError(
                        f"{path}/aliases",
                        f"literal model name {name!r} is already owned by "
                        f"{previous.canonical_model_id!r}",
                    )
                names[name] = group

            deployment_by_id = {item.id: item for item in group.deployments}
            request_default_profile_ids = {
                item.profile_id for item in group.request_defaults
            }
            group_wires: dict[str, WireProfile] = {}
            for deployment in group.deployments:
                previous_model = deployment_ids.get(deployment.id)
                if previous_model is not None:
                    raise CatalogValidationError(
                        f"{path}/deployments",
                        f"deployment ID {deployment.id!r} is already used by "
                        f"{previous_model!r}",
                    )
                deployment_ids[deployment.id] = group.canonical_model_id

            for profile in group.profiles:
                previous_model = profile_ids.get(profile.id)
                if previous_model is not None:
                    raise CatalogValidationError(
                        f"{path}/profiles",
                        f"profile ID {profile.id!r} is already used by "
                        f"{previous_model!r}",
                    )
                profile_ids[profile.id] = group.canonical_model_id
                deployment = deployment_by_id[profile.deployment_id]
                wire = WireProfile(
                    canonical_model_id=group.canonical_model_id,
                    aliases=group.aliases,
                    profile_id=profile.id,
                    deployment_id=deployment.id,
                    task=profile.task,
                    endpoint=profile.endpoint,
                    default_for_endpoint=profile.default_for_endpoint,
                    is_request_default=(
                        profile.id in request_default_profile_ids
                    ),
                    backend=deployment.backend,
                    artifact=deployment.artifact,
                    loader=deployment.loader,
                    model_metadata=group.metadata,
                    deployment_metadata=deployment.metadata,
                    profile_metadata=profile.metadata,
                )
                wires.append(wire)
                group_wires[profile.id] = wire
                if profile.default_for_endpoint:
                    defaults[(group.canonical_model_id, profile.endpoint)] = wire

            for request_default in group.request_defaults:
                previous = request_defaults.get(request_default.endpoint)
                if previous is not None:
                    raise CatalogValidationError(
                        f"{path}/request_defaults",
                        f"endpoint {request_default.endpoint.value!r} has "
                        "multiple request defaults: "
                        f"{previous.profile_id!r} and "
                        f"{request_default.profile_id!r}",
                    )
                request_defaults[request_default.endpoint] = group_wires[
                    request_default.profile_id
                ]

        ordered_wires = tuple(
            sorted(
                wires,
                key=lambda wire: (
                    wire.canonical_model_id,
                    wire.endpoint.value,
                    wire.profile_id,
                ),
            )
        )
        by_backend = {
            backend: tuple(
                wire for wire in ordered_wires if wire.backend.type is backend
            )
            for backend in BackendType
        }
        by_task = {
            task: tuple(wire for wire in ordered_wires if wire.task is task)
            for task in ModelTask
        }
        by_endpoint = {
            endpoint: tuple(
                wire for wire in ordered_wires if wire.endpoint is endpoint
            )
            for endpoint in ModelEndpoint
        }
        for endpoint in REQUEST_DEFAULT_ENDPOINTS:
            if by_endpoint[endpoint] and endpoint not in request_defaults:
                raise CatalogValidationError(
                    "/request_defaults",
                    f"endpoint {endpoint.value!r} is present but has no "
                    "request default",
                )

        digest_document = {
            "catalog_schema_version": SCHEMA_VERSION,
            "manifests": [
                group.to_manifest_dict(schema_version=SCHEMA_VERSION)
                for group in ordered
            ],
        }
        try:
            canonical = canonical_json(digest_document)
        except JCSCanonicalizationError as exc:
            raise CatalogValidationError(
                "/",
                f"cannot canonicalize catalog digest preimage: {exc}",
            ) from exc
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        self._groups = ordered
        self._names = MappingProxyType(names)
        self._wire_profiles = ordered_wires
        self._wire_by_profile_id = MappingProxyType(
            {wire.profile_id: wire for wire in ordered_wires}
        )
        self._wire_by_backend = MappingProxyType(by_backend)
        self._wire_by_task = MappingProxyType(by_task)
        self._wire_by_endpoint = MappingProxyType(by_endpoint)
        self._defaults = MappingProxyType(defaults)
        self._request_defaults = MappingProxyType(request_defaults)
        self._catalog_digest = f"sha256:{digest}"
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ModelCatalog is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_manifests(
        cls,
        manifests: Iterable[Mapping[str, object]],
    ) -> "ModelCatalog":
        return cls(
            parse_manifest(manifest, source=f"manifest[{index}]")
            for index, manifest in enumerate(manifests)
        )

    @property
    def groups(self) -> tuple[ModelGroup, ...]:
        return self._groups

    @property
    def wire_profiles(self) -> tuple[WireProfile, ...]:
        return self._wire_profiles

    @property
    def catalog_digest(self) -> str:
        return self._catalog_digest

    def resolve(self, model_name: object) -> ModelGroup | None:
        """Resolve only an exact canonical ID or explicitly declared alias."""

        if type(model_name) is not str:
            return None
        return self._names.get(model_name)

    def require(self, model_name: object) -> ModelGroup:
        group = self.resolve(model_name)
        if group is None:
            raise CatalogLookupError(model_name)
        return group

    def wire_profile(self, profile_id: object) -> WireProfile | None:
        if type(profile_id) is not str:
            return None
        return self._wire_by_profile_id.get(profile_id)

    def for_backend(self, backend: BackendType | str) -> tuple[WireProfile, ...]:
        resolved = _exact_enum(backend, BackendType)
        if resolved is None:
            return ()
        return self._wire_by_backend[resolved]

    def for_task(self, task: ModelTask | str) -> tuple[WireProfile, ...]:
        resolved = _exact_enum(task, ModelTask)
        if resolved is None:
            return ()
        return self._wire_by_task[resolved]

    def for_endpoint(
        self,
        endpoint: ModelEndpoint | str,
    ) -> tuple[WireProfile, ...]:
        resolved = _exact_enum(endpoint, ModelEndpoint)
        if resolved is None:
            return ()
        return self._wire_by_endpoint[resolved]

    def default_for_endpoint(
        self,
        model_name: object,
        endpoint: ModelEndpoint | str,
    ) -> WireProfile | None:
        group = self.resolve(model_name)
        resolved_endpoint = _exact_enum(endpoint, ModelEndpoint)
        if group is None or resolved_endpoint is None:
            return None
        return self._defaults.get(
            (group.canonical_model_id, resolved_endpoint)
        )

    def request_default_for_endpoint(
        self,
        endpoint: ModelEndpoint | str,
    ) -> WireProfile | None:
        """Return the single declared request default for an exact endpoint."""

        resolved_endpoint = _exact_enum(endpoint, ModelEndpoint)
        if resolved_endpoint is None:
            return None
        return self._request_defaults.get(resolved_endpoint)
