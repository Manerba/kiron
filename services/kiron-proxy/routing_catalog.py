"""Pure, immutable Catalog projection for managed proxy routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.model_catalog import (
    BackendType,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
    WireProfile,
)


_ENDPOINT_ORDER = (
    ModelEndpoint.EMBED,
    ModelEndpoint.EMBED_LATE,
    ModelEndpoint.EMBED_COLBERT,
    ModelEndpoint.RERANK,
    ModelEndpoint.SCORE,
)
_TASK_FOR_ENDPOINT = MappingProxyType(
    {
        ModelEndpoint.EMBED: ModelTask.EMBEDDING,
        ModelEndpoint.EMBED_LATE: ModelTask.EMBEDDING,
        ModelEndpoint.EMBED_COLBERT: ModelTask.EMBEDDING,
        ModelEndpoint.RERANK: ModelTask.RERANK,
        ModelEndpoint.SCORE: ModelTask.NLI,
    }
)
_LOADERS_FOR_ROUTE = MappingProxyType(
    {
        (ModelEndpoint.EMBED, BackendType.KIRON_EMBEDDINGS): frozenset(
            (
                LoaderType.SENTENCE_TRANSFORMERS,
                LoaderType.TRANSFORMERS_LAST_TOKEN,
            )
        ),
        (ModelEndpoint.EMBED, BackendType.OLLAMA): frozenset(
            (LoaderType.OLLAMA,)
        ),
        (ModelEndpoint.EMBED_LATE, BackendType.KIRON_EMBEDDINGS): frozenset(
            (LoaderType.SENTENCE_TRANSFORMERS,)
        ),
        (ModelEndpoint.EMBED_COLBERT, BackendType.KIRON_EMBEDDINGS): frozenset(
            (LoaderType.COLBERT_XMOD,)
        ),
        (ModelEndpoint.RERANK, BackendType.KIRON_DEBERTA): frozenset(
            (LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN)
        ),
        (ModelEndpoint.SCORE, BackendType.KIRON_DEBERTA): frozenset(
            (LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN)
        ),
    }
)
_BACKEND_PARAMETER_FIELDS = frozenset(
    ("implementation", "implementation_revision", "model_name")
)
_REQUEST_DEFAULT_ENDPOINTS = frozenset(
    (ModelEndpoint.RERANK, ModelEndpoint.SCORE)
)


class ProxyRoutingCatalogError(ValueError):
    """A controlled, path-aware error in the proxy Catalog projection."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"invalid kiron-proxy Catalog at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class ProxyRoute:
    """One selected group-default deployment for one managed endpoint."""

    canonical_model_id: str
    aliases: tuple[str, ...]
    profile_id: str
    deployment_id: str
    task: ModelTask
    endpoint: ModelEndpoint
    backend: BackendType
    backend_model_name: str
    loader_type: LoaderType

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.canonical_model_id, *self.aliases)


@dataclass(frozen=True, slots=True)
class ProxyRoutingView:
    """Exact, read-only routing index built from one injected Catalog."""

    catalog: ModelCatalog
    routes: tuple[ProxyRoute, ...]
    _routes_by_endpoint: Mapping[ModelEndpoint, tuple[ProxyRoute, ...]] = field(
        repr=False
    )
    _names_by_endpoint: Mapping[
        ModelEndpoint, Mapping[str, ProxyRoute]
    ] = field(repr=False)
    _request_defaults: Mapping[ModelEndpoint, ProxyRoute] = field(repr=False)

    @property
    def catalog_digest(self) -> str:
        return self.catalog.catalog_digest

    @property
    def endpoints(self) -> tuple[ModelEndpoint, ...]:
        return tuple(
            endpoint
            for endpoint in _ENDPOINT_ORDER
            if self._routes_by_endpoint.get(endpoint)
        )

    def endpoint(self, value: object) -> ModelEndpoint | None:
        if isinstance(value, ModelEndpoint):
            endpoint = value
        elif type(value) is str:
            try:
                endpoint = ModelEndpoint(value)
            except ValueError:
                return None
        else:
            return None
        return endpoint if endpoint in self._routes_by_endpoint else None

    def resolve(
        self,
        model_name: object,
        endpoint: ModelEndpoint | str,
        *,
        backend: BackendType | None = None,
    ) -> ProxyRoute | None:
        resolved_endpoint = self.endpoint(endpoint)
        if resolved_endpoint is None or type(model_name) is not str:
            return None
        route = self._names_by_endpoint[resolved_endpoint].get(model_name)
        if route is None or (backend is not None and route.backend is not backend):
            return None
        return route

    def routes_for_endpoint(
        self,
        endpoint: ModelEndpoint | str,
        *,
        backend: BackendType | None = None,
    ) -> tuple[ProxyRoute, ...]:
        resolved_endpoint = self.endpoint(endpoint)
        if resolved_endpoint is None:
            return ()
        routes = self._routes_by_endpoint[resolved_endpoint]
        if backend is None:
            return routes
        return tuple(route for route in routes if route.backend is backend)

    def available_models(
        self,
        endpoint: ModelEndpoint | str,
        *,
        backend: BackendType | None = None,
    ) -> tuple[str, ...]:
        return tuple(
            route.backend_model_name
            for route in self.routes_for_endpoint(endpoint, backend=backend)
        )

    def request_default(self, endpoint: ModelEndpoint | str) -> ProxyRoute:
        resolved_endpoint = self.endpoint(endpoint)
        route = (
            self._request_defaults.get(resolved_endpoint)
            if resolved_endpoint is not None
            else None
        )
        if route is None:
            rendered = (
                resolved_endpoint.value
                if resolved_endpoint is not None
                else repr(endpoint)
            )
            raise ProxyRoutingCatalogError(
                f"/proxy/request_defaults/{rendered}",
                "endpoint has no managed request default",
            )
        return route


def _fail(path: str, detail: str) -> None:
    raise ProxyRoutingCatalogError(path, detail)


def _closed_backend_parameters(
    wire: WireProfile,
    *,
    path: str,
) -> Mapping[str, Any]:
    value = wire.backend.parameters
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    if any(type(key) is not str for key in value):
        _fail(path, "object names must be strings")
    actual = set(value)
    missing = sorted(_BACKEND_PARAMETER_FIELDS - actual)
    unknown = sorted(actual - _BACKEND_PARAMETER_FIELDS)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")
    if unknown:
        _fail(path, f"unknown fields: {', '.join(unknown)}")
    return value


def _route_from_wire(wire: WireProfile) -> ProxyRoute:
    path = f"/catalog/{wire.canonical_model_id}/{wire.profile_id}"
    expected_task = _TASK_FOR_ENDPOINT.get(wire.endpoint)
    if expected_task is None or wire.task is not expected_task:
        _fail(
            f"{path}/route",
            "unsupported task/endpoint pair "
            f"{wire.task.value!r}:{wire.endpoint.value!r}",
        )
    allowed_loaders = _LOADERS_FOR_ROUTE.get((wire.endpoint, wire.backend.type))
    if allowed_loaders is None:
        _fail(
            f"{path}/backend/type",
            f"backend {wire.backend.type.value!r} cannot serve "
            f"endpoint {wire.endpoint.value!r}",
        )
    if wire.loader.type not in allowed_loaders:
        _fail(
            f"{path}/loader/type",
            f"loader {wire.loader.type.value!r} cannot serve "
            f"{wire.backend.type.value!r}:{wire.endpoint.value!r}",
        )
    parameters = _closed_backend_parameters(
        wire,
        path=f"{path}/backend/parameters",
    )
    backend_model_name = parameters["model_name"]
    if type(backend_model_name) is not str or not backend_model_name:
        _fail(
            f"{path}/backend/parameters/model_name",
            "must be a non-empty string",
        )
    if backend_model_name not in (wire.canonical_model_id, *wire.aliases):
        _fail(
            f"{path}/backend/parameters/model_name",
            "must be an exact canonical model ID or declared alias",
        )
    return ProxyRoute(
        canonical_model_id=wire.canonical_model_id,
        aliases=tuple(wire.aliases),
        profile_id=wire.profile_id,
        deployment_id=wire.deployment_id,
        task=wire.task,
        endpoint=wire.endpoint,
        backend=wire.backend.type,
        backend_model_name=backend_model_name,
        loader_type=wire.loader.type,
    )


def build_proxy_routing_view(catalog: ModelCatalog) -> ProxyRoutingView:
    """Build the proxy view solely from an already-loaded Catalog instance."""

    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")

    # Validate every declared managed wire, including non-default alternative
    # deployments. A malformed unused route must not hide behind a valid default.
    validated_wires = {
        wire.profile_id: _route_from_wire(wire)
        for wire in catalog.wire_profiles
    }

    routes: list[ProxyRoute] = []
    by_endpoint: dict[ModelEndpoint, list[ProxyRoute]] = {
        endpoint: [] for endpoint in _ENDPOINT_ORDER
    }
    names_by_endpoint: dict[ModelEndpoint, dict[str, ProxyRoute]] = {
        endpoint: {} for endpoint in _ENDPOINT_ORDER
    }
    selected_by_profile: dict[str, ProxyRoute] = {}

    for group in catalog.groups:
        group_endpoints = {
            wire.endpoint
            for wire in catalog.wire_profiles
            if wire.canonical_model_id == group.canonical_model_id
        }
        for endpoint in _ENDPOINT_ORDER:
            if endpoint not in group_endpoints:
                continue
            wire = catalog.default_for_endpoint(group.canonical_model_id, endpoint)
            if wire is None:
                _fail(
                    f"/catalog/{group.canonical_model_id}/{endpoint.value}",
                    "endpoint has no group-default profile",
                )
            route = validated_wires[wire.profile_id]
            routes.append(route)
            by_endpoint[endpoint].append(route)
            selected_by_profile[wire.profile_id] = route
            for name in route.input_names:
                previous = names_by_endpoint[endpoint].get(name)
                if previous is not None and previous != route:
                    _fail(
                        f"/proxy/names/{endpoint.value}",
                        f"literal name {name!r} resolves to multiple routes",
                    )
                names_by_endpoint[endpoint][name] = route

    for endpoint in _ENDPOINT_ORDER:
        endpoint_routes = by_endpoint[endpoint]
        if not endpoint_routes:
            _fail(
                f"/proxy/endpoints/{endpoint.value}",
                "Catalog has no managed route for required endpoint",
            )
        backend_names = [route.backend_model_name for route in endpoint_routes]
        if len(backend_names) != len(set(backend_names)):
            _fail(
                f"/proxy/available_models/{endpoint.value}",
                "backend model names must be unique",
            )

    request_defaults: dict[ModelEndpoint, ProxyRoute] = {}
    for endpoint in _REQUEST_DEFAULT_ENDPOINTS:
        wire = catalog.request_default_for_endpoint(endpoint)
        if wire is None:
            _fail(
                f"/proxy/request_defaults/{endpoint.value}",
                "Catalog request default is missing",
            )
        route = selected_by_profile.get(wire.profile_id)
        if route is None:
            _fail(
                f"/proxy/request_defaults/{endpoint.value}",
                "request default must select the group's endpoint-default route",
            )
        request_defaults[endpoint] = route

    frozen_by_endpoint = MappingProxyType(
        {
            endpoint: tuple(by_endpoint[endpoint])
            for endpoint in _ENDPOINT_ORDER
        }
    )
    frozen_names = MappingProxyType(
        {
            endpoint: MappingProxyType(dict(names_by_endpoint[endpoint]))
            for endpoint in _ENDPOINT_ORDER
        }
    )
    return ProxyRoutingView(
        catalog=catalog,
        routes=tuple(routes),
        _routes_by_endpoint=frozen_by_endpoint,
        _names_by_endpoint=frozen_names,
        _request_defaults=MappingProxyType(request_defaults),
    )


PROXY_ROUTING_VIEW = build_proxy_routing_view(MODEL_CATALOG)


__all__ = [
    "PROXY_ROUTING_VIEW",
    "ProxyRoute",
    "ProxyRoutingCatalogError",
    "ProxyRoutingView",
    "build_proxy_routing_view",
]
