"""Strict, immutable Catalog view for the kiron-deberta service."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from string import Formatter
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from kiron_common.model_catalog import (
    Artifact,
    ArtifactType,
    BackendType,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
    WireProfile,
)


_SERVICE_BACKEND = BackendType.KIRON_DEBERTA
_SERVICE_ENDPOINTS = frozenset(
    (ModelEndpoint.RERANK, ModelEndpoint.SCORE)
)
_TASK_FOR_ENDPOINT = MappingProxyType(
    {
        ModelEndpoint.RERANK: ModelTask.RERANK,
        ModelEndpoint.SCORE: ModelTask.NLI,
    }
)
_SERVICE_LOADER_TYPES = frozenset(
    (LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN)
)


class DebertaServiceCatalogError(ValueError):
    """A controlled, path-aware error in the service-specific Catalog view."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"invalid kiron-deberta Catalog at {path}: {detail}")


class LoaderCallable(Protocol):
    def __call__(self, model: "DebertaServiceModel") -> object: ...


@dataclass(frozen=True, slots=True)
class CrossEncoderLoaderParameters:
    max_length: int
    torch_dtype: str


@dataclass(frozen=True, slots=True)
class MankeiLastTokenLoaderParameters:
    backbone_torch_dtype: str
    batch_size: int
    head_filename: str
    head_torch_dtype: str
    max_length: int
    padding_side: str
    pair_template: str
    tokenizer_use_fast: bool


LoaderParameters: TypeAlias = (
    CrossEncoderLoaderParameters | MankeiLastTokenLoaderParameters
)


@dataclass(frozen=True, slots=True)
class DebertaServiceModel:
    canonical_model_id: str
    aliases: tuple[str, ...]
    model_name: str
    artifact: Artifact
    loader_type: LoaderType
    loader_parameters: LoaderParameters
    labels: tuple[str, ...] | None
    rerank_label: str | None
    size: int
    display_order: int
    endpoints: tuple[ModelEndpoint, ...]

    @property
    def max_length(self) -> int:
        return self.loader_parameters.max_length

    @property
    def precision(self) -> str:
        if isinstance(self.loader_parameters, CrossEncoderLoaderParameters):
            return "fp16"
        return "bf16"

    def serves(self, endpoint: ModelEndpoint) -> bool:
        return endpoint in self.endpoints


@dataclass(frozen=True, slots=True)
class DebertaServiceView:
    catalog_digest: str
    models: tuple[DebertaServiceModel, ...]
    _names: Mapping[str, DebertaServiceModel] = field(repr=False)
    _runtime_names: Mapping[str, DebertaServiceModel] = field(repr=False)
    _models_by_endpoint: Mapping[
        ModelEndpoint, tuple[DebertaServiceModel, ...]
    ] = field(repr=False)
    _request_defaults: Mapping[ModelEndpoint, DebertaServiceModel] = field(
        repr=False
    )
    _loader_registry: Mapping[LoaderType, LoaderCallable] = field(repr=False)

    def resolve(
        self,
        model_name: object,
        endpoint: ModelEndpoint | None = None,
    ) -> DebertaServiceModel | None:
        if type(model_name) is not str:
            return None
        model = self._names.get(model_name)
        if model is None:
            return None
        if endpoint is not None and not model.serves(endpoint):
            return None
        return model

    def runtime_model(self, model_name: object) -> DebertaServiceModel | None:
        if type(model_name) is not str:
            return None
        return self._runtime_names.get(model_name)

    def require_runtime_model(self, model_name: object) -> DebertaServiceModel:
        model = self.runtime_model(model_name)
        if model is None:
            raise DebertaServiceCatalogError(
                "/service/runtime_model",
                f"unknown exact service model name {model_name!r}",
            )
        return model

    def models_for_endpoint(
        self, endpoint: ModelEndpoint
    ) -> tuple[DebertaServiceModel, ...]:
        return self._models_by_endpoint.get(endpoint, ())

    def available_model_names(
        self, endpoint: ModelEndpoint | None = None
    ) -> tuple[str, ...]:
        models = self.models if endpoint is None else self.models_for_endpoint(endpoint)
        return tuple(model.model_name for model in models)

    def request_default(self, endpoint: ModelEndpoint) -> DebertaServiceModel:
        model = self._request_defaults.get(endpoint)
        if model is None:
            raise DebertaServiceCatalogError(
                f"/service/request_defaults/{endpoint.value}",
                "endpoint has no kiron-deberta request default",
            )
        return model

    def loader_for(self, model: DebertaServiceModel) -> LoaderCallable:
        loader = self._loader_registry.get(model.loader_type)
        if loader is None:
            raise DebertaServiceCatalogError(
                f"/service/models/{model.model_name}/loader/type",
                f"loader {model.loader_type.value!r} is not registered",
            )
        return loader


@dataclass(slots=True)
class _ModelAccumulator:
    canonical_model_id: str
    aliases: tuple[str, ...]
    model_name: str
    artifact: Artifact
    loader_type: LoaderType
    loader_parameters: LoaderParameters
    labels: tuple[str, ...] | None
    rerank_label: str | None
    size: int
    display_order: int
    endpoints: list[ModelEndpoint]


def _fail(path: str, detail: str) -> None:
    raise DebertaServiceCatalogError(path, detail)


def _closed_mapping(
    value: object,
    *,
    path: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    if any(type(key) is not str for key in value):
        _fail(path, "object names must be strings")
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")
    if unknown:
        _fail(path, f"unknown fields: {', '.join(unknown)}")
    return value


def _non_empty_string(value: object, *, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a non-empty string")
    return value


def _nullable_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, path=path)


def _positive_int(value: object, *, path: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(path, "must be a positive integer")
    return value


def _boolean(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _labels(value: object, *, path: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "must be an array or null")
    parsed = tuple(
        _non_empty_string(item, path=f"{path}/{index}")
        for index, item in enumerate(value)
    )
    if not parsed:
        _fail(path, "must not be empty")
    if len(parsed) != len(set(parsed)):
        _fail(path, "must not contain duplicates")
    return parsed


def _pair_template(value: object, *, path: str) -> str:
    template = _non_empty_string(value, path=path)
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        _fail(path, f"invalid formatting template: {exc}")
    fields: list[str] = []
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in {"query", "passage"}:
            _fail(path, f"unknown template field {field_name!r}")
        if format_spec or conversion:
            _fail(path, "format specifications and conversions are not allowed")
        fields.append(field_name)
    if fields.count("query") != 1 or fields.count("passage") != 1:
        _fail(path, "must contain exactly one {query} and one {passage} field")
    return template


def _parse_backend_model_name(wire: WireProfile, *, path: str) -> str:
    backend = _closed_mapping(
        wire.backend.parameters,
        path=path,
        required=frozenset(
            ("implementation", "implementation_revision", "model_name")
        ),
    )
    _non_empty_string(backend["implementation"], path=f"{path}/implementation")
    _nullable_string(
        backend["implementation_revision"],
        path=f"{path}/implementation_revision",
    )
    model_name = _non_empty_string(
        backend["model_name"], path=f"{path}/model_name"
    )
    if model_name not in (wire.canonical_model_id, *wire.aliases):
        _fail(
            f"{path}/model_name",
            "must be an exact canonical model ID or declared alias",
        )
    return model_name


def _parse_loader_parameters(
    wire: WireProfile, *, path: str
) -> LoaderParameters:
    raw = wire.loader.parameters
    if wire.loader.type is LoaderType.CROSS_ENCODER:
        values = _closed_mapping(
            raw,
            path=path,
            required=frozenset(("max_length", "torch_dtype")),
        )
        torch_dtype = _non_empty_string(
            values["torch_dtype"], path=f"{path}/torch_dtype"
        )
        if torch_dtype != "float16":
            _fail(f"{path}/torch_dtype", "must be 'float16'")
        return CrossEncoderLoaderParameters(
            max_length=_positive_int(
                values["max_length"], path=f"{path}/max_length"
            ),
            torch_dtype=torch_dtype,
        )
    if wire.loader.type is LoaderType.MANKEI_LAST_TOKEN:
        values = _closed_mapping(
            raw,
            path=path,
            required=frozenset(
                (
                    "backbone_torch_dtype",
                    "batch_size",
                    "head_filename",
                    "head_torch_dtype",
                    "max_length",
                    "padding_side",
                    "pair_template",
                    "tokenizer_use_fast",
                )
            ),
        )
        backbone_dtype = _non_empty_string(
            values["backbone_torch_dtype"],
            path=f"{path}/backbone_torch_dtype",
        )
        if backbone_dtype != "bfloat16":
            _fail(f"{path}/backbone_torch_dtype", "must be 'bfloat16'")
        head_dtype = _non_empty_string(
            values["head_torch_dtype"], path=f"{path}/head_torch_dtype"
        )
        if head_dtype != "float32":
            _fail(f"{path}/head_torch_dtype", "must be 'float32'")
        padding_side = _non_empty_string(
            values["padding_side"], path=f"{path}/padding_side"
        )
        if padding_side != "right":
            _fail(f"{path}/padding_side", "must be 'right'")
        return MankeiLastTokenLoaderParameters(
            backbone_torch_dtype=backbone_dtype,
            batch_size=_positive_int(
                values["batch_size"], path=f"{path}/batch_size"
            ),
            head_filename=_non_empty_string(
                values["head_filename"], path=f"{path}/head_filename"
            ),
            head_torch_dtype=head_dtype,
            max_length=_positive_int(
                values["max_length"], path=f"{path}/max_length"
            ),
            padding_side=padding_side,
            pair_template=_pair_template(
                values["pair_template"], path=f"{path}/pair_template"
            ),
            tokenizer_use_fast=_boolean(
                values["tokenizer_use_fast"],
                path=f"{path}/tokenizer_use_fast",
            ),
        )
    _fail(
        path,
        f"loader {wire.loader.type.value!r} is not allowed for backend "
        f"{_SERVICE_BACKEND.value!r}",
    )


def _parse_service_metadata(
    wire: WireProfile, *, path: str
) -> tuple[tuple[str, ...] | None, str | None, int, int]:
    metadata = _closed_mapping(
        wire.deployment_metadata,
        path=path,
        required=frozenset(("service",)),
    )
    service = _closed_mapping(
        metadata["service"],
        path=f"{path}/service",
        required=frozenset(
            ("display_order", "labels", "rerank_label", "size")
        ),
    )
    labels = _labels(service["labels"], path=f"{path}/service/labels")
    rerank_label = _nullable_string(
        service["rerank_label"], path=f"{path}/service/rerank_label"
    )
    if labels is None and rerank_label is not None:
        _fail(
            f"{path}/service/rerank_label",
            "single-score models must use null",
        )
    if labels is not None and rerank_label not in labels:
        _fail(
            f"{path}/service/rerank_label",
            "classification models require a label present in labels",
        )
    return (
        labels,
        rerank_label,
        _positive_int(service["size"], path=f"{path}/service/size"),
        _positive_int(
            service["display_order"], path=f"{path}/service/display_order"
        ),
    )


def _validate_artifacts(
    wire: WireProfile,
    parameters: LoaderParameters,
    *,
    path: str,
) -> None:
    if wire.artifact.type is not ArtifactType.HUGGINGFACE:
        _fail(f"{path}/artifact/type", "kiron_deberta requires HuggingFace")
    if len(wire.artifact.weights) != 1:
        _fail(f"{path}/artifact/weights", "requires exactly one weight artifact")
    weight = wire.artifact.weights[0]
    if weight.path != "model.safetensors":
        _fail(
            f"{path}/artifact/weights/0/path",
            "must be 'model.safetensors'",
        )
    if isinstance(parameters, CrossEncoderLoaderParameters):
        if wire.artifact.auxiliary:
            _fail(
                f"{path}/artifact/auxiliary",
                "cross_encoder does not accept auxiliary artifacts",
            )
        return
    if len(wire.artifact.auxiliary) != 1:
        _fail(
            f"{path}/artifact/auxiliary",
            "mankei_last_token requires exactly one head artifact",
        )
    if wire.artifact.auxiliary[0].path != parameters.head_filename:
        _fail(
            f"{path}/artifact/auxiliary/0/path",
            "must match loader head_filename",
        )


def _validated_registry(
    registry: Mapping[LoaderType, LoaderCallable],
) -> Mapping[LoaderType, LoaderCallable]:
    if not isinstance(registry, Mapping):
        _fail("/service/loader_registry", "must be a mapping")
    result: dict[LoaderType, LoaderCallable] = {}
    for loader_type, loader in registry.items():
        if not isinstance(loader_type, LoaderType):
            _fail(
                "/service/loader_registry",
                f"registry key {loader_type!r} must be a LoaderType",
            )
        if not callable(loader):
            _fail(
                f"/service/loader_registry/{loader_type.value}",
                "registered loader must be callable",
            )
        result[loader_type] = loader
    return MappingProxyType(result)


def build_deberta_service_view(
    catalog: ModelCatalog,
    loader_registry: Mapping[LoaderType, LoaderCallable],
) -> DebertaServiceView:
    """Build the service view solely from an injected, loaded Catalog."""

    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")
    loaders = _validated_registry(loader_registry)
    service_wires = catalog.for_backend(_SERVICE_BACKEND)

    accumulators: dict[str, _ModelAccumulator] = {}
    for wire in service_wires:
        path = f"/catalog/{wire.canonical_model_id}/{wire.deployment_id}"
        expected_task = _TASK_FOR_ENDPOINT.get(wire.endpoint)
        if expected_task is None or wire.task is not expected_task:
            _fail(
                f"{path}/route",
                f"unsupported task/endpoint pair "
                f"{wire.task.value!r}:{wire.endpoint.value!r}",
            )
        if wire.loader.type not in _SERVICE_LOADER_TYPES:
            _fail(
                f"{path}/loader/type",
                f"loader {wire.loader.type.value!r} is not allowed for backend "
                f"{_SERVICE_BACKEND.value!r}",
            )
        if wire.loader.type not in loaders:
            _fail(
                f"{path}/loader/type",
                f"loader {wire.loader.type.value!r} is not registered",
            )
        if not wire.default_for_endpoint:
            _fail(
                f"{path}/profile/default_for_endpoint",
                "service endpoint profiles must be the group endpoint default",
            )

        model_name = _parse_backend_model_name(
            wire, path=f"{path}/backend/parameters"
        )
        loader_parameters = _parse_loader_parameters(
            wire, path=f"{path}/loader/parameters"
        )
        _validate_artifacts(
            wire, loader_parameters, path=path
        )
        labels, rerank_label, size, display_order = _parse_service_metadata(
            wire, path=f"{path}/metadata"
        )

        previous = accumulators.get(model_name)
        if previous is None:
            accumulators[model_name] = _ModelAccumulator(
                canonical_model_id=wire.canonical_model_id,
                aliases=wire.aliases,
                model_name=model_name,
                artifact=wire.artifact,
                loader_type=wire.loader.type,
                loader_parameters=loader_parameters,
                labels=labels,
                rerank_label=rerank_label,
                size=size,
                display_order=display_order,
                endpoints=[wire.endpoint],
            )
            continue

        comparisons = (
            ("canonical_model_id", previous.canonical_model_id, wire.canonical_model_id),
            ("aliases", previous.aliases, wire.aliases),
            ("artifact", previous.artifact, wire.artifact),
            ("loader.type", previous.loader_type, wire.loader.type),
            ("loader.parameters", previous.loader_parameters, loader_parameters),
            ("labels", previous.labels, labels),
            ("rerank_label", previous.rerank_label, rerank_label),
            ("size", previous.size, size),
            ("display_order", previous.display_order, display_order),
        )
        for field_name, expected, actual in comparisons:
            if actual != expected:
                _fail(
                    path,
                    "conflicting deployment data for service model identity "
                    f"{model_name!r}: {field_name}",
                )
        if wire.endpoint in previous.endpoints:
            _fail(
                path,
                "conflicting deployment data for service model identity "
                f"{model_name!r}: duplicate endpoint {wire.endpoint.value!r}",
            )
        previous.endpoints.append(wire.endpoint)

    models = tuple(
        sorted(
            (
                DebertaServiceModel(
                    canonical_model_id=item.canonical_model_id,
                    aliases=tuple(sorted(item.aliases)),
                    model_name=item.model_name,
                    artifact=item.artifact,
                    loader_type=item.loader_type,
                    loader_parameters=item.loader_parameters,
                    labels=item.labels,
                    rerank_label=item.rerank_label,
                    size=item.size,
                    display_order=item.display_order,
                    endpoints=tuple(sorted(item.endpoints, key=lambda endpoint: endpoint.value)),
                )
                for item in accumulators.values()
            ),
            key=lambda model: model.display_order,
        )
    )
    display_orders = [model.display_order for model in models]
    if len(display_orders) != len(set(display_orders)):
        _fail("/service/models", "display_order values must be unique")
    for model in models:
        if set(model.endpoints) != _SERVICE_ENDPOINTS:
            _fail(
                f"/service/models/{model.model_name}/profiles",
                "must configure both /api/rerank and /api/score",
            )

    names: dict[str, DebertaServiceModel] = {}
    runtime_names: dict[str, DebertaServiceModel] = {}
    for model in models:
        if model.model_name in runtime_names:
            _fail(
                "/service/runtime_names",
                f"duplicate runtime model name {model.model_name!r}",
            )
        runtime_names[model.model_name] = model
        for name in (model.canonical_model_id, *model.aliases):
            previous = names.get(name)
            if previous is not None and previous is not model:
                _fail(
                    "/service/names",
                    f"literal name {name!r} resolves to multiple service models",
                )
            names[name] = model

    by_endpoint = {
        endpoint: tuple(model for model in models if model.serves(endpoint))
        for endpoint in _SERVICE_ENDPOINTS
    }
    request_defaults: dict[ModelEndpoint, DebertaServiceModel] = {}
    for endpoint in _SERVICE_ENDPOINTS:
        wire = catalog.request_default_for_endpoint(endpoint)
        if wire is None:
            _fail(
                f"/service/request_defaults/{endpoint.value}",
                "Catalog request default is missing",
            )
        if wire.backend.type is not _SERVICE_BACKEND:
            _fail(
                f"/service/request_defaults/{endpoint.value}",
                "Catalog request default belongs to another backend",
            )
        model_name = _parse_backend_model_name(
            wire,
            path=(
                f"/service/request_defaults/{endpoint.value}/"
                "backend/parameters"
            ),
        )
        model = runtime_names.get(model_name)
        if model is None or not model.serves(endpoint):
            _fail(
                f"/service/request_defaults/{endpoint.value}",
                f"does not resolve to a configured runtime model: {model_name!r}",
            )
        request_defaults[endpoint] = model

    return DebertaServiceView(
        catalog_digest=catalog.catalog_digest,
        models=models,
        _names=MappingProxyType(names),
        _runtime_names=MappingProxyType(runtime_names),
        _models_by_endpoint=MappingProxyType(by_endpoint),
        _request_defaults=MappingProxyType(request_defaults),
        _loader_registry=loaders,
    )


__all__ = [
    "CrossEncoderLoaderParameters",
    "DebertaServiceCatalogError",
    "DebertaServiceModel",
    "DebertaServiceView",
    "LoaderCallable",
    "LoaderParameters",
    "MankeiLastTokenLoaderParameters",
    "build_deberta_service_view",
]
