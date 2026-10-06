"""Strict, immutable Catalog view for the kiron-embeddings service."""

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


_SERVICE_BACKEND = BackendType.KIRON_EMBEDDINGS
_SERVICE_TASK = ModelTask.EMBEDDING
_SERVICE_ENDPOINTS = frozenset(
    (
        ModelEndpoint.EMBED,
        ModelEndpoint.EMBED_LATE,
        ModelEndpoint.EMBED_COLBERT,
    )
)
_SERVICE_LOADER_TYPES = frozenset(
    (
        LoaderType.SENTENCE_TRANSFORMERS,
        LoaderType.TRANSFORMERS_LAST_TOKEN,
        LoaderType.COLBERT_XMOD,
    )
)


class EmbeddingServiceCatalogError(ValueError):
    """A controlled, path-aware error in the service-specific Catalog view."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"invalid kiron-embeddings Catalog at {path}: {detail}")


class LoaderCallable(Protocol):
    def __call__(self, model: "EmbeddingServiceModel") -> object: ...


@dataclass(frozen=True, slots=True)
class SentenceTransformersLoaderParameters:
    additional_role_template: str | None


@dataclass(frozen=True, slots=True)
class TransformersLastTokenLoaderParameters:
    tokenizer_use_fast: bool
    torch_dtype: str


@dataclass(frozen=True, slots=True)
class ColbertXmodLoaderParameters:
    projection_weight_tensor: str
    tokenizer_use_fast: bool


LoaderParameters: TypeAlias = (
    SentenceTransformersLoaderParameters
    | TransformersLastTokenLoaderParameters
    | ColbertXmodLoaderParameters
)


@dataclass(frozen=True, slots=True)
class DiscoveryMetadata:
    family: str
    format: str
    parameter_size: str
    precision: str | None
    size: int


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    profile_id: str
    deployment_id: str
    endpoint: ModelEndpoint
    kind: str
    dimensions: int
    document_max_tokens: int
    query_max_tokens: int
    document_template: str
    query_template: str
    additional_task_roles: tuple[str, ...]
    default_language: str | None = None
    mask_punctuation: bool = False
    padding_side: str | None = None

    def template_for(self, input_type: str) -> str | None:
        if input_type == "search_document":
            return self.document_template
        if input_type == "search_query":
            return self.query_template
        return None


@dataclass(frozen=True, slots=True)
class EmbeddingServiceModel:
    canonical_model_id: str
    aliases: tuple[str, ...]
    model_name: str
    artifact: Artifact
    loader_type: LoaderType
    loader_parameters: LoaderParameters
    discovery: DiscoveryMetadata
    profiles: tuple[EndpointProfile, ...]

    def profile_for(self, endpoint: ModelEndpoint) -> EndpointProfile | None:
        for profile in self.profiles:
            if profile.endpoint is endpoint:
                return profile
        return None

    def require_profile(self, endpoint: ModelEndpoint) -> EndpointProfile:
        profile = self.profile_for(endpoint)
        if profile is None:
            raise EmbeddingServiceCatalogError(
                f"/service/models/{self.model_name}/profiles",
                f"endpoint {endpoint.value!r} is not configured",
            )
        return profile

    @property
    def endpoints(self) -> tuple[ModelEndpoint, ...]:
        return tuple(profile.endpoint for profile in self.profiles)

    @property
    def embedding_length(self) -> int | None:
        profile = self.profile_for(ModelEndpoint.EMBED)
        return profile.dimensions if profile is not None else None

    @property
    def context_length(self) -> int | None:
        profile = self.profile_for(ModelEndpoint.EMBED)
        if profile is None:
            return None
        return max(profile.document_max_tokens, profile.query_max_tokens)

    def format_texts(
        self,
        texts: list[str],
        *,
        endpoint: ModelEndpoint,
        input_type: str | None,
    ) -> list[str]:
        profile = self.require_profile(endpoint)
        role = input_type if input_type is not None else "search_document"
        template = profile.template_for(role)
        if template is None and role in profile.additional_task_roles:
            parameters = self.loader_parameters
            if isinstance(parameters, SentenceTransformersLoaderParameters):
                template = parameters.additional_role_template
        if template is None:
            raise ValueError(
                f"input_type {role!r} has no formatting template for "
                f"endpoint {endpoint.value!r}"
            )
        return [template.format(text=text, input_type=role) for text in texts]


@dataclass(frozen=True, slots=True)
class EmbeddingServiceView:
    catalog_digest: str
    models: tuple[EmbeddingServiceModel, ...]
    _names: Mapping[str, EmbeddingServiceModel] = field(repr=False)
    _runtime_names: Mapping[str, EmbeddingServiceModel] = field(repr=False)
    _models_by_endpoint: Mapping[
        ModelEndpoint, tuple[EmbeddingServiceModel, ...]
    ] = field(repr=False)
    _loader_registry: Mapping[LoaderType, LoaderCallable] = field(repr=False)

    def resolve(
        self,
        model_name: object,
        endpoint: ModelEndpoint | None = None,
    ) -> EmbeddingServiceModel | None:
        if type(model_name) is not str:
            return None
        model = self._names.get(model_name)
        if model is None:
            return None
        if endpoint is not None and model.profile_for(endpoint) is None:
            return None
        return model

    def require_runtime_model(self, model_name: object) -> EmbeddingServiceModel:
        if type(model_name) is not str:
            model = None
        else:
            model = self._runtime_names.get(model_name)
        if model is None:
            raise EmbeddingServiceCatalogError(
                "/service/runtime_model",
                f"unknown exact service model name {model_name!r}",
            )
        return model

    def models_for_endpoint(
        self, endpoint: ModelEndpoint
    ) -> tuple[EmbeddingServiceModel, ...]:
        return self._models_by_endpoint.get(endpoint, ())

    def available_model_names(
        self, endpoint: ModelEndpoint | None = None
    ) -> tuple[str, ...]:
        models = self.models if endpoint is None else self.models_for_endpoint(endpoint)
        return tuple(model.model_name for model in models)

    def loader_for(self, model: EmbeddingServiceModel) -> LoaderCallable:
        loader = self._loader_registry.get(model.loader_type)
        if loader is None:
            raise EmbeddingServiceCatalogError(
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
    discovery: DiscoveryMetadata
    profiles: list[EndpointProfile]


def _fail(path: str, detail: str) -> None:
    raise EmbeddingServiceCatalogError(path, detail)


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


def _string_tuple(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "must be an array")
    result = tuple(
        _non_empty_string(item, path=f"{path}/{index}")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        _fail(path, "must not contain duplicates")
    return result


def _validate_template(value: object, *, path: str) -> str:
    template = _non_empty_string(value, path=path)
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        _fail(path, f"invalid formatting template: {exc}")
    fields: list[str] = []
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in {"text", "input_type"}:
            _fail(path, f"unknown template field {field_name!r}")
        if format_spec or conversion:
            _fail(path, "format specifications and conversions are not allowed")
        fields.append(field_name)
    if fields.count("text") != 1:
        _fail(path, "must contain exactly one {text} field")
    return template


def _parse_loader_parameters(wire: WireProfile, *, path: str) -> LoaderParameters:
    raw = wire.loader.parameters
    if wire.loader.type is LoaderType.SENTENCE_TRANSFORMERS:
        values = _closed_mapping(
            raw,
            path=path,
            required=frozenset(("additional_role_template",)),
        )
        template = _nullable_string(
            values["additional_role_template"],
            path=f"{path}/additional_role_template",
        )
        if template is not None:
            template = _validate_template(
                template, path=f"{path}/additional_role_template"
            )
        return SentenceTransformersLoaderParameters(template)
    if wire.loader.type is LoaderType.TRANSFORMERS_LAST_TOKEN:
        values = _closed_mapping(
            raw,
            path=path,
            required=frozenset(
                ("tokenizer_use_fast", "torch_dtype")
            ),
        )
        torch_dtype = _non_empty_string(
            values["torch_dtype"], path=f"{path}/torch_dtype"
        )
        if torch_dtype != "float32":
            _fail(f"{path}/torch_dtype", "must be 'float32'")
        return TransformersLastTokenLoaderParameters(
            tokenizer_use_fast=_boolean(
                values["tokenizer_use_fast"],
                path=f"{path}/tokenizer_use_fast",
            ),
            torch_dtype=torch_dtype,
        )
    if wire.loader.type is LoaderType.COLBERT_XMOD:
        values = _closed_mapping(
            raw,
            path=path,
            required=frozenset(
                ("projection_weight_tensor", "tokenizer_use_fast")
            ),
        )
        return ColbertXmodLoaderParameters(
            projection_weight_tensor=_non_empty_string(
                values["projection_weight_tensor"],
                path=f"{path}/projection_weight_tensor",
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


def _parse_discovery(wire: WireProfile, *, path: str) -> DiscoveryMetadata:
    metadata = _closed_mapping(
        wire.deployment_metadata,
        path=path,
        required=frozenset(("discovery",)),
    )
    discovery = _closed_mapping(
        metadata["discovery"],
        path=f"{path}/discovery",
        required=frozenset(
            ("family", "format", "parameter_size", "precision", "size")
        ),
    )
    return DiscoveryMetadata(
        family=_non_empty_string(
            discovery["family"], path=f"{path}/discovery/family"
        ),
        format=_non_empty_string(
            discovery["format"], path=f"{path}/discovery/format"
        ),
        parameter_size=_non_empty_string(
            discovery["parameter_size"],
            path=f"{path}/discovery/parameter_size",
        ),
        precision=_nullable_string(
            discovery["precision"], path=f"{path}/discovery/precision"
        ),
        size=_positive_int(discovery["size"], path=f"{path}/discovery/size"),
    )


def _parse_formatting_entry(value: object, *, path: str) -> str:
    entry = _closed_mapping(
        value,
        path=path,
        required=frozenset(("template", "parameters")),
    )
    parameters = _closed_mapping(
        entry["parameters"],
        path=f"{path}/parameters",
        required=frozenset(),
    )
    if parameters:
        _fail(f"{path}/parameters", "must be empty")
    return _validate_template(entry["template"], path=f"{path}/template")


def _parse_endpoint_profile(wire: WireProfile, *, path: str) -> EndpointProfile:
    values = dict(wire.profile_metadata)
    if "created" in values:
        created = values.pop("created")
        if type(created) is not int or created < 0:
            _fail(f"{path}/metadata/created", "must be a nonnegative API creation timestamp")
    metadata = _closed_mapping(
        values,
        path=f"{path}/metadata",
        required=frozenset(
            (
                "kind",
                "dimensions",
                "max_input_tokens",
                "output_normalized",
                "similarity",
                "input_type",
                "formatting_owner",
                "pipeline",
                "verification",
            )
        ),
    )
    if metadata["formatting_owner"] != "server":
        _fail(f"{path}/metadata/formatting_owner", "must be 'server'")

    limits = _closed_mapping(
        metadata["max_input_tokens"],
        path=f"{path}/metadata/max_input_tokens",
        required=frozenset(
            ("unit", "counting", "truncation", "overflow", "by_role")
        ),
    )
    by_role = _closed_mapping(
        limits["by_role"],
        path=f"{path}/metadata/max_input_tokens/by_role",
        required=frozenset(("search_document", "search_query")),
    )
    input_type = _closed_mapping(
        metadata["input_type"],
        path=f"{path}/metadata/input_type",
        required=frozenset(
            (
                "supported",
                "additional_task_roles",
                "aliases",
                "role_sensitive",
                "required",
                "default",
                "missing_role_behavior",
            )
        ),
    )
    additional_roles = _string_tuple(
        input_type["additional_task_roles"],
        path=f"{path}/metadata/input_type/additional_task_roles",
    )

    pipeline = _closed_mapping(
        metadata["pipeline"],
        path=f"{path}/metadata/pipeline",
        required=frozenset(
            ("tokenizer", "formatting", "pooling", "normalization", "parameters")
        ),
    )
    tokenizer = _closed_mapping(
        pipeline["tokenizer"],
        path=f"{path}/metadata/pipeline/tokenizer",
        required=frozenset(("repository", "revision")),
    )
    if tokenizer["repository"] != wire.artifact.repository:
        _fail(
            f"{path}/metadata/pipeline/tokenizer/repository",
            "must match the normalized artifact repository",
        )
    if tokenizer["revision"] != wire.artifact.revision:
        _fail(
            f"{path}/metadata/pipeline/tokenizer/revision",
            "must match the normalized artifact revision",
        )
    formatting = _closed_mapping(
        pipeline["formatting"],
        path=f"{path}/metadata/pipeline/formatting",
        required=frozenset(("search_document", "search_query")),
    )
    document_template = _parse_formatting_entry(
        formatting["search_document"],
        path=f"{path}/metadata/pipeline/formatting/search_document",
    )
    query_template = _parse_formatting_entry(
        formatting["search_query"],
        path=f"{path}/metadata/pipeline/formatting/search_query",
    )

    default_language: str | None = None
    mask_punctuation = False
    padding_side: str | None = None
    pooling = _closed_mapping(
        pipeline["pooling"],
        path=f"{path}/metadata/pipeline/pooling",
        required=frozenset(("method", "parameters")),
    )
    if wire.loader.type is LoaderType.COLBERT_XMOD:
        if pooling["method"] != "token_projection":
            _fail(
                f"{path}/metadata/pipeline/pooling/method",
                "colbert_xmod requires 'token_projection'",
            )
        pooling_parameters = _closed_mapping(
            pooling["parameters"],
            path=f"{path}/metadata/pipeline/pooling/parameters",
            required=frozenset(
                (
                    "default_language_adapter",
                    "mask_punctuation",
                    "remove_special_tokens",
                    "row_order",
                )
            ),
        )
        default_language = _non_empty_string(
            pooling_parameters["default_language_adapter"],
            path=(
                f"{path}/metadata/pipeline/pooling/parameters/"
                "default_language_adapter"
            ),
        )
        mask_punctuation = _boolean(
            pooling_parameters["mask_punctuation"],
            path=f"{path}/metadata/pipeline/pooling/parameters/mask_punctuation",
        )
        projection = _closed_mapping(
            pipeline["parameters"],
            path=f"{path}/metadata/pipeline/parameters",
            required=frozenset(("projection_dimensions",)),
        )
        if projection["projection_dimensions"] != metadata["dimensions"]:
            _fail(
                f"{path}/metadata/pipeline/parameters/projection_dimensions",
                "must match profile dimensions",
            )
    elif wire.loader.type is LoaderType.TRANSFORMERS_LAST_TOKEN:
        precision = _closed_mapping(
            pipeline["parameters"],
            path=f"{path}/metadata/pipeline/parameters",
            required=frozenset(("torch_dtype",)),
        )
        if precision["torch_dtype"] != "float32":
            _fail(f"{path}/metadata/pipeline/parameters/torch_dtype", "must be 'float32'")
        if pooling["method"] != "last_token":
            _fail(
                f"{path}/metadata/pipeline/pooling/method",
                "transformers_last_token requires 'last_token'",
            )
        pooling_parameters = _closed_mapping(
            pooling["parameters"],
            path=f"{path}/metadata/pipeline/pooling/parameters",
            required=frozenset(("padding_side", "select")),
        )
        padding_side = _non_empty_string(
            pooling_parameters["padding_side"],
            path=f"{path}/metadata/pipeline/pooling/parameters/padding_side",
        )
        if padding_side != "right":
            _fail(
                f"{path}/metadata/pipeline/pooling/parameters/padding_side",
                "must be 'right'",
            )

    return EndpointProfile(
        profile_id=wire.profile_id,
        deployment_id=wire.deployment_id,
        endpoint=wire.endpoint,
        kind=_non_empty_string(metadata["kind"], path=f"{path}/metadata/kind"),
        dimensions=_positive_int(
            metadata["dimensions"], path=f"{path}/metadata/dimensions"
        ),
        document_max_tokens=_positive_int(
            by_role["search_document"],
            path=f"{path}/metadata/max_input_tokens/by_role/search_document",
        ),
        query_max_tokens=_positive_int(
            by_role["search_query"],
            path=f"{path}/metadata/max_input_tokens/by_role/search_query",
        ),
        document_template=document_template,
        query_template=query_template,
        additional_task_roles=additional_roles,
        default_language=default_language,
        mask_punctuation=mask_punctuation,
        padding_side=padding_side,
    )


def _parse_backend_model_name(wire: WireProfile, *, path: str) -> str:
    backend = _closed_mapping(
        wire.backend.parameters,
        path=path,
        required=frozenset(
            ("implementation", "implementation_revision", "model_name")
        ),
    )
    _non_empty_string(
        backend["implementation"], path=f"{path}/implementation"
    )
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


def build_embedding_service_view(
    catalog: ModelCatalog,
    loader_registry: Mapping[LoaderType, LoaderCallable],
) -> EmbeddingServiceView:
    """Build the service view solely from an injected, already loaded Catalog."""

    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")
    loaders = _validated_registry(loader_registry)

    task_profile_ids = {
        wire.profile_id for wire in catalog.for_task(_SERVICE_TASK)
    }
    endpoint_profile_ids = {
        wire.profile_id
        for endpoint in _SERVICE_ENDPOINTS
        for wire in catalog.for_endpoint(endpoint)
    }
    service_wires = catalog.for_backend(_SERVICE_BACKEND)

    accumulators: dict[str, _ModelAccumulator] = {}
    for wire in service_wires:
        path = f"/catalog/{wire.canonical_model_id}/{wire.deployment_id}"
        if wire.profile_id not in task_profile_ids:
            _fail(
                f"{path}/task",
                f"backend {_SERVICE_BACKEND.value!r} requires task "
                f"{_SERVICE_TASK.value!r}",
            )
        if wire.profile_id not in endpoint_profile_ids:
            _fail(
                f"{path}/endpoint",
                f"endpoint {wire.endpoint.value!r} is not served by "
                f"backend {_SERVICE_BACKEND.value!r}",
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
        if wire.artifact.type is not ArtifactType.HUGGINGFACE:
            _fail(
                f"{path}/artifact/type",
                "kiron_embeddings requires a HuggingFace artifact",
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
        discovery = _parse_discovery(wire, path=f"{path}/metadata")
        profile = _parse_endpoint_profile(wire, path=f"{path}/profile")

        if isinstance(
            loader_parameters, TransformersLastTokenLoaderParameters
        ) and discovery.precision != "F32":
            _fail(
                f"{path}/metadata/discovery/precision",
                "transformers_last_token float32 configuration requires 'F32'",
            )
        if isinstance(loader_parameters, ColbertXmodLoaderParameters):
            if len(wire.artifact.weights) != 1:
                _fail(
                    f"{path}/artifact/weights",
                    "colbert_xmod requires exactly one projection weight artifact",
                )
        if profile.additional_task_roles:
            if not isinstance(
                loader_parameters, SentenceTransformersLoaderParameters
            ) or loader_parameters.additional_role_template is None:
                _fail(
                    f"{path}/loader/parameters/additional_role_template",
                    "additional task roles require an explicit formatting template",
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
                discovery=discovery,
                profiles=[profile],
            )
            continue

        comparisons = (
            ("canonical_model_id", previous.canonical_model_id, wire.canonical_model_id),
            ("aliases", previous.aliases, wire.aliases),
            ("artifact", previous.artifact, wire.artifact),
            ("loader.type", previous.loader_type, wire.loader.type),
            ("loader.parameters", previous.loader_parameters, loader_parameters),
            ("metadata.discovery", previous.discovery, discovery),
        )
        for field_name, expected, actual in comparisons:
            if actual != expected:
                _fail(
                    path,
                    "conflicting deployment data for service model identity "
                    f"{model_name!r}: {field_name}",
                )
        if any(item.endpoint is profile.endpoint for item in previous.profiles):
            _fail(
                path,
                "conflicting deployment data for service model identity "
                f"{model_name!r}: duplicate endpoint {profile.endpoint.value!r}",
            )
        if any(item.dimensions != profile.dimensions for item in previous.profiles):
            _fail(
                path,
                "conflicting deployment data for service model identity "
                f"{model_name!r}: dimensions",
            )
        previous.profiles.append(profile)

    models = tuple(
        sorted(
            (
                EmbeddingServiceModel(
                    canonical_model_id=item.canonical_model_id,
                    aliases=tuple(sorted(item.aliases)),
                    model_name=item.model_name,
                    artifact=item.artifact,
                    loader_type=item.loader_type,
                    loader_parameters=item.loader_parameters,
                    discovery=item.discovery,
                    profiles=tuple(
                        sorted(item.profiles, key=lambda profile: profile.endpoint.value)
                    ),
                )
                for item in accumulators.values()
            ),
            key=lambda model: model.model_name,
        )
    )

    names: dict[str, EmbeddingServiceModel] = {}
    runtime_names: dict[str, EmbeddingServiceModel] = {}
    for model in models:
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
        endpoint: tuple(
            model for model in models if model.profile_for(endpoint) is not None
        )
        for endpoint in _SERVICE_ENDPOINTS
    }
    return EmbeddingServiceView(
        catalog_digest=catalog.catalog_digest,
        models=models,
        _names=MappingProxyType(names),
        _runtime_names=MappingProxyType(runtime_names),
        _models_by_endpoint=MappingProxyType(by_endpoint),
        _loader_registry=loaders,
    )


__all__ = [
    "ColbertXmodLoaderParameters",
    "DiscoveryMetadata",
    "EmbeddingServiceCatalogError",
    "EmbeddingServiceModel",
    "EmbeddingServiceView",
    "EndpointProfile",
    "LoaderCallable",
    "LoaderParameters",
    "SentenceTransformersLoaderParameters",
    "TransformersLastTokenLoaderParameters",
    "build_embedding_service_view",
]
