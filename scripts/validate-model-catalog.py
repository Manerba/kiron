#!/usr/bin/env python3
"""Offline architecture validator for KIron's built-in model Catalog.

The validator reads packaged JSON resources and constructs every production
projection with explicit inert loader registries.  It never probes the network,
the model cache, Docker, systemd, or runtime state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMMON_SOURCE = ROOT / "services" / "kiron-common"
PRODUCTION_GOLDEN = (
    COMMON_SOURCE
    / "tests"
    / "fixtures"
    / "model_catalog"
    / "production-catalog-v1.digest.json"
)

# Source-tree execution must not depend on a service venv or an editable
# install.  sys.dont_write_bytecode also makes the operational preflight
# read-only even when the caller forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
if str(COMMON_SOURCE) not in sys.path:
    sys.path.insert(0, str(COMMON_SOURCE))

from kiron_common.embedding_registry import build_embedding_registry  # noqa: E402
from kiron_common.model_catalog import (  # noqa: E402
    BackendType,
    DEFAULT_MANIFEST_PACKAGE,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
    load_catalog_from_package,
)
from kiron_common.model_state import build_model_state_view  # noqa: E402


class CatalogArchitectureError(ValueError):
    """A cross-projection or production-golden invariant failed."""


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CatalogArchitectureError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _inert_loader(_model: object) -> object:
    raise AssertionError("Catalog validation must never invoke a model loader")


def _manifest_index(package: str) -> tuple[tuple[str, ...], dict[str, str]]:
    root = resources.files(package)
    items = tuple(
        sorted(
            (
                item
                for item in root.iterdir()
                if item.is_file() and item.name.endswith(".model.json")
            ),
            key=lambda item: item.name,
        )
    )
    names = tuple(item.name for item in items)
    by_model: dict[str, str] = {}
    for item in items:
        try:
            document = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CatalogArchitectureError(
                f"cannot index validated manifest {package}:{item.name}: {exc}"
            ) from exc
        canonical = document.get("canonical_model_id") if isinstance(document, dict) else None
        if type(canonical) is not str or not canonical:
            raise CatalogArchitectureError(
                f"{package}:{item.name} has no canonical_model_id"
            )
        previous = by_model.get(canonical)
        if previous is not None:
            raise CatalogArchitectureError(
                f"{canonical!r} is declared by both {previous!r} and {item.name!r}"
            )
        by_model[canonical] = item.name
    return names, by_model


def _production_golden_check(summary: dict[str, Any], golden_path: Path) -> None:
    try:
        expected = json.loads(golden_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogArchitectureError(
            f"cannot read production golden {golden_path}: {exc}"
        ) from exc

    actual_groups = [
        {
            "canonical_model_id": row["canonical_model_id"],
            "aliases": row["aliases"],
            "profile_ids": sorted(
                profile["id"] for profile in row["profiles"]
            ),
        }
        for row in summary["model_matrix"]
    ]
    comparisons = {
        "catalog_digest": summary["catalog_digest"],
        "group_count": summary["counts"]["catalog_groups"],
        "profile_count": summary["counts"]["catalog_profiles"],
        "manifest_files": summary["manifest_files"],
        "groups": actual_groups,
    }
    for field, actual in comparisons.items():
        if actual != expected.get(field):
            raise CatalogArchitectureError(
                f"production golden mismatch for {field}: "
                f"expected {expected.get(field)!r}, got {actual!r}"
            )


def _matrix_row(group: object, manifest_name: str) -> dict[str, Any]:
    document = group.to_manifest_dict(schema_version=1)
    return {
        "manifest": manifest_name,
        "canonical_model_id": document["canonical_model_id"],
        "aliases": document["aliases"],
        "deployments": document["deployments"],
        "profiles": document["profiles"],
        "request_defaults": document["request_defaults"],
    }


def validate_catalog_architecture(
    *,
    manifest_package: str = DEFAULT_MANIFEST_PACKAGE,
    enforce_production_golden: bool = True,
    golden_path: Path = PRODUCTION_GOLDEN,
) -> dict[str, Any]:
    """Build and cross-check all production projections without side effects."""

    catalog = load_catalog_from_package(manifest_package)
    manifest_files, manifest_by_model = _manifest_index(manifest_package)
    catalog_names = {group.canonical_model_id for group in catalog.groups}
    if set(manifest_by_model) != catalog_names:
        raise CatalogArchitectureError(
            "manifest resources and parsed Catalog groups are not one-to-one"
        )

    embedding_registry = build_embedding_registry(catalog)
    embedding_registry.validate()
    state_view = build_model_state_view(catalog)

    embeddings_module = _load_source_module(
        "_kiron_catalog_validator_embeddings_view",
        ROOT / "services" / "kiron-embeddings" / "catalog_view.py",
    )
    deberta_module = _load_source_module(
        "_kiron_catalog_validator_deberta_view",
        ROOT / "services" / "kiron-deberta" / "catalog_view.py",
    )
    routing_module = _load_source_module(
        "_kiron_catalog_validator_proxy_routing",
        ROOT / "services" / "kiron-proxy" / "routing_catalog.py",
    )

    embedding_loader_registry = {
        LoaderType.SENTENCE_TRANSFORMERS: _inert_loader,
        LoaderType.TRANSFORMERS_LAST_TOKEN: _inert_loader,
        LoaderType.COLBERT_XMOD: _inert_loader,
    }
    deberta_loader_registry = {
        LoaderType.CROSS_ENCODER: _inert_loader,
        LoaderType.MANKEI_LAST_TOKEN: _inert_loader,
    }
    embedding_view = embeddings_module.build_embedding_service_view(
        catalog,
        embedding_loader_registry,
    )
    deberta_view = deberta_module.build_deberta_service_view(
        catalog,
        deberta_loader_registry,
    )
    routing_view = routing_module.build_proxy_routing_view(catalog)

    digests = {
        catalog.catalog_digest,
        embedding_registry.catalog_digest,
        state_view.catalog_digest,
        embedding_view.catalog_digest,
        deberta_view.catalog_digest,
        routing_view.catalog_digest,
    }
    if digests != {catalog.catalog_digest}:
        raise CatalogArchitectureError(
            f"projection digest mismatch: {sorted(str(item) for item in digests)!r}"
        )

    catalog_deployments = {
        deployment.id
        for group in catalog.groups
        for deployment in group.deployments
    }
    state_deployments = [
        deployment_id
        for definition in state_view.models
        for deployment_id in definition.deployment_ids
    ]
    if len(state_deployments) != len(set(state_deployments)):
        raise CatalogArchitectureError(
            "a deployment is captured by more than one Model-State definition"
        )
    if set(state_deployments) != catalog_deployments:
        raise CatalogArchitectureError(
            "Model-State view does not capture every Catalog deployment"
        )

    embedding_profile_ids = {
        wire.profile_id for wire in catalog.for_task(ModelTask.EMBEDDING)
    }
    registry_profile_ids = {
        profile["profile_id"]
        for group in embedding_registry.groups
        for profile in group.capabilities["profiles"]
    }
    if registry_profile_ids != embedding_profile_ids:
        raise CatalogArchitectureError(
            "Embedding Registry does not capture every embedding profile"
        )

    embedding_service_profile_ids = {
        profile.profile_id
        for model in embedding_view.models
        for profile in model.profiles
    }
    expected_embedding_service_profiles = {
        wire.profile_id
        for wire in catalog.for_backend(BackendType.KIRON_EMBEDDINGS)
    }
    if embedding_service_profile_ids != expected_embedding_service_profiles:
        raise CatalogArchitectureError(
            "Embeddings Service view does not capture every service profile"
        )

    expected_deberta = {
        (
            definition.backend_model_name,
            tuple(endpoint.value for endpoint in definition.endpoints),
        )
        for definition in state_view.for_backend(BackendType.KIRON_DEBERTA)
    }
    actual_deberta = {
        (
            model.model_name,
            tuple(endpoint.value for endpoint in model.endpoints),
        )
        for model in deberta_view.models
    }
    if actual_deberta != expected_deberta:
        raise CatalogArchitectureError(
            "DeBERTa Service view does not capture every service model/endpoint"
        )

    endpoint_counts = {
        endpoint.value: len(routing_view.routes_for_endpoint(endpoint))
        for endpoint in ModelEndpoint
    }
    summary: dict[str, Any] = {
        "status": "ok",
        "catalog_schema_version": 1,
        "catalog_digest": catalog.catalog_digest,
        "manifest_package": manifest_package,
        "manifest_files": list(manifest_files),
        "counts": {
            "catalog_groups": len(catalog.groups),
            "catalog_profiles": len(catalog.wire_profiles),
            "embedding_registry_groups": len(embedding_registry.groups),
            "embedding_registry_profiles": embedding_registry.profile_count,
            "model_state_definitions": len(state_view.models),
            "embedding_service_models": len(embedding_view.models),
            "deberta_service_models": len(deberta_view.models),
            "proxy_routes": len(routing_view.routes),
        },
        "loader_registries": {
            "kiron_embeddings": sorted(
                loader.value for loader in embedding_loader_registry
            ),
            "kiron_deberta": sorted(
                loader.value for loader in deberta_loader_registry
            ),
        },
        "proxy_endpoint_routes": endpoint_counts,
        "request_defaults": {
            endpoint.value: routing_view.request_default(endpoint).canonical_model_id
            for endpoint in (ModelEndpoint.RERANK, ModelEndpoint.SCORE)
        },
        "model_matrix": [
            _matrix_row(group, manifest_by_model[group.canonical_model_id])
            for group in catalog.groups
        ],
    }
    if enforce_production_golden:
        _production_golden_check(summary, golden_path)
    return summary


def _text_summary(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    endpoint_counts = summary["proxy_endpoint_routes"]
    lines = [
        "catalog_validation=ok",
        f"catalog_digest={summary['catalog_digest']}",
        f"catalog_manifests={len(summary['manifest_files'])}",
        f"catalog_groups={counts['catalog_groups']}",
        f"catalog_profiles={counts['catalog_profiles']}",
        f"embedding_registry_groups={counts['embedding_registry_groups']}",
        f"embedding_registry_profiles={counts['embedding_registry_profiles']}",
        f"model_state_definitions={counts['model_state_definitions']}",
        f"embedding_service_models={counts['embedding_service_models']}",
        f"deberta_service_models={counts['deberta_service_models']}",
        f"proxy_routes={counts['proxy_routes']}",
        "proxy_endpoint_routes="
        + ",".join(
            f"{endpoint}={endpoint_counts[endpoint]}"
            for endpoint in sorted(endpoint_counts)
        ),
        "embedding_loaders="
        + ",".join(summary["loader_registries"]["kiron_embeddings"]),
        "deberta_loaders="
        + ",".join(summary["loader_registries"]["kiron_deberta"]),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate KIron's complete built-in model Catalog offline."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the stable complete model/projection matrix as JSON",
    )
    args = parser.parse_args(argv)
    try:
        summary = validate_catalog_architecture()
    except Exception as exc:
        print("catalog_validation=failed", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_text_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
