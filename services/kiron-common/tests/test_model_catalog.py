from __future__ import annotations

import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import FrozenInstanceError
from importlib import resources
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from kiron_common.model_catalog import (
    ArtifactType,
    BackendType,
    CatalogLookupError,
    CatalogResourceError,
    CatalogValidationError,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
    RequestDefault,
    load_catalog,
    load_catalog_from_package,
    parse_manifest,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "model_catalog"


def _fixture(name: str = "alpha.model.json") -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _hf_manifest(
    *,
    canonical_model_id: str = "beta-reranker:latest",
    stem: str = "beta",
    task: str = "rerank",
    endpoint: str = "/api/rerank",
    request_default: bool = True,
) -> dict:
    return {
        "schema_version": 2,
        "canonical_model_id": canonical_model_id,
        "aliases": [stem],
        "deployments": [
            {
                "id": f"{stem}.service",
                "backend": {"type": "kiron_deberta", "parameters": {}}, "runtime_profile": None,
                "artifact": {
                    "type": "huggingface", "format": "hf_weights", "projector": None,
                    "repository": f"org/{stem}",
                    "revision": "b" * 40,
                    "manifest_digest": None,
                    "trust_remote_code": False,
                    "weights": [
                        {"path": "model.safetensors", "sha256": "4" * 64, "size_bytes": None}
                    ],
                    "auxiliary": [],
                    "metadata": {},
                },
                "routes": [{"task": task, "endpoint": endpoint}],
                "loader": {"type": "cross_encoder", "parameters": {}},
                "metadata": {},
            }
        ],
        "profiles": [
            {
                "id": f"{stem}.primary",
                "deployment_id": f"{stem}.service",
                "task": task,
                "endpoint": endpoint,
                "default_for_endpoint": True,
                "metadata": {},
            }
        ],
        "request_defaults": (
            [{"endpoint": endpoint, "profile_id": f"{stem}.primary"}]
            if request_default
            else []
        ),
        "metadata": {"family": stem},
    }


def _embedding_manifest(*, stem: str) -> dict:
    manifest = _fixture()
    manifest["canonical_model_id"] = f"Org/{stem.title()}:Q4"
    manifest["aliases"] = [stem]
    manifest["metadata"] = {"display_name": stem.title()}
    for deployment in manifest["deployments"]:
        deployment["id"] = deployment["id"].replace("alpha", stem)
        repository = deployment["artifact"]["repository"]
        if repository is not None:
            deployment["artifact"]["repository"] = repository.replace(
                "alpha", stem
            )
    for profile in manifest["profiles"]:
        profile["id"] = profile["id"].replace("alpha", stem)
        profile["deployment_id"] = profile["deployment_id"].replace(
            "alpha", stem
        )
    return manifest


def _rerank_and_score_manifest() -> dict:
    manifest = _hf_manifest(
        canonical_model_id="multi-service:latest",
        stem="multi",
    )
    score_deployment = copy.deepcopy(manifest["deployments"][0])
    score_deployment["id"] = "multi.score-service"
    score_deployment["routes"] = [
        {"task": "nli", "endpoint": "/api/score"}
    ]
    score_profile = {
        "id": "multi.score",
        "deployment_id": "multi.score-service",
        "task": "nli",
        "endpoint": "/api/score",
        "default_for_endpoint": True,
        "metadata": {},
    }
    manifest["deployments"].append(score_deployment)
    manifest["profiles"].append(score_profile)
    manifest["request_defaults"].append(
        {"endpoint": "/api/score", "profile_id": "multi.score"}
    )
    return manifest


def _catalog() -> ModelCatalog:
    return ModelCatalog.from_manifests([_fixture(), _hf_manifest()])


def _assert_invalid(manifest: dict, match: str) -> CatalogValidationError:
    with pytest.raises(CatalogValidationError, match=match) as raised:
        ModelCatalog.from_manifests([manifest])
    return raised.value


def test_schema_contract_is_packaged_next_to_the_runtime_parser():
    schema_resource = resources.files("kiron_common.model_catalog").joinpath(
        "model-manifest-v2.schema.json"
    )
    schema = json.loads(schema_resource.read_text(encoding="utf-8"))

    assert schema["$id"].endswith("model-manifest-v2.schema.json")
    assert schema["properties"]["schema_version"] == {"const": 2}
    assert schema["additionalProperties"] is False
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(_fixture())) == []
    assert list(Draft202012Validator(schema).iter_errors(_hf_manifest())) == []
    assert schema["$defs"]["backend"]["properties"]["type"]["enum"] == [
        item.value for item in BackendType
    ]
    assert schema["$defs"]["loader"]["properties"]["type"]["enum"] == [
        item.value for item in LoaderType
    ]
    assert schema["$defs"]["profile"]["properties"]["task"]["enum"] == [
        item.value for item in ModelTask
    ]
    assert schema["$defs"]["profile"]["properties"]["endpoint"]["enum"] == [
        item.value for item in ModelEndpoint
    ]
    assert schema["$defs"]["requestDefault"]["properties"]["endpoint"][
        "enum"
    ] == [ModelEndpoint.RERANK.value, ModelEndpoint.SCORE.value]


def test_normalized_catalog_builds_expanded_typed_query_views():
    catalog = _catalog()

    assert [group.canonical_model_id for group in catalog.groups] == [
        "Org/Alpha:Q4",
        "beta-reranker:latest",
    ]
    assert catalog.resolve("alpha") is catalog.require("Org/Alpha:Q4")
    assert catalog.require("beta").canonical_model_id == "beta-reranker:latest"

    embedding_profiles = catalog.for_task(ModelTask.EMBEDDING)
    assert [profile.profile_id for profile in embedding_profiles] == [
        "alpha.ollama.dense",
        "alpha.service.dense",
    ]
    assert len(catalog.for_backend("kiron_embeddings")) == 1
    assert len(catalog.for_backend(BackendType.OLLAMA)) == 1
    assert len(catalog.for_backend("kiron_deberta")) == 1
    assert len(catalog.for_endpoint(ModelEndpoint.EMBED)) == 2
    assert len(catalog.for_endpoint("/api/rerank")) == 1

    default = catalog.default_for_endpoint("alpha", "/api/embed")
    assert default is not None
    assert default.profile_id == "alpha.service.dense"
    assert default.deployment_id == "alpha.service"
    assert default.backend.type is BackendType.KIRON_EMBEDDINGS
    assert default.artifact.type is ArtifactType.HUGGINGFACE
    assert default.artifact.repository == "org/alpha"
    assert default.loader.type is LoaderType.SENTENCE_TRANSFORMERS
    assert default.to_dict()["artifact"]["revision"] == "a" * 40
    assert catalog.wire_profile("alpha.ollama.dense").loader.type is LoaderType.OLLAMA
    request_default = catalog.request_default_for_endpoint("/api/rerank")
    assert request_default is not None
    assert request_default.profile_id == "beta.primary"
    assert request_default.is_request_default is True
    assert request_default.to_dict()["is_request_default"] is True
    assert default.is_request_default is False
    assert catalog.request_default_for_endpoint("/api/embed") is None
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", catalog.catalog_digest)


@pytest.mark.parametrize(
    "candidate",
    [
        "Alpha",
        "alpha ",
        "org/alpha",
        "Org/Alpha:latest",
        "Alpha:Q4",
        "Org/Alpha:Q4 ",
        None,
        1,
    ],
)
def test_alias_resolution_is_only_exact_and_literal(candidate):
    catalog = _catalog()

    assert catalog.resolve(candidate) is None
    assert catalog.default_for_endpoint(candidate, "/api/embed") is None


def test_require_and_filter_lookups_fail_without_normalization_or_guessing():
    catalog = _catalog()

    with pytest.raises(CatalogLookupError, match="no exact canonical ID"):
        catalog.require("Alpha")
    assert catalog.for_backend("KIRON_EMBEDDINGS") == ()
    assert catalog.for_task("Embedding") == ()
    assert catalog.for_endpoint("api/embed") == ()
    assert catalog.wire_profile("ALPHA.SERVICE.DENSE") is None


@pytest.mark.parametrize(
    "candidate",
    ["api/rerank", "/API/rerank", "/api/rerank/", " /api/rerank", None, 1],
)
def test_request_default_endpoint_resolution_is_exact(candidate):
    catalog = _catalog()

    assert catalog.request_default_for_endpoint(candidate) is None
    assert (
        catalog.request_default_for_endpoint(ModelEndpoint.RERANK).profile_id
        == "beta.primary"
    )


def test_models_catalog_and_nested_json_values_are_deeply_immutable():
    catalog = _catalog()
    group = catalog.require("alpha")
    wire = catalog.default_for_endpoint("alpha", "/api/embed")
    assert wire is not None

    with pytest.raises(FrozenInstanceError):
        group.canonical_model_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        catalog.require("beta").request_defaults[0].profile_id = (  # type: ignore[misc]
            "changed"
        )
    assert isinstance(catalog.require("beta").request_defaults[0], RequestDefault)
    with pytest.raises(TypeError):
        group.metadata["display_name"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        wire.loader.parameters["batch_size"] = 1  # type: ignore[index]
    assert wire.loader.parameters["tags"] == ("A", "b")
    with pytest.raises(AttributeError, match="immutable"):
        catalog._groups = ()  # type: ignore[attr-defined]

    detached = wire.to_dict()
    detached["backend"]["parameters"]["queue"] = "parallel"
    detached["metadata"]["profile"]["dimensions"] = 1
    assert wire.backend.parameters["queue"] == "serial"
    assert wire.profile_metadata["dimensions"] == 768


def test_two_reranker_groups_have_one_unambiguous_request_default():
    first = _hf_manifest(
        canonical_model_id="first-reranker:latest",
        stem="first",
    )
    second = _hf_manifest(
        canonical_model_id="second-reranker:latest",
        stem="second",
        request_default=False,
    )

    catalog = ModelCatalog.from_manifests([second, first])

    request_default = catalog.request_default_for_endpoint("/api/rerank")
    assert request_default is not None
    assert request_default.profile_id == "first.primary"
    assert request_default.canonical_model_id == "first-reranker:latest"
    assert request_default.is_request_default is True
    assert catalog.wire_profile("second.primary").is_request_default is False
    assert (
        catalog.default_for_endpoint("first", "/api/rerank").profile_id
        == "first.primary"
    )
    assert (
        catalog.default_for_endpoint("second", "/api/rerank").profile_id
        == "second.primary"
    )


def test_score_request_default_resolves_to_an_expanded_wire_profile():
    catalog = ModelCatalog.from_manifests([_rerank_and_score_manifest()])

    score_default = catalog.request_default_for_endpoint(ModelEndpoint.SCORE)
    assert score_default is not None
    assert score_default.profile_id == "multi.score"
    assert score_default.task is ModelTask.NLI
    assert score_default.endpoint is ModelEndpoint.SCORE
    assert score_default.is_request_default is True


@pytest.mark.parametrize(
    ("task", "endpoint"),
    [("rerank", "/api/rerank"), ("nli", "/api/score")],
)
def test_required_endpoint_without_request_default_fails_fast(task, endpoint):
    manifest = _hf_manifest(
        task=task,
        endpoint=endpoint,
        request_default=False,
    )

    _assert_invalid(manifest, "present but has no request default")


def test_multiple_request_defaults_for_one_endpoint_fail_fast():
    first = _hf_manifest(
        canonical_model_id="first-reranker:latest",
        stem="first",
    )
    second = _hf_manifest(
        canonical_model_id="second-reranker:latest",
        stem="second",
    )

    with pytest.raises(CatalogValidationError, match="multiple request defaults"):
        ModelCatalog.from_manifests([first, second])

    duplicate_in_one_manifest = _hf_manifest()
    duplicate_in_one_manifest["request_defaults"].append(
        {"endpoint": "/api/rerank", "profile_id": "beta.primary"}
    )
    _assert_invalid(
        duplicate_in_one_manifest,
        "multiple request defaults for the same endpoint",
    )


def test_request_default_must_reference_a_profile_serving_its_endpoint():
    manifest = _hf_manifest()
    manifest["request_defaults"][0]["endpoint"] = "/api/score"

    _assert_invalid(manifest, "serves endpoint '/api/rerank', not '/api/score'")

    manifest = _hf_manifest()
    manifest["request_defaults"][0]["profile_id"] = "missing.profile"
    _assert_invalid(manifest, "references unknown profile")


def test_embedding_groups_need_no_catalog_request_default():
    alpha = _fixture()
    gamma = _embedding_manifest(stem="gamma")

    catalog = ModelCatalog.from_manifests([gamma, alpha])

    assert catalog.request_default_for_endpoint("/api/embed") is None
    assert (
        catalog.default_for_endpoint("alpha", "/api/embed").profile_id
        == "alpha.service.dense"
    )
    assert (
        catalog.default_for_endpoint("gamma", "/api/embed").profile_id
        == "gamma.service.dense"
    )


def test_embedding_endpoints_cannot_declare_request_defaults():
    manifest = _fixture()
    manifest["request_defaults"] = [
        {"endpoint": "/api/embed", "profile_id": "alpha.service.dense"}
    ]

    _assert_invalid(manifest, "request defaults are allowed only")


def test_digest_is_independent_of_manifest_and_set_like_array_order():
    alpha = _fixture()
    beta = _hf_manifest()
    baseline = ModelCatalog.from_manifests([alpha, beta])
    assert baseline.catalog_digest == (
        "sha256:0c6160273374f66580fa003ab92a33d2e14626944b9c806f54c036f587ce5d95"
    )

    reordered_alpha = copy.deepcopy(alpha)
    reordered_alpha["aliases"].reverse()
    reordered_alpha["deployments"].reverse()
    reordered_alpha["profiles"].reverse()
    reordered_alpha["metadata"] = {"display_name": "Alpha"}
    reordered = ModelCatalog.from_manifests([beta, reordered_alpha])

    assert reordered.catalog_digest == baseline.catalog_digest
    assert reordered.wire_profiles == baseline.wire_profiles

    changed = copy.deepcopy(alpha)
    changed["metadata"]["display_name"] = "Changed"
    assert (
        ModelCatalog.from_manifests([changed, beta]).catalog_digest
        != baseline.catalog_digest
    )


def test_digest_changes_when_request_default_semantics_change():
    first = _hf_manifest(
        canonical_model_id="first-reranker:latest",
        stem="first",
    )
    second = _hf_manifest(
        canonical_model_id="second-reranker:latest",
        stem="second",
        request_default=False,
    )
    first_default = ModelCatalog.from_manifests([first, second])

    moved_first = copy.deepcopy(first)
    moved_second = copy.deepcopy(second)
    moved_first["request_defaults"] = []
    moved_second["request_defaults"] = [
        {"endpoint": "/api/rerank", "profile_id": "second.primary"}
    ]
    second_default = ModelCatalog.from_manifests([moved_second, moved_first])

    assert first_default.catalog_digest != second_default.catalog_digest
    assert (
        first_default.request_default_for_endpoint("/api/rerank").profile_id
        == "first.primary"
    )
    assert (
        second_default.request_default_for_endpoint("/api/rerank").profile_id
        == "second.primary"
    )


def test_digest_ignores_request_default_declaration_order():
    manifest = _rerank_and_score_manifest()
    baseline = ModelCatalog.from_manifests([manifest])
    reordered = copy.deepcopy(manifest)
    reordered["request_defaults"].reverse()

    reordered_catalog = ModelCatalog.from_manifests([reordered])

    assert reordered_catalog.catalog_digest == baseline.catalog_digest
    assert reordered_catalog.wire_profiles == baseline.wire_profiles


def test_empty_packaged_catalog_is_valid_without_registering_production_models(
    tmp_path, monkeypatch
):
    package_name = _write_resource_package(
        tmp_path,
        {},
        package_name="fixture_empty_catalog_resources",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    catalog = load_catalog_from_package(package_name)

    assert catalog.groups == ()
    assert catalog.wire_profiles == ()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", catalog.catalog_digest)


def test_packaged_production_catalog_matches_complete_digest_golden():
    expected = _fixture("production-catalog-v1.digest.json")
    manifest_root = resources.files("kiron_common.model_catalog.manifests")
    manifest_files = sorted(
        item.name
        for item in manifest_root.iterdir()
        if item.name.endswith(".model.json")
    )
    catalog = load_catalog()
    actual_groups = [
        {
            "canonical_model_id": group.canonical_model_id,
            "aliases": list(group.aliases),
            "profile_ids": sorted(profile.id for profile in group.profiles),
        }
        for group in catalog.groups
    ]

    assert expected["fixture_version"] == 1
    assert expected["catalog_schema_version"] == 2
    assert manifest_files == expected["manifest_files"]
    assert len(catalog.groups) == expected["group_count"] == 13
    assert len(catalog.wire_profiles) == expected["profile_count"] == 23
    assert actual_groups == expected["groups"]
    assert catalog.catalog_digest == expected["catalog_digest"]
    assert {
        endpoint.value: catalog.request_default_for_endpoint(endpoint).canonical_model_id
        for endpoint in (ModelEndpoint.RERANK, ModelEndpoint.SCORE)
    } == {
        "/api/rerank": "ms-marco-MiniLM-L-6-v2",
        "/api/score": "ms-marco-MiniLM-L-6-v2",
    }


def test_duplicate_canonical_ids_and_aliases_fail_across_manifests():
    alpha = _fixture()
    duplicate = _hf_manifest(canonical_model_id="Org/Alpha:Q4", stem="beta")
    with pytest.raises(CatalogValidationError, match="literal model name"):
        ModelCatalog.from_manifests([alpha, duplicate])

    collision = _hf_manifest(stem="beta")
    collision["aliases"] = ["alpha"]
    with pytest.raises(CatalogValidationError, match="literal model name"):
        ModelCatalog.from_manifests([alpha, collision])


def test_duplicate_deployment_and_profile_ids_fail_globally():
    alpha = _fixture()
    beta = _hf_manifest()
    beta["deployments"][0]["id"] = "alpha.service"
    beta["profiles"][0]["deployment_id"] = "alpha.service"
    with pytest.raises(CatalogValidationError, match="deployment ID"):
        ModelCatalog.from_manifests([alpha, beta])

    beta = _hf_manifest()
    beta["profiles"][0]["id"] = "alpha.service.dense"
    beta["request_defaults"][0]["profile_id"] = "alpha.service.dense"
    with pytest.raises(CatalogValidationError, match="profile ID"):
        ModelCatalog.from_manifests([alpha, beta])


def test_duplicate_ids_and_aliases_fail_within_one_manifest():
    manifest = _fixture()
    manifest["aliases"].append("alpha")
    _assert_invalid(manifest, "must not contain duplicates")

    manifest = _fixture()
    manifest["deployments"].append(copy.deepcopy(manifest["deployments"][0]))
    _assert_invalid(manifest, "duplicate deployment IDs")

    manifest = _fixture()
    manifest["profiles"].append(copy.deepcopy(manifest["profiles"][0]))
    _assert_invalid(manifest, "duplicate profile IDs")


def test_unknown_and_inconsistent_profile_references_fail_fast():
    manifest = _fixture()
    manifest["profiles"][0]["deployment_id"] = "missing.deployment"
    _assert_invalid(manifest, "references unknown deployment")

    manifest = _fixture()
    manifest["profiles"][0]["endpoint"] = "/api/embed_late"
    _assert_invalid(manifest, "route is not declared")

    manifest = _fixture()
    manifest["deployments"][0]["routes"].append(
        {"task": "embedding", "endpoint": "/api/embed_late"}
    )
    _assert_invalid(manifest, "routes without profiles")


def test_every_present_endpoint_requires_exactly_one_default():
    manifest = _fixture()
    manifest["profiles"][0]["default_for_endpoint"] = False
    _assert_invalid(manifest, "requires exactly one default; found 0")

    manifest = _fixture()
    manifest["profiles"][1]["default_for_endpoint"] = True
    _assert_invalid(manifest, "requires exactly one default; found 2")


@pytest.mark.parametrize(
    ("task", "endpoint"),
    [
        ("embedding", "/api/rerank"),
        ("rerank", "/api/embed"),
        ("nli", "/api/embed_colbert"),
        ("rerank", "/api/score"),
        ("nli", "/api/rerank"),
    ],
)
def test_invalid_task_endpoint_combinations_fail_fast(task, endpoint):
    manifest = _hf_manifest(task=task, endpoint=endpoint)
    _assert_invalid(manifest, "allows only endpoints")


@pytest.mark.parametrize(
    "endpoint",
    ["/api/embed", "/api/embed_late", "/api/embed_colbert"],
)
def test_embedding_task_accepts_each_adr_0009_endpoint(endpoint):
    manifest = _fixture()
    manifest["deployments"] = [manifest["deployments"][0]]
    manifest["profiles"] = [manifest["profiles"][0]]
    manifest["deployments"][0]["routes"] = [
        {"task": "embedding", "endpoint": endpoint}
    ]
    manifest["profiles"][0]["endpoint"] = endpoint

    parsed = parse_manifest(manifest)

    assert parsed.profiles[0].task is ModelTask.EMBEDDING
    assert parsed.profiles[0].endpoint.value == endpoint


def test_unknown_schema_versions_fields_enums_and_wrong_types_fail_closed():
    manifest = _fixture()
    manifest["schema_version"] = 1
    _assert_invalid(manifest, "unsupported schema version")

    manifest = _fixture()
    manifest["unexpected"] = True
    _assert_invalid(manifest, "unknown fields")

    manifest = _fixture()
    del manifest["request_defaults"]
    _assert_invalid(manifest, "missing required fields: request_defaults")

    manifest = _fixture()
    manifest["deployments"][0]["loader"]["type"] = "magic_fallback"
    _assert_invalid(manifest, "unknown value 'magic_fallback'")

    manifest = _fixture()
    manifest["profiles"][0]["default_for_endpoint"] = 1
    _assert_invalid(manifest, "must be a boolean")


@pytest.mark.parametrize(
    "revision",
    [None, "main", "latest", "a" * 39, "A" * 40],
)
def test_huggingface_artifacts_require_full_immutable_revisions(revision):
    manifest = _fixture()
    manifest["deployments"][0]["artifact"]["revision"] = revision
    _assert_invalid(manifest, "immutable 40-character lowercase revision")


def test_trust_remote_code_is_allowed_only_with_an_immutable_revision():
    manifest = _fixture()
    manifest["deployments"][0]["artifact"]["trust_remote_code"] = True
    parse_manifest(manifest)

    manifest["deployments"][0]["artifact"]["revision"] = "main"
    _assert_invalid(manifest, "trust_remote_code requires an immutable")


def test_ollama_artifacts_require_content_identity_and_matching_loader():
    manifest = _fixture()
    manifest["deployments"][1]["artifact"]["manifest_digest"] = None
    _assert_invalid(manifest, "require an immutable manifest digest")

    manifest = _fixture()
    manifest["deployments"][1]["loader"]["type"] = "cross_encoder"
    _assert_invalid(manifest, "requires the ollama loader ID")


def test_duplicate_artifact_paths_fail_within_and_across_roles():
    manifest = _fixture()
    weight = manifest["deployments"][0]["artifact"]["weights"][0]
    manifest["deployments"][0]["artifact"]["weights"].append(copy.deepcopy(weight))
    _assert_invalid(manifest, "duplicate artifact path")

    manifest = _fixture()
    manifest["deployments"][0]["artifact"]["auxiliary"].append(
        copy.deepcopy(manifest["deployments"][0]["artifact"]["weights"][0])
    )
    _assert_invalid(manifest, "weight and auxiliary paths overlap")


def test_parameter_maps_reject_non_json_non_finite_and_unsafe_integers():
    manifest = _fixture()
    manifest["deployments"][0]["loader"]["parameters"]["bad"] = {1, 2}
    _assert_invalid(manifest, "is not JSON")

    manifest = _fixture()
    manifest["metadata"]["bad"] = float("nan")
    _assert_invalid(manifest, "must be finite")

    manifest = _fixture()
    manifest["metadata"]["bad"] = 9_007_199_254_740_992
    _assert_invalid(manifest, "outside the I-JSON interoperable range")


def test_manifest_strings_reject_lone_surrogates_but_accept_non_bmp_scalars():
    manifest = _fixture()
    manifest["aliases"].append("valid-😀")
    parsed = parse_manifest(manifest)
    assert "valid-😀" in parsed.aliases

    manifest["aliases"].append("invalid-\ud800")
    _assert_invalid(manifest, "lone Unicode surrogate")


def _write_resource_package(
    tmp_path: Path,
    documents: dict[str, dict],
    *,
    package_name: str = "fixture_catalog_resources",
) -> str:
    sys.modules.pop(package_name, None)
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for filename, document in documents.items():
        (package / filename).write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8"
        )
    return package_name


def test_resource_loader_uses_package_resources_not_cwd_and_sorts_files(
    tmp_path, monkeypatch
):
    alpha = _fixture()
    beta = _hf_manifest()
    package_name = _write_resource_package(
        tmp_path,
        {
            "z-alpha.model.json": alpha,
            "a-beta.model.json": beta,
            "ignored.json": {"schema_version": 999},
        },
        package_name="fixture_catalog_order_a",
    )
    reverse_package_name = _write_resource_package(
        tmp_path,
        {
            "a-alpha.model.json": alpha,
            "z-beta.model.json": beta,
        },
        package_name="fixture_catalog_order_b",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    loaded = load_catalog_from_package(package_name)
    reverse_loaded = load_catalog_from_package(reverse_package_name)
    direct = ModelCatalog.from_manifests([beta, alpha])

    assert loaded.catalog_digest == direct.catalog_digest
    assert reverse_loaded.catalog_digest == direct.catalog_digest
    assert [group.canonical_model_id for group in loaded.groups] == [
        "Org/Alpha:Q4",
        "beta-reranker:latest",
    ]


def test_wheel_install_preserves_all_production_manifest_resources(tmp_path):
    expected = _fixture("production-catalog-v1.digest.json")
    project_root = Path(__file__).resolve().parents[1]
    source_copy = tmp_path / "source"
    shutil.copytree(
        project_root,
        source_copy,
        ignore=shutil.ignore_patterns(
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    environment = os.environ.copy()
    environment["PIP_NO_INDEX"] = "1"
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source_copy),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    wheel_path, = wheel_dir.glob("kiron_common-*.whl")
    resource_prefix = "kiron_common/model_catalog/manifests/"
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_manifests = sorted(
            name.removeprefix(resource_prefix)
            for name in archive.namelist()
            if name.startswith(resource_prefix) and name.endswith(".model.json")
        )
    assert wheel_manifests == expected["manifest_files"]

    install_dir = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheel_path),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    smoke_environment = environment.copy()
    smoke_environment["PYTHONPATH"] = str(install_dir)
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from importlib import resources; "
            "from kiron_common.catalog_consistency import check_catalog_digests; "
            "from kiron_common.model_catalog import load_catalog; "
            "from kiron_common.model_state import build_model_state_view; "
            "root = resources.files('kiron_common.model_catalog.manifests'); "
            "names = sorted(item.name for item in root.iterdir() "
            "if item.name.endswith('.model.json')); "
            "catalog = load_catalog(); "
            "state_view = build_model_state_view(catalog); "
            "digest_report = check_catalog_digests(catalog.catalog_digest, "
            "{'service': catalog.catalog_digest}, required_services=('service',)); "
            "print(json.dumps({'manifest_files': names, "
            "'group_count': len(catalog.groups), "
            "'profile_count': len(catalog.wire_profiles), "
            "'state_model_count': len(state_view.models), "
            "'digest_consistent': digest_report.consistent, "
            "'catalog_digest': catalog.catalog_digest}, sort_keys=True))",
        ],
        check=True,
        cwd=tmp_path,
        env=smoke_environment,
        capture_output=True,
        text=True,
    )
    installed = json.loads(smoke.stdout)
    assert installed == {
        "manifest_files": expected["manifest_files"],
        "group_count": 13,
        "profile_count": 23,
        "state_model_count": 17,
        "digest_consistent": True,
        "catalog_digest": expected["catalog_digest"],
    }


def test_resource_loader_rejects_duplicate_json_object_names(tmp_path, monkeypatch):
    package_name = _write_resource_package(tmp_path, {})
    package = tmp_path / package_name
    (package / "broken.model.json").write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    with pytest.raises(CatalogResourceError, match="duplicate JSON object name"):
        load_catalog_from_package(package_name)


def test_resource_loader_reports_missing_packages_without_filesystem_fallback():
    with pytest.raises(CatalogResourceError, match="cannot enumerate package"):
        load_catalog_from_package("kiron_common.does_not_exist")
