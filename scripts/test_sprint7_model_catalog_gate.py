"""Sprint-7 offline Catalog, architecture, and pre-mutation gates."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMON_SOURCE = ROOT / "services" / "kiron-common"
MANIFEST_ROOT = (
    COMMON_SOURCE / "kiron_common" / "model_catalog" / "manifests"
)
VALIDATOR_PATH = ROOT / "scripts" / "validate-model-catalog.py"
EXPECTED_DIGEST = (
    "sha256:c91229d7ea472b49d87f6344dbfb640fc760f43e8cace398421d5b364452e6f6"
)

if str(COMMON_SOURCE) not in sys.path:
    sys.path.insert(0, str(COMMON_SOURCE))

from kiron_common.embedding_registry import build_embedding_registry  # noqa: E402
from kiron_common.local_model_registry import (  # noqa: E402
    LocalModelProvider,
    RegistryEntry,
)
from kiron_common.model_catalog import (  # noqa: E402
    BackendType,
    LoaderType,
    ModelEndpoint,
    ModelTask,
    load_catalog,
    load_catalog_from_package,
)
from kiron_common.model_state import (  # noqa: E402
    BackendRuntimeSnapshot,
    HuggingFaceRevision,
    LocalModelInventory,
    RuntimeInventory,
    RuntimeState,
    build_model_state_view,
)


def _load_python_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_python_file("_sprint7_catalog_validator", VALIDATOR_PATH)


def _production_documents() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(MANIFEST_ROOT.glob("*.model.json"))
    }


def _write_resource_package(
    tmp_path: Path,
    package_name: str,
    documents: dict[str, dict],
) -> str:
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, document in documents.items():
        (package / name).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    importlib.invalidate_caches()
    return package_name


def _projection_modules(prefix: str):
    embeddings = _load_python_file(
        f"_{prefix}_embeddings_catalog_view",
        ROOT / "services" / "kiron-embeddings" / "catalog_view.py",
    )
    deberta = _load_python_file(
        f"_{prefix}_deberta_catalog_view",
        ROOT / "services" / "kiron-deberta" / "catalog_view.py",
    )
    routing = _load_python_file(
        f"_{prefix}_proxy_routing_catalog",
        ROOT / "services" / "kiron-proxy" / "routing_catalog.py",
    )
    discovery = _load_python_file(
        f"_{prefix}_proxy_model_discovery",
        ROOT / "services" / "kiron-proxy" / "model_discovery.py",
    )
    return embeddings, deberta, routing, discovery


def _never_load(_model: object) -> object:
    raise AssertionError("offline Catalog test invoked a model loader")


def test_operational_validator_is_stable_complete_and_read_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("validator attempted a write or network access")

    for method in (
        "mkdir",
        "touch",
        "write_bytes",
        "write_text",
        "rename",
        "replace",
        "unlink",
    ):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)

    summary = VALIDATOR.validate_catalog_architecture()

    assert summary["status"] == "ok"
    assert summary["catalog_digest"] == EXPECTED_DIGEST
    assert summary["counts"] == {
        "catalog_groups": 13,
        "catalog_profiles": 23,
        "embedding_registry_groups": 8,
        "embedding_registry_profiles": 13,
        "model_state_definitions": 17,
        "embedding_service_models": 6,
        "deberta_service_models": 5,
        "proxy_routes": 19,
    }
    assert len(summary["manifest_files"]) == len(summary["model_matrix"]) == 13
    assert summary["proxy_endpoint_routes"] == {
        "/api/embed": 7,
        "/api/embed_late": 1,
        "/api/embed_colbert": 1,
        "/api/rerank": 5,
        "/api/score": 5,
    }


def test_operational_validator_cli_json_is_byte_stable():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    command = [sys.executable, str(VALIDATOR_PATH), "--json"]
    first = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["catalog_digest"] == EXPECTED_DIGEST
    assert payload["counts"]["proxy_routes"] == 19


def _mutate_unknown_loader(documents: dict[str, dict]) -> None:
    documents["mxbai-embed-large.model.json"]["deployments"][0]["loader"][
        "type"
    ] = "unknown_loader"


def _mutate_incompatible_loader(documents: dict[str, dict]) -> None:
    documents["mxbai-embed-large.model.json"]["deployments"][0]["loader"][
        "type"
    ] = "cross_encoder"


def _mutate_unknown_backend(documents: dict[str, dict]) -> None:
    documents["mxbai-embed-large.model.json"]["deployments"][0]["backend"][
        "type"
    ] = "unknown_backend"


def _mutate_unknown_endpoint(documents: dict[str, dict]) -> None:
    documents["mxbai-embed-large.model.json"]["profiles"][0][
        "endpoint"
    ] = "/api/unknown"


def _mutate_invalid_default(documents: dict[str, dict]) -> None:
    documents["ms-marco-minilm-l-6-v2.model.json"]["request_defaults"][0][
        "profile_id"
    ] = "missing-profile"


def _mutate_invalid_reference(documents: dict[str, dict]) -> None:
    documents["mxbai-embed-large.model.json"]["profiles"][0][
        "deployment_id"
    ] = "missing-deployment"


@pytest.mark.parametrize(
    "case,mutator",
    (
        ("unknown_loader", _mutate_unknown_loader),
        ("incompatible_loader", _mutate_incompatible_loader),
        ("unknown_backend", _mutate_unknown_backend),
        ("unknown_endpoint", _mutate_unknown_endpoint),
        ("invalid_default", _mutate_invalid_default),
        ("invalid_reference", _mutate_invalid_reference),
    ),
)
def test_validator_fails_closed_for_invalid_cross_projection_contracts(
    tmp_path,
    monkeypatch,
    case,
    mutator,
):
    documents = _production_documents()
    mutator(documents)
    package = _write_resource_package(
        tmp_path,
        f"invalid_catalog_{case}",
        documents,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(Exception):
        VALIDATOR.validate_catalog_architecture(
            manifest_package=package,
            enforce_production_golden=False,
        )


def _new_standard_embedding_manifest() -> dict:
    source = _production_documents()["mxbai-embed-large.model.json"]
    document = copy.deepcopy(source)
    canonical = "fixture-standard-embed:latest"
    alias = "fixture-standard-embed"
    repository = "fixture.invalid/kiron-standard-embed"
    revision = "1" * 40
    weight_sha = "2" * 64
    deployment = document["deployments"][0]
    deployment["id"] = "kiron-fixture-standard-dense-v1.deployment"
    deployment["backend"]["parameters"]["model_name"] = alias
    deployment["artifact"]["repository"] = repository
    deployment["artifact"]["revision"] = revision
    deployment["artifact"]["weights"][0]["sha256"] = weight_sha
    profile = document["profiles"][0]
    profile["id"] = "kiron-fixture-standard-dense-v1"
    profile["deployment_id"] = deployment["id"]
    profile["metadata"]["pipeline"]["tokenizer"] = {
        "repository": repository,
        "revision": revision,
    }
    profile["metadata"]["verification"]["evidence"] = [
        f"hf-readme:{repository}@{revision}",
        f"local-weight-sha256:{weight_sha}",
        "runtime-pipeline:fixture-sentence-transformers-cls-fp32-l2",
    ]
    document.update(
        canonical_model_id=canonical,
        aliases=[alias],
        deployments=[deployment],
        profiles=[profile],
        request_defaults=[],
    )
    return document


def test_manifest_only_add_model_flows_through_every_pure_projection(
    tmp_path,
    monkeypatch,
):
    production_digest = load_catalog().catalog_digest
    documents = _production_documents()
    documents["fixture-standard-embed.model.json"] = (
        _new_standard_embedding_manifest()
    )
    package = _write_resource_package(
        tmp_path,
        "manifest_only_add_model_resources",
        documents,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    catalog = load_catalog_from_package(package)
    registry = build_embedding_registry(catalog)
    state_view = build_model_state_view(catalog)
    embeddings, deberta, routing, discovery = _projection_modules("add_model")
    embedding_view = embeddings.build_embedding_service_view(
        catalog,
        {
            LoaderType.SENTENCE_TRANSFORMERS: _never_load,
            LoaderType.TRANSFORMERS_LAST_TOKEN: _never_load,
            LoaderType.COLBERT_XMOD: _never_load,
        },
    )
    deberta_view = deberta.build_deberta_service_view(
        catalog,
        {
            LoaderType.CROSS_ENCODER: _never_load,
            LoaderType.MANKEI_LAST_TOKEN: _never_load,
        },
    )
    routing_view = routing.build_proxy_routing_view(catalog)

    canonical = "fixture-standard-embed:latest"
    alias = "fixture-standard-embed"
    repository = "fixture.invalid/kiron-standard-embed"
    revision = "1" * 40
    assert registry.resolve(canonical) is registry.resolve(alias)
    assert embedding_view.resolve(canonical) is embedding_view.resolve(alias)
    service_model = embedding_view.resolve(alias, ModelEndpoint.EMBED)
    assert service_model is not None
    assert service_model.loader_type is LoaderType.SENTENCE_TRANSFORMERS
    assert service_model.model_name == alias
    assert deberta_view.resolve(alias, ModelEndpoint.RERANK) is None

    canonical_route = routing_view.resolve(canonical, ModelEndpoint.EMBED)
    alias_route = routing_view.resolve(alias, ModelEndpoint.EMBED)
    assert canonical_route is alias_route
    assert canonical_route is not None
    assert canonical_route.backend is BackendType.KIRON_EMBEDDINGS
    assert canonical_route.backend_model_name == alias
    assert canonical_route.endpoint is ModelEndpoint.EMBED
    assert routing_view.resolve(alias, ModelEndpoint.RERANK) is None
    assert routing_view.resolve("unknown-fixture-model", ModelEndpoint.EMBED) is None

    local = LocalModelInventory(
        huggingface_revisions=frozenset(
            (HuggingFaceRevision(repository, revision),)
        )
    )
    runtime = RuntimeInventory(
        {
            BackendType.KIRON_EMBEDDINGS: BackendRuntimeSnapshot(
                known=True,
                loaded_names=frozenset(),
            )
        }
    )
    state = next(
        item
        for item in state_view.states(local, runtime)
        if item.definition.backend_model_name == alias
    )
    assert state.configured is True
    assert state.installed is True
    assert state.runtime_state is RuntimeState.UNLOADED
    assert load_catalog().catalog_digest == production_digest == EXPECTED_DIGEST
    assert not any(
        "fixture-standard-embed" in path.read_text(encoding="utf-8")
        for path in (
            ROOT / "services" / "kiron-embeddings" / "main.py",
            ROOT / "services" / "kiron-deberta" / "main.py",
            ROOT / "services" / "kiron-proxy" / "proxy.py",
        )
    )


def _assignment_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    return names[0] if len(names) == 1 else None


def _enclosing_scope(tree: ast.AST, target: ast.AST) -> str:
    parents = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    node = target
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
    return "<module>"


def test_ast_gate_keeps_managed_registrations_out_of_service_entrypoints():
    forbidden_builders = {
        "load_catalog",
        "load_catalog_from_package",
        "ModelCatalog",
        "EmbeddingProfileRegistry",
        "build_embedding_registry",
        "build_model_state_view",
    }
    dead_exports = {
        "EMBED_SERVICE_MODELS",
        "COLBERT_MODELS",
        "OLLAMA_EMBED_MODELS",
    }
    entrypoints = (
        ROOT / "services" / "kiron-embeddings" / "main.py",
        ROOT / "services" / "kiron-deberta" / "main.py",
        ROOT / "services" / "kiron-proxy" / "proxy.py",
        ROOT / "services" / "kiron-proxy" / "vram_lease.py",
    )
    registration_fields = {
        "canonical_model_id",
        "aliases",
        "deployments",
        "profiles",
        "request_defaults",
        "backend",
        "loader",
    }
    for path in entrypoints:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assigned = {
            name
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            if (name := _assignment_name(node)) is not None
        }
        assert assigned.isdisjoint(dead_exports), path
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            literal_keys = {
                child.value
                for child in ast.walk(value)
                if isinstance(child, ast.Constant)
                and type(child.value) is str
                and child.value in registration_fields
            }
            assert not literal_keys, (path, node.lineno, sorted(literal_keys))
            for call in (
                child for child in ast.walk(value) if isinstance(child, ast.Call)
            ):
                if isinstance(call.func, ast.Name):
                    assert call.func.id not in forbidden_builders, (
                        path,
                        node.lineno,
                        call.func.id,
                    )

    vram_tree = ast.parse(
        (ROOT / "services" / "kiron-proxy" / "vram_lease.py").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        or node.name != "apply_ollama_embed_fallback"
        for node in ast.walk(vram_tree)
    )


def test_ast_gate_names_only_the_generic_benchmark_inventory_exceptions():
    app_path = ROOT / "services" / "kiron-proxy" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    exceptions = {
        "_STATIC_BENCHMARKS": {"_fetch_one"},
        "_OLLAMA_TO_HF_MAP": {"_fetch_one"},
        "_OLLAMA_TO_EVALPLUS_MAP": {"_find_evalplus_match"},
    }
    module_assignments = {
        name: node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        if (name := _assignment_name(node)) is not None
    }
    assert set(exceptions) <= set(module_assignments)
    for name, allowed_scopes in exceptions.items():
        assignment = module_assignments[name]
        assert isinstance(assignment.value, ast.Dict)
        scopes = {
            _enclosing_scope(tree, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
        }
        assert scopes == allowed_scopes

    forbidden_consumers = (
        "routing_catalog.py",
        "model_discovery.py",
        "proxy.py",
        "vram_lease.py",
    )
    for filename in forbidden_consumers:
        consumer_tree = ast.parse(
            (ROOT / "services" / "kiron-proxy" / filename).read_text(
                encoding="utf-8"
            )
        )
        referenced = {
            node.id for node in ast.walk(consumer_tree) if isinstance(node, ast.Name)
        }
        assert referenced.isdisjoint(exceptions), filename


def test_repo_python_has_no_second_declarative_managed_model_registry():
    allowed_literal_metadata = {
        (
            ROOT / "services" / "kiron-proxy" / "app.py",
            "_STATIC_BENCHMARKS",
        ),
        (
            ROOT / "services" / "kiron-proxy" / "app.py",
            "_OLLAMA_TO_HF_MAP",
        ),
        (
            ROOT / "services" / "kiron-proxy" / "app.py",
            "_OLLAMA_TO_EVALPLUS_MAP",
        ),
    }
    production_files = sorted(
        {
            *(
                path
                for path in (ROOT / "services").rglob("*.py")
                if "tests" not in path.parts
                and not path.name.startswith("test_")
                and "__pycache__" not in path.parts
            ),
            *(
                path
                for path in (ROOT / "scripts").glob("*.py")
                if not path.name.startswith("test_")
            ),
            *(ROOT / "docs").glob("*.py"),
        }
    )
    registration_fields = {
        "canonical_model_id",
        "aliases",
        "deployments",
        "profiles",
        "request_defaults",
        "backend",
        "artifact",
        "routes",
        "loader",
    }
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            name = _assignment_name(node)
            value = node.value
            if name is None or value is None:
                continue
            if (path, name) in allowed_literal_metadata:
                continue

            literal_keys = {
                child.value
                for child in ast.walk(value)
                if isinstance(child, ast.Constant)
                and type(child.value) is str
                and child.value in registration_fields
            }
            assert len(literal_keys) < 3, (
                path,
                node.lineno,
                name,
                sorted(literal_keys),
            )

            if isinstance(value, ast.Dict) and len(value.keys) >= 2:
                nested_keys = {
                    child.value
                    for child in ast.walk(value)
                    if isinstance(child, ast.Constant)
                    and type(child.value) is str
                    and child.value
                    in {"hf", "repository", "revision", "model_name"}
                }
                assert not nested_keys, (
                    path,
                    node.lineno,
                    name,
                    sorted(nested_keys),
                )

            upper_name = name.upper()
            model_registry_name = (
                upper_name == "MODELS"
                or upper_name.endswith("_MODELS")
                or ("MODEL" in upper_name and "REGISTR" in upper_name)
                or ("MODEL" in upper_name and "DEFAULT" in upper_name)
            )
            if model_registry_name and isinstance(
                value,
                (ast.Dict, ast.List, ast.Set, ast.Tuple),
            ):
                strings = {
                    child.value
                    for child in ast.walk(value)
                    if isinstance(child, ast.Constant)
                    and type(child.value) is str
                    and child.value
                }
                assert not strings, (path, node.lineno, name, sorted(strings))

            for call in (
                child for child in ast.walk(value) if isinstance(child, ast.Call)
            ):
                called = (
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else call.func.attr
                    if isinstance(call.func, ast.Attribute)
                    else None
                )
                if called not in {"load_catalog", "load_catalog_from_package"}:
                    continue
                assert (
                    path
                    == ROOT
                    / "services"
                    / "kiron-common"
                    / "kiron_common"
                    / "embedding_registry.py"
                    and name == "MODEL_CATALOG"
                    and called == "load_catalog"
                ), (path, node.lineno, name, called)


def test_every_manifest_is_one_source_row_and_all_projections_capture_it():
    summary = VALIDATOR.validate_catalog_architecture()
    resource_names = sorted(path.name for path in MANIFEST_ROOT.glob("*.model.json"))
    matrix_names = sorted(row["manifest"] for row in summary["model_matrix"])
    matrix_models = [row["canonical_model_id"] for row in summary["model_matrix"]]
    assert resource_names == summary["manifest_files"] == matrix_names
    assert len(matrix_models) == len(set(matrix_models)) == len(resource_names)
    assert summary["counts"]["catalog_profiles"] == 23
    assert summary["counts"]["model_state_definitions"] == 17
    assert summary["counts"]["proxy_routes"] == 19


def test_registered_ollama_inventory_stays_outside_managed_catalog_metadata():
    catalog = load_catalog()
    state_view = build_model_state_view(catalog)
    _embeddings, _deberta, _routing, discovery = _projection_modules(
        "generic_inventory"
    )
    payload = discovery.build_local_models_payload(
        state_view=state_view,
        huggingface_revisions=frozenset(),
        ollama_tag_rows=[
            {
                "name": "qwen3:8b",
                "size": 5_000_000_000,
                "details": {"family": "qwen", "parameter_size": "8B"},
            }
        ],
        ollama_ps={"models": []},
        embedding_health={"status": "ok", "loaded_models": []},
        embedding_reachable=True,
        deberta_health={"status": "ok", "loaded_models": []},
        deberta_reachable=True,
        registrations=(
            RegistryEntry.create(
                provider=LocalModelProvider.OLLAMA,
                reference="qwen3:8b",
                display_name="qwen3:8b",
                loader=LoaderType.OLLAMA,
            ),
        ),
        ollama_show_by_name={},
    )
    row = next(item for item in payload["models"] if item["name"] == "qwen3:8b")
    assert row["configured"] is False
    assert row["installed"] is True
    assert row["backend"] == BackendType.OLLAMA.value
    assert row["source"] == "ollama"
    assert row["locally_registered"] is True
    assert row["catalog_managed"] is False
    assert "canonical_model_id" not in row
    assert "kiron_capabilities" not in row
    assert state_view.resolve("qwen3:8b", BackendType.OLLAMA) is None


def test_non_registry_bge_m3_comment_is_preserved():
    text = (ROOT / "services" / "kiron-embeddings" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "xlm-roberta, BGE-M3" in text


MUTATING_COMMANDS = (
    "docker",
    "rsync",
    "mkdir",
    "chown",
    "chmod",
    "cp",
    "systemctl",
    "rm",
    "mv",
    "find",
)


def _write_wrapper(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _instrumented_env(tmp_path: Path, *, validator_fails: bool) -> tuple[dict, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "mutations.log"
    for command in MUTATING_COMMANDS:
        _write_wrapper(
            bin_dir / command,
            f'printf "%s\\n" "{command} $*" >> "$KIRON_TEST_MUTATION_LOG"\n'
            "exit 97\n",
        )
    real_python = shutil.which("python3")
    assert real_python is not None
    if validator_fails:
        python_body = (
            'case "${1:-}" in\n'
            "  */validate-model-catalog.py) exit 42 ;;\n"
            "esac\n"
            f'exec "{real_python}" "$@"\n'
        )
    else:
        python_body = f'exec "{real_python}" "$@"\n'
    _write_wrapper(bin_dir / "python3", python_body)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["KIRON_TEST_MUTATION_LOG"] = str(log)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env, log


@pytest.mark.parametrize("script_name", ("deploy-local.sh", "setup-venvs.sh"))
def test_invalid_catalog_stops_scripts_before_any_mutation_or_service_stop(
    tmp_path,
    script_name,
):
    env, log = _instrumented_env(tmp_path, validator_fails=True)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Model-Catalog" in result.stderr
    assert not log.exists() or not log.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("script_name", ("deploy-local.sh", "setup-venvs.sh"))
def test_missing_required_group_stops_before_mutation_stop_or_venv_build(
    tmp_path,
    script_name,
):
    env, log = _instrumented_env(tmp_path, validator_fails=False)
    real_id = shutil.which("id")
    assert real_id is not None
    id_wrapper = Path(env["PATH"].split(":", 1)[0]) / "id"
    _write_wrapper(
        id_wrapper,
        'if [ "${1:-}" = "-nG" ] && [ "${2:-}" = "kiron-embeddings" ]; then\n'
        '  echo "kiron-embeddings kiron-models kiron-config video render"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{real_id}" "$@"\n',
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script_name)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "kiron-embeddings Gruppen falsch" in result.stderr
    assert "install-system-configs.sh" in result.stderr
    assert not log.exists() or not log.read_text(encoding="utf-8").strip()
