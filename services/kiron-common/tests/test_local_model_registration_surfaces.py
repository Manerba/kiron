from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

import ast
import json
from pathlib import Path

from kiron_common import local_ollama_api as ollama_api_module
from kiron_common.local_model_registry import (
    ModelRegistrationService,
    RuntimeModelRegistry,
)
from kiron_common.local_model_registry.cli import run_cli
from kiron_common.local_model_registry import cli as cli_module
from kiron_common.local_model_registry.composition import (
    build_model_registration_service,
)


class _LocalOllama:
    def __init__(self, *names: str) -> None:
        self.names = names
        self.calls: list[tuple[str, str | None]] = []

    def list_models(self) -> object:
        self.calls.append(("list", None))
        return {"models": [{"name": name, "digest": "a" * 64} for name in self.names]}

    def show_model(self, name: str) -> object:
        self.calls.append(("show", name))
        return {"model": name, "details": {"family": "unit"}}


def _hf_model(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        '{"architectures":["UnitModel"],"model_type":"unit"}\n',
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"local-test-weights")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def _service(
    path: Path,
    ollama: _LocalOllama,
    *,
    model_root: Path | None = None,
) -> ModelRegistrationService:
    return build_model_registration_service(
        registry=RuntimeModelRegistry(path),
        ollama=ollama,
        huggingface_model_root=model_root or path.parent,
    )


def test_shared_composition_registers_ollama_and_huggingface_locally(
    tmp_path: Path,
) -> None:
    ollama = _LocalOllama("Unit/Model:Q4_0")
    service = _service(tmp_path / "registry.json", ollama)
    hf_path = _hf_model(tmp_path / "hf-model")

    ollama_entry = service.register_model(
        runtime_provider="ollama",
        reference="unit/model:q4_0",
    )
    hf_entry = service.register_model(
        runtime_provider="kiron_embeddings",
        reference=str(hf_path),
        loader="sentence_transformers",
    )

    assert ollama_entry.reference == "Unit/Model:Q4_0"
    assert hf_entry.reference == str(hf_path)
    assert hf_entry.display_name == "hf-model"
    assert ollama.calls == [
        ("list", None),
        ("show", "Unit/Model:Q4_0"),
        ("list", None),
    ]
    assert service.list_models() == tuple(
        sorted((ollama_entry, hf_entry), key=lambda entry: entry.id)
    )


def test_local_ollama_adapter_is_fixed_loopback_inventory_only(
    monkeypatch,
) -> None:
    configurations: list[dict[str, object]] = []
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _Response:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self.payload

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            configurations.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(
            self,
            method: str,
            path: str,
            **kwargs: object,
        ) -> _Response:
            calls.append((method, path, kwargs))
            return _Response({"models": []} if path == "/api/tags" else {"model": "demo"})

    monkeypatch.setattr(ollama_api_module.httpx, "Client", _Client)
    adapter = ollama_api_module.LocalOllamaAPI()

    assert adapter.list_models() == {"models": []}
    assert adapter.show_model("demo") == {"model": "demo"}
    assert configurations == [
        {
            "base_url": "http://127.0.0.1:11435",
            "timeout": 10.0,
            "trust_env": False,
        },
        {
            "base_url": "http://127.0.0.1:11435",
            "timeout": 10.0,
            "trust_env": False,
        },
    ]
    assert calls == [
        ("GET", "/api/tags", {}),
        ("POST", "/api/show", {"json": {"model": "demo"}}),
    ]


def test_cli_register_and_list_are_stable_json_values_and_persistent(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    service = _service(registry_path, _LocalOllama("demo:latest"))

    code, registered = run_cli(
        ["register", "--runtime-provider", "ollama", "--reference", "DEMO"],
        service=service,
        effective_uid=0,
    )
    assert code == 0
    assert registered["status"] == "registered"
    assert registered["model"]["reference"] == "demo:latest"

    restarted = _service(registry_path, _LocalOllama("demo:latest"))
    code, listed = run_cli(
        ["list"],
        service=restarted,
        effective_uid=0,
    )
    assert code == 0
    assert listed == {"status": "ok", "models": [registered["model"]]}


def test_cli_rejects_non_root_before_touching_the_service(tmp_path: Path) -> None:
    class _ForbiddenService:
        def register_model(self, **_kwargs):
            raise AssertionError("service must not be called")

        def list_models(self):
            raise AssertionError("service must not be called")

    code, payload = run_cli(
        ["list"],
        service=_ForbiddenService(),  # type: ignore[arg-type]
        effective_uid=1000,
    )

    assert code != 0
    assert payload == {
        "status": "error",
        "error": {
            "code": "root_required",
            "message": "local model registry CLI requires root",
        },
    }


def test_console_entry_point_rejects_non_root_before_building_service(
    monkeypatch,
    capsys,
) -> None:
    def forbidden_composition():
        raise AssertionError("composition must not be built for non-root")

    monkeypatch.setattr(cli_module.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        cli_module,
        "build_model_registration_service",
        forbidden_composition,
    )

    assert cli_module.main(["list"]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "error": {
            "code": "root_required",
            "message": "local model registry CLI requires root",
        },
    }


def test_cli_has_no_http_token_or_application_auth_logic() -> None:
    cli_path = (
        Path(__file__).parents[1]
        / "kiron_common"
        / "local_model_registry"
        / "cli.py"
    )
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not ({"httpx", "requests", "urllib", "socket"} & imports)
    lowered = source.lower()
    for forbidden in ("authorization", "bearer", "token", "credential"):
        assert forbidden not in lowered
