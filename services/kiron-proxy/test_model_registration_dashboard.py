from __future__ import annotations

import os
from pathlib import Path
import sys

from starlette.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as app_module  # noqa: E402

from kiron_common.local_model_registry import (  # noqa: E402
    HuggingFaceLocalValidator,
    LocalLoaderMetadata,
    ModelRegistrationService,
    OllamaLocalValidator,
    RegistrationValidators,
    RuntimeModelRegistry,
)
from kiron_common.local_model_registry.cli import run_cli  # noqa: E402
from kiron_common.local_model_registry.composition import (  # noqa: E402
    build_model_registration_service,
)


def _service(
    path: Path,
    *,
    names: tuple[str, ...],
    model_root: Path | None = None,
) -> ModelRegistrationService:
    return ModelRegistrationService(
        RuntimeModelRegistry(path),
        RegistrationValidators(
            ollama=OllamaLocalValidator(
                list_models=lambda: {
                    "models": [{"name": name} for name in names]
                },
                show_model=lambda name: {"model": name, "details": {}},
            ),
            huggingface=HuggingFaceLocalValidator(
                inspect_loader=lambda path, _loader: LocalLoaderMetadata(
                    display_name=path.name
                ),
                model_root=model_root or path.parent,
            ),
        ),
    )


class _LocalOllama:
    def __init__(self, *names: str) -> None:
        self.names = names
        self.calls: list[tuple[str, str | None]] = []

    def list_models(self) -> object:
        self.calls.append(("list", None))
        return {"models": [{"name": name} for name in self.names]}

    def show_model(self, name: str) -> object:
        self.calls.append(("show", name))
        return {"model": name, "details": {}}


def _composed_service(
    path: Path,
    *,
    names: tuple[str, ...] = (),
    model_root: Path | None = None,
) -> ModelRegistrationService:
    return build_model_registration_service(
        registry=RuntimeModelRegistry(path),
        ollama=_LocalOllama(*names),
        huggingface_model_root=model_root or path.parent,
    )


def _hf_model(path: Path, *, weights: bool = True) -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    if weights:
        (path / "model.safetensors").write_bytes(b"local-only-test-weights")
    return path


def test_dashboard_registration_is_basic_protected() -> None:
    response = TestClient(app_module.app).post(
        "/api/models/register",
        json={"provider": "ollama", "reference": "demo"},
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic ")


def test_dashboard_registration_candidates_are_basic_protected() -> None:
    response = TestClient(app_module.app).get(
        "/api/models/registration-candidates"
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic ")


def test_dashboard_lists_only_unknown_local_registration_candidates(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "local-models"
    model_root.mkdir()
    available_hf = _hf_model(model_root / "available-hf")
    registered_hf = _hf_model(model_root / "registered-hf")
    catalog_reference = app_module.MODEL_STATE_VIEW.for_backend(
        app_module.BackendType.OLLAMA
    )[0].input_names[0]
    service = _service(
        tmp_path / "registry.json",
        names=(
            catalog_reference,
            "qwen3.5:latest",
            "already:latest",
            "remote-only:cloud",
        ),
        model_root=model_root,
    )
    service.register_model(provider="ollama", reference="already")
    service.register_model(
        provider="huggingface",
        reference=str(registered_hf),
        loader="sentence_transformers",
    )
    previous = app_module.get_model_registration_service()
    app_module.set_model_registration_service(service)
    try:
        response = TestClient(app_module.app).get(
            "/api/models/registration-candidates",
            auth=("admin", "admin"),
        )
    finally:
        app_module.set_model_registration_service(previous)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "candidates": [
            {
                "provider": "huggingface",
                "reference": str(available_hf),
                "display_name": "available-hf",
            },
            {
                "provider": "ollama",
                "reference": "qwen3.5:latest",
                "display_name": "qwen3.5:latest",
            },
        ],
    }


def test_dashboard_json_framework_boundary_is_fastapi_422() -> None:
    client = TestClient(app_module.app)

    missing = client.post(
        "/api/models/register",
        auth=("admin", "admin"),
    )
    malformed = client.post(
        "/api/models/register",
        auth=("admin", "admin"),
        content=b'{"provider":',
        headers={"content-type": "application/json"},
    )

    assert missing.status_code == 422
    assert malformed.status_code == 422


def test_dashboard_and_cli_return_the_same_entry_and_domain_errors(
    tmp_path: Path,
) -> None:
    dashboard = _service(tmp_path / "dashboard.json", names=("Demo:Q4_0",))
    cli = _service(tmp_path / "cli.json", names=("Demo:Q4_0",))
    previous = app_module.get_model_registration_service()
    app_module.set_model_registration_service(dashboard)
    try:
        client = TestClient(app_module.app)
        response = client.post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={"provider": "ollama", "reference": "demo:q4_0"},
        )
        cli_code, cli_payload = run_cli(
            [
                "register",
                "--provider",
                "ollama",
                "--reference",
                "demo:q4_0",
            ],
            service=cli,
            effective_uid=0,
        )
        assert response.status_code == 201
        assert cli_code == 0
        assert response.json() == cli_payload

        missing_dashboard = _service(
            tmp_path / "missing-dashboard.json",
            names=(),
        )
        missing_cli = _service(tmp_path / "missing-cli.json", names=())
        app_module.set_model_registration_service(missing_dashboard)
        response = client.post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={"provider": "ollama", "reference": "missing"},
        )
        cli_code, cli_payload = run_cli(
            [
                "register",
                "--provider",
                "ollama",
                "--reference",
                "missing",
            ],
            service=missing_cli,
            effective_uid=0,
        )
        assert response.status_code == 404
        assert cli_code != 0
        assert response.json()["error"]["code"] == "model_not_found"
        assert cli_payload["error"]["code"] == "model_not_found"
    finally:
        app_module.set_model_registration_service(previous)


def test_dashboard_registration_persists_for_a_new_service_instance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    service = _service(path, names=("persisted:latest",))
    previous = app_module.get_model_registration_service()
    app_module.set_model_registration_service(service)
    try:
        response = TestClient(app_module.app).post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={"provider": "ollama", "reference": "persisted"},
        )
        assert response.status_code == 201
    finally:
        app_module.set_model_registration_service(previous)

    restarted = _service(path, names=("persisted:latest",))
    assert [entry.to_dict() for entry in restarted.list_models()] == [
        response.json()["model"]
    ]


def test_dashboard_and_cli_share_hf_canonicalization_and_domain_errors(
    tmp_path: Path,
) -> None:
    client = TestClient(app_module.app)
    previous = app_module.get_model_registration_service()
    valid = _hf_model(tmp_path / "valid-hf")
    invalid = _hf_model(tmp_path / "invalid-hf", weights=False)
    missing = tmp_path / "missing-hf"
    try:
        dashboard = _composed_service(tmp_path / "dashboard-valid.json")
        cli = _composed_service(tmp_path / "cli-valid.json")
        app_module.set_model_registration_service(dashboard)
        response = client.post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={
                "provider": "huggingface",
                "reference": str(valid),
                "loader": "sentence_transformers",
            },
        )
        cli_code, cli_payload = run_cli(
            [
                "register",
                "--provider",
                "huggingface",
                "--reference",
                str(valid),
                "--loader",
                "sentence_transformers",
            ],
            service=cli,
            effective_uid=0,
        )
        assert response.status_code == 201
        assert cli_code == 0
        assert response.json() == cli_payload

        duplicate_response = client.post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={
                "provider": "huggingface",
                "reference": str(valid),
                "loader": "sentence_transformers",
            },
        )
        duplicate_code, duplicate_payload = run_cli(
            [
                "register",
                "--provider",
                "huggingface",
                "--reference",
                str(valid),
                "--loader",
                "sentence_transformers",
            ],
            service=cli,
            effective_uid=0,
        )
        assert duplicate_response.status_code == 409
        assert duplicate_code == 3
        assert duplicate_response.json()["error"] == duplicate_payload["error"]

        for index, (reference, expected_status, expected_code) in enumerate(
            (
                (missing, 404, "model_not_found"),
                (invalid, 422, "loader_metadata_invalid"),
            )
        ):
            dashboard_error = _composed_service(
                tmp_path / f"dashboard-error-{index}.json"
            )
            cli_error = _composed_service(tmp_path / f"cli-error-{index}.json")
            app_module.set_model_registration_service(dashboard_error)
            response = client.post(
                "/api/models/register",
                auth=("admin", "admin"),
                json={
                    "provider": "huggingface",
                    "reference": str(reference),
                    "loader": "sentence_transformers",
                },
            )
            cli_code, cli_payload = run_cli(
                [
                    "register",
                    "--provider",
                    "huggingface",
                    "--reference",
                    str(reference),
                    "--loader",
                    "sentence_transformers",
                ],
                service=cli_error,
                effective_uid=0,
            )
            assert response.status_code == expected_status
            assert cli_code == 3
            assert response.json()["error"] == cli_payload["error"]
            assert response.json()["error"]["code"] == expected_code
    finally:
        app_module.set_model_registration_service(previous)


def test_shared_registry_survives_new_dashboard_and_cli_service_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared-registry.json"
    valid = _hf_model(tmp_path / "shared-hf")
    previous = app_module.get_model_registration_service()
    try:
        dashboard = _composed_service(path, names=("shared:latest",))
        app_module.set_model_registration_service(dashboard)
        response = TestClient(app_module.app).post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={"provider": "ollama", "reference": "shared"},
        )
        assert response.status_code == 201

        restarted_cli = _composed_service(path, names=("shared:latest",))
        code, listed = run_cli(
            ["list"],
            service=restarted_cli,
            effective_uid=0,
        )
        assert code == 0
        assert listed["models"] == [response.json()["model"]]

        code, registered_hf = run_cli(
            [
                "register",
                "--provider",
                "huggingface",
                "--reference",
                str(valid),
                "--loader",
                "cross_encoder",
            ],
            service=restarted_cli,
            effective_uid=0,
        )
        assert code == 0

        restarted_dashboard = _composed_service(path, names=("shared:latest",))
        app_module.set_model_registration_service(restarted_dashboard)
        assert [
            entry.to_dict()
            for entry in app_module.get_model_registration_service().list_models()
        ] == sorted(
            (response.json()["model"], registered_hf["model"]),
            key=lambda entry: entry["id"],
        )
    finally:
        app_module.set_model_registration_service(previous)


def test_corrupt_registry_is_fail_closed_with_surface_parity(
    tmp_path: Path,
) -> None:
    dashboard_path = tmp_path / "corrupt-dashboard.json"
    cli_path = tmp_path / "corrupt-cli.json"
    for path in (dashboard_path, cli_path):
        path.write_bytes(b"not-json")
        path.chmod(0o640)

    dashboard = _composed_service(dashboard_path, names=("local:latest",))
    cli = _composed_service(cli_path, names=("local:latest",))
    previous = app_module.get_model_registration_service()
    app_module.set_model_registration_service(dashboard)
    try:
        response = TestClient(app_module.app).post(
            "/api/models/register",
            auth=("admin", "admin"),
            json={"provider": "ollama", "reference": "local"},
        )
        cli_code, cli_payload = run_cli(
            ["register", "--provider", "ollama", "--reference", "local"],
            service=cli,
            effective_uid=0,
        )
        assert response.status_code == 500
        assert cli_code == 3
        assert response.json()["error"] == cli_payload["error"]
        assert response.json()["error"]["code"] == "registry_corrupt"
    finally:
        app_module.set_model_registration_service(previous)
