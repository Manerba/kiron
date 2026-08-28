from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import socket
import subprocess
import urllib.request

import pytest

from kiron_common.local_model_registry import (
    DEFAULT_HUGGINGFACE_MODEL_ROOT,
    DuplicateModelError,
    HuggingFaceLocalValidator,
    InvalidLoaderError,
    InvalidReferenceError,
    LoaderMetadataError,
    LocalLoaderMetadata,
    LocalModelNotFoundError,
    LocalModelProvider,
    LocalValidationError,
    ModelRegistrationService,
    OllamaLocalValidator,
    RegistrationValidators,
    RegistryEntry,
    RuntimeModelRegistry,
    list_models,
    read_model,
    register_model,
)
from kiron_common.model_catalog import LoaderType


def _validators(
    *,
    models: object | None = None,
    show=None,
    loader_probe=None,
    model_root: Path = DEFAULT_HUGGINGFACE_MODEL_ROOT,
) -> RegistrationValidators:
    listed = models if models is not None else {"models": []}
    return RegistrationValidators(
        ollama=OllamaLocalValidator(
            list_models=lambda: listed,
            show_model=show or (lambda _name: {"details": {}}),
        ),
        huggingface=HuggingFaceLocalValidator(
            inspect_loader=loader_probe
            or (lambda _path, _loader: LocalLoaderMetadata()),
            model_root=model_root,
        ),
    )


def test_registry_entry_is_immutable_minimal_and_stably_identified() -> None:
    first = RegistryEntry.create(
        provider=LocalModelProvider.OLLAMA,
        reference="example/model:latest",
        display_name="example/model:latest",
        loader=LoaderType.OLLAMA,
    )
    second = RegistryEntry.create(
        provider=LocalModelProvider.OLLAMA,
        reference="example/model:latest",
        display_name="A different presentation",
        loader=LoaderType.OLLAMA,
    )

    assert first.id == second.id
    assert first.id == (
        "local.464204f04e7d95aae1f3ab7cce6ad099"
        "4ef34f87d142b1c03b84827fbc7f6a76"
    )
    assert first.to_dict() == {
        "id": first.id,
        "provider": "ollama",
        "reference": "example/model:latest",
        "display_name": "example/model:latest",
        "loader": "ollama",
    }
    with pytest.raises(FrozenInstanceError):
        first.reference = "changed"  # type: ignore[misc]


def test_ollama_registers_only_an_exact_locally_listed_and_shown_model(
    tmp_path: Path,
) -> None:
    shown: list[str] = []

    def show(name: str) -> object:
        shown.append(name)
        return {"details": {"family": "llama"}}

    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    entry = register_model(
        registry,
        _validators(
            models={"models": [{"name": "example/model:Q4_0"}]},
            show=show,
        ),
        provider=" OLLAMA ",
        reference=" EXAMPLE/MODEL:q4_0 ",
    )

    assert entry.provider is LocalModelProvider.OLLAMA
    assert entry.reference == "example/model:Q4_0"
    assert entry.display_name == "example/model:Q4_0"
    assert entry.loader is LoaderType.OLLAMA
    assert shown == ["example/model:Q4_0"]
    assert list_models(registry) == (entry,)
    assert read_model(registry, entry.id) == entry


def test_ollama_missing_stops_before_show_and_does_not_persist(
    tmp_path: Path,
) -> None:
    shown: list[str] = []
    registry = RuntimeModelRegistry(tmp_path / "registry.json")

    with pytest.raises(LocalModelNotFoundError) as caught:
        register_model(
            registry,
            _validators(
                models={"models": [{"name": "other:latest"}]},
                show=lambda name: shown.append(name),
            ),
            provider="ollama",
            reference="missing",
        )

    assert caught.value.code == "model_not_found"
    assert shown == []
    assert list_models(registry) == ()


@pytest.mark.parametrize(
    "reference",
    (
        "https://registry.example/model",
        "http://127.0.0.1/model",
        "model name",
        "../model",
        "model@sha256:deadbeef",
    ),
)
def test_ollama_rejects_nonlocal_or_noncanonical_references_before_callbacks(
    tmp_path: Path,
    reference: str,
) -> None:
    calls: list[str] = []
    validators = RegistrationValidators(
        ollama=OllamaLocalValidator(
            list_models=lambda: calls.append("list"),
            show_model=lambda _name: calls.append("show"),
        ),
        huggingface=HuggingFaceLocalValidator(
            inspect_loader=lambda _path, _loader: LocalLoaderMetadata()
        ),
    )

    with pytest.raises(InvalidReferenceError):
        register_model(
            RuntimeModelRegistry(tmp_path / "registry.json"),
            validators,
            provider="ollama",
            reference=reference,
        )
    assert calls == []


def test_ollama_requires_closed_list_and_show_shapes(tmp_path: Path) -> None:
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    with pytest.raises(LocalValidationError):
        register_model(
            registry,
            _validators(models=[{"name": "model:latest"}]),
            provider="ollama",
            reference="model",
        )

    with pytest.raises(LocalValidationError):
        register_model(
            registry,
            _validators(
                models={"models": [{"name": "model:latest"}]},
                show=lambda _name: {"model": "different:latest"},
            ),
            provider="ollama",
            reference="model",
        )
    with pytest.raises(LocalValidationError):
        register_model(
            registry,
            _validators(
                models={"models": [{"name": "model:latest"}]},
                show=lambda _name: [],
            ),
            provider="ollama",
            reference="model",
        )


def test_huggingface_canonicalizes_local_directory_and_checks_loader_metadata(
    tmp_path: Path,
) -> None:
    model = tmp_path / "models" / "demo-model"
    model.mkdir(parents=True)
    observed: list[tuple[Path, LoaderType]] = []

    def inspect(path: Path, loader: LoaderType) -> LocalLoaderMetadata:
        observed.append((path, loader))
        return LocalLoaderMetadata(display_name="Demo model")

    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    entry = register_model(
        registry,
        _validators(loader_probe=inspect, model_root=model.parent),
        provider="huggingface",
        reference=f"{model.parent}/./demo-model",
        loader=" sentence_transformers ",
    )

    assert entry.provider is LocalModelProvider.HUGGINGFACE
    assert entry.reference == str(model.resolve())
    assert entry.display_name == "Demo model"
    assert entry.loader is LoaderType.SENTENCE_TRANSFORMERS
    assert observed == [(model.resolve(), LoaderType.SENTENCE_TRANSFORMERS)]


def test_huggingface_uses_directory_name_as_minimal_display_fallback(
    tmp_path: Path,
) -> None:
    model = tmp_path / "local-model"
    model.mkdir()
    entry = register_model(
        RuntimeModelRegistry(tmp_path / "registry.json"),
        _validators(model_root=tmp_path),
        provider="huggingface",
        reference=str(model),
        loader=LoaderType.CROSS_ENCODER,
    )
    assert entry.display_name == "local-model"


@pytest.mark.parametrize("kind", ("missing", "symlink", "file"))
def test_huggingface_rejects_missing_symlink_and_non_directory_before_loader(
    tmp_path: Path,
    kind: str,
) -> None:
    target = tmp_path / "model"
    if kind == "file":
        target.write_text("not a model", encoding="utf-8")
    elif kind == "symlink":
        real = tmp_path / "real"
        real.mkdir()
        target.symlink_to(real, target_is_directory=True)
    calls: list[Path] = []
    validator = _validators(
        loader_probe=lambda path, _loader: (
            calls.append(path),
            LocalLoaderMetadata(),
        )[1],
        model_root=tmp_path,
    )

    error = LocalModelNotFoundError if kind == "missing" else InvalidReferenceError
    with pytest.raises(error):
        register_model(
            RuntimeModelRegistry(tmp_path / "registry.json"),
            validator,
            provider="huggingface",
            reference=str(target),
            loader="cross_encoder",
        )
    assert calls == []


def test_huggingface_rejects_relative_url_ollama_loader_and_loader_failure(
    tmp_path: Path,
) -> None:
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    for reference in (
        "relative/model",
        "https://example.invalid/model",
        str(tmp_path / "bad\nmodel"),
    ):
        with pytest.raises(InvalidReferenceError):
            register_model(
                registry,
                _validators(model_root=tmp_path),
                provider="huggingface",
                reference=reference,
                loader="cross_encoder",
            )

    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(InvalidLoaderError):
        register_model(
            registry,
            _validators(model_root=tmp_path),
            provider="huggingface",
            reference=str(model),
            loader="ollama",
        )
    with pytest.raises(LoaderMetadataError) as caught:
        register_model(
            registry,
            _validators(
                loader_probe=lambda _path, _loader: (_ for _ in ()).throw(
                    RuntimeError("credential-canary")
                ),
                model_root=tmp_path,
            ),
            provider="huggingface",
            reference=str(model),
            loader="cross_encoder",
        )
    assert "credential-canary" not in str(caught.value)
    assert list_models(registry) == ()


def test_duplicate_semantics_use_provider_and_canonical_reference(
    tmp_path: Path,
) -> None:
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    validators = _validators(
        models={"models": [{"name": "demo:latest"}]}
    )
    first = register_model(
        registry,
        validators,
        provider="ollama",
        reference="DEMO",
    )
    with pytest.raises(DuplicateModelError) as caught:
        register_model(
            registry,
            validators,
            provider="ollama",
            reference="demo:latest",
        )
    assert caught.value.code == "duplicate_model"
    assert list_models(registry) == (first,)


def test_bound_service_and_function_share_the_exact_use_case(tmp_path: Path) -> None:
    validators = _validators(
        models={"models": [
            {"name": "functional:latest"},
            {"name": "bound:latest"},
        ]}
    )
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    functional = register_model(
        registry,
        validators,
        provider="ollama",
        reference="functional",
    )
    service = ModelRegistrationService(registry, validators)
    bound = service.register_model(
        provider="ollama",
        reference="bound",
    )

    assert service.list_models() == list_models(registry)
    assert service.read_model(functional.id) == functional
    assert service.read_model(bound.id) == bound


def test_registration_candidates_are_local_sorted_and_exclude_registered(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "local-models"
    model_root.mkdir()
    registered_hf = model_root / "registered-hf"
    registered_hf.mkdir()
    available_hf = model_root / "available-hf"
    available_hf.mkdir()
    (model_root / ".uploading").mkdir()
    (model_root / "not-a-model").write_text("file", encoding="utf-8")
    (model_root / "linked-model").symlink_to(available_hf, target_is_directory=True)

    validators = _validators(
        models={
            "models": [
                {"name": "Zulu:Q4_0"},
                {"name": "registered:latest"},
                {"name": "alpha:latest"},
                {"name": "remote:cloud"},
            ]
        },
        model_root=model_root,
    )
    service = ModelRegistrationService(
        RuntimeModelRegistry(tmp_path / "registry.json"),
        validators,
    )
    service.register_model(provider="ollama", reference="registered")
    service.register_model(
        provider="huggingface",
        reference=str(registered_hf),
        loader="cross_encoder",
    )

    assert [candidate.to_dict() for candidate in service.list_candidates()] == [
        {
            "provider": "huggingface",
            "reference": str(available_hf),
            "display_name": "available-hf",
        },
        {
            "provider": "ollama",
            "reference": "alpha:latest",
            "display_name": "alpha:latest",
        },
        {
            "provider": "ollama",
            "reference": "Zulu:Q4_0",
            "display_name": "Zulu:Q4_0",
        },
    ]


def test_huggingface_registration_is_limited_to_direct_model_roots(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "local-models"
    model_root.mkdir()
    direct = model_root / "direct"
    direct.mkdir()
    nested = direct / "nested"
    nested.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service = ModelRegistrationService(
        RuntimeModelRegistry(tmp_path / "registry.json"),
        _validators(model_root=model_root),
    )

    for forbidden in (model_root, nested, outside):
        with pytest.raises(InvalidReferenceError):
            service.register_model(
                provider="huggingface",
                reference=str(forbidden),
                loader="sentence_transformers",
            )

    registered = service.register_model(
        provider="huggingface",
        reference=str(direct),
        loader="sentence_transformers",
    )
    assert registered.reference == str(direct)


def test_candidate_discovery_fails_closed_for_malformed_local_inventory(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "local-models"
    model_root.mkdir()
    service = ModelRegistrationService(
        RuntimeModelRegistry(tmp_path / "registry.json"),
        _validators(
            models={"models": [{"name": "duplicate"}, {"name": "DUPLICATE:latest"}]},
            model_root=model_root,
        ),
    )

    with pytest.raises(LocalValidationError):
        service.list_candidates()


def test_registration_executes_no_network_download_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden: list[str] = []

    def fail_network(*_args, **_kwargs):
        forbidden.append("network")
        raise AssertionError("network use")

    def fail_process(*_args, **_kwargs):
        forbidden.append("process")
        raise AssertionError("process use")

    monkeypatch.setattr(socket, "socket", fail_network)
    monkeypatch.setattr(socket, "getaddrinfo", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr(subprocess, "run", fail_process)
    monkeypatch.setattr(subprocess, "Popen", fail_process)
    model = tmp_path / "model"
    model.mkdir()
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    validators = _validators(
        models={"models": [{"name": "local:latest"}]},
        model_root=tmp_path,
    )

    register_model(
        registry,
        validators,
        provider="ollama",
        reference="local",
    )
    register_model(
        registry,
        validators,
        provider="huggingface",
        reference=str(model),
        loader="sentence_transformers",
    )
    assert forbidden == []
