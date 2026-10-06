"""Local provider composition shared by the dashboard and root CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path

from kiron_common.local_ollama_api import DEFAULT_OLLAMA_BASE_URL, LocalOllamaAPI
from kiron_common.model_catalog import LoaderType
from kiron_common.prism_runtime_policy import Policy, PolicyError

from .models import LocalLoaderMetadata, HF_LOADERS
from .gguf import DEFAULT_GGUF_MODEL_ROOT, GGUFLocalValidator, GGUFRegistrationPolicy
from .registry import RuntimeModelRegistry
from .service import ModelRegistrationService, RegistrationValidators
from .validators import (
    DEFAULT_HUGGINGFACE_MODEL_ROOT,
    HuggingFaceLocalValidator,
    OllamaLocalValidator,
)


_CONFIG_LIMIT = 1024 * 1024
DEFAULT_PRISM_POLICY_PATH = Path("/usr/lib/kiron/data/prism-runtime-policy.json")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
_TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    }
)


def _regular_local_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def inspect_huggingface_loader_metadata(
    path: Path,
    loader: LoaderType,
) -> LocalLoaderMetadata:
    """Inspect only files already present in one canonical local directory."""

    if loader not in HF_LOADERS:
        raise ValueError("unsupported local Hugging Face loader")
    config_path = path / "config.json"
    if not _regular_local_file(config_path):
        raise ValueError("config.json is missing")
    if config_path.stat().st_size > _CONFIG_LIMIT:
        raise ValueError("config.json is too large")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config.json must contain an object")

    files = tuple(item for item in path.iterdir() if _regular_local_file(item))
    if not any(item.name.endswith(_WEIGHT_SUFFIXES) for item in files):
        raise ValueError("local model weights are missing")
    if not any(item.name in _TOKENIZER_FILES for item in files):
        raise ValueError("local tokenizer metadata is missing")
    return LocalLoaderMetadata(display_name=path.name)


def build_model_registration_service(
    *,
    registry: RuntimeModelRegistry | None = None,
    ollama: object | None = None,
    inspect_loader: Callable[[Path, LoaderType], LocalLoaderMetadata] | None = None,
    huggingface_model_root: Path = DEFAULT_HUGGINGFACE_MODEL_ROOT,
    gguf_model_root: Path = DEFAULT_GGUF_MODEL_ROOT,
    gguf_profiles: Mapping[str, GGUFRegistrationPolicy] | None = None,
    prism_policy_path: Path | None = DEFAULT_PRISM_POLICY_PATH,
) -> ModelRegistrationService:
    """Build the sole local registration use-case composition."""

    # Explicit injected profiles serve isolated compositions. Production CLI and
    # dashboard use the same root-owned policy and parser as the controller.
    if gguf_profiles is None and prism_policy_path is not None:
        policy_path = Path(prism_policy_path)
        try:
            policy_path.lstat()
        except FileNotFoundError:
            pass  # Optional policy absent: GGUF remains unconfigured.
        else:
            policy = Policy.load(policy_path)
            if len(policy.artifact_roots) != 1:
                raise PolicyError("registration requires exactly one configured GGUF root")
            gguf_model_root = policy.artifact_roots[0]
            gguf_profiles = policy.registration_profiles()
    resolved_registry = registry if registry is not None else RuntimeModelRegistry()
    resolved_ollama = ollama if ollama is not None else LocalOllamaAPI()
    list_models = getattr(resolved_ollama, "list_models", None)
    show_model = getattr(resolved_ollama, "show_model", None)
    if not callable(list_models) or not callable(show_model):
        raise TypeError("ollama must provide local list and show operations")
    inspector = inspect_loader or inspect_huggingface_loader_metadata
    return ModelRegistrationService(
        resolved_registry,
        RegistrationValidators(
            gguf=GGUFLocalValidator(model_root=gguf_model_root, profiles=gguf_profiles),
            ollama=OllamaLocalValidator(
                list_models=list_models,
                show_model=show_model,
            ),
            huggingface=HuggingFaceLocalValidator(
                inspect_loader=inspector,
                model_root=huggingface_model_root,
            ),
        ),
    )


__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_HUGGINGFACE_MODEL_ROOT",
    "LocalOllamaAPI",
    "build_model_registration_service",
    "inspect_huggingface_loader_metadata",
]
