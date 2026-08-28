"""First-party SFT trainer entrypoint for kitt-worker."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping
import zipfile

import job_stubs


KIRON_SFT_TRAINER_ENTRYPOINT = "adr-0008.sft.v2"
RESOURCE_CATALOG_ENV = "KITT_SFT_RESOURCE_CATALOG"
RESOURCE_CATALOG_SCHEMA_VERSION = "kiron_sft_resources_v1"
REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
)
EXIT_CONFIG_INVALID = 2
EXIT_DEPENDENCY_MISSING = 3
EXIT_TRAINING_FAILED = 4
SUPPORTED_HYPERPARAMETERS = {
    "batch_size",
    "dataset_text_field",
    "eval_steps",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "learning_rate",
    "lora_alpha",
    "lora_dropout",
    "lora_r",
    "lr_scheduler_type",
    "max_batch_size",
    "max_context_tokens",
    "logging_steps",
    "micro_batch_size",
    "max_seq_length",
    "max_sequence_length",
    "max_steps",
    "num_train_epochs",
    "optimizer",
    "packing",
    "optim",
    "per_device_train_batch_size",
    "quantization_mode",
    "save_steps",
    "seed",
    "target_modules",
    "warmup_steps",
    "weight_decay",
}
HYPERPARAMETER_ALIASES = {
    "batch_size": "per_device_train_batch_size",
    "max_batch_size": "per_device_train_batch_size",
    "max_context_tokens": "max_seq_length",
    "max_sequence_length": "max_seq_length",
    "micro_batch_size": "per_device_train_batch_size",
    "optimizer": "optim",
}


@dataclass(frozen=True, slots=True)
class ResolvedResources:
    dataset_version_uid: str
    dataset_path: Path
    dataset_text_field: str
    dataset_sha256: str | None
    model_ref: str
    model_path: Path
    model_revision: str
    model_hash_sha256: str
    tokenizer_hash_sha256: str
    license_alignment_hash_sha256: str
    training_profile_hash_sha256: str
    lora: Mapping[str, Any]
    training_args: Mapping[str, Any]
    output_artifacts: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    max_steps: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_seq_length: int
    logging_steps: int
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    quantization_mode: str
    optim: str
    gradient_checkpointing: bool
    warmup_steps: int
    weight_decay: float
    lr_scheduler_type: str
    allow_merge: bool
    gguf_converter_path: Path | None
    gguf_quantization: str


class TrainerFailure(RuntimeError):
    def __init__(self, exit_code: int) -> None:
        super().__init__(str(exit_code))
        self.exit_code = exit_code


def main() -> int:
    try:
        run_from_environment()
    except TrainerFailure as exc:
        return exc.exit_code
    except Exception:
        return EXIT_TRAINING_FAILED
    return 0


def run_from_environment() -> None:
    spec_path = _required_path_env("KITT_JOB_SPEC_PATH")
    output_dir = _required_path_env("KITT_OUTPUT_DIR")
    runner_nonce = os.environ.get("KITT_RUNNER_NONCE")
    catalog_path = _required_path_env(RESOURCE_CATALOG_ENV)
    job_uid = os.environ.get("KITT_JOB_UID", "")
    if not runner_nonce:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    spec = _load_job_spec(spec_path)
    resources = resolve_resources(spec, catalog_path)
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    _run_training(
        spec=spec,
        resources=resources,
        output_dir=output_dir,
        runner_nonce=runner_nonce,
        job_uid=job_uid,
    )


def resolve_resources(
    spec: Mapping[str, Any],
    catalog_path: Path,
) -> ResolvedResources:
    catalog = _load_catalog(catalog_path)
    dataset_ref = spec["input_reference"]["dataset_version_uid"]
    model_ref = spec["base_or_parent_model"]["external_parent_ref"]
    profile_ref = (
        f"{spec['training_profile']['profile_uid']}:"
        f"{spec['training_profile']['version_label']}"
    )
    dataset_entry = _catalog_entry(catalog, "datasets", dataset_ref)
    model_entry = _catalog_entry(catalog, "models", model_ref)
    profile_entry = _catalog_entry(catalog, "training_profiles", profile_ref)
    if (
        profile_entry.get("profile_hash_sha256")
        != spec["training_profile"]["profile_hash_sha256"]
    ):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    dataset_path = _catalog_path(dataset_entry.get("path"), catalog_path)
    model_path = _catalog_path(model_entry.get("path"), catalog_path)
    if not dataset_path.is_file() or not model_path.is_dir():
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    dataset_format = dataset_entry.get("format", "jsonl")
    if dataset_format != "jsonl":
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    text_field = dataset_entry.get("text_field", "text")
    if not isinstance(text_field, str) or not _safe_name(text_field):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    dataset_sha256 = _optional_hash(dataset_entry.get("sha256"))
    if dataset_sha256 is not None and _sha256_file(dataset_path) != dataset_sha256:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    lora = profile_entry.get("lora")
    training_args = profile_entry.get("training_args")
    output_artifacts = profile_entry.get("output_artifacts", {})
    if (
        not isinstance(lora, Mapping)
        or not isinstance(training_args, Mapping)
        or not isinstance(output_artifacts, Mapping)
    ):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    training_args = _merged_training_args(
        profile_args=training_args,
        spec_hyperparameters=spec["hyperparameters"],
    )
    return ResolvedResources(
        dataset_version_uid=str(dataset_ref),
        dataset_path=dataset_path,
        dataset_text_field=str(text_field),
        dataset_sha256=dataset_sha256,
        model_ref=str(model_ref),
        model_path=model_path,
        model_revision=_safe_catalog_text(model_entry.get("revision"), default="local"),
        model_hash_sha256=_catalog_hash(model_entry, "model_hash_sha256"),
        tokenizer_hash_sha256=_catalog_hash(model_entry, "tokenizer_hash_sha256"),
        license_alignment_hash_sha256=_catalog_hash(
            model_entry,
            "license_alignment_hash_sha256",
        ),
        training_profile_hash_sha256=str(profile_entry["profile_hash_sha256"]),
        lora=lora,
        training_args=training_args,
        output_artifacts=output_artifacts,
    )


def _merged_training_args(
    *,
    profile_args: Mapping[str, Any],
    spec_hyperparameters: object,
) -> dict[str, Any]:
    if not isinstance(spec_hyperparameters, Mapping):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    merged = dict(profile_args)
    for key, value in spec_hyperparameters.items():
        if not isinstance(key, str) or key not in SUPPORTED_HYPERPARAMETERS:
            raise TrainerFailure(EXIT_CONFIG_INVALID)
        merged[HYPERPARAMETER_ALIASES.get(key, key)] = value
    return merged


def _run_training(
    *,
    spec: Mapping[str, Any],
    resources: ResolvedResources,
    output_dir: Path,
    runner_nonce: str,
    job_uid: str,
) -> None:
    stack = _import_training_stack()
    settings = _training_settings(resources)
    output_roles = set(spec.get("output_roles", []))
    texts = _load_jsonl_texts(resources.dataset_path, text_field=resources.dataset_text_field)
    if not texts:
        raise TrainerFailure(EXIT_CONFIG_INVALID)

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        resources.model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        fallback_token = tokenizer.eos_token or tokenizer.unk_token
        if fallback_token is None:
            raise TrainerFailure(EXIT_CONFIG_INVALID)
        tokenizer.pad_token = fallback_token

    model = _load_model(stack, resources=resources, settings=settings)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if settings.quantization_mode == "qlora_4bit":
        model = stack["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=settings.gradient_checkpointing,
        )
    lora_config = stack["LoraConfig"](
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = stack["get_peft_model"](model, lora_config)

    dataset = stack["Dataset"].from_list([{"text": text} for text in texts])

    def tokenize(batch: Mapping[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=settings.max_seq_length,
            padding=False,
        )
        return dict(encoded)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
    training_output_dir = output_dir / "trainer"
    training_output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    args = _training_arguments(stack, settings=settings, output_dir=training_output_dir)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized,
        "data_collator": stack["DataCollatorForLanguageModeling"](
            tokenizer=tokenizer,
            mlm=False,
        ),
    }
    trainer_signature = inspect.signature(stack["Trainer"])
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = stack["Trainer"](**trainer_kwargs)
    result = trainer.train()
    trained_steps = int(getattr(trainer.state, "global_step", 0) or settings.max_steps)
    if trained_steps <= 0:
        raise TrainerFailure(EXIT_TRAINING_FAILED)
    train_loss = _safe_loss(getattr(result, "metrics", {}).get("train_loss"))

    artifacts: dict[str, dict[str, Any]] = {}
    adapter_path = _write_adapter_artifact(
        output_dir=output_dir,
        model=model,
        base_model_ref=resources.model_ref,
    )
    artifacts["adapter"] = _artifact_entry(adapter_path)

    if "merged_weights" in output_roles:
        del trainer
        del model
        _release_accelerator_memory(stack)
        merged_path = _write_merged_weights_artifact(
            output_dir=output_dir,
            stack=stack,
            resources=resources,
            settings=settings,
            adapter_dir=output_dir / "adapter",
            tokenizer=tokenizer,
        )
        artifacts["merged_weights"] = _artifact_entry(merged_path)
    if "gguf" in output_roles:
        gguf_path = _write_gguf_artifact(output_dir=output_dir, settings=settings)
        artifacts["gguf"] = _artifact_entry(gguf_path)
    if "ollama_modelfile" in output_roles:
        modelfile_path = _write_modelfile(output_dir=output_dir)
        artifacts["ollama_modelfile"] = _artifact_entry(modelfile_path)
    if "training_log" in output_roles:
        training_log_path = _write_training_log(
            output_dir=output_dir,
            spec=spec,
            trained_steps=trained_steps,
            dataset_examples=len(texts),
        )
        artifacts["training_log"] = _artifact_entry(training_log_path)
    if "metrics_jsonl" in output_roles:
        metrics_path = _write_metrics(
            output_dir=output_dir,
            spec=spec,
            train_loss=train_loss,
            trained_steps=trained_steps,
            dataset_examples=len(texts),
        )
        artifacts["metrics_jsonl"] = _artifact_entry(metrics_path)
    if "run_lock" in output_roles:
        run_lock_path = _write_local_run_lock(
            output_dir=output_dir,
            spec=spec,
            resources=resources,
            artifacts=artifacts,
            job_uid=job_uid,
        )
        artifacts["run_lock"] = _artifact_entry(run_lock_path)

    manifest = _manifest(
        spec=spec,
        resources=resources,
        runner_nonce=runner_nonce,
        trained_steps=trained_steps,
        train_loss=train_loss,
        dataset_examples=len(texts),
        artifacts=artifacts,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_model(
    stack: Mapping[str, Any],
    *,
    resources: ResolvedResources,
    settings: TrainingSettings,
) -> Any:
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if settings.quantization_mode == "qlora_4bit":
        torch = stack["torch"]
        if not torch.cuda.is_available():
            raise TrainerFailure(EXIT_TRAINING_FAILED)
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = "auto"
    return stack["AutoModelForCausalLM"].from_pretrained(resources.model_path, **kwargs)


def _training_arguments(
    stack: Mapping[str, Any],
    *,
    settings: TrainingSettings,
    output_dir: Path,
) -> Any:
    torch = stack["torch"]
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "max_steps": settings.max_steps,
        "num_train_epochs": settings.num_train_epochs,
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "learning_rate": settings.learning_rate,
        "logging_steps": settings.logging_steps,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "seed": settings.seed,
        "disable_tqdm": True,
        "gradient_checkpointing": settings.gradient_checkpointing,
        "optim": settings.optim,
        "warmup_steps": settings.warmup_steps,
        "weight_decay": settings.weight_decay,
        "lr_scheduler_type": settings.lr_scheduler_type,
    }
    if settings.quantization_mode == "qlora_4bit":
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        kwargs["bf16"] = bf16
        kwargs["fp16"] = not bf16
    signature = inspect.signature(stack["TrainingArguments"])
    return stack["TrainingArguments"](
        **{key: value for key, value in kwargs.items() if key in signature.parameters}
    )


def _write_adapter_artifact(*, output_dir: Path, model: Any, base_model_ref: str) -> Path:
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    _patch_adapter_config(
        adapter_dir / "adapter_config.json",
        base_model_ref=base_model_ref,
    )
    adapter_path = output_dir / "adapter.zip"
    _zip_directory(adapter_dir, adapter_path)
    return adapter_path


def _write_merged_weights_artifact(
    *,
    output_dir: Path,
    stack: Mapping[str, Any],
    resources: ResolvedResources,
    settings: TrainingSettings,
    adapter_dir: Path,
    tokenizer: Any,
) -> Path:
    if not settings.allow_merge:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    if not adapter_dir.is_dir():
        raise TrainerFailure(EXIT_TRAINING_FAILED)
    merged_dir = output_dir / "merged_weights"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    base_model = _load_merge_base_model(stack, resources=resources)
    try:
        peft_model = stack["PeftModel"].from_pretrained(
            base_model,
            adapter_dir,
            is_trainable=False,
        )
        if not hasattr(peft_model, "merge_and_unload"):
            raise TrainerFailure(EXIT_TRAINING_FAILED)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        tokenizer.save_pretrained(merged_dir)
    finally:
        try:
            del merged_model
        except UnboundLocalError:
            pass
        try:
            del peft_model
        except UnboundLocalError:
            pass
        del base_model
        _release_accelerator_memory(stack)
    target_path = output_dir / "merged_weights.zip"
    _zip_directory(merged_dir, target_path)
    return target_path


def _load_merge_base_model(stack: Mapping[str, Any], *, resources: ResolvedResources) -> Any:
    torch = stack["torch"]
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if hasattr(torch, "float16"):
        kwargs["torch_dtype"] = torch.float16
    return stack["AutoModelForCausalLM"].from_pretrained(resources.model_path, **kwargs)


def _write_gguf_artifact(*, output_dir: Path, settings: TrainingSettings) -> Path:
    if settings.gguf_converter_path is None:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    merged_dir = output_dir / "merged_weights"
    if not merged_dir.is_dir():
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    target_path = output_dir / "model.gguf"
    proc = subprocess.run(
        [
            sys.executable,
            str(settings.gguf_converter_path),
            str(merged_dir),
            "--outfile",
            str(target_path),
            "--outtype",
            settings.gguf_quantization,
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0 or not target_path.is_file():
        raise TrainerFailure(EXIT_TRAINING_FAILED)
    return target_path


def _release_accelerator_memory(stack: Mapping[str, Any]) -> None:
    gc.collect()
    torch = stack.get("torch")
    cuda = None if torch is None else getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
            if hasattr(cuda, "ipc_collect"):
                cuda.ipc_collect()
    except Exception:
        pass


def _write_modelfile(*, output_dir: Path) -> Path:
    path = output_dir / "Modelfile"
    path.write_text(
        "FROM ./model.gguf\nPARAMETER temperature 0\n",
        encoding="utf-8",
    )
    return path


def _write_training_log(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    trained_steps: int,
    dataset_examples: int,
) -> Path:
    path = output_dir / "training_log.jsonl"
    _append_jsonl(
        path,
        {
            "event": "training_completed",
            "run_uid": spec["run_uid"],
            "trained_steps": trained_steps,
            "dataset_examples": dataset_examples,
            "timestamp": _utc_now(),
        },
    )
    return path


def _write_metrics(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    train_loss: float,
    trained_steps: int,
    dataset_examples: int,
) -> Path:
    path = output_dir / "metrics.jsonl"
    _append_jsonl(
        path,
        {
            "schema_version": "kiron_sft_metrics_v1",
            "run_uid": spec["run_uid"],
            "train_loss": train_loss,
            "trained_steps": trained_steps,
            "dataset_examples": dataset_examples,
        },
    )
    return path


def _write_local_run_lock(
    *,
    output_dir: Path,
    spec: Mapping[str, Any],
    resources: ResolvedResources,
    artifacts: Mapping[str, Mapping[str, Any]],
    job_uid: str,
) -> Path:
    payload = {
        "schema_version": "kitt_run_lock_v1",
        "lock_kind": "diagnostic",
        "run_outcome": "succeeded",
        "run": {
            "run_uid": spec["run_uid"],
            "run_type": spec["run_type"],
            "training_phase": "sft",
            "training_reference_mode": "dataset_version",
            "run_outcome": "succeeded",
            "base_model_ref": resources.model_ref,
            "job_uid": job_uid or "unknown",
            "job_spec_hash_sha256": job_stubs.job_spec_hash(dict(spec)),
            "capability_snapshot_hash_sha256": spec["capability_snapshot_hash_sha256"],
        },
        "dataset": {
            "dataset_version_uid": resources.dataset_version_uid,
            "content_hash_sha256": resources.dataset_sha256 or _sha256_file(
                resources.dataset_path
            ),
        },
        "base_model": {
            "logical_ref": resources.model_ref,
            "revision": resources.model_revision,
            "model_hash_sha256": resources.model_hash_sha256,
            "license_alignment_hash_sha256": resources.license_alignment_hash_sha256,
        },
        "tokenizer": {
            "logical_ref": resources.model_ref,
            "revision": resources.model_revision,
            "tokenizer_hash_sha256": resources.tokenizer_hash_sha256,
        },
        "artifacts": [
            {
                "artifact_type": role,
                "role": role,
                "sha256": str(entry["sha256"]),
                "size_bytes": int(entry["size_bytes"]),
            }
            for role, entry in sorted(artifacts.items())
        ],
    }
    path = output_dir / "run_lock.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _load_job_spec(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerFailure(EXIT_CONFIG_INVALID) from exc
    try:
        job_stubs.validate_job_spec(data)
    except job_stubs.JobValidationError as exc:
        raise TrainerFailure(EXIT_CONFIG_INVALID) from exc
    return data


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerFailure(EXIT_CONFIG_INVALID) from exc
    if not isinstance(data, dict) or data.get("schema_version") != RESOURCE_CATALOG_SCHEMA_VERSION:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return data


def _catalog_entry(
    catalog: Mapping[str, Any],
    section: str,
    ref: str,
) -> Mapping[str, Any]:
    section_data = catalog.get(section)
    if not isinstance(section_data, Mapping):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    entry = section_data.get(ref)
    if not isinstance(entry, Mapping):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return entry


def _catalog_path(raw: object, catalog_path: Path) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    try:
        resolved = Path(raw).resolve(strict=True)
        catalog_root = catalog_path.parent.resolve(strict=True)
    except OSError as exc:
        raise TrainerFailure(EXIT_CONFIG_INVALID) from exc
    if not (resolved == catalog_root or resolved.is_relative_to(catalog_root)):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return resolved


def _training_settings(resources: ResolvedResources) -> TrainingSettings:
    profile_args = resources.training_args
    quantization_mode = str(profile_args.get("quantization_mode", "none"))
    if quantization_mode not in {"none", "qlora_4bit"}:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    default_optim = "paged_adamw_8bit" if quantization_mode == "qlora_4bit" else "adamw_torch"
    gguf_converter = resources.output_artifacts.get("gguf_converter_path")
    return TrainingSettings(
        max_steps=_int_setting(profile_args, "max_steps", 1, 1, 100_000),
        num_train_epochs=_float_setting(profile_args, "num_train_epochs", 1.0, 0.001, 1000.0),
        per_device_train_batch_size=_int_setting(
            profile_args,
            "per_device_train_batch_size",
            1,
            1,
            128,
        ),
        gradient_accumulation_steps=_int_setting(
            profile_args,
            "gradient_accumulation_steps",
            1,
            1,
            1024,
        ),
        learning_rate=_float_setting(profile_args, "learning_rate", 2e-4, 1e-8, 10.0),
        max_seq_length=_int_setting(profile_args, "max_seq_length", 256, 8, 8192),
        logging_steps=_int_setting(profile_args, "logging_steps", 1, 1, 10_000),
        seed=_int_setting(profile_args, "seed", 1, 0, 2**32 - 1),
        lora_r=_int_setting(
            profile_args,
            "lora_r",
            _int_setting(resources.lora, "r", 8, 1, 1024),
            1,
            1024,
        ),
        lora_alpha=_int_setting(
            profile_args,
            "lora_alpha",
            _int_setting(resources.lora, "lora_alpha", 16, 1, 4096),
            1,
            4096,
        ),
        lora_dropout=_float_setting(
            profile_args,
            "lora_dropout",
            _float_setting(resources.lora, "lora_dropout", 0.0, 0.0, 1.0),
            0.0,
            1.0,
        ),
        target_modules=_target_modules(
            profile_args.get("target_modules", resources.lora.get("target_modules"))
        ),
        quantization_mode=quantization_mode,
        optim=_safe_catalog_text(profile_args.get("optim"), default=default_optim),
        gradient_checkpointing=_bool_setting(profile_args, "gradient_checkpointing", True),
        warmup_steps=_int_setting(profile_args, "warmup_steps", 0, 0, 100_000),
        weight_decay=_float_setting(profile_args, "weight_decay", 0.0, 0.0, 1.0),
        lr_scheduler_type=_safe_catalog_text(
            profile_args.get("lr_scheduler_type"),
            default="linear",
        ),
        allow_merge=_bool_setting(resources.output_artifacts, "allow_merge", False),
        gguf_converter_path=(
            None
            if gguf_converter is None
            else _executable_path(gguf_converter)
        ),
        gguf_quantization=_safe_catalog_text(
            resources.output_artifacts.get("gguf_quantization"),
            default="f16",
        ),
    )


def _int_setting(
    source: Mapping[str, Any],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum or raw > maximum:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return raw


def _float_setting(
    source: Mapping[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = source.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    value = float(raw)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return value


def _bool_setting(source: Mapping[str, Any], name: str, default: bool) -> bool:
    raw = source.get(name, default)
    if not isinstance(raw, bool):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return raw


def _target_modules(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    modules = tuple(item for item in raw if isinstance(item, str) and _safe_name(item))
    if len(modules) != len(raw):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return modules


def _load_jsonl_texts(path: Path, *, text_field: str) -> list[str]:
    texts: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TrainerFailure(EXIT_CONFIG_INVALID)
                text = row.get(text_field)
                if not isinstance(text, str) or not text.strip():
                    raise TrainerFailure(EXIT_CONFIG_INVALID)
                texts.append(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerFailure(EXIT_CONFIG_INVALID) from exc
    return texts


def _import_training_stack() -> dict[str, Any]:
    try:
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch
        import transformers  # noqa: F401
        import trl  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import DataCollatorForLanguageModeling
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise TrainerFailure(EXIT_DEPENDENCY_MISSING) from exc
    return {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "torch": torch,
    }


def _patch_adapter_config(path: Path, *, base_model_ref: str) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerFailure(EXIT_TRAINING_FAILED) from exc
    if not isinstance(data, dict):
        raise TrainerFailure(EXIT_TRAINING_FAILED)
    data["base_model_name_or_path"] = base_model_ref
    path.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _zip_directory(source_dir: Path, target_path: Path) -> None:
    with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            archive.write(path, path.relative_to(source_dir).as_posix())


def _manifest(
    *,
    spec: Mapping[str, Any],
    resources: ResolvedResources,
    runner_nonce: str,
    trained_steps: int,
    train_loss: float,
    dataset_examples: int,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "kiron_sft_runner_manifest_v1",
        "run_type": spec["run_type"],
        "run_uid": spec["run_uid"],
        "job_spec_hash_sha256": job_stubs.job_spec_hash(dict(spec)),
        "runner_nonce_sha256": _sha256_text(runner_nonce),
        "trained_steps": trained_steps,
        "training_provenance": {
            "kind": "kiron_sft_training_run",
            "provenance_version": 1,
            "base_model_ref": resources.model_ref,
            "training_profile_hash_sha256": spec["training_profile"][
                "profile_hash_sha256"
            ],
            "input_reference_hash_sha256": _json_hash(spec["input_reference"]),
            "trainer_entrypoint": {
                "kind": "kiron_sft_trainer",
                "version": KIRON_SFT_TRAINER_ENTRYPOINT,
            },
            "optimizer_steps": trained_steps,
            "dataset_examples_seen": max(1, dataset_examples),
            "final_train_loss": train_loss,
        },
        "artifacts": dict(artifacts),
    }


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _required_path_env(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    path = Path(raw)
    if not path.is_absolute():
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return path


def _catalog_hash(entry: Mapping[str, Any], name: str) -> str:
    value = entry.get(name)
    if not isinstance(value, str) or not _is_hash(value):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return value


def _optional_hash(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _is_hash(value):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return value


def _is_hash(value: str) -> bool:
    return len(value) == 64 and value == value.lower() and all(
        char in "0123456789abcdef" for char in value
    )


def _safe_catalog_text(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not _safe_text(value):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return value


def _executable_path(value: object) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    path = Path(value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise TrainerFailure(EXIT_CONFIG_INVALID)
    return path


def _safe_name(value: str) -> bool:
    return value.replace("_", "").replace("-", "").isalnum() and 1 <= len(value) <= 96


def _safe_text(value: str) -> bool:
    return bool(value) and "\x00" not in value and len(value) <= 256


def _safe_loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    loss = float(value)
    return loss if math.isfinite(loss) else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


if __name__ == "__main__":
    sys.exit(main())
