from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

import runners
import sft_trainer


KITT_SPEC = {
    "schema_version": "kitt_job_spec_v1",
    "run_uid": "kitt-run-1",
    "run_type": "sft",
    "training_profile": {
        "profile_uid": "profile-1",
        "version_label": "v1",
        "profile_hash_sha256": "a" * 64,
    },
    "input_reference": {
        "mode": "dataset_version",
        "dataset_version_uid": "5136e167-dbb7-49a6-a752-6fe1e962adb7",
    },
    "base_or_parent_model": {"external_parent_ref": "Qwen/Qwen2.5-7B-Instruct"},
    "hyperparameters": {
        "learning_rate": 0.0002,
        "max_steps": 1,
        "max_seq_length": 16,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "seed": 1,
    },
    "output_roles": [
        "adapter",
        "merged_weights",
        "gguf",
        "ollama_modelfile",
        "training_log",
        "metrics_jsonl",
        "run_lock",
    ],
    "capability_snapshot_hash_sha256": "b" * 64,
}


def _write_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "alpha beta gamma"}),
                json.dumps({"text": "beta gamma delta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_catalog(root: Path, *, model_dir: Path, dataset_path: Path) -> Path:
    converter = root / "convert_to_gguf.py"
    converter.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import sys",
                "outfile = Path(sys.argv[sys.argv.index('--outfile') + 1])",
                "(outfile.parent / 'converter_python.txt').write_text(sys.executable)",
                "outfile.write_bytes(b'GGUF' + b'\\0' * 4096)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    converter.chmod(0o755)
    catalog = {
        "schema_version": "kiron_sft_resources_v1",
        "datasets": {
            "5136e167-dbb7-49a6-a752-6fe1e962adb7": {
                "path": str(dataset_path),
                "format": "jsonl",
                "text_field": "text",
            }
        },
        "models": {
            "Qwen/Qwen2.5-7B-Instruct": {
                "path": str(model_dir),
                "revision": "tiny-test",
                "model_hash_sha256": "d" * 64,
                "tokenizer_hash_sha256": "e" * 64,
                "license_alignment_hash_sha256": "f" * 64,
            }
        },
        "training_profiles": {
            "profile-1:v1": {
                "profile_hash_sha256": "a" * 64,
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": ["c_attn"],
                },
                "training_args": {
                    "max_steps": 1,
                    "num_train_epochs": 1.0,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 0.0002,
                    "max_seq_length": 16,
                    "logging_steps": 1,
                    "seed": 1,
                    "quantization_mode": "none",
                    "gradient_checkpointing": False,
                },
                "output_artifacts": {
                    "allow_merge": True,
                    "gguf_converter_path": str(converter),
                    "gguf_quantization": "f16",
                },
            }
        },
    }
    path = root / "sft_resources.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _write_tiny_model(model_dir: Path, texts: list[str]) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    fast.save_pretrained(model_dir)
    config = GPT2Config(
        vocab_size=len(fast),
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=fast.eos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    GPT2LMHeadModel(config).save_pretrained(model_dir)


def test_resolve_resources_accepts_catalog_refs_under_catalog_root(tmp_path):
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    dataset_path = tmp_path / "datasets" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True)
    _write_dataset(dataset_path)
    catalog_path = _write_catalog(tmp_path, model_dir=model_dir, dataset_path=dataset_path)

    resources = sft_trainer.resolve_resources(KITT_SPEC, catalog_path)

    assert resources.dataset_path == dataset_path
    assert resources.model_path == model_dir
    assert resources.dataset_text_field == "text"


def test_resolve_resources_rejects_profile_hash_mismatch(tmp_path):
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    dataset_path = tmp_path / "datasets" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True)
    _write_dataset(dataset_path)
    catalog_path = _write_catalog(tmp_path, model_dir=model_dir, dataset_path=dataset_path)
    bad_spec = {
        **KITT_SPEC,
        "training_profile": {
            **KITT_SPEC["training_profile"],
            "profile_hash_sha256": "c" * 64,
        },
    }

    with pytest.raises(sft_trainer.TrainerFailure) as exc:
        sft_trainer.resolve_resources(bad_spec, catalog_path)

    assert exc.value.exit_code == sft_trainer.EXIT_CONFIG_INVALID


def test_resolve_resources_rejects_unsupported_hyperparameters(tmp_path):
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    dataset_path = tmp_path / "datasets" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True)
    _write_dataset(dataset_path)
    catalog_path = _write_catalog(tmp_path, model_dir=model_dir, dataset_path=dataset_path)
    bad_spec = {
        **KITT_SPEC,
        "hyperparameters": {
            **KITT_SPEC["hyperparameters"],
            "unsupported_knob": 1,
        },
    }

    with pytest.raises(sft_trainer.TrainerFailure) as exc:
        sft_trainer.resolve_resources(bad_spec, catalog_path)

    assert exc.value.exit_code == sft_trainer.EXIT_CONFIG_INVALID


def test_first_party_sft_trainer_runs_tiny_lora_training(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    pytest.importorskip("trl")
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    dataset_path = tmp_path / "datasets" / "train.jsonl"
    dataset_path.parent.mkdir(parents=True)
    _write_dataset(dataset_path)
    _write_tiny_model(model_dir, ["alpha beta gamma", "beta gamma delta"])
    catalog_path = _write_catalog(tmp_path, model_dir=model_dir, dataset_path=dataset_path)
    spec_path = tmp_path / "job_spec.json"
    output_dir = tmp_path / "output"
    runner_nonce = "runner-nonce"
    spec_path.write_text(json.dumps(KITT_SPEC), encoding="utf-8")
    monkeypatch.setenv("KITT_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("KITT_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("KITT_RUNNER_NONCE", runner_nonce)
    monkeypatch.setenv("KITT_SFT_RESOURCE_CATALOG", str(catalog_path))

    assert sft_trainer.main() == 0

    artifacts = (
        runners.RunnerArtifact("manifest", output_dir / "manifest.json"),
        runners.RunnerArtifact("adapter", output_dir / "adapter.zip"),
        runners.RunnerArtifact("merged_weights", output_dir / "merged_weights.zip"),
        runners.RunnerArtifact("gguf", output_dir / "model.gguf"),
        runners.RunnerArtifact("ollama_modelfile", output_dir / "Modelfile"),
        runners.RunnerArtifact("training_log", output_dir / "training_log.jsonl"),
        runners.RunnerArtifact("metrics_jsonl", output_dir / "metrics.jsonl"),
        runners.RunnerArtifact("run_lock", output_dir / "run_lock.json"),
    )
    nonce_hash = hashlib.sha256(runner_nonce.encode()).hexdigest()
    for artifact in artifacts:
        assert (
            runners.validate_artifact_file(
                artifact,
                spec=KITT_SPEC,
                runner_nonce_sha256=nonce_hash,
            )
            is None
        )
    assert (
        runners.validate_artifact_bundle(
            artifacts,
            spec=KITT_SPEC,
            runner_nonce_sha256=nonce_hash,
        )
        is None
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_provenance"]["base_model_ref"] == "Qwen/Qwen2.5-7B-Instruct"
    assert manifest["trained_steps"] == 1
    modelfile_text = (output_dir / "Modelfile").read_text(encoding="utf-8")
    assert modelfile_text == "FROM ./model.gguf\nPARAMETER temperature 0\n"
    assert "ADAPTER" not in modelfile_text
    assert (output_dir / "converter_python.txt").read_text(encoding="utf-8") == sys.executable
    with zipfile.ZipFile(output_dir / "merged_weights.zip") as archive:
        names = {Path(info.filename).name for info in archive.infolist()}
    assert "config.json" in names
    assert "tokenizer.json" in names
    assert set(manifest["artifacts"]) == {
        "adapter",
        "merged_weights",
        "gguf",
        "ollama_modelfile",
        "training_log",
        "metrics_jsonl",
        "run_lock",
    }
