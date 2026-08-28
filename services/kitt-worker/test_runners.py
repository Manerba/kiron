from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import sys
import time
import zipfile

import pytest

import publish_test_support
import queue_store
import runners

KITT_SPEC = {
    "schema_version": "kitt_job_spec_v1",
    "run_uid": "kitt-run-1",
    "run_type": "sft",
    "training_profile": {
        "profile_uid": "profile-1",
        "version_label": "v1",
        "profile_hash_sha256": "a" * 64,
    },
    "input_reference": {"mode": "dataset_version", "dataset_version_uid": "5136e167-dbb7-49a6-a752-6fe1e962adb7"},
    "base_or_parent_model": {"external_parent_ref": "Qwen/Qwen2.5-7B-Instruct"},
    "hyperparameters": {"learning_rate": 0.0002},
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


@pytest.fixture
def allow_test_sft_command(monkeypatch):
    monkeypatch.setattr(runners, "_command_argv", lambda command: shlex.split(command))


def _lora_safetensors_payload(*, zero_filled: bool = False) -> bytes:
    tensor_header = json.dumps(
        {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [8, 256],
                "data_offsets": [0, 8192],
            },
            "base_model.model.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [256, 8],
                "data_offsets": [8192, 16384],
            },
        },
        separators=(",", ":"),
    ).encode()
    tensor_header += b" " * ((8 - (len(tensor_header) % 8)) % 8)
    tensor_bytes = (
        b"\0" * 16384
        if zero_filled
        else bytes((index % 251) + 1 for index in range(16384))
    )
    return len(tensor_header).to_bytes(8, "little") + tensor_header + tensor_bytes


def _write_adapter_zip(
    path,
    *,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    zero_filled: bool = False,
) -> None:
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("adapter_config.json", json.dumps(adapter_config))
        archive.writestr(
            "adapter_model.safetensors",
            _lora_safetensors_payload(zero_filled=zero_filled),
        )


def test_fake_runner_succeeds_without_side_effects(tmp_path):
    async def run():
        cancel = asyncio.Event()
        runner = runners.FakeRunner()
        outcome = await runner.run(object(), cancel)
        return runner, outcome

    runner, outcome = asyncio.run(run())

    assert runner.started is True
    assert outcome.state == "succeeded"
    assert list(tmp_path.iterdir()) == []


def test_fake_runner_reports_failure_as_codes():
    async def run():
        cancel = asyncio.Event()
        runner = runners.FakeRunner(
            result="failed",
            failure_code="runner_failed",
            failure_class="runner",
        )
        outcome = await runner.run(object(), cancel)
        return outcome

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_failed"
    assert outcome.failure_class == "runner"


def test_fake_runner_observes_cooperative_cancel():
    async def run():
        cancel = asyncio.Event()
        runner = runners.FakeRunner(delay_seconds=10)
        task = asyncio.create_task(runner.run(object(), cancel))
        await asyncio.sleep(0)
        cancel.set()
        outcome = await task
        return runner, outcome

    runner, outcome = asyncio.run(run())

    assert runner.cancel_requested is True
    assert outcome.state == "canceled"


def test_sft_subprocess_runner_executes_command_and_collects_artifacts(
    tmp_path,
    monkeypatch,
    allow_test_sft_command,
):
    monkeypatch.setenv("KITT_WORKER_TOKEN", "do-not-inherit")
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import hashlib",
                "import json",
                "import os",
                "import zipfile",
                "spec = json.loads(Path(os.environ['KITT_JOB_SPEC_PATH']).read_text())",
                "canonical_spec = json.dumps(",
                "    spec,",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "job_spec_hash = hashlib.sha256(canonical_spec.encode()).hexdigest()",
                "canonical_input = json.dumps(",
                "    spec['input_reference'],",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "input_hash = hashlib.sha256(canonical_input.encode()).hexdigest()",
                "runner_nonce_hash = hashlib.sha256(",
                "    os.environ['KITT_RUNNER_NONCE'].encode()",
                ").hexdigest()",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "model_path = out / 'adapter.zip'",
                "tensor_header = json.dumps({",
                "    'base_model.model.q_proj.lora_A.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [8, 256],",
                "        'data_offsets': [0, 8192],",
                "    },",
                "    'base_model.model.q_proj.lora_B.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [256, 8],",
                "        'data_offsets': [8192, 16384],",
                "    },",
                "}, separators=(',', ':')).encode()",
                "tensor_header += b' ' * ((8 - (len(tensor_header) % 8)) % 8)",
                "adapter_payload = (",
                "    len(tensor_header).to_bytes(8, 'little')",
                "    + tensor_header",
                "    + bytes((index % 251) + 1 for index in range(16384))",
                ")",
                "adapter_config = {",
                "    'peft_type': 'LORA',",
                "    'base_model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',",
                "    'r': 8,",
                "    'lora_alpha': 16,",
                "    'target_modules': ['q_proj'],",
                "}",
                "with zipfile.ZipFile(model_path, 'w') as archive:",
                "    archive.writestr('adapter_config.json', json.dumps(adapter_config))",
                "    archive.writestr('adapter_model.safetensors', adapter_payload)",
                "model_bytes = model_path.read_bytes()",
                "merged_path = out / 'merged_weights.zip'",
                "with zipfile.ZipFile(merged_path, 'w') as archive:",
                "    archive.writestr('config.json', json.dumps({'model_type': 'qwen2'}))",
                "    archive.writestr('tokenizer.json', '{}')",
                "    archive.writestr('model.safetensors', bytes((index % 251) + 1 for index in range(4096)))",
                "merged_bytes = merged_path.read_bytes()",
                "gguf_path = out / 'model.gguf'",
                "gguf_path.write_bytes(b'GGUF' + bytes((index % 251) + 1 for index in range(4096)))",
                "gguf_bytes = gguf_path.read_bytes()",
                "modelfile_path = out / 'Modelfile'",
                "modelfile_path.write_text('FROM Qwen/Qwen2.5-7B-Instruct\\nADAPTER ./adapter.zip\\n')",
                "modelfile_bytes = modelfile_path.read_bytes()",
                "training_log_path = out / 'training_log.jsonl'",
                "training_log_path.write_text(json.dumps({'event': 'training_completed'}) + '\\n')",
                "training_log_bytes = training_log_path.read_bytes()",
                "metrics_path = out / 'metrics.jsonl'",
                "metrics_path.write_text(json.dumps({'schema_version': 'kiron_sft_metrics_v1'}) + '\\n')",
                "metrics_bytes = metrics_path.read_bytes()",
                "run_lock_path = out / 'run_lock.json'",
                "run_lock_path.write_text(json.dumps({'schema_version': 'kitt_run_lock_v1'}))",
                "run_lock_bytes = run_lock_path.read_bytes()",
                "manifest = {",
                "    'schema_version': 'kiron_sft_runner_manifest_v1',",
                "    'run_type': spec['run_type'],",
                "    'run_uid': spec['run_uid'],",
                "    'job_spec_hash_sha256': job_spec_hash,",
                "    'runner_nonce_sha256': runner_nonce_hash,",
                "    'trained_steps': 1,",
                "    'secret_present': 'KITT_WORKER_TOKEN' in os.environ,",
                "    'training_provenance': {",
                "        'kind': 'kiron_sft_training_run',",
                "        'provenance_version': 1,",
                "        'base_model_ref': spec['base_or_parent_model']['external_parent_ref'],",
                "        'training_profile_hash_sha256': (",
                "            spec['training_profile']['profile_hash_sha256']",
                "        ),",
                "        'input_reference_hash_sha256': input_hash,",
                "        'trainer_entrypoint': {",
                "            'kind': 'kiron_sft_trainer',",
                "            'version': 'adr-0008.sft.v2',",
                "        },",
                "        'optimizer_steps': 1,",
                "        'dataset_examples_seen': 1,",
                "        'final_train_loss': 1.0,",
                "    },",
                "    'artifacts': {",
                "        'adapter': {",
                "            'sha256': hashlib.sha256(model_bytes).hexdigest(),",
                "            'size_bytes': len(model_bytes),",
                "        },",
                "        'merged_weights': {",
                "            'sha256': hashlib.sha256(merged_bytes).hexdigest(),",
                "            'size_bytes': len(merged_bytes),",
                "        },",
                "        'gguf': {",
                "            'sha256': hashlib.sha256(gguf_bytes).hexdigest(),",
                "            'size_bytes': len(gguf_bytes),",
                "        },",
                "        'ollama_modelfile': {",
                "            'sha256': hashlib.sha256(modelfile_bytes).hexdigest(),",
                "            'size_bytes': len(modelfile_bytes),",
                "        },",
                "        'training_log': {",
                "            'sha256': hashlib.sha256(training_log_bytes).hexdigest(),",
                "            'size_bytes': len(training_log_bytes),",
                "        },",
                "        'metrics_jsonl': {",
                "            'sha256': hashlib.sha256(metrics_bytes).hexdigest(),",
                "            'size_bytes': len(metrics_bytes),",
                "        },",
                "        'run_lock': {",
                "            'sha256': hashlib.sha256(run_lock_bytes).hexdigest(),",
                "            'size_bytes': len(run_lock_bytes),",
                "        },",
                "    },",
                "}",
                "Path(out / 'manifest.json').write_text(json.dumps(manifest))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "succeeded"
    assert {artifact.role for artifact in outcome.artifacts} == {
        "manifest",
        "adapter",
        "merged_weights",
        "gguf",
        "ollama_modelfile",
        "training_log",
        "metrics_jsonl",
        "run_lock",
    }
    for artifact in outcome.artifacts:
        assert artifact.path.is_file()
    manifest = next(artifact.path for artifact in outcome.artifacts if artifact.role == "manifest")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == "kiron_sft_runner_manifest_v1"
    assert manifest_data["run_type"] == "sft"
    assert manifest_data["run_uid"] == "kitt-run-1"
    assert manifest_data["trained_steps"] == 1
    assert manifest_data["secret_present"] is False
    assert set(manifest_data["artifacts"]) == {
        "adapter",
        "merged_weights",
        "gguf",
        "ollama_modelfile",
        "training_log",
        "metrics_jsonl",
        "run_lock",
    }


def test_sft_subprocess_runner_rejects_untrusted_marker_script(tmp_path):
    command = tmp_path / "kiron_sft_trainer.py"
    command.write_text(
        "\n".join(
            [
                'KIRON_SFT_TRAINER_ENTRYPOINT = "adr-0008.sft.v2"',
                "KITT_JOB_SPEC_PATH = 'KITT_JOB_SPEC_PATH'",
                "KITT_OUTPUT_DIR = 'KITT_OUTPUT_DIR'",
                "FILES = ('adapter_config.json', 'adapter_model.safetensors')",
                "PACKAGES = ('transformers', 'peft', 'trl')",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-untrusted-runner",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_config_invalid"
    assert outcome.failure_class == "runner"


def test_sft_subprocess_runner_rejects_fake_python_with_trusted_script(tmp_path):
    fake_python = tmp_path / "python-fake"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from pathlib import Path",
                "import os",
                "out = os.environ.get('KITT_OUTPUT_DIR')",
                "if out:",
                "    Path(out, 'fake-python-ran').write_text('ran')",
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    trusted_script = Path(__file__).resolve().with_name("sft_trainer.py")
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{fake_python} {trusted_script}",
    )
    job = SimpleNamespace(
        job_uid="job-fake-python",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_config_invalid"
    assert outcome.failure_class == "runner"
    assert not (tmp_path / "data" / "work" / job.job_uid / "output").exists()


def test_sft_subprocess_runner_requires_requested_role_artifacts(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import hashlib",
                "import json",
                "import os",
                "import zipfile",
                "spec = json.loads(Path(os.environ['KITT_JOB_SPEC_PATH']).read_text())",
                "canonical_spec = json.dumps(",
                "    spec,",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "job_spec_hash = hashlib.sha256(canonical_spec.encode()).hexdigest()",
                "canonical_input = json.dumps(",
                "    spec['input_reference'],",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "input_hash = hashlib.sha256(canonical_input.encode()).hexdigest()",
                "runner_nonce_hash = hashlib.sha256(",
                "    os.environ['KITT_RUNNER_NONCE'].encode()",
                ").hexdigest()",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "model_path = out / 'adapter.zip'",
                "tensor_header = json.dumps({",
                "    'base_model.model.q_proj.lora_A.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [8, 256],",
                "        'data_offsets': [0, 8192],",
                "    },",
                "    'base_model.model.q_proj.lora_B.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [256, 8],",
                "        'data_offsets': [8192, 16384],",
                "    },",
                "}, separators=(',', ':')).encode()",
                "tensor_header += b' ' * ((8 - (len(tensor_header) % 8)) % 8)",
                "adapter_payload = (",
                "    len(tensor_header).to_bytes(8, 'little')",
                "    + tensor_header",
                "    + bytes((index % 251) + 1 for index in range(16384))",
                ")",
                "adapter_config = {",
                "    'peft_type': 'LORA',",
                "    'base_model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',",
                "    'r': 8,",
                "    'lora_alpha': 16,",
                "    'target_modules': ['q_proj'],",
                "}",
                "with zipfile.ZipFile(model_path, 'w') as archive:",
                "    archive.writestr('adapter_config.json', json.dumps(adapter_config))",
                "    archive.writestr('adapter_model.safetensors', adapter_payload)",
                "model_bytes = model_path.read_bytes()",
                "manifest = {",
                "    'schema_version': 'kiron_sft_runner_manifest_v1',",
                "    'run_type': spec['run_type'],",
                "    'run_uid': spec['run_uid'],",
                "    'job_spec_hash_sha256': job_spec_hash,",
                "    'runner_nonce_sha256': runner_nonce_hash,",
                "    'trained_steps': 1,",
                "    'training_provenance': {",
                "        'kind': 'kiron_sft_training_run',",
                "        'provenance_version': 1,",
                "        'base_model_ref': spec['base_or_parent_model']['external_parent_ref'],",
                "        'training_profile_hash_sha256': (",
                "            spec['training_profile']['profile_hash_sha256']",
                "        ),",
                "        'input_reference_hash_sha256': input_hash,",
                "        'trainer_entrypoint': {",
                "            'kind': 'kiron_sft_trainer',",
                "            'version': 'adr-0008.sft.v2',",
                "        },",
                "        'optimizer_steps': 1,",
                "        'dataset_examples_seen': 1,",
                "        'final_train_loss': 1.0,",
                "    },",
                "    'artifacts': {",
                "        'adapter': {",
                "            'sha256': hashlib.sha256(model_bytes).hexdigest(),",
                "            'size_bytes': len(model_bytes),",
                "        },",
                "    },",
                "}",
                "Path(out / 'manifest.json').write_text(json.dumps(manifest))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-missing-metrics",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_artifact_missing"
    assert outcome.failure_class == "artifact"


def test_sft_subprocess_runner_rejects_manifest_only_success(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import hashlib",
                "import json",
                "import os",
                "spec = json.loads(Path(os.environ['KITT_JOB_SPEC_PATH']).read_text())",
                "canonical_spec = json.dumps(",
                "    spec,",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "job_spec_hash = hashlib.sha256(canonical_spec.encode()).hexdigest()",
                "canonical_input = json.dumps(",
                "    spec['input_reference'],",
                "    sort_keys=True,",
                "    separators=(',', ':'),",
                "    ensure_ascii=False,",
                "    allow_nan=False,",
                ")",
                "input_hash = hashlib.sha256(canonical_input.encode()).hexdigest()",
                "runner_nonce_hash = hashlib.sha256(",
                "    os.environ['KITT_RUNNER_NONCE'].encode()",
                ").hexdigest()",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "manifest = {",
                "    'schema_version': 'kiron_sft_runner_manifest_v1',",
                "    'run_type': spec['run_type'],",
                "    'run_uid': spec['run_uid'],",
                "    'job_spec_hash_sha256': job_spec_hash,",
                "    'runner_nonce_sha256': runner_nonce_hash,",
                "    'trained_steps': 1,",
                "    'training_provenance': {",
                "        'kind': 'kiron_sft_training_run',",
                "        'provenance_version': 1,",
                "        'base_model_ref': spec['base_or_parent_model']['external_parent_ref'],",
                "        'training_profile_hash_sha256': (",
                "            spec['training_profile']['profile_hash_sha256']",
                "        ),",
                "        'input_reference_hash_sha256': input_hash,",
                "        'trainer_entrypoint': {",
                "            'kind': 'kiron_sft_trainer',",
                "            'version': 'adr-0008.sft.v2',",
                "        },",
                "        'optimizer_steps': 1,",
                "        'dataset_examples_seen': 1,",
                "        'final_train_loss': 1.0,",
                "    },",
                "    'artifacts': {'manifest': {'sha256': '0' * 64, 'size_bytes': 1}},",
                "}",
                "Path(out / 'manifest.json').write_text(json.dumps(manifest))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-manifest-only",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_artifact_missing"
    assert outcome.failure_class == "artifact"


def test_sft_subprocess_runner_rejects_empty_required_artifacts(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "Path(out / 'manifest.json').write_bytes(b'')",
                "Path(out / 'adapter.zip').write_bytes(b'')",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-empty-artifacts",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "artifact_empty"
    assert outcome.failure_class == "artifact"


def test_sft_subprocess_runner_rejects_nonempty_dummy_artifacts(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "Path(out / 'manifest.json').write_text('{}')",
                "Path(out / 'adapter.zip').write_bytes(b'x')",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-dummy-artifacts",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_artifact_invalid"
    assert outcome.failure_class == "artifact"


def test_sft_subprocess_runner_rejects_zip_with_dummy_weight(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import hashlib",
                "import json",
                "import os",
                "import zipfile",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "model_path = out / 'adapter.zip'",
                "with zipfile.ZipFile(model_path, 'w') as archive:",
                "    archive.writestr('adapter_config.json', '{\"model_type\":\"test\"}')",
                "    archive.writestr('adapter_model.safetensors', b'x' * 2048)",
                "model_bytes = model_path.read_bytes()",
                "manifest = {",
                "    'schema_version': 'kiron_sft_runner_manifest_v1',",
                "    'run_type': 'sft',",
                "    'run_uid': 'kitt-run-1',",
                "    'trained_steps': 1,",
                "    'artifacts': {",
                "        'adapter': {",
                "            'sha256': hashlib.sha256(model_bytes).hexdigest(),",
                "            'size_bytes': len(model_bytes),",
                "        },",
                "    },",
                "}",
                "Path(out / 'manifest.json').write_text(json.dumps(manifest))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-dummy-zip",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_artifact_invalid"
    assert outcome.failure_class == "artifact"


def test_sft_subprocess_runner_rejects_tiny_lora_named_safetensors(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import hashlib",
                "import json",
                "import os",
                "import zipfile",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "model_path = out / 'adapter.zip'",
                "tensor_header = json.dumps({",
                "    'x.lora_A.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [1, 1],",
                "        'data_offsets': [0, 4],",
                "    },",
                "    'x.lora_B.weight': {",
                "        'dtype': 'F32',",
                "        'shape': [1, 1],",
                "        'data_offsets': [4, 8],",
                "    },",
                "}, separators=(',', ':')).encode()",
                "tensor_header += b' ' * ((8 - (len(tensor_header) % 8)) % 8)",
                "adapter_payload = (",
                "    len(tensor_header).to_bytes(8, 'little')",
                "    + tensor_header",
                "    + b'\\0' * 8",
                ")",
                "adapter_config = {",
                "    'peft_type': 'LORA',",
                "    'base_model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',",
                "    'r': 1,",
                "    'lora_alpha': 1,",
                "    'target_modules': ['q_proj'],",
                "}",
                "with zipfile.ZipFile(model_path, 'w') as archive:",
                "    archive.writestr('adapter_config.json', json.dumps(adapter_config))",
                "    archive.writestr('adapter_model.safetensors', adapter_payload)",
                "model_bytes = model_path.read_bytes()",
                "manifest = {",
                "    'schema_version': 'kiron_sft_runner_manifest_v1',",
                "    'run_type': 'sft',",
                "    'run_uid': 'kitt-run-1',",
                "    'trained_steps': 1,",
                "    'artifacts': {",
                "        'adapter': {",
                "            'sha256': hashlib.sha256(model_bytes).hexdigest(),",
                "            'size_bytes': len(model_bytes),",
                "        },",
                "    },",
                "}",
                "Path(out / 'manifest.json').write_text(json.dumps(manifest))",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-tiny-lora",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        return await runner.run(job, cancel)

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "runner_artifact_invalid"
    assert outcome.failure_class == "artifact"


def test_artifact_validator_rejects_adapter_parent_model_mismatch(tmp_path):
    artifact_path = tmp_path / "adapter.zip"
    _write_adapter_zip(artifact_path, base_model="wrong-parent:1")

    assert (
        runners.validate_artifact_file(
            runners.RunnerArtifact("adapter", artifact_path),
            spec=KITT_SPEC,
        )
        == "runner_artifact_invalid"
    )


def test_artifact_validator_rejects_merged_weights_without_tokenizer(tmp_path):
    artifact_path = tmp_path / "merged_weights.zip"
    with zipfile.ZipFile(artifact_path, "w") as archive:
        archive.writestr("config.json", json.dumps({"model_type": "qwen2"}))
        archive.writestr(
            "model.safetensors",
            bytes((index % 251) + 1 for index in range(4096)),
        )

    assert (
        runners.validate_artifact_file(
            runners.RunnerArtifact("merged_weights", artifact_path),
            spec=KITT_SPEC,
        )
        == "runner_artifact_invalid"
    )


def test_artifact_validator_rejects_zero_filled_lora_tensors(tmp_path):
    artifact_path = tmp_path / "adapter.zip"
    _write_adapter_zip(artifact_path, zero_filled=True)

    assert (
        runners.validate_artifact_file(
            runners.RunnerArtifact("adapter", artifact_path),
            spec=KITT_SPEC,
        )
        == "runner_artifact_invalid"
    )


def test_sft_subprocess_runner_task_cancel_kills_child_process(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "import signal",
                "import subprocess",
                "import sys",
                "import time",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "child_code = '; '.join([",
                "    'import signal, time',",
                "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)',",
                "    'time.sleep(30)',",
                "])",
                "helper_code = '; '.join([",
                "    'from pathlib import Path',",
                "    'import os, signal, subprocess, sys',",
                "    f'child_code = {child_code!r}',",
                "    'argv = [sys.executable, \"-c\", child_code]',",
                "    'child = subprocess.Popen(argv, start_new_session=True)',",
                "    'pid_path = Path(os.environ[\"KITT_OUTPUT_DIR\"]) / \"pid\"',",
                "    'pid_path.write_text(str(child.pid))',",
                "])",
                "subprocess.run([sys.executable, '-c', helper_code], check=True)",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-cancel",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        task = asyncio.create_task(runner.run(job, cancel))
        pid_path = tmp_path / "data" / "work" / job.job_uid / "output" / "pid"
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        pid = int(pid_path.read_text(encoding="utf-8"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(run())

    assert _wait_pid_gone(pid)


def test_sft_subprocess_runner_cancel_kills_reparented_env_child(
    tmp_path,
    allow_test_sft_command,
):
    command = tmp_path / "runner_command.py"
    command.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "import signal",
                "import subprocess",
                "import sys",
                "import time",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "out = Path(os.environ['KITT_OUTPUT_DIR'])",
                "out.mkdir(parents=True, exist_ok=True)",
                "child_code = '; '.join([",
                "    'import signal, time',",
                "    'signal.signal(signal.SIGTERM, signal.SIG_IGN)',",
                "    'time.sleep(30)',",
                "])",
                "helper_code = '; '.join([",
                "    'import signal, time',",
                "    'from pathlib import Path',",
                "    'import os, subprocess, sys',",
                "    f'child_code = {child_code!r}',",
                "    'argv = [sys.executable, \"-c\", child_code]',",
                "    'child = subprocess.Popen(argv, start_new_session=True)',",
                "    'pid_path = Path(os.environ[\"KITT_OUTPUT_DIR\"]) / \"pid\"',",
                "    'pid_path.write_text(str(child.pid))',",
                "])",
                "subprocess.run([sys.executable, '-c', helper_code], check=True)",
                "while not (out / 'pid').exists():",
                "    time.sleep(0.01)",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        data_dir=tmp_path / "data",
        sft_command=f"{sys.executable} {command}",
    )
    job = SimpleNamespace(
        job_uid="job-sft-runner-cancel",
        run_uid="run-worker-1",
        canonical_job_spec_json=json.dumps(KITT_SPEC),
    )

    async def run():
        cancel = asyncio.Event()
        runner = runners.SftSubprocessRunner(cfg)
        task = asyncio.create_task(runner.run(job, cancel))
        pid_path = tmp_path / "data" / "work" / job.job_uid / "output" / "pid"
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        pid = int(pid_path.read_text(encoding="utf-8"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(run())

    assert _wait_pid_gone(pid)


def test_ollama_publish_runner_creates_and_verifies_tag(tmp_path, monkeypatch):
    store = _store(tmp_path)
    calls = _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-runner",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "succeeded"
    assert outcome.publish_result is not None
    assert outcome.publish_result["target_ref"] == publish_test_support.PUBLISH_TARGET_REF
    assert outcome.publish_result["idempotent"] is False
    assert outcome.publish_result["ollama_digest"] == "sha256:testdigest"
    gguf_ref = next(
        artifact["artifact_ref"]
        for artifact in artifacts
        if artifact["role"] == "gguf"
    )
    gguf_record = store.get_artifact_by_ref(gguf_ref)
    staged_gguf = tmp_path / "staging" / "objects" / gguf_record.artifact_uid
    workspace_gguf = tmp_path / "work" / "job-publish-runner" / "publish" / "model.gguf"
    assert staged_gguf.stat().st_ino == workspace_gguf.stat().st_ino
    assert calls == [
        ["show", publish_test_support.PUBLISH_TARGET_REF],
        ["create", publish_test_support.PUBLISH_TARGET_REF, "-f", "Modelfile"],
        ["show", publish_test_support.PUBLISH_TARGET_REF],
    ]


def test_ollama_publish_runner_rejects_duplicate_tag_without_matching_result(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="show_exists")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-duplicate",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "duplicate_tag_conflict"
    assert outcome.failure_class == "publish"


def test_ollama_publish_runner_rejects_existing_tag_digest_mismatch(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="show_exists")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    source_fingerprint = runners._source_artifact_fingerprint(spec)
    store.put_job(
        job_uid="job-publish-existing",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(
        job_uid="job-publish-existing",
        owner="worker.01",
        runner_kind="ollama_publish",
    )
    store.mark_publish_succeeded(
        job_uid="job-publish-existing",
        owner="worker.01",
        publish_job_uid=spec["publish_job_uid"],
        target_ref=spec["target"]["target_ref"],
        publish_spec_hash_sha256=spec["publish_spec_hash_sha256"],
        source_artifact_fingerprint_sha256=source_fingerprint,
        provenance_fingerprint_sha256=runners._publish_provenance_fingerprint(
            spec=spec,
            source_artifact_fingerprint_sha256=source_fingerprint,
        ),
        ollama_digest="sha256:previousdigest",
        idempotent=False,
    )
    job = store.put_job(
        job_uid="job-publish-digest-mismatch",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "duplicate_tag_conflict"
    assert outcome.failure_class == "publish"


def test_ollama_publish_runner_reports_disk_full_without_success(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="disk_full")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-disk-full",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "disk_full"
    assert outcome.failure_class == "publish"


def test_ollama_publish_runner_rejects_unsucceeded_source_job(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
        complete_source_job=False,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-source-not-succeeded",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "lineage_mismatch"
    assert outcome.failure_class == "artifact"


def test_ollama_publish_runner_rejects_cleanup_pending_artifact(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    gguf_ref = next(
        artifact["artifact_ref"]
        for artifact in artifacts
        if artifact["role"] == "gguf"
    )
    gguf_record = store.get_artifact_by_ref(gguf_ref)
    store.mark_artifact_cleanup_pending(artifact_uid=gguf_record.artifact_uid)
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-cleanup-pending-artifact",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "artifact_missing"
    assert outcome.failure_class == "artifact"


def test_ollama_publish_runner_rejects_unbound_run_lock(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
        bind_run_lock=False,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-unbound-run-lock",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "lineage_mismatch"
    assert outcome.failure_class == "artifact"


def test_ollama_publish_runner_rejects_unversioned_run_lock(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
        run_lock_schema_version=None,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-unversioned-run-lock",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "lineage_mismatch"
    assert outcome.failure_class == "artifact"


def test_ollama_publish_runner_cancel_after_create_is_unknown(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _install_fake_ollama(monkeypatch, mode="create_canceled")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    spec, _keyring = publish_test_support.signed_publish_spec(artifacts=artifacts)
    job = store.put_job(
        job_uid="job-publish-create-canceled",
        job_spec=spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "publish_verification_unknown"
    assert outcome.failure_class == "publish"


def test_ollama_publish_runner_revalidates_current_keyring_before_publish(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    calls = _install_fake_ollama(monkeypatch, mode="create_success")
    cfg = publish_test_support.publish_cfg(tmp_path)
    artifacts = publish_test_support.stage_publish_artifacts(
        tmp_path=tmp_path,
        store=store,
        cfg=cfg,
    )
    old_spec, _old_keyring = publish_test_support.signed_publish_spec(
        artifacts=artifacts,
    )
    _new_spec, new_keyring = publish_test_support.signed_publish_spec(
        artifacts=artifacts,
    )
    keyring_path = tmp_path / "publish-verify.json"
    _write_publish_keyring_file(keyring_path, new_keyring)
    cfg.publish_verify_keys_file = keyring_path
    cfg.publish_verify_keyring_config_error_code = None
    job = store.put_job(
        job_uid="job-publish-stale-signature",
        job_spec=old_spec,
        capability_hash_sha256=publish_test_support.CAPABILITY_HASH,
    )

    async def run():
        return await runners.OllamaPublishRunner(cfg, store).run(job, asyncio.Event())

    outcome = asyncio.run(run())

    assert outcome.state == "failed"
    assert outcome.failure_code == "publish_spec_signature_invalid"
    assert outcome.failure_class == "publish"
    assert calls == []


def _write_publish_keyring_file(path: Path, keyring) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": keyring.current.key_id,
                    "public_key_hex": keyring.current.public_key.hex(),
                    "active": True,
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_ollama_env_allows_only_loopback_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    assert runners._ollama_env()["OLLAMA_HOST"] == "http://127.0.0.1:11434"

    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.20:11434")

    assert "OLLAMA_HOST" not in runners._ollama_env()


def test_ollama_env_rejects_nonlocal_and_sensitive_hosts(monkeypatch):
    for host in (
        "http://example.com:11434",
        "0.0.0.0:11434",
        "http://localhost:11434/path",
        "http://localhost:11434?token=value",
    ):
        monkeypatch.setenv("OLLAMA_HOST", host)
        assert "OLLAMA_HOST" not in runners._ollama_env()


def test_ollama_env_sets_home_from_worker_data_dir(tmp_path):
    cfg = publish_test_support.publish_cfg(tmp_path)

    assert runners._ollama_env(cfg)["HOME"] == str(tmp_path.resolve(strict=False))


def _store(tmp_path):
    return queue_store.QueueStore.open(
        tmp_path,
        queue_store.QueueSettings(
            max_jobs=10,
            lease_ttl_seconds=30,
            resume_limit=2,
            sqlite_busy_timeout_ms=1000,
        ),
    )


def _install_fake_ollama(monkeypatch, *, mode: str) -> list[list[str]]:
    calls: list[list[str]] = []
    created = False

    async def fake_run_ollama(self, args, context, cancel_event):
        nonlocal created
        calls.append(list(args))
        if cancel_event.is_set():
            return runners._OllamaCommandResult(1, b"", b"", canceled=True)
        if args[:1] == ["show"]:
            if mode == "show_exists" or created:
                return runners._OllamaCommandResult(
                    0,
                    b'{"digest":"sha256:testdigest"}',
                    b"",
                )
            return runners._OllamaCommandResult(1, b"", b"")
        if args[:1] == ["create"]:
            if mode == "create_canceled":
                return runners._OllamaCommandResult(1, b"", b"", canceled=True)
            if mode == "disk_full":
                return runners._OllamaCommandResult(
                    1,
                    b"",
                    b"no space left on device",
                )
            assert args == [
                "create",
                publish_test_support.PUBLISH_TARGET_REF,
                "-f",
                "Modelfile",
            ]
            assert (context.workspace / "Modelfile").read_text(
                encoding="utf-8"
            ).startswith("FROM ./model.gguf")
            created = True
            return runners._OllamaCommandResult(0, b"", b"")
        return runners._OllamaCommandResult(2, b"", b"bad args")

    monkeypatch.setattr(runners.OllamaPublishRunner, "_run_ollama", fake_run_ollama)
    return calls


def _wait_pid_gone(pid: int, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if not _pid_exists(pid):
            return True
        time.sleep(0.02)
    return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8",
            errors="replace",
        )
        state = raw.rsplit(")", 1)[1].strip().split()[0]
    except (OSError, IndexError):
        return False
    if state == "Z":
        return False
    return True
