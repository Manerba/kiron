from __future__ import annotations

import pytest

import job_stubs


VALID_SPEC = {
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
    "labels": {"kitt_job": "job.v1", "tenant": "kiron"},
    "metadata": {"priority": "low", "dry_run": True, "ordinal": 1},
}
KITT_SPEC = VALID_SPEC


def _join(*parts: str) -> str:
    return "".join(parts)


def test_canonical_job_spec_hash_is_stable_for_semantically_equal_objects():
    left_json, left_hash = job_stubs.canonicalize_job_spec(
        {**VALID_SPEC, "metadata": {"ordinal": 1, "priority": "low"}}
    )
    right_json, right_hash = job_stubs.canonicalize_job_spec(
        {**VALID_SPEC, "metadata": {"priority": "low", "ordinal": 1}}
    )

    assert left_json == right_json
    assert left_hash == right_hash
    assert len(left_hash) == 64


def test_allowed_job_spec_accepts_only_explicit_fields():
    canonical, digest = job_stubs.canonicalize_job_spec(VALID_SPEC)

    assert "idempotency_key" not in canonical
    assert digest == job_stubs.job_spec_hash(VALID_SPEC)


def test_legacy_metadata_only_job_spec_is_rejected():
    with pytest.raises(job_stubs.JobValidationError):
        job_stubs.canonicalize_job_spec(
            {
                "job_kind": "metadata_only",
                "model_ref": "model.v1",
                "dataset_ref": "dataset.v1",
                "trainer_profile_ref": "trainer.v1",
                "limits": {"max_steps": 0},
                "labels": {"kitt_job": "job.v1"},
                "metadata": {"operator": "kiron"},
            }
        )


def test_kitt_job_spec_accepts_only_sft_and_known_output_roles():
    job_stubs.canonicalize_job_spec(KITT_SPEC)
    job_stubs.canonicalize_job_spec(
        {
            **KITT_SPEC,
            "output_roles": [
                "adapter",
                "merged_weights",
                "gguf",
                "ollama_modelfile",
                "training_log",
                "metrics_jsonl",
                "run_lock",
            ],
        }
    )

    for bad_spec in (
        {**KITT_SPEC, "run_type": "dpo"},
        {**KITT_SPEC, "run_type": "cp"},
        {**KITT_SPEC, "run_type": "eval"},
        {**KITT_SPEC, "output_roles": ["manifest"]},
        {**KITT_SPEC, "output_roles": ["metrics_jsonl"]},
        {**KITT_SPEC, "output_roles": ["adapter", "adapter"]},
        {**KITT_SPEC, "output_roles": ["adapter", "unknown_role"]},
    ):
        with pytest.raises(job_stubs.JobValidationError):
            job_stubs.canonicalize_job_spec(bad_spec)


@pytest.mark.parametrize(
    "bad_spec",
    [
        {},
        {key: value for key, value in VALID_SPEC.items() if key != "schema_version"},
        {key: value for key, value in VALID_SPEC.items() if key != "run_uid"},
        {key: value for key, value in VALID_SPEC.items() if key != "run_type"},
        {key: value for key, value in VALID_SPEC.items() if key != "training_profile"},
        {key: value for key, value in VALID_SPEC.items() if key != "input_reference"},
        {key: value for key, value in VALID_SPEC.items() if key != "base_or_parent_model"},
        {key: value for key, value in VALID_SPEC.items() if key != "hyperparameters"},
        {key: value for key, value in VALID_SPEC.items() if key != "output_roles"},
        {**VALID_SPEC, "training_profile": {}},
        {**VALID_SPEC, "input_reference": {}},
        {**VALID_SPEC, "base_or_parent_model": {}},
        {**VALID_SPEC, "hyperparameters": {}},
        {**VALID_SPEC, "training_profile": {"profile_uid": "profile-1"}},
        {
            **VALID_SPEC,
            "training_profile": {
                "profile_uid": "profile-1",
                "version_label": "v1",
                "profile_hash_sha256": "A" * 64,
            },
        },
        {
            **VALID_SPEC,
            "input_reference": {
                "mode": "file",
                "dataset_version_uid": "5136e167-dbb7-49a6-a752-6fe1e962adb7",
            },
        },
        {
            **VALID_SPEC,
            "input_reference": {
                "mode": "dataset_version",
                "dataset_ref": "5136e167-dbb7-49a6-a752-6fe1e962adb7",
            },
        },
        {
            **VALID_SPEC,
            "base_or_parent_model": {"model_ref": "Qwen/Qwen2.5-7B-Instruct"},
        },
        {**VALID_SPEC, "hyperparameters": {"learning_rate": None}},
        {**VALID_SPEC, "hyperparameters": {"dry_run": True}},
        {
            key: value
            for key, value in VALID_SPEC.items()
            if key != "capability_snapshot_hash_sha256"
        },
        {"job_kind": "metadata_only"},
        {"unknown": "value"},
        {**VALID_SPEC, _join("pro", "mpt"): "raw"},
        {**VALID_SPEC, "labels": {_join("api", "_", "key"): "value"}},
        {**VALID_SPEC, "labels": {"nested": {"value": "x"}}},
        {**VALID_SPEC, "labels": {"items": ["free", "text"]}},
        {
            **VALID_SPEC,
            "base_or_parent_model": {
                "external_parent_ref": _join(
                    "/", "usr", "/", "lib", "/", "kiron", "/", "data", "/", "model"
                )
            },
        },
        {
            **VALID_SPEC,
            "input_reference": {
                "mode": "dataset_version",
                "dataset_version_uid": _join(
                    "file",
                    ":",
                    "//",
                    "tmp",
                    "/",
                    "dataset.json",
                ),
            },
        },
        {
            **VALID_SPEC,
            "training_profile": {
                "profile_uid": _join("post", "gresql", "://", "user:pass@db/kitt")
            },
        },
        {**VALID_SPEC, "metadata": {_join("raw", "_", "prompt"): "value"}},
        {**VALID_SPEC, "metadata": {"note": "contains whitespace"}},
        {**VALID_SPEC, "metadata": {"nullable": None}},
        {**VALID_SPEC, "labels": {"nullable": None}},
        {**VALID_SPEC, "metadata": {_join("to", "ken", "_hint"): "value"}},
        {**VALID_SPEC, "metadata": {"safe": _join("to", "ken")}},
        {
            **VALID_SPEC,
            "base_or_parent_model": {"external_parent_ref": _join("model.", "to", "ken", "ized")},
        },
        {
            **VALID_SPEC,
            "input_reference": {
                "mode": "dataset_version",
                "dataset_version_uid": _join("dataset.", "sec", "ret", "ive"),
            },
        },
        {**VALID_SPEC, "metadata": {"safe": "value/with/slash"}},
        {**VALID_SPEC, "metadata": {"safe": "value?query"}},
        {**VALID_SPEC, "metadata": {"safe": "value#fragment"}},
        {**VALID_SPEC, "metadata": {"safe": "value..parent"}},
    ],
)
def test_job_spec_rejects_secrets_paths_urls_raw_text_and_unknown_shapes(bad_spec):
    with pytest.raises(job_stubs.JobValidationError):
        job_stubs.canonicalize_job_spec(bad_spec)


def test_last_checkpoint_ref_uses_strict_logical_ref_allowlist():
    assert job_stubs.validate_last_checkpoint_ref("checkpoint:run-1")
    for value in (
        _join("/", "usr", "/", "lib", "/", "kiron", "/", "data", "/", "checkpoint"),
        "checkpoint/one",
        "checkpoint one",
        _join("https", "://", "example.invalid", "/", "checkpoint"),
        _join("checkpoint?sig", "nature=raw"),
        "-checkpoint",
        _join("to", "ken"),
        _join("checkpoint:", "to", "ken", "ized"),
        _join("checkpoint:", "sec", "ret", "ive"),
        _join("checkpoint:", "key", "id", "ed"),
    ):
        assert not job_stubs.validate_last_checkpoint_ref(value)


def test_lease_owner_uses_internal_safe_owner_allowlist():
    assert job_stubs.validate_lease_owner("worker.01")
    for value in (
        "x",
        "Worker.01",
        "worker/01",
        "worker 01",
        _join("worker?to", "ken=raw"),
        _join("/", "run", "/", "kiron", "/", "kitt-worker"),
        _join("sec", "ret", "-worker"),
        _join("worker.", "to", "ken", "ized"),
        _join("worker.", "key", "id", "ed"),
    ):
        assert not job_stubs.validate_lease_owner(value)
