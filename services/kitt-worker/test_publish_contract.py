from __future__ import annotations

import hashlib

import pytest

import publish_contract
import publish_test_support


def test_publish_signature_verifies_with_public_key_only():
    spec, keyring = publish_test_support.signed_publish_spec()
    envelope = spec["signature_envelope"]

    assert envelope["schema_version"] == "kitt_publish_spec_signature_v1"
    assert envelope["algorithm"] == "ed25519"
    assert len(keyring.current.public_key) == 32
    assert len(envelope["signature_base64url"]) > 80
    assert envelope["signature_digest_sha256"] == hashlib.sha256(
        publish_contract._base64url_decode_signature(envelope["signature_base64url"])
    ).hexdigest()
    publish_contract.verify_publish_spec_signature(spec, keyring=keyring)


def test_publish_signature_rejects_wrong_purpose_and_algorithm():
    spec, keyring = publish_test_support.signed_publish_spec()
    wrong_purpose = {
        **spec,
        "signature_envelope": {
            **spec["signature_envelope"],
            "purpose": "kitt_job_signature_v1",
        },
    }
    wrong_algorithm = {
        **spec,
        "signature_envelope": {
            **spec["signature_envelope"],
            "algorithm": "HMAC-SHA256",
        },
    }

    with pytest.raises(publish_contract.PublishSpecError) as purpose_error:
        publish_contract.verify_publish_spec_signature(
            wrong_purpose,
            keyring=keyring,
        )
    with pytest.raises(publish_contract.PublishSpecError) as algorithm_error:
        publish_contract.verify_publish_spec_signature(
            wrong_algorithm,
            keyring=keyring,
        )

    assert purpose_error.value.reason_code == "publish_spec_signature_shape_invalid"
    assert (
        algorithm_error.value.reason_code
        == "publish_spec_signature_algorithm_unsupported"
    )


def test_publish_signature_rejects_hash_tampering_and_unknown_key():
    spec, keyring = publish_test_support.signed_publish_spec()
    tampered = {**spec, "publish_intent_uid": "publish-intent-2"}
    unknown_key = {
        **spec,
        "signature_envelope": {
            **spec["signature_envelope"],
            "key_id": "unknown01",
        },
    }

    with pytest.raises(publish_contract.PublishSpecError) as tamper_error:
        publish_contract.verify_publish_spec_signature(tampered, keyring=keyring)
    with pytest.raises(publish_contract.PublishSpecError) as key_error:
        publish_contract.verify_publish_spec_signature(unknown_key, keyring=keyring)

    assert tamper_error.value.reason_code == "publish_spec_hash_mismatch"
    assert key_error.value.reason_code == "publish_spec_signature_key_unknown"
