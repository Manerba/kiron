from __future__ import annotations

import logging
from pathlib import Path

import monitoring


ROOT = Path(__file__).resolve().parents[2]


def _join(*parts: str) -> str:
    return "".join(parts)


def test_redaction_removes_secrets_paths_tracebacks_and_raw_payload_markers():
    token = _join("kidvalue01", ".", "s" * 40)
    token_hash = "a" * 64
    local_path = _join("/", "usr", "/", "lib", "/", "kiron", "/", "data", "/", "kitt-worker")
    dsn = _join("postgres", "://", "user:pass", "@", "db.local", "/", "kiron")
    signed = _join("https", "://", "example.invalid", "/artifact?signature=abc")
    traceback_text = (
        "Traceback (most recent call last):\n"
        f"  File \"{local_path}/main.py\", line 1, in <module>\n"
        "RuntimeError: boom"
    )
    raw = (
        f"Authorization: Bearer {token} hash={token_hash} path={local_path} "
        f"dsn={dsn} url={signed} raw_prompt=hello raw_response=world "
        f"cp_raw_text=secret artifact_bytes=deadbeef {traceback_text}"
    )

    redacted = monitoring.sanitize_log_text(raw)

    for forbidden in (
        token,
        token_hash,
        local_path,
        dsn,
        signed,
        "Authorization",
        "Bearer",
        "raw_prompt",
        "raw_response",
        "cp_raw_text",
        "artifact_bytes",
        "Traceback",
        "main.py",
        "RuntimeError",
    ):
        assert forbidden not in redacted


def test_log_redaction_removes_cp_raw_text_payload_and_exception_class():
    redacted = monitoring.sanitize_log_text("cp_raw_text=plainpayload RuntimeError: boom")

    for forbidden in ("cp_raw_text", "plainpayload", "RuntimeError"):
        assert forbidden not in redacted


def test_log_redaction_removes_json_colon_and_nested_raw_payload_values():
    message = (
        '"raw_prompt": "hello secret", "artifact_bytes": "deadbeef" '
        'canonical_job_spec_json={"cp_raw_text":"nested body",'
        '"raw_response":"nested world"} '
        "cp_raw_text: multi word payload"
    )

    redacted = monitoring.sanitize_log_text(message)

    for forbidden in (
        "raw_prompt",
        "artifact_bytes",
        "canonical_job_spec_json",
        "cp_raw_text",
        "raw_response",
        "hello secret",
        "deadbeef",
        "nested body",
        "nested world",
        "multi word payload",
    ):
        assert forbidden not in redacted


def test_redaction_failure_event_drops_original_payload():
    events: list[dict] = []

    monitoring.emit_event(
        "kitt_worker_alert",
        {
            "alert_code": "queue_full",
            "state": "active",
            "severity": "warning",
            "reason_code": "queue_full",
            "source": "queue",
            "unsafe_path": _join("/", "opt", "/", "kiron", "/", "secret.txt"),
        },
        event_sink=events.append,
    )

    assert events == [
        {
            "event": "kitt_worker_alert",
            "alert_code": "redaction_failure",
            "state": "active",
            "severity": "critical",
            "reason_code": "redaction_failure",
        }
    ]
    assert "unsafe_path" not in str(events)
    assert "/opt/kiron" not in str(events)


def test_event_payload_rejects_sensitive_request_id_substrings():
    events: list[dict] = []

    monitoring.emit_event(
        "kitt_worker_alert",
        {
            "alert_code": "queue_full",
            "state": "active",
            "severity": "warning",
            "reason_code": "queue_full",
            "request_id": f"trace:kidvalue01.{'s' * 40}",
        },
        event_sink=events.append,
    )

    assert events == [
        {
            "event": "kitt_worker_alert",
            "alert_code": "redaction_failure",
            "state": "active",
            "severity": "critical",
            "reason_code": "redaction_failure",
        }
    ]
    assert "kidvalue01" not in str(events)


def test_logging_filter_redacts_root_kitt_worker_and_uvicorn_loggers(caplog):
    monitoring.install_logging_redaction(logging.INFO)
    path = _join("/", "opt", "/", "kiron", "/", "services", "/", "kitt-worker", "/", "main.py")
    message = (
        f"Traceback (most recent call last): File \"{path}\" "
        f"Authorization: Bearer kidvalue01.{'s' * 40} "
        "cp_raw_text=plainpayload RuntimeError: boom"
    )

    with caplog.at_level(logging.ERROR):
        logging.getLogger().error(message, exc_info=True)
        logging.getLogger("kitt_worker").error(message, exc_info=True)
        logging.getLogger("uvicorn.error").error(message, exc_info=True)

    for forbidden in (
        path,
        "Authorization",
        "Bearer",
        "kidvalue01",
        "Traceback",
        "File",
        "cp_raw_text",
        "plainpayload",
        "RuntimeError",
    ):
        assert forbidden not in caplog.text


def test_uvicorn_access_logger_is_disabled_by_redaction_install():
    monitoring.install_logging_redaction(logging.INFO)

    access_log = logging.getLogger("uvicorn.access")

    assert access_log.disabled is True
    assert access_log.propagate is False


def test_kitt_worker_sources_do_not_use_logger_exception_for_journald_paths():
    service_files = [
        ROOT / "services" / "kitt-worker" / "main.py",
        ROOT / "services" / "kitt-worker" / "contract.py",
        ROOT / "services" / "kitt-worker" / "executor.py",
        ROOT / "services" / "kitt-worker" / "v1.py",
    ]
    service_text = "\n".join(path.read_text(encoding="utf-8") for path in service_files)

    assert "logger.exception" not in service_text
    assert "traceback.format_exception" not in service_text
