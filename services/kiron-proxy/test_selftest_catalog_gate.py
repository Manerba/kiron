from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import selftest


def test_selftest_mismatch_fails_closed_before_pytest() -> None:
    mismatch = "sha256:" + "0" * 64
    report = {
        "consistent": False,
        "status": "inconsistent",
        "expected_digest": "sha256:" + "1" * 64,
        "services": {
            "kiron-deberta": {
                "reported_digest": mismatch,
                "state": "mismatch",
                "error": "simulated mismatch",
            },
        },
        "errors": ["simulated mismatch"],
    }
    selftest._test_status.update({
        "running": True,
        "progress": 0,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "current_test": "",
        "duration": 0.0,
        "results": [],
        "error": None,
    })

    with mock.patch.object(
        selftest,
        "check_live_catalog_consistency",
        return_value=report,
    ), mock.patch.object(selftest.subprocess, "run") as collect, mock.patch.object(
        selftest.subprocess,
        "Popen",
    ) as execute, mock.patch.object(selftest, "_save_results"):
        selftest._run_tests()

    collect.assert_not_called()
    execute.assert_not_called()
    assert selftest._test_status["running"] is False
    assert selftest._test_status["failed"] == 1
    assert selftest._test_status["progress"] == 1
    assert "fail-closed" in selftest._test_status["error"]
    assert mismatch in selftest._test_status["results"][0]["details"]


def test_successful_gate_does_not_hide_pytest_startup_failure() -> None:
    class FailedPytest:
        stdout: list[str] = []

        def wait(self, timeout):
            assert timeout == 600
            return 2

    selftest._test_status.update({
        "running": True,
        "progress": 0,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "current_test": "",
        "duration": 0.0,
        "results": [],
        "error": None,
    })

    with mock.patch.object(
        selftest,
        "check_live_catalog_consistency",
        return_value={"consistent": True},
    ), mock.patch.object(
        selftest.subprocess,
        "run",
        return_value=SimpleNamespace(stdout=""),
    ), mock.patch.object(
        selftest.subprocess,
        "Popen",
        return_value=FailedPytest(),
    ), mock.patch.object(selftest, "_save_results"):
        selftest._run_tests()

    assert selftest._test_status["passed"] == 1
    assert selftest._test_status["progress"] == 1
    assert "kein Test ausgefuehrt" in selftest._test_status["error"]
