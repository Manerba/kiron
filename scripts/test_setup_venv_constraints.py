"""Exercise the real setup preflight/build shell with temporary, inert commands."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/setup-venvs.sh").read_text()
SERVICES = ("kiron-proxy", "kiron-docling", "kiron-embeddings", "kiron-deberta",
            "kiron-prism", "kitt-worker")
PREFLIGHT = SOURCE.split("\nservice_group() {", 1)[0]
BUILD = SOURCE.split("# #886 Phase 1:", 1)[1].split("\nfor svc", 1)[1]
BUILD = "for svc" + BUILD.split("# #886 Phase 2:", 1)[0]


def constraints(tmp_path):
    root = tmp_path / "constraints"
    root.mkdir()
    for service in SERVICES:
        (root / (service + ".txt")).write_text(
            "# exact operator pins\npip==26.2.1\nsetuptools==84.0.0\nwheel==0.48.0\n"
            "torch==2.7.1+cu128\n"
        )
    return root


def run_preflight(root):
    env = {**os.environ, "KIRON_VENV_CONSTRAINTS_DIR": str(root)}
    return subprocess.run(["bash", "-c", PREFLIGHT + "\necho PRECHECK_OK\n"],
                          env=env, text=True, capture_output=True, timeout=10)


def test_all_six_exact_constraint_files_are_validated_before_any_stop(tmp_path):
    root = constraints(tmp_path)
    result = run_preflight(root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PRECHECK_OK"
    assert SOURCE.index("\nvalidate_venv_constraints\n") < SOURCE.index("WAS_ACTIVE=()")


@pytest.mark.parametrize("bad", ["missing", "extra", "symlink", "directory", "fifo",
                                  "empty", "range", "duplicate", "url", "option",
                                  "invalid_version", "oversize"])
def test_bad_constraint_set_fails_before_setup_actions(tmp_path, bad):
    root = constraints(tmp_path)
    path = root / "kitt-worker.txt"
    if bad in {"missing", "symlink", "directory", "fifo"}:
        path.unlink()
    if bad == "extra":
        (root / "unexpected.txt").write_text("pip==26.2.1\n")
    elif bad == "symlink":
        path.symlink_to(root / "kiron-proxy.txt")
    elif bad == "directory":
        path.mkdir()
    elif bad == "fifo":
        os.mkfifo(path)
    elif bad == "empty":
        path.write_text("")
    elif bad == "oversize":
        path.write_text("#" * (1024 * 1024 + 1))
    elif bad in {"range", "duplicate", "url", "option", "invalid_version"}:
        suffix = {"range": "httpx>=0.28", "duplicate": "PIP==26.2.1",
                  "url": "httpx @ https://example.invalid/httpx.whl",
                  "option": "--extra-index-url https://example.invalid",
                  "invalid_version": "httpx==banana"}[bad]
        path.write_text(path.read_text() + suffix + "\n")
    result = run_preflight(root)
    assert result.returncode != 0
    assert "FEHLER:" in result.stderr
    assert "PRECHECK_OK" not in result.stdout


def test_constraint_directory_must_be_absolute_and_not_symlinked(tmp_path):
    root = constraints(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    assert run_preflight(link).returncode != 0
    assert run_preflight("relative-path").returncode != 0


@pytest.mark.parametrize("configured", [True, False])
def test_each_pip_operation_has_own_pins_and_preserves_outer_environment(tmp_path, configured):
    root = constraints(tmp_path)
    work = tmp_path / "work"
    for service in SERVICES:
        directory = work / "services" / service
        directory.mkdir(parents=True)
        (directory / "requirements.txt").write_text("httpx>=0.27\n")
    spy = tmp_path / "spy"
    log = tmp_path / "calls.jsonl"
    spy.write_text(f"#!{sys.executable}\n" +
                   "import json,os,sys\n"
                   "with open(os.environ['CALL_LOG'],'a') as stream:\n"
                   " stream.write(json.dumps({'argv':sys.argv,'constraint':os.environ.get('PIP_CONSTRAINT'),"
                   "'build_constraint':os.environ.get('PIP_BUILD_CONSTRAINT')})+'\\n')\n")
    spy.chmod(0o700)
    wrapper = tmp_path / "entrypoint"
    wrapper.write_text("#!/bin/sh\nexit 0\n")
    shell = '''set -e
SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kiron-prism kitt-worker)
SRC="$WORK"
DST="$WORK"
COMMON_SRC="$WORK/common"
REGISTRY_CLI_ENTRYPOINT_SOURCE="$WRAPPER"
python3() {
    test "$1" = -m && test "$2" = venv
    mkdir -p "$3/bin"
    ln -s "$SPY" "$3/bin/pip"
    ln -s "$SPY" "$3/bin/python"
}
'''
    env = {**os.environ, "WORK": str(work), "SPY": str(spy), "WRAPPER": str(wrapper),
           "CALL_LOG": str(log), "PIP_CONSTRAINT": "/outer/runtime-pins.txt",
           "PIP_BUILD_CONSTRAINT": "/outer/build-pins.txt",
           "KIRON_VENV_CONSTRAINTS_DIR": str(root) if configured else ""}
    # install's owner arguments belong to production permissions, not this
    # unprivileged temporary fixture; only the root CLI wrapper copy is inert.
    shell += 'install() { :; }\n'
    shell += BUILD + '\nprintf "OUTER=%s,%s\\n" "$PIP_CONSTRAINT" "$PIP_BUILD_CONSTRAINT"\n'
    result = subprocess.run(["bash", "-c", shell], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    pip_rows = [row for row in rows if row["argv"][0].endswith("/pip")]
    assert len(pip_rows) == 18
    for service in SERVICES:
        calls = [row for row in pip_rows if f"/{service}/" in row["argv"][0]]
        assert len(calls) == 3
        assert any("--upgrade" in row["argv"] for row in calls)
        assert any("-r" in row["argv"] for row in calls)
        assert any(str(work / "common") in row["argv"] for row in calls)
        for row in calls:
            assert row["constraint"] == (str(root / (service + ".txt")) if configured else "/outer/runtime-pins.txt")
            assert row["build_constraint"] == (str(root / (service + ".txt")) if configured else "/outer/build-pins.txt")
    assert "OUTER=/outer/runtime-pins.txt,/outer/build-pins.txt" in result.stdout
