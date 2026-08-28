"""Kiron Selbsttest — pytest-Suite als Background-Thread mit Status-Polling.

Adaptiert von kiara/web/api_testsuite.py. Faehrt pytest im eigenen Service-
Verzeichnis aus (test_*.py direkt neben diesem Modul), parst stdout live und
stellt Fortschritt + Details ueber /api/tests/{run,status} bereit.
"""
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from catalog_health import check_live_catalog_consistency

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tests", tags=["selftest"])

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent.parent
_PYTEST_BIN = str(_SERVICE_DIR / "venv" / "bin" / "pytest")
_TESTS_DIR = str(_SERVICE_DIR)
_RESULTS_FILE = _PROJECT_ROOT / "data" / "kiron-proxy" / "selftest_results.json"

_test_status = {
    "running": False,
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
}
_status_lock = threading.Lock()

# Pytest-Output: "test_foo.py::Class::test_bar PASSED     [  7%]"
_RESULT_RE = re.compile(
    r"^(\S+\.py::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s"
)
# "71 tests collected in 0.11s"
_COLLECT_RE = re.compile(r"(\d+)\s+tests?\s+collected")


def _save_results():
    try:
        _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {k: v for k, v in _test_status.items() if k != "current_test"}
        snapshot["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = _RESULTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        os.chmod(tmp, 0o640)
        tmp.replace(_RESULTS_FILE)
    except Exception as exc:
        logger.warning("Selftest-Ergebnis konnte nicht gespeichert werden: %s", exc)


def _load_results():
    try:
        if _RESULTS_FILE.is_file():
            data = json.loads(_RESULTS_FILE.read_text())
            for key in ("progress", "total", "passed", "failed", "errors",
                        "skipped", "duration", "results", "error"):
                if key in data:
                    _test_status[key] = data[key]
    except Exception as exc:
        logger.warning("Selftest-Ergebnis konnte nicht geladen werden: %s", exc)


_load_results()


def _run_tests():
    try:
        start = time.monotonic()
        with _status_lock:
            _test_status["current_test"] = "catalog_consistency::managed_services"
            _test_status["total"] = 1

        catalog_report = check_live_catalog_consistency()
        catalog_ok = catalog_report.get("consistent") is True
        with _status_lock:
            _test_status["progress"] = 1
            if catalog_ok:
                _test_status["passed"] = 1
                _test_status["results"].append({
                    "name": "catalog_consistency::managed_services",
                    "status": "passed",
                    "details": "",
                })
            else:
                details = json.dumps(
                    catalog_report,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                _test_status["failed"] = 1
                _test_status["results"].append({
                    "name": "catalog_consistency::managed_services",
                    "status": "failed",
                    "details": details,
                })
                _test_status["error"] = (
                    "Catalog-Konsistenz-Gate fehlgeschlagen; pytest wurde "
                    "fail-closed nicht gestartet."
                )
                _test_status["duration"] = round(time.monotonic() - start, 1)
        if not catalog_ok:
            return

        # Phase 1: Tests zaehlen
        collect = subprocess.run(
            [_PYTEST_BIN, "--collect-only", "-q",
             "--continue-on-collection-errors", _TESTS_DIR],
            capture_output=True, text=True, timeout=60,
            cwd=_TESTS_DIR,
        )
        for line in collect.stdout.splitlines():
            m = _COLLECT_RE.search(line)
            if m:
                with _status_lock:
                    _test_status["total"] = int(m.group(1)) + 1
                break

        # Phase 2: Tests ausfuehren
        env = {**os.environ, "COLUMNS": "500"}
        proc = subprocess.Popen(
            [_PYTEST_BIN, _TESTS_DIR, "-v", "--tb=short", "--no-header",
             "--continue-on-collection-errors",
             "--timeout=30", "--timeout-method=signal"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=_TESTS_DIR,
            env=env,
        )

        tail_buffer = []
        in_summary = False
        in_failures = False
        current_block_name = ""
        current_block_lines = []
        traceback_blocks = {}
        summary_details = {}

        for line in proc.stdout:
            line = line.rstrip("\n")
            tail_buffer.append(line)
            if len(tail_buffer) > 30:
                tail_buffer.pop(0)

            if "= short test summary info =" in line:
                in_summary = True
                if current_block_name and current_block_lines:
                    traceback_blocks[current_block_name] = "\n".join(
                        current_block_lines,
                    )[:3000]
                in_failures = False
                continue
            if "= FAILURES =" in line or "= ERRORS =" in line:
                in_failures = True
                continue

            if in_summary:
                for prefix in ("FAILED ", "ERROR "):
                    if line.startswith(prefix):
                        rest = line[len(prefix):]
                        if " - " in rest:
                            tname, detail = rest.split(" - ", 1)
                        else:
                            tname, detail = rest, ""
                        summary_details[tname.strip()] = detail.strip()
                        break
                continue

            if in_failures:
                if line.startswith("___") and line.endswith("___"):
                    if current_block_name and current_block_lines:
                        traceback_blocks[current_block_name] = "\n".join(
                            current_block_lines,
                        )[:3000]
                    current_block_name = line.strip("_ ").strip()
                    current_block_lines = []
                else:
                    current_block_lines.append(line)
                continue

            m = _RESULT_RE.match(line)
            if not m:
                continue

            name, status = m.group(1), m.group(2)
            with _status_lock:
                _test_status["current_test"] = name
                _test_status["progress"] += 1

                if status in ("PASSED", "XPASS"):
                    _test_status["passed"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "passed", "details": ""},
                    )
                elif status == "FAILED":
                    _test_status["failed"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "failed", "details": ""},
                    )
                elif status == "ERROR":
                    _test_status["errors"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "error", "details": ""},
                    )
                elif status in ("SKIPPED", "XFAIL"):
                    _test_status["skipped"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "skipped", "details": ""},
                    )

        try:
            rc = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with _status_lock:
                _test_status["error"] = "Testsuite nach 600s abgebrochen (Timeout)"
                _test_status["duration"] = round(time.monotonic() - start, 1)
                _test_status["running"] = False
            _save_results()
            return

        with _status_lock:
            _assign_failure_details(traceback_blocks, summary_details)

            # progress==1 bedeutet: nur das vorangestellte Catalog-Gate lief,
            # aber pytest konnte keinen Test ausfuehren.
            if rc != 0 and _test_status["progress"] == 1:
                error_lines = [
                    ln for ln in tail_buffer
                    if ln.strip() and not ln.startswith("=")
                    and "warning" not in ln.lower()
                ]
                detail = "\n".join(error_lines[-10:]) if error_lines else ""
                _test_status["error"] = (
                    f"pytest abgebrochen (Exit-Code {rc}), "
                    f"kein Test ausgefuehrt.\n{detail}"
                ).strip()

            _test_status["duration"] = round(time.monotonic() - start, 1)

    except FileNotFoundError:
        with _status_lock:
            _test_status["error"] = f"pytest nicht gefunden: {_PYTEST_BIN}"
    except subprocess.TimeoutExpired:
        with _status_lock:
            _test_status["error"] = "Testsammlung Timeout (>60s)"
    except Exception as exc:
        with _status_lock:
            _test_status["error"] = str(exc)
    finally:
        with _status_lock:
            _test_status["running"] = False
            _test_status["current_test"] = ""
        _save_results()


def _assign_failure_details(traceback_blocks: dict, summary_details: dict):
    failed_results = [
        r for r in _test_status["results"]
        if r["status"] in ("failed", "error") and not r["details"]
    ]
    # Pytest nutzt "." zwischen Klasse und Methode im FAILURES-Header,
    # Ergebnis-Namen "::" — normalisieren fuers Matching.
    norm_blocks = {}
    for bk, bv in traceback_blocks.items():
        norm_blocks[bk.replace(".", "::")] = bv

    for result in failed_results:
        name = result["name"]
        parts = name.split("::", 1)
        suffix = parts[1] if len(parts) > 1 else name

        for block_key, block_text in norm_blocks.items():
            if suffix in block_key or block_key in suffix:
                result["details"] = block_text
                break

        if not result["details"]:
            detail = summary_details.get(name, "")
            if not detail:
                for skey, sval in summary_details.items():
                    if name in skey or skey.endswith(name):
                        detail = sval
                        break
            result["details"] = detail


@router.post("/run")
async def run_tests():
    with _status_lock:
        if _test_status["running"]:
            raise HTTPException(status_code=409, detail="Selbsttest laeuft bereits")
        _test_status.update({
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

    threading.Thread(target=_run_tests, daemon=True).start()
    return {"message": "Selbsttest gestartet"}


@router.get("/status")
async def get_status():
    with _status_lock:
        status = dict(_test_status)
        status["results"] = [dict(r) for r in status.get("results", [])]
    return status
