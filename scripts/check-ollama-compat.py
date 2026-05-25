#!/usr/bin/env python3
"""Ollama compatibility validator for Kiron.

Runs a candidate Ollama image in an isolated Docker container and writes a
JSON plus Markdown report under data/ollama_compat_reports/.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "data" / "ollama_compat_reports"
TEST_PORT = 11445
CONTAINER_PORT = 11435
VOLUME = "kiron_ollama_compat_models"
FIXTURE_MODEL = "smollm2:135m"
VALIDATOR_VERSION = "v1"


@dataclass
class Check:
    ok: bool
    severity: str
    message: str
    data: dict[str, Any] | None = None


def _canonical_model_name(name: str) -> str:
    """Add an implicit ``:latest`` tag like Ollama does internally.

    `/api/ps` always lists the canonical name with tag, so both sides must
    be canonicalized before comparison. Only the last path segment carries
    the tag — a ``:`` in a registry host like ``localhost:5000/llama3``
    is not a tag separator. Mirrors kiron_common.ollama_compat to keep the
    validator a self-contained standalone script.
    """
    if not name:
        return name
    last = name.rsplit("/", 1)[-1]
    return name if ":" in last else f"{name}:latest"


def capability_num_gpu_zero_effective(capabilities: dict[str, Any]) -> bool:
    """Return True only for the validator's nested safe-offload field."""
    check = capabilities.get("num_gpu_zero_chat_generate")
    if not isinstance(check, dict):
        return False
    data = check.get("data")
    return isinstance(data, dict) and data.get("num_gpu_zero_effective") is True


def find_matching_report(
    report_dir: pathlib.Path,
    image: str,
    digest: str,
) -> tuple[pathlib.Path, bool] | None:
    """Find the newest green report for image+digest.

    The second return value is the strict `num_gpu=0` handoff safety flag.
    A top-level/check-level `ok=true` is not sufficient; deploy must read
    `capabilities.num_gpu_zero_chat_generate.data.num_gpu_zero_effective`.
    """
    # Misconfigured --report-dir (z.B. /usr/lib/kiron statt /opt/kiron) darf
    # nicht als "kein Match" durchgereicht werden — sonst sagt deploy-local.sh
    # "Validator vorher ausfuehren", obwohl die wahre Ursache ein falscher Pfad
    # ist. Explizit eskalieren, damit main()'s try/except auf exit 2 mit
    # traceback laeuft und deploy-local.sh den stderr ausgibt.
    if not report_dir.exists():
        raise FileNotFoundError(f"report directory does not exist: {report_dir}")
    if not report_dir.is_dir():
        raise NotADirectoryError(f"report path is not a directory: {report_dir}")
    matches: list[tuple[tuple[str, str, float], pathlib.Path, dict[str, Any]]] = []
    for path in report_dir.glob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("image") != image and data.get("compose_tag") != image:
            continue
        if data.get("image_digest") != digest:
            continue
        if data.get("report_status") != "passed" or data.get("upgrade_allowed") is not True:
            continue
        # Schema-Drift-Schutz: ein zukuenftiger v2-Validator kann
        # capability-Felder (z.B. num_gpu_zero_effective) strikter
        # definieren. Ohne Versionsabgleich wuerden v1-Reports den Gate
        # weiterhin clearen. Ein Bump von VALIDATOR_VERSION invalidiert
        # alte Reports automatisch und erzwingt einen Re-Validate-Lauf.
        if data.get("validator_version") != VALIDATOR_VERSION:
            continue
        capabilities = data.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}
        tested_at = data.get("tested_at")
        if not isinstance(tested_at, str):
            tested_at = ""
        # tested_at und Dateiname (started_at) sind stabil; mtime nur als letzte
        # Fallback-Quelle, weil checkout/rsync/touch sie zuruecksetzen.
        sort_key = (tested_at, path.name, path.stat().st_mtime)
        matches.append((sort_key, path, capabilities))
    if not matches:
        return None
    _, path, capabilities = sorted(matches)[-1]
    return path, capability_num_gpu_zero_effective(capabilities)


class Validator:
    def __init__(self, image: str, mode: str, pull_large: bool) -> None:
        self.image = image
        self.mode = mode
        self.pull_large = pull_large
        self.started_at = time.strftime("%Y%m%d-%H%M%S")
        self.container = f"kiron-ollama-compat-{time.time_ns()}"
        self.base_url = f"http://127.0.0.1:{TEST_PORT}"
        self.digest = ""
        self.capabilities: dict[str, Check] = {}
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def run(self) -> int:
        cleanup_needed = False
        try:
            self._cleanup_stale()
            self._docker(["pull", self.image])
            self.digest = self._image_digest()
            self._docker(["volume", "create", VOLUME])
            cleanup_needed = True
            self._start_container()
            self._wait_ready()
            self._run_standard_checks(FIXTURE_MODEL)
            if self.mode == "deep":
                self._deep_placeholder()
            report_status = self._report_status()
            self._write_reports(report_status)
            return 0 if report_status == "passed" else 1
        finally:
            if cleanup_needed:
                self._cleanup_container()

    def _docker(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["docker", *args]
        return subprocess.run(cmd, text=True, capture_output=True, check=check)

    def _image_digest(self) -> str:
        proc = self._docker([
            "image",
            "inspect",
            self.image,
            "--format",
            "{{range .RepoDigests}}{{println .}}{{end}}",
        ])
        # Lexikografisch kleinster RepoDigest -- deterministisch unabhaengig von
        # Docker-Storage-Reihenfolge; muss mit deploy-local.sh:docker_image_snapshot
        # uebereinstimmen, sonst schlaegt der Report-Lookup fehl. Leere Zeilen
        # werden parallel zur sed '/^[[:space:]]*$/d' im Bash-Pendant gefiltert.
        lines = sorted(line for line in proc.stdout.splitlines() if line.strip())
        digest = lines[0] if lines else ""
        if not digest:
            raise RuntimeError(f"no RepoDigest for {self.image}")
        return digest

    def _cleanup_stale(self) -> None:
        proc = self._docker(["ps", "-a", "--format", "{{.Names}}|{{.State}}"], check=False)
        for line in proc.stdout.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            name, state = parts
            if not name.startswith("kiron-ollama-compat-"):
                continue
            # Nur wirklich tote Container fegen. paused/created/removing gehoeren zu
            # parallelen Sibling-Validatoren (created waehrend docker-run-Anlauf,
            # paused fuer Operator-Debug, removing waehrend laufendem rm) — die
            # duerfen nicht via docker rm -f mid-flight gekillt werden.
            if state not in ("exited", "dead"):
                continue
            self._docker(["rm", "-f", name], check=False)

    def _start_container(self) -> None:
        self._docker([
            "run",
            "-d",
            "--name",
            self.container,
            "-p",
            f"127.0.0.1:{TEST_PORT}:{CONTAINER_PORT}",
            "-v",
            f"{VOLUME}:/root/.ollama",
            "-e",
            f"OLLAMA_HOST=0.0.0.0:{CONTAINER_PORT}",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=all",
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "--gpus",
            "all",
            self.image,
        ])

    def _cleanup_container(self) -> None:
        self._docker(["rm", "-f", self.container], check=False)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        last = None
        while time.monotonic() < deadline:
            try:
                self._json("GET", "/api/version")
                return
            except Exception as exc:
                last = exc
                time.sleep(1)
        raise RuntimeError(f"Ollama test container did not become ready: {last}")

    def _json(self, method: str, path: str, payload: dict | None = None, timeout: float = 120.0):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

    def _poll_ps(self) -> Any:
        # /api/ps-Polls duerfen einen transienten 5xx oder Connection-Blip nicht
        # in einen Traceback verwandeln -- die 30s-Schleifen sollen Transients
        # absorbieren. None signalisiert "noch nicht / erneut versuchen", die
        # Aufrufer schlafen 1s und pollen weiter.
        try:
            return self._json("GET", "/api/ps")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def _pull_fixture(self, model: str) -> None:
        # /api/pull mit stream:False liefert auf Partial-Failures (manifest parse,
        # Registry-5xx mid-blob) HTTP 200 mit {"error": "..."} statt erfolgreichem
        # {"status": "success"}. Ohne Inspektion wird der gescheiterte Pull als
        # Erfolg gewertet, der naechste _generate_check schlaegt mit
        # model-not-found fehl, und das Failure landet im Report unter dem
        # falschen Check.
        response = self._json("POST", "/api/pull", {"name": model, "stream": False}, timeout=600.0)
        if not isinstance(response, dict):
            raise RuntimeError(f"pull returned non-dict response: {response!r}")
        if response.get("error"):
            raise RuntimeError(f"pull failed: {response['error']}")
        if response.get("status") != "success":
            raise RuntimeError(f"pull did not return status=success: {response!r}")

    def _pull_fixture_check(self, model: str) -> Check:
        self._pull_fixture(model)
        return Check(True, "info", "fixture pull ok")

    def _check(self, name: str, func) -> None:
        try:
            self.capabilities[name] = func()
        except Exception as exc:
            self.capabilities[name] = Check(False, "fail", str(exc))
        result = self.capabilities[name]
        if not result.ok and result.severity == "fail":
            self.failures.append(f"{name}: {result.message}")
        elif not result.ok:
            self.warnings.append(f"{name}: {result.message}")

    def _run_standard_checks(self, model: str) -> None:
        # Initialer Pull als _check, damit ein transienter /api/pull-Fehler
        # einen Report mit failed-Capability erzeugt statt eine Exception ins
        # finally von run() zu propagieren -- ohne Wrapper wuerde der Container
        # ohne _write_reports abgeraeumt und das Deploy-Gate blockierte ohne
        # Diagnose.
        self._check("fixture_initial_pull", lambda: self._pull_fixture_check(model))
        # gpu_visible_in_container und baseline_gpu_resident schuetzen
        # num_gpu_zero_chat_generate gegen CPU-only Container, in denen
        # size_vram=0 trivial waere und num_gpu_zero_effective faelschlich
        # gruen meldet -- generate liefe ohnehin auf CPU.
        self._check("gpu_visible_in_container", lambda: self._gpu_visible_check())
        self._check("version_endpoint", lambda: self._version_check())
        self._check("tags_shape", lambda: self._tags_check())
        self._check("ps_shape", lambda: self._ps_check(require_size_vram=False))
        self._check("generate_nonstream", lambda: self._generate_check(model))
        self._check("chat_nonstream", lambda: self._chat_check(model))
        self._check("baseline_gpu_resident", lambda: self._baseline_gpu_resident_check(model))
        self._check("stream_ndjson", lambda: self._stream_check(model))
        self._check("think_false", lambda: self._think_false_check(model))
        self._check("keep_alive_zero_unload", lambda: self._unload_check(model))
        # Bei einem transienten /api/pull-Fehler darf die Exception nicht aus
        # _run_standard_checks heraus propagieren -- sonst wird _write_reports
        # nie aufgerufen und die zuvor gesammelten Capabilities gehen verloren.
        self._check("fixture_repull", lambda: self._pull_fixture_check(model))
        self._check("num_gpu_zero_chat_generate", lambda: self._num_gpu_zero_check(model))
        self._check("ps_size_vram", lambda: self._ps_check(require_size_vram=True))
        self._check("error_shape_normalizable", lambda: self._error_shape_check())

    def _version_check(self) -> Check:
        data = self._json("GET", "/api/version")
        version = data.get("version") if isinstance(data, dict) else None
        return Check(isinstance(version, str) and bool(version), "fail", "version ok", data)

    def _tags_check(self) -> Check:
        data = self._json("GET", "/api/tags")
        ok = isinstance(data, dict) and isinstance(data.get("models"), list)
        if ok:
            ok = all(
                isinstance(m, dict)
                and isinstance(m.get("name"), str)
                and isinstance(m.get("size"), int) and not isinstance(m.get("size"), bool)
                and isinstance(m.get("details"), dict)
                for m in data["models"]
            )
        return Check(ok, "fail", "tags shape ok" if ok else "invalid tags shape")

    def _ps_check(self, *, require_size_vram: bool) -> Check:
        data = self._json("GET", "/api/ps")
        ok = isinstance(data, dict) and isinstance(data.get("models"), list)
        if ok:
            for item in data.get("models", []):
                if not isinstance(item, dict):
                    ok = False
                    break
                if require_size_vram and not (isinstance(item.get("size_vram"), int) and not isinstance(item.get("size_vram"), bool)):
                    ok = False
                    break
        return Check(ok, "fail", "ps shape ok" if ok else "invalid ps shape")

    def _generate_check(self, model: str) -> Check:
        data = self._json("POST", "/api/generate", {"model": model, "prompt": "hi", "stream": False})
        ok = isinstance(data, dict) and ("response" in data or data.get("done") is True)
        return Check(ok, "fail", "generate ok" if ok else "generate shape invalid")

    def _chat_check(self, model: str) -> Check:
        data = self._json("POST", "/api/chat", {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        ok = isinstance(data, dict) and isinstance(data.get("message"), dict)
        return Check(ok, "fail", "chat ok" if ok else "chat shape invalid")

    def _stream_check(self, model: str) -> Check:
        req = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps({"model": model, "prompt": "hi", "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        done = False
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                item = json.loads(raw)
                if isinstance(item, dict) and item.get("done") is True:
                    done = True
                    break
        return Check(done, "fail", "stream done ok" if done else "stream ended without done")

    def _think_false_check(self, model: str) -> Check:
        data = self._json("POST", "/api/generate", {"model": model, "prompt": "hi", "stream": False, "think": False})
        ok = isinstance(data, dict) and "error" not in data
        return Check(ok, "fail", "think:false accepted" if ok else "think:false rejected")

    def _unload_check(self, model: str) -> Check:
        self._json("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False})
        canonical = _canonical_model_name(model)
        deadline = time.monotonic() + 30
        saw_valid_shape = False
        while time.monotonic() < deadline:
            ps = self._poll_ps()
            if ps is None:
                time.sleep(1)
                continue
            # Strikte Shape-Validierung: ohne diese koennte ein malformed /api/ps
            # (models nicht-Liste, oder Eintraege ohne string name) eine leere
            # loaded-Liste erzeugen und faelschlich "unload verified" zurueckgeben.
            if not isinstance(ps, dict) or not isinstance(ps.get("models"), list):
                time.sleep(1)
                continue
            items = ps["models"]
            if not all(isinstance(m, dict) and isinstance(m.get("name"), str) for m in items):
                time.sleep(1)
                continue
            saw_valid_shape = True
            loaded = [_canonical_model_name(m["name"]) for m in items]
            if canonical not in loaded:
                return Check(True, "info", "unload verified")
            time.sleep(1)
        if not saw_valid_shape:
            return Check(False, "fail", "ps shape never valid during unload poll")
        return Check(False, "fail", "model still listed in /api/ps after unload")

    def _gpu_visible_check(self) -> Check:
        proc = self._docker(["exec", self.container, "nvidia-smi", "-L"], check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return Check(False, "fail", f"nvidia-smi -L failed (rc={proc.returncode}): {detail}")
        output = (proc.stdout or "").strip()
        gpu_lines = [line for line in output.splitlines() if line.startswith("GPU")]
        if not gpu_lines:
            return Check(False, "fail", f"nvidia-smi -L produced no GPU lines: {output!r}")
        return Check(
            True,
            "info",
            "GPU visible in container",
            {"nvidia_smi_l": output, "gpu_count": len(gpu_lines)},
        )

    def _baseline_gpu_resident_check(self, model: str) -> Check:
        canonical = _canonical_model_name(model)
        deadline = time.monotonic() + 30
        last_size: Any = None
        while time.monotonic() < deadline:
            ps = self._poll_ps()
            if ps is None:
                time.sleep(1)
                continue
            for item in ps.get("models", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str) and _canonical_model_name(name) == canonical:
                    size_vram = item.get("size_vram")
                    if isinstance(size_vram, int) and not isinstance(size_vram, bool):
                        last_size = size_vram
                        if size_vram > 0:
                            return Check(
                                True,
                                "info",
                                f"baseline load GPU-resident size_vram={size_vram}",
                                {"size_vram": size_vram},
                            )
            time.sleep(1)
        return Check(
            False,
            "fail",
            f"baseline load not GPU-resident within 30s (last size_vram={last_size})",
            {"size_vram": last_size},
        )

    def _num_gpu_zero_check(self, model: str) -> Check:
        # Capability deckt beide Pfade ab und der Runtime-Handoff
        # (capability_num_gpu_zero_effective → vram_lease) gibt CPU-Offload
        # fuer Chat und Generate gemeinsam frei. Daher muessen auch beide
        # API-Pfade gegen `options.num_gpu=0` verifiziert werden, sonst
        # blieben /api/chat-Regressionen unentdeckt (#861).
        if not self._num_gpu_zero_path_effective(
            model,
            "/api/generate",
            {"prompt": "hi"},
        ):
            return Check(False, "fail", "num_gpu=0 not effective on /api/generate within 30s", {"num_gpu_zero_effective": False})
        # Zwischen den Pfaden entladen, damit /api/chat den Load wirklich
        # selbst triggert und nicht das vom Generate-Lauf bereits CPU-only
        # geladene Modell mitnutzt.
        if not self._unload_for_path_check(model):
            return Check(False, "fail", "model still loaded after keep_alive=0 between paths", {"num_gpu_zero_effective": False})
        if not self._num_gpu_zero_path_effective(
            model,
            "/api/chat",
            {"messages": [{"role": "user", "content": "hi"}]},
        ):
            return Check(False, "fail", "num_gpu=0 not effective on /api/chat within 30s", {"num_gpu_zero_effective": False})
        return Check(True, "info", "num_gpu=0 effective on /api/generate and /api/chat", {"num_gpu_zero_effective": True})

    def _num_gpu_zero_path_effective(self, model: str, path: str, extra: dict[str, Any]) -> bool:
        payload = {"model": model, "stream": False, "options": {"num_gpu": 0}, **extra}
        self._json("POST", path, payload)
        canonical = _canonical_model_name(model)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ps = self._poll_ps()
            if ps is None:
                time.sleep(1)
                continue
            for item in ps.get("models", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str) and _canonical_model_name(name) == canonical:
                    size_vram = item.get("size_vram")
                    if (
                        size_vram == 0
                        and isinstance(size_vram, int)
                        and not isinstance(size_vram, bool)
                    ):
                        return True
            time.sleep(1)
        return False

    def _unload_for_path_check(self, model: str) -> bool:
        self._json("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False})
        canonical = _canonical_model_name(model)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ps = self._poll_ps()
            if ps is None:
                time.sleep(1)
                continue
            loaded = [
                _canonical_model_name(m.get("name"))
                for m in ps.get("models", [])
                if isinstance(m, dict) and isinstance(m.get("name"), str)
            ]
            if canonical not in loaded:
                return True
            time.sleep(1)
        return False

    def _error_shape_check(self) -> Check:
        try:
            self._json("POST", "/api/generate", {"model": "kiron-missing-model", "prompt": "hi", "stream": False})
        except urllib.error.HTTPError as exc:
            try:
                data = json.loads(exc.read().decode("utf-8"))
            except Exception:
                return Check(False, "warn", "error body not JSON")
            return Check(isinstance(data, dict) and "error" in data, "warn", "error shape normalizable")
        return Check(False, "warn", "unknown model did not error")

    def _deep_placeholder(self) -> None:
        if not self.pull_large:
            self.warnings.append("deep mode incomplete: --pull-large not set")

    def _report_status(self) -> str:
        if self.mode == "deep" and not self.pull_large:
            return "incomplete"
        return "failed" if self.failures else "passed"

    def _write_reports(self, status: str) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        # Erst Repo-Anteil abschneiden, dann Tag — sonst landet bei tag-losen
        # Refs mit Registry-Port (z.B. 'localhost:5000/ollama') ein '/' im stem
        # und _atomic_write schreibt in ein nicht existierendes Unterverzeichnis.
        image_tag = self.image.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        # Digest-Praefix verhindert Kollision wenn zwei Validatoren denselben
        # Tag aus verschiedenen Registries in derselben Sekunde schreiben —
        # sonst ueberschreibt os.replace den ersten Report und
        # find_matching_report findet ihn ueber image_digest nicht mehr.
        digest_short = self.digest.rsplit(":", 1)[-1][:12]
        stem = f"{self.started_at}-ollama-{image_tag}-{digest_short}"
        report = {
            "image": self.image,
            "compose_tag": self.image,
            "image_digest": self.digest,
            "published_version": None,
            "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validator_version": VALIDATOR_VERSION,
            "mode": self.mode,
            "fixture_models": [FIXTURE_MODEL],
            "host_ollama_binary_version": self._host_ollama_version(),
            "production_container_image": self._prod_container_field(".Config.Image"),
            "production_container_id": self._prod_container_field(".Id"),
            "capabilities": {k: asdict(v) for k, v in self.capabilities.items()},
            "failures": self.failures,
            "warnings": self.warnings,
            "report_status": status,
            "upgrade_allowed": status == "passed",
        }
        json_path = REPORT_DIR / f"{stem}.json"
        md_path = REPORT_DIR / f"{stem}.md"
        self._atomic_write(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        md = [
            f"# Ollama Compat Report {self.image}",
            "",
            f"- Status: `{status}`",
            f"- Digest: `{self.digest}`",
            f"- Mode: `{self.mode}`",
            f"- Failures: {len(self.failures)}",
            f"- Warnings: {len(self.warnings)}",
            "",
        ]
        for failure in self.failures:
            md.append(f"- FAIL: {failure}")
        for warning in self.warnings:
            md.append(f"- WARN: {warning}")
        self._atomic_write(md_path, "\n".join(md) + "\n")
        print(json_path)
        print(md_path)

    def _atomic_write(self, path: pathlib.Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, path)

    def _host_ollama_version(self) -> str | None:
        try:
            proc = subprocess.run(["/usr/local/bin/ollama", "--version"], text=True, capture_output=True)
        except (FileNotFoundError, OSError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or proc.stderr.strip() or None

    def _prod_container_field(self, field: str) -> str | None:
        proc = self._docker(["inspect", "-f", "{{" + field + "}}", "kiron-ollama"], check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--find-report", action="store_true",
                        help="print newest matching report path and strict num_gpu_zero safety")
    parser.add_argument("--digest")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--mode", choices=("standard", "deep"), default="standard")
    parser.add_argument("--pull-large", action="store_true")
    args = parser.parse_args(argv)

    if args.find_report:
        if not args.digest:
            parser.error("--digest is required with --find-report")
        # Exit codes: 0 = match, 1 = no match (clean), 2 = unexpected crash.
        # Bash deploy gate distinguishes these for accurate diagnostics.
        try:
            match = find_matching_report(
                pathlib.Path(args.report_dir),
                args.image,
                args.digest,
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 2
        if match is None:
            return 1
        path, safe = match
        print(path)
        print("true" if safe else "false")
        return 0

    validator = Validator(args.image, args.mode, args.pull_large)

    def handle_signal(signum, frame):
        # Restore defaults first, sonst kann ein zweites Signal den Handler
        # mid-_cleanup_container reentrant aufrufen (docker rm -f doppelt) und
        # einen Zombie-Container hinterlassen.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        validator._cleanup_container()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return validator.run()


if __name__ == "__main__":
    raise SystemExit(main())
