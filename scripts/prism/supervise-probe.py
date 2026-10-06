#!/usr/bin/env python3
"""Supervise one bounded, unprivileged Prism test process; never touch service units."""
import argparse
import fcntl
import grp
import hashlib
import http.client
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import stat
import subprocess
import threading
import time
from datetime import datetime, timezone

LOCK_PATH = Path("/tmp/kiron-prism-probe.lock")
MAX_GPU_BYTES = 11 * 1024**3
MIN_RAM_BYTES = 2 * 1024**3
INTERVAL = 2.0
PRODUCTION_PORTS = {5001, 8505, 11434, 11435, 11436, 11437, 11440, 11441, 11442}


def canonical_path(value, *, directory=False, missing=False):
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or ".." in path.parts:
        raise ValueError("paths must be absolute, canonical and free of symlinks")
    if missing and not path.exists():
        return path
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("unexpected path type")
    return path


def validate(args):
    for name in ("binary", "model"):
        setattr(args, name, canonical_path(getattr(args, name)))
    if not os.access(args.binary, os.X_OK):
        raise ValueError("binary is not executable")
    if args.mmproj:
        args.mmproj = canonical_path(args.mmproj)
    args.output_dir = canonical_path(args.output_dir, directory=True, missing=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty; evidence is never overwritten")
    if args.port in PRODUCTION_PORTS or not 18000 <= args.port <= 18999:
        raise ValueError("only isolated test ports 18000..18999 are allowed")
    if not 0 <= args.gpu_layers <= 65:
        raise ValueError("gpu-layers must be 0..65 (64 blocks plus output layer)")
    if not 1 <= args.threads <= 4:
        raise ValueError("threads must be 1..4; batch threads use the same limit")
    if not 1 <= args.startup_timeout <= 180 or not 1 <= args.max_lifetime <= 1800:
        raise ValueError("startup timeout must be 1..180s and lifetime 1..1800s")
    if args.library_path:
        for directory in args.library_path.split(":"):
            canonical_path(directory, directory=True)


def command(args):
    result = [str(args.binary), "--model", str(args.model), "--alias", "bonsai-probe", "--host", "127.0.0.1",
              "--port", str(args.port), "--log-verbosity", "4", "--ctx-size", "1024", "--parallel", "1",
              "--batch-size", "128", "--ubatch-size", "128", "--threads", str(args.threads),
              "--threads-batch", str(args.threads), "--n-gpu-layers", str(args.gpu_layers),
              "--jinja", "--fit", "off", "--cache-ram", "0", "--no-warmup", "--no-context-shift",
              "--reasoning", "off", "--reasoning-format", "deepseek",
              "--no-agent", "--no-webui", "--no-ui-mcp-proxy"]
    if args.gpu_layers == 0:
        result += ["--device", "none"]
    result += (["--mmproj", str(args.mmproj), "--no-mmproj-offload"]
               if args.mmproj else ["--no-mmproj"])
    return result


def environment(args):
    result = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    if args.library_path:
        result["LD_LIBRARY_PATH"] = args.library_path
    return result


def resources(pid=None):
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram = int(memory["MemAvailable"].split()[0]) * 1024
    gpu = subprocess.run(
        ["/usr/bin/nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=2,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    rows = [tuple(int(value.strip()) * 1024**2 for value in line.split(","))
            for line in gpu.stdout.splitlines() if line.strip()]
    if not rows or any(len(row) != 2 or row[0] < 0 or row[1] <= 0 or row[0] > row[1] for row in rows):
        raise ValueError("invalid GPU telemetry")
    rss = None
    if pid is not None:
        process = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
        rss = int(process["VmRSS"].split()[0]) * 1024
    return {"ram_available_bytes": ram, "gpu_used_bytes": sum(row[0] for row in rows),
            "gpus": [{"used_bytes": row[0], "total_bytes": row[1]} for row in rows],
            "process_rss_bytes": rss}


def resource_failure(measured):
    if measured["gpu_used_bytes"] > MAX_GPU_BYTES:
        return "gpu_limit_exceeded"
    if measured["ram_available_bytes"] < MIN_RAM_BYTES:
        return "ram_limit_exceeded"
    return None


def health(port, timeout=1.0):
    """Bound total time, including a response that never stops trickling bytes."""
    start = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    timer, expired = None, threading.Event()
    try:
        connection.connect()
        sock = connection.sock

        def expire():
            expired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            return {"ready": False, "error": "health_total_timeout"}
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read(65537)
        if expired.is_set():
            return {"ready": False, "error": "health_total_timeout"}
        if len(body) > 65536:
            return {"ready": False, "error": "health_body_limit"}
        payload = json.loads(body)
        return {"ready": response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ok",
                "status": response.status, "body": payload}
    except (OSError, ValueError, http.client.HTTPException) as error:
        return {"ready": False, "error": "health_total_timeout" if expired.is_set() else type(error).__name__}
    finally:
        if timer:
            timer.cancel()
        connection.close()


def stop_process(process):
    """Kill only our start_new_session process group, and reap the direct child."""
    if process.returncode is not None:  # poll() may already have reaped an exited child.
        return
    for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
            return  # Do not signal a reaped leader's potentially reused PGID.
        except subprocess.TimeoutExpired:
            continue
    process.wait(timeout=1)


def write_json(path, value):
    with path.open("x") as target:
        json.dump(value, target, indent=2)
        target.write("\n")


def probe(args):
    validate(args)
    identity = ({"user": pwd.getpwnam("nobody").pw_uid, "group": grp.getgrnam("kiron-common").gr_gid,
                 "extra_groups": []} if os.geteuid() == 0 else {})
    with os.fdopen(os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a") as lock:
        info = os.fstat(lock.fileno())
        if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("unsafe probe lock")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as port_check:
            port_check.bind(("127.0.0.1", args.port))
        initial = resources()
        if failure := resource_failure(initial):
            raise ValueError(f"preflight: {failure}")
        args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        argv, env = command(args), environment(args)
        launch = {"argv": argv, "environment": env, "identity": identity,
                  "startup_timeout": args.startup_timeout, "max_lifetime": args.max_lifetime,
                  "started_at": datetime.now(timezone.utc).isoformat(), "preflight": initial,
                  "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  "model_bytes": args.model.stat().st_size,
                  "model_mtime_ns": args.model.stat().st_mtime_ns}
        write_json(args.output_dir / "launch.json", launch)
        stopped, previous, process = [], {}, None
        start, ready_at, reason, samples = time.monotonic(), None, "start_failed", 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))
        try:
            with (args.output_dir / "stdout.log").open("xb", buffering=0) as log, \
                    (args.output_dir / "metrics.jsonl").open("x") as metrics, \
                    (args.output_dir / "health.jsonl").open("x") as checks:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, cwd="/", start_new_session=True, umask=0o077, **identity)
                while True:
                    tick = time.monotonic()
                    elapsed = time.monotonic() - start
                    if stopped or (args.output_dir / "STOP").exists():
                        reason = "signal" if stopped else "stop_file"
                        break
                    if elapsed >= args.max_lifetime:
                        reason = "max_lifetime"
                        break
                    if process.poll() is not None:
                        reason = "process_exited"
                        break
                    measured = {"elapsed_seconds": elapsed, "timestamp": datetime.now(timezone.utc).isoformat(),
                                **resources(process.pid)}
                    metrics.write(json.dumps(measured) + "\n")
                    metrics.flush()
                    samples += 1
                    if failure := resource_failure(measured):
                        reason = failure
                        break
                    status = health(args.port)
                    checks.write(json.dumps({"elapsed_seconds": elapsed, **status}) + "\n")
                    checks.flush()
                    elapsed = time.monotonic() - start
                    if ready_at is None and elapsed >= args.startup_timeout:
                        reason = "startup_timeout"
                        break
                    if status["ready"] and ready_at is None:
                        ready_at = elapsed
                        write_json(args.output_dir / "ready.json", {"pid": process.pid, "ready_after_seconds": ready_at})
                    if (args.output_dir / "stdout.log").stat().st_size > 64 * 1024**2:
                        reason = "log_size_limit"
                        break
                    remaining = args.max_lifetime - (time.monotonic() - start)
                    if ready_at is None:
                        remaining = min(remaining, args.startup_timeout - (time.monotonic() - start))
                    time.sleep(max(0, min(INTERVAL - (time.monotonic() - tick), remaining)))
        except Exception as error:
            reason = f"supervisor_error:{type(error).__name__}"
            raise
        finally:
            try:
                if process is not None:
                    stop_process(process)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                write_json(args.output_dir / "result.json", {"reason": reason, "ready_after_seconds": ready_at,
                           "samples": samples, "elapsed_seconds": time.monotonic() - start,
                           "exit_code": process.returncode if process is not None else None})
        return 0 if reason in {"stop_file", "signal", "max_lifetime"} and ready_at is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("binary", "model", "output-dir"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--library-path")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2, help="inference and batch threads, 1..4 (default: 2)")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--max-lifetime", type=int, default=600)
    parser.add_argument("--startup-timeout", type=int, default=180)
    try:
        parser.exit(probe(parser.parse_args()))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Prism probe failed: {error}\n")


if __name__ == "__main__":
    main()
