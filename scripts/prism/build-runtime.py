#!/usr/bin/env python3
"""Prepare or build the locked Prism source profile; default action is read-only."""
import argparse
import fcntl
import grp
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LOCK = HERE / "source-toolchain-lock.json"
ROOT = Path("/usr/lib/kiron/test-runtimes/prism/source-builds/cuda12.8-sm86-tokens-v1")
INPUT_CACHE = Path("/usr/lib/kiron/test-runtimes/prism/source-builds/cuda12.8-sm86-v1/prepared")
MIN_FREE = 13_000_000_000
RESERVE = 4_000_000_000
MAX_LOG = 64 * 1024**2
MIN_RAM = 2 * 1024**3


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recipe_identity():
    return {name: digest(HERE / name) for name in
            ("build-runtime.py", "host-toolchain.py", "install-runtime.py")}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def safe_path(path):
    if not path.is_absolute() or ".." in path.parts or ROOT not in (path, *path.parents):
        raise ValueError("only the fixed isolated source-build root is allowed")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink in build path")
    return path


def clean_environment(lock, source=None):
    env = dict(lock["environment"])
    if source is not None:
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0=str(source))
    return env


def verify_host(lock):
    for value in lock["empty_include_roots"]:
        path = Path(value)
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError("unsupported local compiler headers outside the pinned package inputs")
    if module("host-toolchain").snapshot_host() != lock["host"]:
        raise ValueError("host toolchain drift: inspect and explicitly renew the reviewed lock")


def preflight(lock):
    safe_path(ROOT)
    patchset(lock)
    ancestor = next(p for p in (ROOT, *ROOT.parents) if p.exists())
    if shutil.disk_usage(ancestor).free < MIN_FREE:
        raise ValueError("source build needs at least 13 GB free, including reserve")
    verify_host(lock)


def group_running(pgid):
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def stop(process):
    # run() uses WNOWAIT, keeping an exited leader unreaped: its PID/PGID cannot
    # be reused while we terminate surviving make/nvcc descendants.
    if process.returncode is not None:
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGTERM:
                until = time.monotonic() + 5
                while group_running(process.pid) and time.monotonic() < until:
                    time.sleep(0.1)
        process.wait(timeout=5)  # Reap only after the final signal to the reserved PGID.
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def ram_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("RAM telemetry unavailable")


def run(argv, *, cwd, env, log, deadline):
    """One process group; absolute operation deadline also covers each command."""
    if time.monotonic() >= deadline:
        raise TimeoutError("operation deadline exceeded before spawn")
    with log.open("ab", buffering=0) as output:
        output.write((json.dumps({"argv": argv}) + "\n").encode())
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("operation deadline exceeded")
                if shutil.disk_usage(ROOT).free < RESERVE or log.stat().st_size > MAX_LOG:
                    raise ValueError("disk reserve or log limit exceeded")
                if ram_available() < MIN_RAM:
                    raise ValueError("available RAM below 2 GiB")
                time.sleep(0.2)
        finally:
            stop(process)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv)


class FixedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in
            {"developer.download.nvidia.com", "files.pythonhosted.org"}
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("unapproved toolchain URL")


def check_artifact(path, artifact):
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size != artifact["bytes"]:
        raise ValueError("toolchain archive size/type mismatch")
    if digest(path) != artifact["sha256"]:
        raise ValueError("toolchain archive SHA-256 mismatch")


def download(artifact, path):
    safe_path(path)
    check_url(artifact["url"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), FixedRedirect())
    with opener.open(artifact["url"], timeout=15) as response, path.open("xb") as output:
        remaining = artifact["bytes"]
        while True:
            block = response.read(min(1024**2, remaining + 1))
            if not block:
                break
            remaining -= len(block)
            if remaining < 0:
                raise ValueError("download exceeds pinned size")
            output.write(block)
    check_artifact(path, artifact)  # Parent process enforces the total download deadline.


def extract(path, destination, artifact):
    check_artifact(path, artifact)  # No extracted entry before complete integrity check.
    if artifact["format"] == "tar.xz":
        with tarfile.open(path, "r:xz") as archive:
            planner = module("install-runtime")
            planner.MAX_MEMBERS, planner.MAX_UNPACKED = 15000, 512 * 1024**2
            plan = planner.archive_plan(archive)
            if any(PurePosixPath(name).parts[0] != artifact["prefix"] for name in plan):
                raise ValueError("unexpected NVIDIA archive root")
            # Namespace/symlink validation above precedes the stdlib data filter.
            archive.extractall(destination, members=plan.values(), filter="data")
    else:
        with zipfile.ZipFile(path) as archive:
            members, names, total = archive.infolist(), set(), 0
            for member in members:
                name = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                total += member.file_size
                if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                        or name.as_posix() in names or stat.S_ISLNK(mode)
                        or total > 128 * 1024**2 or len(members) > 5000):
                    raise ValueError("unsafe CMake wheel")
                names.add(name.as_posix())
            for member in members:
                target = destination / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if member.external_attr >> 16 & 0o111 else 0o644)


def merge(source, destination, label):
    """Merge pinned components without silently replacing a different header/library."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        target = destination / path.name
        if path.name.lower().startswith("license"):
            target = destination / f"{label}-{path.name}"
        if path.is_dir() and not path.is_symlink():
            merge(path, target, label)
        elif target.exists() or target.is_symlink():
            if (path.is_symlink() != target.is_symlink()
                    or (os.readlink(path) != os.readlink(target) if path.is_symlink()
                        else digest(path) != digest(target))):
                raise ValueError("conflicting toolchain component files")
        else:
            shutil.move(path, target)


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or relative.as_posix() == "prepared.json":
            continue
        if path.is_symlink():
            result[str(relative)] = {"link": os.readlink(path)}
        elif path.is_file():
            result[str(relative)] = {"sha256": digest(path), "mode": stat.S_IMODE(path.stat().st_mode)}
    return result


def git_output(source, args, env):
    return subprocess.check_output(["/usr/bin/git", "-C", str(source), *args],
                                   env=env, timeout=15, text=True).strip()


def patchset(lock):
    patches = lock["source"]["patches"]
    if type(patches) is not list or len(patches) > 8:
        raise ValueError("invalid source patchset")
    targets = set()
    for patch in patches:
        if type(patch) is not dict or set(patch) != {"path", "sha256", "target", "before_sha256", "after_sha256"}:
            raise ValueError("invalid source patch record")
        relative, target = PurePosixPath(patch["path"]), PurePosixPath(patch["target"])
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "patches"
                or target.is_absolute() or ".." in target.parts or ".git" in target.parts
                or not target.parts or str(target) in targets or any(c.isspace() for c in str(target))):
            raise ValueError("unsafe or duplicate source patch target")
        targets.add(str(target))
        path = HERE / relative
        if (any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file()
                or path.stat().st_size > 1024 * 1024 or digest(path) != patch["sha256"]):
            raise ValueError("source patch digest/type mismatch")
        for field in ("before_sha256", "after_sha256"):
            value = patch[field]
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid source patch file digest")
    return patches


def verify_source(source, lock, env):
    patches = patchset(lock)
    if git_output(source, ["rev-parse", "HEAD"], env) != lock["source"]["commit"]:
        raise ValueError("source checkout is not the exact pinned commit")
    expected = "\n".join(sorted(" M " + patch["target"] for patch in patches)).strip()
    if git_output(source, ["status", "--porcelain", "--untracked-files=all"], env) != expected:
        raise ValueError("source checkout differs from the exact pinned patchset")
    for patch in patches:
        path = source / patch["target"]
        if path.is_symlink() or not path.is_file() or digest(path) != patch["after_sha256"]:
            raise ValueError("patched source file digest mismatch")


def apply_source_patches(source, lock, env, log, deadline):
    if git_output(source, ["status", "--porcelain", "--untracked-files=all"], env):
        raise ValueError("source must be clean before applying the pinned patchset")
    for patch in patchset(lock):
        target = source / patch["target"]
        if target.is_symlink() or not target.is_file() or digest(target) != patch["before_sha256"]:
            raise ValueError("source patch preimage mismatch")
        path = HERE / patch["path"]
        changed = git_output(source, ["apply", "--numstat", str(path)], env).splitlines()
        if len(changed) != 1 or changed[0].split("\t")[-1] != patch["target"]:
            raise ValueError("source patch changes an undeclared file")
        run(["/usr/bin/git", "-C", str(source), "-c", "core.hooksPath=/dev/null",
             "apply", "--check", "--whitespace=error", str(path)], cwd=source, env=env, log=log, deadline=deadline)
        run(["/usr/bin/git", "-C", str(source), "-c", "core.hooksPath=/dev/null",
             "apply", "--whitespace=error", str(path)], cwd=source, env=env, log=log, deadline=deadline)
    verify_source(source, lock, env)


def verify_prepared(prepared, lock):
    safe_path(prepared)
    record = json.loads((prepared / "prepared.json").read_text())
    if record.get("recipe") != recipe_identity():
        raise ValueError("prepared recipe drift: prepare a fresh tree with the reviewed code")
    if record["lock"] != lock or record["inventory"] != inventory(prepared):
        raise ValueError("prepared toolchain/source drift")
    env = clean_environment(lock)
    verify_source(prepared / "source", lock, env)


def link_cuda_layout(cuda):
    library = cuda / "lib"
    if library.is_symlink() or not library.is_dir():
        raise ValueError("CUDA lib must be a real directory in the prepared tree")
    # NVIDIA component archives use lib; the pinned nvcc.profile searches lib64.
    (cuda / "lib64").symlink_to("lib", target_is_directory=True)


def cache_input(relative):
    """A fixed immutable previous input cache, never a build output or fallback."""
    path = INPUT_CACHE / relative
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("unsafe source input cache")
    return path


def prepare(lock, report, deadline, *, reuse_verified_inputs=False):
    recipe = recipe_identity()
    destination = ROOT / "prepared"
    if destination.exists():
        verify_prepared(destination, lock)
        return destination
    env, log = clean_environment(lock), report / "prepare.log"
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=ROOT) as temporary:
        stage = Path(temporary)
        for name, artifact in lock["artifacts"].items():
            packed = stage / f"{name}.archive"
            if reuse_verified_inputs:
                cached = cache_input(f"{name}.archive")
                check_artifact(cached, artifact)
                shutil.copyfile(cached, packed)
                check_artifact(packed, artifact)
            else:
                run(["/usr/bin/python3", str(Path(__file__).resolve()), "_download", name, str(packed)],
                    cwd=ROOT, env=env, log=log, deadline=min(deadline, time.monotonic() + 900))
            extract(packed, stage / "parts" / name, artifact)
            if artifact["format"] == "tar.xz":
                merge(stage / "parts" / name / artifact["prefix"], stage / "cuda", name)
        cublas = Path(lock["host"]["cublas"]["path"])
        for header in sorted((cublas / "include").glob("*.h")):
            (stage / "cuda" / "include" / header.name).symlink_to(header)
        library = stage / "cuda" / "lib"
        library.mkdir(exist_ok=True)
        link_cuda_layout(stage / "cuda")
        for name in ("libcublas.so.12", "libcublasLt.so.12"):
            (library / name).symlink_to(cublas / "lib" / name)
            (library / name.removesuffix(".12")).symlink_to(name)
        source = stage / "source"
        fetch_source = cache_input("source/.git").parent.as_uri() if reuse_verified_inputs else "origin"
        commands = [["init", str(source)], ["-C", str(source), "remote", "add", "origin", lock["source"]["url"]],
                    ["-C", str(source), "-c", "fetch.fsckObjects=true", "fetch", "--depth=1", fetch_source, lock["source"]["commit"]],
                    ["-C", str(source), "checkout", "--detach", "FETCH_HEAD"]]
        for args in commands:
            run(["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *args], cwd=ROOT, env=env, log=log, deadline=deadline)
        if git_output(source, ["rev-parse", "HEAD"], env) != lock["source"]["commit"]:
            raise ValueError("wrong source commit")
        apply_source_patches(source, lock, env, log, deadline)
        for path in stage.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        stage.chmod(0o755)
        if recipe_identity() != recipe:
            raise ValueError("preparation recipe changed during operation")
        write_json(stage / "prepared.json", {"lock": lock, "recipe": recipe, "inventory": inventory(stage),
            "input_cache": str(INPUT_CACHE) if reuse_verified_inputs else None})
        os.rename(stage, destination)
    return destination


def build_commands(lock, prepared, output):
    cmake = str(prepared / "parts" / "cmake" / "cmake" / "data" / "bin" / "cmake")
    options = dict(lock["cmake_options"])
    options.update(CUDAToolkit_ROOT=str(prepared / "cuda"),
                   CMAKE_CUDA_COMPILER=str(prepared / "cuda" / "bin" / "nvcc"))
    configure = [cmake, "-S", str(prepared / "source"), "-B", str(output), "-G", "Unix Makefiles"]
    configure += [f"-D{name}={value}" for name, value in sorted(options.items())]
    return [configure, [cmake, "--build", str(output), "--target", "llama-server", "--parallel", "2"]]


def build(lock, report, deadline):
    recipe = recipe_identity()
    prepared = ROOT / "prepared"
    verify_prepared(prepared, lock)
    if os.geteuid() != 0:
        raise ValueError("build requires root supervisor for network namespace and unprivileged child")
    uid, gid = pwd.getpwnam("nobody").pw_uid, grp.getgrnam("kiron-common").gr_gid
    output = report / "build"
    output.mkdir(mode=0o755)
    os.chown(output, uid, gid)
    env = clean_environment(lock, prepared / "source")
    env["TMPDIR"] = str(output)
    isolation = ["/usr/bin/unshare", "--net", "--", "/usr/bin/setpriv", "--reuid", str(uid),
                 "--regid", str(gid), "--clear-groups", "--no-new-privs", "--"]
    commands = [isolation + command for command in build_commands(lock, prepared, output)]
    write_json(report / "launch.json", {"commands": commands, "environment": env, "lock": lock, "recipe": recipe})
    succeeded, build_failure = False, None
    try:
        for index, command in enumerate(commands):
            try:
                run(command, cwd=output, env=env, log=report / "build.log", deadline=deadline)
            except subprocess.CalledProcessError as error:
                # run() has already stopped/reaped this command's entire group.
                # Retain only ordinary build-tool failures, never configure,
                # signal, deadline or resource-limit failures. No resume path.
                if (index == 1 and error.returncode > 0 and time.monotonic() < deadline
                        and shutil.disk_usage(ROOT).free >= RESERVE
                        and ram_available() >= MIN_RAM
                        and (report / "build.log").stat().st_size <= MAX_LOG):
                    build_failure = error
                raise
        verify_host(lock)
        verify_prepared(prepared, lock)
        if recipe_identity() != recipe:
            raise ValueError("build recipe changed during operation")
        if not (output / "bin" / "llama-server").is_file():
            raise ValueError("build completed without llama-server")
        write_json(report / "binary-manifest.json", inventory(output / "bin"))
        succeeded = True
    finally:
        cache = output / "CMakeCache.txt"
        if cache.exists():
            shutil.copyfile(cache, report / "CMakeCache.txt")
        if not succeeded:
            if build_failure is None:
                shutil.rmtree(output)
            else:
                # A root-only diagnostic quarantine, never an installable binary
                # manifest or a build directory accepted by a subsequent run.
                output.chmod(0o700)
                os.chown(output, 0, 0)
                os.rename(output, report / "failed-build")
                write_json(report / "failed-build.json", {"status": "failed",
                           "returncode": build_failure.returncode,
                           "diagnostics": "failed-build", "reusable": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "build", "_download"), nargs="?", default="plan")
    parser.add_argument("worker_args", nargs="*")
    parser.add_argument("--timeout", type=int, default=7200, help="total operation seconds, 1..7200")
    parser.add_argument("--reuse-verified-inputs", action="store_true", help="prepare only: fixed verified local archive/Git cache, no download fallback")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    recipe = recipe_identity()
    try:
        if not 1 <= args.timeout <= 7200:
            raise ValueError("timeout must be 1..7200 seconds")
        if args.action == "_download":
            name, path = args.worker_args
            download(lock["artifacts"][name], Path(path))
            return
        if args.worker_args:
            raise ValueError("unexpected positional arguments")
        if args.reuse_verified_inputs and args.action != "prepare":
            raise ValueError("verified input reuse is only a prepare option")
        preflight(lock)
        if args.action == "plan":
            print(json.dumps({"root": str(ROOT), "required_free_bytes": MIN_FREE,
                              "source": lock["source"], "host_verified": True}, indent=2))
            return
        ROOT.mkdir(parents=True, exist_ok=True, mode=0o755)
        with os.fdopen(os.open(ROOT / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a") as guard:
            info = os.fstat(guard.fileno())
            if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("unsafe build lock")
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = Path(tempfile.mkdtemp(prefix=f"{args.action}-", dir=ROOT))
            report.chmod(0o755)
            started, status = time.monotonic(), "failed"
            previous = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                deadline = started + min(args.timeout, 1800 if args.action == "prepare" else 7200)
                if args.action == "prepare":
                    prepare(lock, report, deadline, reuse_verified_inputs=args.reuse_verified_inputs)
                else:
                    build(lock, report, deadline)
                status = "complete"
            finally:
                signal.signal(signal.SIGTERM, previous)
                write_json(report / "result.json", {"status": status, "action": args.action,
                           "elapsed_seconds": time.monotonic() - started, "lock_sha256": digest(LOCK), "recipe": recipe})
            print(json.dumps({"report": str(report), "status": status}))
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        parser.exit(1, f"Prism source build failed: {error}\n")


if __name__ == "__main__":
    main()
