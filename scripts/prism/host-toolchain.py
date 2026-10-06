#!/usr/bin/env python3
"""Read-only host build-input snapshot, not a claim of hermetic/bit-identical builds.

All installed, version-satisfying alternatives are included conservatively:
dpkg does not retain a unique chosen provider for an OR dependency. Only package
paths in the listed build roots are fingerprinted; documentation and caches are
excluded. No environment variables, credentials, or /etc file contents are read.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

SEEDS = ("gcc-13", "g++-13", "make", "binutils", "libc6-dev", "git", "coreutils", "dash", "python3.12", "util-linux")
ROOTS = ("/usr/bin", "/bin", "/usr/include", "/usr/lib", "/lib")
CUBLAS_PATH = Path("/usr/lib/kiron/test-venvs/prism-runtime/lib/python3.12/site-packages/nvidia/cublas")
DRIVER_PATH = Path("/usr/lib/x86_64-linux-gnu/libcuda.so")
SKIP_COMPONENTS = {"__pycache__", ".cache", "cache", "doc", "docs"}
DEPENDENCY = re.compile(r"^([a-z0-9][a-z0-9+.-]*)(?::([a-z0-9-]+))?(?:\s*\((<<|<=|=|>=|>>)\s*([^()]+)\))?$")


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30,
                          env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}).stdout


def installed_packages():
    fields = ("binary:Package", "Version", "Architecture", "Multi-Arch", "Status", "Depends", "Pre-Depends", "Provides")
    output = command(["/usr/bin/dpkg-query", "--show", "--showformat=" + "\t".join("${" + field + "}" for field in fields) + "\n"])
    packages = {}
    for line in output.splitlines():
        name, version, architecture, multiarch, status_value, depends, predepends, provides = line.split("\t")
        if status_value.endswith(" ok installed"):
            packages[name] = {"version": version, "architecture": architecture, "multiarch": multiarch,
                              "depends": ",".join(filter(None, (depends, predepends))), "provides": provides}
    return packages


def parse_dependency(value):
    match = DEPENDENCY.fullmatch(value.strip())
    if not match:
        raise ValueError(f"unsupported installed dependency syntax: {value}")
    return match.groups()


def version_satisfies(version, operator, required):
    if operator is None:
        return True
    if version is None:  # An unversioned Provides cannot satisfy a versioned Depends.
        return False
    result = subprocess.run(["/usr/bin/dpkg", "--compare-versions", version, operator, required], capture_output=True, timeout=5,
                            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    if result.returncode not in (0, 1):
        raise RuntimeError("dpkg version comparison failed")
    return result.returncode == 0


def dependency_closure(packages, native_arch, seeds=SEEDS, compare=version_satisfies):
    providers = {}
    for name, package in packages.items():
        providers.setdefault(name.split(":")[0], []).append((name, package["version"]))
        for provided in filter(None, package["provides"].split(",")):
            alias, qualifier, operator, version = parse_dependency(provided)
            if qualifier or operator not in (None, "="):
                raise ValueError(f"unsupported Provides: {provided}")
            providers.setdefault(alias, []).append((name, version))

    def resolve(group, architecture):
        matches = set()
        for alternative in group.split("|"):
            base, qualifier, operator, required = parse_dependency(alternative)
            wanted_arch = native_arch if qualifier == "native" else qualifier or architecture
            for name, version in providers.get(base, []):
                package = packages[name]
                arch_ok = (qualifier == "any" or package["architecture"] in ("all", wanted_arch)
                           or (qualifier is None and package["multiarch"] == "foreign"))
                if arch_ok and compare(version, operator, required):
                    matches.add(name)
        if not matches:
            raise RuntimeError(f"no installed provider satisfies dependency: {group}")
        return matches

    pending, selected = set(), set()
    for seed in seeds:
        pending.update(resolve(seed, native_arch))
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        package = packages[name]
        architecture = native_arch if package["architecture"] == "all" else package["architecture"]
        for group in filter(None, package["depends"].split(",")):
            pending.update(resolve(group, architecture) - selected)
    return sorted(selected)


def relevant(path):
    value = str(path)
    return (any(value == root or value.startswith(root + "/") for root in ROOTS)
            and not SKIP_COMPONENTS.intersection(Path(path).parts)
            and Path(path).suffix not in (".pyc", ".pyo"))


def fingerprint(path):
    """Hash a regular file or the link itself; never hash /etc via a symlink."""
    path = Path(path)
    before = path.lstat()
    record = {"path": str(path), "mode": f"{stat.S_IMODE(before.st_mode):04o}"}
    if stat.S_ISLNK(before.st_mode):
        record.update(kind="symlink", target=os.readlink(path))
    elif stat.S_ISREG(before.st_mode):
        if not relevant(path.resolve(strict=True)):
            raise RuntimeError(f"build input resolves outside allowed roots: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        record.update(kind="file", size=before.st_size, sha256=digest.hexdigest())
    elif stat.S_ISDIR(before.st_mode):
        return None
    else:
        raise RuntimeError(f"unexpected build input type: {path}")
    after = path.lstat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode):
        raise RuntimeError(f"build input changed while fingerprinting: {path}")
    return record


def aggregate(paths):
    digest, count = hashlib.sha256(), 0
    for path in sorted(set(map(str, paths))):
        if not relevant(path):
            continue
        record = fingerprint(path)
        if record is not None:
            digest.update((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
            count += 1
    return {"files_sha256": digest.hexdigest(), "files_count": count}


def driver_snapshot():
    path, seen, chain = DRIVER_PATH, set(), []
    while path.is_symlink():
        if str(path) in seen or not relevant(path):
            raise RuntimeError("invalid driver symlink chain")
        seen.add(str(path))
        record = fingerprint(path)
        chain.append(record)
        target = Path(record["target"])
        path = Path(os.path.normpath(target if target.is_absolute() else path.parent / target))
    if not relevant(path):
        raise RuntimeError("driver resolves outside allowed roots")
    record = fingerprint(path)
    if record is None or record["kind"] != "file":
        raise RuntimeError("CUDA driver is not a regular file")
    return {"path": str(DRIVER_PATH), "resolved_path": str(path), "sha256": record["sha256"],
            "mode": record["mode"], "symlink_chain": chain}


def snapshot_host():
    packages = installed_packages()
    selected = dependency_closure(packages, command(["/usr/bin/dpkg", "--print-architecture"]).strip())
    paths = command(["/usr/bin/dpkg-query", "--listfiles", *selected]).splitlines()
    file_digest = aggregate(path for path in paths if path.startswith("/"))
    if not CUBLAS_PATH.is_dir() or CUBLAS_PATH.is_symlink():
        raise RuntimeError("expected installed cuBLAS wheel directory is missing or a symlink")
    cublas = aggregate(CUBLAS_PATH.rglob("*"))
    if cublas["files_count"] == 0:
        raise RuntimeError("empty cuBLAS tree")
    driver = driver_snapshot()
    after = installed_packages()
    if after != packages:
        raise RuntimeError("installed package metadata changed during snapshot")
    return {"schema_version": 1, "packages": {name: packages[name]["version"] for name in selected}, **file_digest,
            "cublas": {"path": str(CUBLAS_PATH), "tree_sha256": cublas["files_sha256"], "files_count": cublas["files_count"]},
            "driver": driver}


if __name__ == "__main__":
    print(json.dumps(snapshot_host(), indent=2, sort_keys=True))
