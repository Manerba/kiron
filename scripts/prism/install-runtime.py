#!/usr/bin/env python3
"""Verify and atomically unpack the pinned Prism release into an isolated test root."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request

LOCK_PATH = Path(__file__).with_name("upstream-lock.json")
TEST_ROOT = Path("/usr/lib/kiron/test-runtimes/prism")
MAX_UNPACKED = 2 * 1024**3
RESERVE = 2 * 1024**3
MAX_MEMBERS = 4096
CHUNK = 1024**2


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
    return digest.hexdigest()


def load_lock():
    lock = json.loads(LOCK_PATH.read_text())
    tag, archive = lock["release_tag"], lock["archive"]
    expected = f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{tag}/llama-{tag}-bin-linux-cuda-12.8-x64.tar.gz"
    if (not re.fullmatch(r"prism-b[0-9]+-[0-9a-f]{7,40}", tag)
            or archive["url"] != expected
            or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
            or type(archive["bytes"]) is not int or not 0 < archive["bytes"] < MAX_UNPACKED):
        raise ValueError("invalid pinned release lock")
    return lock


def check_archive(path, lock):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != lock["archive"]["bytes"]:
        raise ValueError("archive type/size mismatch")
    with path.open("rb") as stream:
        if sha256(stream) != lock["archive"]["sha256"]:
            raise ValueError("archive SHA-256 mismatch")


class GitHubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlsplit(newurl)
        if (url.scheme != "https" or url.hostname not in
                {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
                or url.username or url.password or url.port not in (None, 443)):
            raise ValueError("unapproved download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(path, lock):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), GitHubRedirect())
    expected, count, deadline = lock["archive"]["bytes"], 0, time.monotonic() + 900
    with opener.open(lock["archive"]["url"], timeout=30) as source, path.open("xb") as target:
        declared = source.headers.get("Content-Length")
        if declared is not None and int(declared) != expected:
            raise ValueError("download Content-Length mismatch")
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("download exceeded 900 seconds")
            block = source.read(min(CHUNK, expected - count + 1))
            if not block:
                break
            count += len(block)
            if count > expected:
                raise ValueError("download exceeds pinned size")
            target.write(block)
    check_archive(path, lock)


def archive_plan(archive):
    """Validate the entire namespace before extracting even the first file."""
    plan, total = {}, 0
    for member in archive:
        path = PurePosixPath(member.name)
        if path == PurePosixPath(".") and member.isdir():
            continue
        if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                or not path.parts or not (member.isdir() or member.isfile() or member.issym())):
            raise ValueError("unsafe archive member")
        name = path.as_posix()
        if name in plan or len(plan) >= MAX_MEMBERS:
            raise ValueError("duplicate member or member limit exceeded")
        total += member.size
        if member.size < 0 or total > MAX_UNPACKED:
            raise ValueError("unpacked size limit exceeded")
        plan[name] = member
    for name, member in plan.items():
        for parent in PurePosixPath(name).parents:
            if str(parent) in plan and not plan[str(parent)].isdir():
                raise ValueError("archive ancestor is not a directory")
        if member.issym():
            current, seen = name, set()
            while plan[current].issym():
                if current in seen:
                    raise ValueError("cyclic archive symlink")
                seen.add(current)
                link = plan[current].linkname
                current = posixpath.normpath(posixpath.join(posixpath.dirname(current), link))
                if (not link or PurePosixPath(link).is_absolute() or "\\" in link
                        or current.startswith("../") or current not in plan):
                    raise ValueError("escaping or dangling archive symlink")
            if not plan[current].isfile():
                raise ValueError("symlink target is not a packaged regular file")
    return plan


def unpack_or_verify(archive_path, runtime, *, verify=False):
    with tarfile.open(archive_path, "r:gz") as archive:
        plan = archive_plan(archive)
        files, expected = {}, set()
        for name in plan:
            expected.add(name)
            expected.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
        if verify and {p.relative_to(runtime).as_posix() for p in runtime.rglob("*")} != expected:
            raise ValueError("installed namespace differs from pinned archive")
        for name, member in plan.items():
            target = runtime / name
            if verify:
                mode = target.lstat().st_mode
                valid = (stat.S_ISDIR(mode) if member.isdir() else
                         stat.S_ISLNK(mode) if member.issym() else stat.S_ISREG(mode))
                if not valid:
                    raise ValueError("installed member type mismatch")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                if not verify:
                    target.mkdir(exist_ok=True)
                continue
            if member.issym():
                if verify:
                    if os.readlink(target) != member.linkname:
                        raise ValueError("installed symlink mismatch")
                else:
                    target.symlink_to(member.linkname)
                files[name] = {"symlink": member.linkname}
                continue
            file_mode = 0o755 if member.mode & 0o111 else 0o644
            with archive.extractfile(member) as source:
                if verify:
                    expected_hash = sha256(source)
                else:
                    with target.open("xb") as output:
                        shutil.copyfileobj(source, output, CHUNK)
                    target.chmod(file_mode)
            with target.open("rb") as installed:
                actual_hash = sha256(installed)
            if verify and (actual_hash != expected_hash or stat.S_IMODE(target.stat().st_mode) != file_mode):
                raise ValueError("installed file hash/mode mismatch")
            files[name] = {"sha256": actual_hash, "mode": oct(file_mode)}
    servers = [name for name, member in plan.items() if member.isfile() and PurePosixPath(name).name == "llama-server"]
    if len(servers) != 1:
        raise ValueError("archive must contain exactly one llama-server")
    return {"server": servers[0], "files": files}


def validate_root(root):
    if not root.is_absolute():
        raise ValueError("test root must be absolute")
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("test root must not contain symlinks")
    for production in (Path("/usr/lib/kiron/runtimes"), Path("/usr/lib/kiron/services")):
        if root == production or production in root.parents:
            raise ValueError("production destination forbidden")


def verify_install(destination, lock):
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("installation is not a regular directory")
    if (destination / "runtime").is_symlink() or not (destination / "runtime").is_dir():
        raise ValueError("runtime is not a regular directory")
    check_archive(destination / "archive.tar.gz", lock)
    inventory = unpack_or_verify(destination / "archive.tar.gz", destination / "runtime", verify=True)
    record = json.loads((destination / "install-record.json").read_text())
    if record != {"upstream": lock, "inventory": inventory}:
        raise ValueError("installation record mismatch")
    return destination / "runtime" / inventory["server"]


def install(root, lock, *, archive_path=None, dry_run=False, verify=False):
    root = Path(os.path.abspath(root))
    validate_root(root)
    destination = root / lock["release_tag"]
    if destination.exists() or destination.is_symlink():
        return verify_install(destination, lock)
    if verify:
        raise ValueError("installation does not exist")
    ancestor = next(p for p in (root, *root.parents) if p.exists())
    needed = lock["archive"]["bytes"] + MAX_UNPACKED + RESERVE
    if shutil.disk_usage(ancestor).free < needed:
        raise ValueError(f"insufficient free disk space: need {needed} bytes including reserve")
    if dry_run:
        return {"destination": str(destination), "required_free_bytes": needed, "upstream": lock}
    root.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(root / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists() or destination.is_symlink():
            return verify_install(destination, lock)
        with tempfile.TemporaryDirectory(prefix=".prism-stage-", dir=root) as temporary:
            stage = Path(temporary) / "bundle"
            stage.mkdir()
            packed = stage / "archive.tar.gz"
            if archive_path is None:
                download(packed, lock)
            else:
                check_archive(Path(archive_path), lock)
                shutil.copyfile(archive_path, packed)
            check_archive(packed, lock)
            runtime = stage / "runtime"
            runtime.mkdir()
            inventory = unpack_or_verify(packed, runtime)
            (stage / "install-record.json").write_text(json.dumps({"upstream": lock, "inventory": inventory}, indent=2) + "\n")
            os.rename(stage, destination)
    return destination / "runtime" / inventory["server"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TEST_ROOT)
    parser.add_argument("--archive", type=Path, help="already downloaded archive; identical pin checks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = install(args.root, load_lock(), archive_path=args.archive, dry_run=args.dry_run, verify=args.verify)
        print(json.dumps(result if isinstance(result, dict) else {"server": str(result)}, indent=2))
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Prism install failed: {error}\n")


if __name__ == "__main__":
    main()
