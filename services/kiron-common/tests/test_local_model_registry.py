from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import grp
import json
import multiprocessing
import os
from pathlib import Path
import pwd
import shutil
import stat
import tempfile

import pytest

from kiron_common.local_model_registry import (
    DEFAULT_REGISTRY_PATH,
    DuplicateModelError,
    LocalModelProvider,
    RegistryAccessError,
    RegistryCorruptionError,
    RegistryEntry,
    RegistryFilePolicy,
    RuntimeModelRegistry,
    stable_registry_id,
)
from kiron_common.model_catalog import LoaderType


def _entry(index: int) -> RegistryEntry:
    return RegistryEntry.create(
        provider=LocalModelProvider.OLLAMA,
        reference=f"example/model-{index}:latest",
        display_name=f"model-{index}",
        loader=LoaderType.OLLAMA,
    )


def _process_add(registry_path: str, index: int) -> None:
    RuntimeModelRegistry(Path(registry_path)).add(_entry(index))


def _write_registry_fixture(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o640)


def _service_registry_call(
    registry_path: str,
    uid: int,
    gid: int,
    operation: str,
    index: int | None,
    connection,
) -> None:
    try:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
        registry = RuntimeModelRegistry(Path(registry_path))
        if operation == "add":
            assert index is not None
            registry.add(_entry(index))
        connection.send(
            (
                "ok",
                tuple(entry.reference for entry in registry.list()),
            )
        )
    except BaseException as error:
        connection.send(
            (
                "error",
                type(error).__name__,
                getattr(error, "code", None),
                str(error),
            )
        )
    finally:
        connection.close()


def _service_identity() -> tuple[int, int, int]:
    if os.geteuid() != 0:
        pytest.skip("cross-user registry probe requires root")
    try:
        service = pwd.getpwnam("kiron-proxy")
        shared_group = grp.getgrnam("kiron-config")
    except KeyError:
        pytest.skip("KIron service identities are not installed")
    assert service.pw_uid != 0
    assert shared_group.gr_gid not in os.getgrouplist(
        service.pw_name,
        service.pw_gid,
    )
    return service.pw_uid, service.pw_gid, shared_group.gr_gid


def _run_as_service(
    registry_path: Path,
    *,
    uid: int,
    gid: int,
    operation: str,
    index: int | None = None,
) -> tuple[str, ...]:
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_service_registry_call,
        args=(str(registry_path), uid, gid, operation, index, child),
    )
    process.start()
    child.close()
    assert parent.poll(10), "service registry probe did not return"
    result = parent.recv()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert result[0] == "ok", result
    return result[1]


def _assert_policy(path: Path, *, uid: int, gid: int) -> None:
    info = path.stat(follow_symlinks=False)
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
    assert info.st_uid == uid
    assert info.st_gid == gid
    assert stat.S_IMODE(info.st_mode) == 0o640


def _cross_user_directory(*, uid: int, gid: int) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="kiron-registry-cross-user-"))
    os.chown(directory, uid, gid)
    directory.chmod(0o2750)
    return directory


def test_default_registry_is_a_runtime_path_outside_the_source_tree() -> None:
    assert DEFAULT_REGISTRY_PATH.is_absolute()
    assert DEFAULT_REGISTRY_PATH == Path(
        "/usr/lib/kiron/data/shared/local-model-registry.json"
    )
    assert not DEFAULT_REGISTRY_PATH.is_relative_to(Path("/opt/kiron"))


def test_registry_round_trip_is_closed_deterministic_and_restart_persistent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    registry = RuntimeModelRegistry(path)
    high = _entry(2)
    low = _entry(1)
    registry.add(high)
    registry.add(low)

    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    document = json.loads(payload)
    assert document == {
        "version": 1,
        "entries": [
            item.to_dict()
            for item in sorted((high, low), key=lambda entry: entry.id)
        ],
    }
    assert RuntimeModelRegistry(path).list() == tuple(
        sorted((high, low), key=lambda entry: entry.id)
    )
    assert path.stat().st_mode & 0o777 == 0o640


def test_root_first_preserves_dashboard_read_and_write_access() -> None:
    service_uid, service_gid, shared_gid = _service_identity()
    directory = _cross_user_directory(uid=service_uid, gid=shared_gid)
    path = directory / "registry.json"
    try:
        RuntimeModelRegistry(path).add(_entry(1))
        _assert_policy(path, uid=service_uid, gid=shared_gid)
        _assert_policy(
            path.with_name("registry.json.lock"),
            uid=service_uid,
            gid=shared_gid,
        )

        assert _run_as_service(
            path,
            uid=service_uid,
            gid=service_gid,
            operation="list",
        ) == (_entry(1).reference,)
        assert set(
            _run_as_service(
                path,
                uid=service_uid,
                gid=service_gid,
                operation="add",
                index=2,
            )
        ) == {_entry(1).reference, _entry(2).reference}
        _assert_policy(path, uid=service_uid, gid=shared_gid)
        _assert_policy(
            path.with_name("registry.json.lock"),
            uid=service_uid,
            gid=shared_gid,
        )
    finally:
        shutil.rmtree(directory)


def test_dashboard_first_then_root_then_dashboard_preserves_shared_access() -> None:
    service_uid, service_gid, shared_gid = _service_identity()
    directory = _cross_user_directory(uid=service_uid, gid=shared_gid)
    path = directory / "registry.json"
    try:
        assert _run_as_service(
            path,
            uid=service_uid,
            gid=service_gid,
            operation="add",
            index=1,
        ) == (_entry(1).reference,)
        _assert_policy(path, uid=service_uid, gid=shared_gid)
        _assert_policy(
            path.with_name("registry.json.lock"),
            uid=service_uid,
            gid=shared_gid,
        )

        root_registry = RuntimeModelRegistry(path)
        assert root_registry.list() == (_entry(1),)
        root_registry.add(_entry(2))
        _assert_policy(path, uid=service_uid, gid=shared_gid)

        assert set(
            _run_as_service(
                path,
                uid=service_uid,
                gid=service_gid,
                operation="add",
                index=3,
            )
        ) == {
            _entry(1).reference,
            _entry(2).reference,
            _entry(3).reference,
        }
        _assert_policy(path, uid=service_uid, gid=shared_gid)
        _assert_policy(
            path.with_name("registry.json.lock"),
            uid=service_uid,
            gid=shared_gid,
        )
    finally:
        shutil.rmtree(directory)


def test_absent_registry_reads_as_empty_and_unknown_id_reads_none(
    tmp_path: Path,
) -> None:
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    assert registry.list() == ()
    assert registry.get("local." + "0" * 64) is None


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"not json",
        b"\xff",
        b'{"version":1,"entries":[],"extra":true}\n',
        b'{"version":true,"entries":[]}\n',
        b'{"version":1,"entries":{},"entries":[]}\n',
        b'{"version":1,"entries":[{"id":"local.0000000000000000000000000000000000000000000000000000000000000000","provider":"ollama","reference":"model:latest","display_name":"\\ud800","loader":"ollama"}]}\n',
        b'{"version":1,"entries":[{"id":"local.0000000000000000000000000000000000000000000000000000000000000000","provider":"ollama","reference":"model:latest","display_name":"model","loader":"ollama","extra":"x"}]}\n',
    ),
)
def test_corrupt_registry_is_rejected_without_fallback(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "registry.json"
    _write_registry_fixture(path, payload)
    with pytest.raises(RegistryCorruptionError) as caught:
        RuntimeModelRegistry(path).list()
    assert caught.value.code == "registry_corrupt"


def test_registry_rejects_wrong_stable_id_duplicate_identity_and_unsorted_rows(
    tmp_path: Path,
) -> None:
    first = _entry(1)
    second = _entry(2)
    documents = (
        {
            "version": 1,
            "entries": [{**first.to_dict(), "id": "local." + "0" * 64}],
        },
        {"version": 1, "entries": [first.to_dict(), first.to_dict()]},
        {
            "version": 1,
            "entries": [
                item.to_dict()
                for item in sorted(
                    (first, second),
                    key=lambda entry: entry.id,
                    reverse=True,
                )
            ],
        },
    )
    for index, document in enumerate(documents):
        path = tmp_path / f"corrupt-{index}.json"
        _write_registry_fixture(path, json.dumps(document).encode("utf-8"))
        with pytest.raises(RegistryCorruptionError):
            RuntimeModelRegistry(path).list()


def test_registry_rejects_noncanonical_references_even_with_matching_ids(
    tmp_path: Path,
) -> None:
    invalid = (
        RegistryEntry.create(
            provider=LocalModelProvider.OLLAMA,
            reference="model:latest",
            display_name="model",
            loader=LoaderType.OLLAMA,
        ).to_dict()
    )
    invalid["reference"] = "model"
    invalid["id"] = stable_registry_id(
        LocalModelProvider.OLLAMA,
        invalid["reference"],
    )
    path = tmp_path / "noncanonical.json"
    _write_registry_fixture(
        path,
        json.dumps({"version": 1, "entries": [invalid]}).encode("utf-8"),
    )
    with pytest.raises(RegistryCorruptionError):
        RuntimeModelRegistry(path).list()


def test_duplicate_add_is_deterministic_and_preserves_bytes(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = RuntimeModelRegistry(path)
    entry = _entry(1)
    registry.add(entry)
    before = path.read_bytes()
    with pytest.raises(DuplicateModelError):
        registry.add(entry)
    assert path.read_bytes() == before


def test_atomic_replace_failure_preserves_previous_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "registry.json"
    registry = RuntimeModelRegistry(path)
    registry.add(_entry(1))
    before = path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        registry.add(_entry(2))
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".local-model-registry.*.tmp")) == []


def test_atomic_temporary_is_normalized_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "registry.json"
    observed: list[tuple[int, int, int, int]] = []
    real_replace = os.replace

    def inspect_replace(source, target):
        info = os.stat(source, follow_symlinks=False)
        observed.append(
            (
                info.st_uid,
                info.st_gid,
                stat.S_IMODE(info.st_mode),
                info.st_nlink,
            )
        )
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", inspect_replace)
    RuntimeModelRegistry(path).add(_entry(1))

    parent = tmp_path.stat()
    assert observed == [(parent.st_uid, parent.st_gid, 0o640, 1)]
    _assert_policy(path, uid=parent.st_uid, gid=parent.st_gid)


def test_unachievable_ownership_is_a_static_safe_registry_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "registry.json"
    attempted: list[tuple[int, int]] = []

    def reject_chown(_descriptor: int, uid: int, gid: int) -> None:
        attempted.append((uid, gid))
        raise PermissionError("ownership-canary")

    monkeypatch.setattr(os, "fchown", reject_chown)
    policy = RegistryFilePolicy(
        owner_uid=os.getuid() + 1,
        group_gid=os.getgid(),
    )
    with pytest.raises(RegistryAccessError) as caught:
        RuntimeModelRegistry(path, file_policy=policy).add(_entry(1))

    assert caught.value.code == "registry_access_failed"
    assert str(caught.value) == "local model registry access failed"
    assert "ownership-canary" not in str(caught.value)
    if os.geteuid() == 0:
        assert attempted == [(policy.owner_uid, policy.group_gid)]
    else:
        assert attempted == []
    assert not path.exists()
    assert not path.with_name("registry.json.lock").exists()
    assert list(tmp_path.glob(".local-model-registry.*.tmp")) == []


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "directory", "fifo"))
def test_hostile_existing_lock_types_fail_closed(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "registry.json"
    lock_path = path.with_name("registry.json.lock")
    source = tmp_path / "hostile-lock-source"
    if kind == "symlink":
        source.write_bytes(b"")
        lock_path.symlink_to(source)
    elif kind == "hardlink":
        source.write_bytes(b"")
        source.chmod(0o640)
        os.link(source, lock_path)
    elif kind == "directory":
        lock_path.mkdir()
    else:
        os.mkfifo(lock_path, mode=0o640)

    with pytest.raises(RegistryCorruptionError):
        RuntimeModelRegistry(path).list()


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "directory", "fifo"))
def test_hostile_existing_registry_types_fail_closed(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "registry.json"
    source = tmp_path / "hostile-registry-source"
    if kind == "symlink":
        _write_registry_fixture(source, b'{"version":1,"entries":[]}\n')
        path.symlink_to(source)
    elif kind == "hardlink":
        _write_registry_fixture(source, b'{"version":1,"entries":[]}\n')
        os.link(source, path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path, mode=0o640)

    with pytest.raises(RegistryCorruptionError):
        RuntimeModelRegistry(path).list()


def test_existing_wrong_policy_fails_with_static_access_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    lock_path = path.with_name("registry.json.lock")
    lock_path.write_bytes(b"")
    lock_path.chmod(0o666)

    with pytest.raises(RegistryAccessError) as caught:
        RuntimeModelRegistry(path).list()
    assert caught.value.code == "registry_access_failed"
    assert str(caught.value) == "local model registry access failed"


def test_thread_concurrency_has_no_lost_updates(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"

    def add(index: int) -> None:
        RuntimeModelRegistry(path).add(_entry(index))

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(add, range(24)))
    assert {item.reference for item in RuntimeModelRegistry(path).list()} == {
        f"example/model-{index}:latest" for index in range(24)
    }


def test_process_file_lock_has_no_lost_updates(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_process_add, args=(str(path), index))
        for index in range(12)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert len(RuntimeModelRegistry(path).list()) == 12


def test_registry_path_must_be_absolute_and_parent_must_preexist(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        RuntimeModelRegistry(Path("relative.json"))
    registry = RuntimeModelRegistry(tmp_path / "missing" / "registry.json")
    with pytest.raises(FileNotFoundError):
        registry.add(_entry(1))


def test_registry_package_has_no_network_download_or_process_capability() -> None:
    package = Path(__file__).parents[1] / "kiron_common" / "local_model_registry"
    forbidden_import_roots = {
        "httpx",
        "requests",
        "socket",
        "subprocess",
        "urllib",
        "huggingface_hub",
    }
    forbidden_names = {
        "download",
        "snapshot_download",
        "pull",
        "git",
        "curl",
        "wget",
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".", 1)[0] not in forbidden_import_roots
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden_import_roots
            elif isinstance(node, ast.Name):
                assert node.id.lower() not in forbidden_names
            elif isinstance(node, ast.Attribute):
                assert node.attr.lower() not in forbidden_names
