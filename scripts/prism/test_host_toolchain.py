"""Fixture tests for host input inventory; no package or toolchain mutation."""
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("host_toolchain", Path(__file__).with_name("host-toolchain.py"))
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


def package(version="1", architecture="amd64", depends="", provides="", multiarch=""):
    return dict(version=version, architecture=architecture, depends=depends, provides=provides, multiarch=multiarch)


class PackageTests(unittest.TestCase):
    def test_inventory_includes_predepends_and_held_installed_packages(self):
        lines = ["root\t1\tamd64\t\thold ok installed\tlibx (>= 1)\tdash\tcompiler",
                 "removed\t2\tamd64\t\tdeinstall ok config-files\t\t\t"]
        with patch.object(host, "command", return_value="\n".join(lines) + "\n"):
            result = host.installed_packages()
        self.assertEqual(set(result), {"root"})
        self.assertEqual(result["root"]["depends"], "libx (>= 1),dash")
        self.assertEqual(result["root"]["provides"], "compiler")

    def test_closure_resolves_versions_virtual_providers_architectures_and_cycles(self):
        packages = {
            "root": package(depends="libdev (>= 2) | fallback, api (>= 2), hosttool:native, archlib, foreign-tool"),
            "libdev:amd64": package("2", depends="root"), "fallback": package(depends="leaf"), "leaf": package(),
            "api-old": package(provides="api (= 1)"), "api-unversioned": package(provides="api"),
            "api-new": package(provides="api (= 3)"),
            "hosttool:amd64": package(), "hosttool:arm64": package(architecture="arm64"),
            "archlib:amd64": package(multiarch="same"), "archlib:i386": package(architecture="i386", multiarch="same"),
            "foreign-tool:arm64": package(architecture="arm64", multiarch="foreign"),
        }
        def compare(version, operator, required):
            return operator is None or (version is not None and int(version) >= int(required))
        selected = host.dependency_closure(packages, "amd64", seeds=("root",), compare=compare)
        self.assertEqual(selected, sorted(["root", "libdev:amd64", "fallback", "leaf", "api-new", "hosttool:amd64",
                                           "archlib:amd64", "foreign-tool:arm64"]))

    def test_missing_dependency_and_unsupported_syntax_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "no installed provider"):
            host.dependency_closure({"root": package(depends="missing")}, "amd64", seeds=("root",))
        with self.assertRaises(ValueError):
            host.parse_dependency("pkg [unresolved-source-architecture]")

    def test_debian_versions_use_dpkg_semantics(self):
        self.assertTrue(host.version_satisfies("1:1.0", ">>", "9.9"))
        self.assertFalse(host.version_satisfies("1.0~rc1", ">=", "1.0"))
        self.assertFalse(host.version_satisfies(None, "=", "1"))

    def test_subprocess_timeouts_are_bounded_and_propagated(self):
        with patch.object(host.subprocess, "run", return_value=SimpleNamespace(stdout="fixture", returncode=0)) as run:
            self.assertEqual(host.command(["/usr/bin/dpkg-query", "--version"]), "fixture")
            self.assertTrue(host.version_satisfies("1", "=", "1"))
            self.assertEqual([call.kwargs["timeout"] for call in run.call_args_list], [30, 5])
        with patch.object(host.subprocess, "run", side_effect=subprocess.TimeoutExpired("dpkg", 5)):
            with self.assertRaises(subprocess.TimeoutExpired):
                host.command(["/usr/bin/dpkg-query", "--version"])
            with self.assertRaises(subprocess.TimeoutExpired):
                host.version_satisfies("1", "=", "1")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        roots = patch.object(host, "ROOTS", (str(self.root),))
        roots.start()
        self.addCleanup(roots.stop)

    def file(self, name, content=b"fixture"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_aggregate_is_order_independent_and_sensitive_to_mode_content_and_links(self):
        first, second = self.file("compiler"), self.file("header", b"header")
        link = self.root / "cc"
        link.symlink_to(first.name)
        initial = host.aggregate([first, second, link])
        self.assertEqual(initial, host.aggregate([link, second, first, first]))
        self.assertEqual(initial["files_count"], 3)
        first.chmod(0o755)
        self.assertNotEqual(initial["files_sha256"], host.aggregate([first, second, link])["files_sha256"])
        first.chmod(0o644)
        second.write_bytes(b"changed")
        self.assertNotEqual(initial["files_sha256"], host.aggregate([first, second, link])["files_sha256"])
        before = host.aggregate([link])
        link.unlink()
        link.symlink_to(second.name)
        self.assertNotEqual(before, host.aggregate([link]))

    def test_cache_docs_and_outside_roots_are_excluded(self):
        paths = [self.file("compiler"), self.file("docs/manual"), self.file("__pycache__/module.pyc"),
                 self.file("cache/object"), Path("/etc/not-a-build-input")]
        self.assertEqual(host.aggregate(paths)["files_count"], 1)

    def test_symlink_fingerprints_target_name_without_reading_target_contents(self):
        link = self.root / "external"
        link.symlink_to("/etc/no-file-content-should-be-read")
        record = host.fingerprint(link)
        self.assertEqual(record["kind"], "symlink")
        self.assertEqual(record["target"], "/etc/no-file-content-should-be-read")
        self.assertNotIn("sha256", record)

    def test_change_during_file_hash_is_rejected(self):
        path = self.file("compiler")
        before = path.stat()
        after = SimpleNamespace(st_ino=before.st_ino, st_size=before.st_size, st_mtime_ns=before.st_mtime_ns + 1, st_mode=before.st_mode)
        with patch.object(Path, "lstat", side_effect=[before, after]), self.assertRaisesRegex(RuntimeError, "changed while"):
            host.fingerprint(path)

    def test_driver_records_full_link_chain_and_target_binary(self):
        binary = self.file("libcuda.so.123", b"driver")
        one, unversioned = self.root / "libcuda.so.1", self.root / "libcuda.so"
        one.symlink_to(binary.name)
        unversioned.symlink_to(one.name)
        with patch.object(host, "DRIVER_PATH", unversioned):
            result = host.driver_snapshot()
        self.assertEqual(result["resolved_path"], str(binary))
        self.assertEqual(result["sha256"], hashlib.sha256(b"driver").hexdigest())
        self.assertEqual([item["target"] for item in result["symlink_chain"]], [one.name, binary.name])

    def test_snapshot_composes_package_tree_cublas_and_driver_and_rejects_inventory_races(self):
        compiler = self.file("bin/compiler", b"compiler")
        cublas = self.root / "cublas"
        self.file("cublas/lib/libcublas.so.12", b"blas")
        self.file("cublas/include/cublas.h", b"declarations")
        self.file("cublas/__pycache__/ignored.pyc")
        driver = self.file("libcuda.so", b"driver")
        inventory = {"compiler": package("13")}
        def commands(args):
            if args[1] == "--print-architecture":
                return "amd64\n"
            self.assertEqual(args[1:], ["--listfiles", "compiler"])
            return str(compiler) + "\n/etc/excluded\n"
        with patch.object(host, "installed_packages", return_value=inventory), \
             patch.object(host, "dependency_closure", return_value=["compiler"]), \
             patch.object(host, "command", side_effect=commands), \
             patch.object(host, "CUBLAS_PATH", cublas), patch.object(host, "DRIVER_PATH", driver):
            result = host.snapshot_host()
            self.assertEqual(result["packages"], {"compiler": "13"})
            self.assertEqual(result["files_count"], 1)
            self.assertEqual(result["cublas"]["files_count"], 2)
            self.assertEqual(result["schema_version"], 1)
            changed = deepcopy(inventory)
            changed["compiler"]["version"] = "14"
            with patch.object(host, "installed_packages", side_effect=[inventory, changed]), \
                 self.assertRaisesRegex(RuntimeError, "metadata changed"):
                host.snapshot_host()


if __name__ == "__main__":
    unittest.main()
