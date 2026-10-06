"""Offline integrity/containment tests; every installation stays in a tempdir."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prism_install", Path(__file__).with_name("install-runtime.py"))
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "runtimes"
        self.archive = self.base / "release.tar.gz"
        self.lock = copy.deepcopy(installer.load_lock())

    def pack(self, members=None):
        rows = members if members is not None else [
            ("package/llama-server", b"fake executable", None),
            ("package/libtest.so.1.0", b"fake shared library", None),
            ("package/libtest.so.1", None, "libtest.so.1.0"),
            ("package/libtest.so", None, "libtest.so.1"),
        ]
        with tarfile.open(self.archive, "w:gz") as output:
            for name, body, link in rows:
                member = tarfile.TarInfo(name)
                member.mode = 0o755 if name.endswith("llama-server") else 0o644
                if link is not None:
                    member.type, member.linkname = tarfile.SYMTYPE, link
                elif body is None:
                    member.type = tarfile.FIFOTYPE
                else:
                    member.size = len(body)
                output.addfile(member, io.BytesIO(body) if body is not None else None)
        data = self.archive.read_bytes()
        self.lock["archive"].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def install(self, **kwargs):
        return installer.install(self.root, self.lock, archive_path=self.archive, **kwargs)

    def test_verified_install_and_reuse_with_packaged_relative_symlink_chain(self):
        self.pack()
        server = self.install()
        before = server.stat().st_mtime_ns
        self.assertEqual(server.read_bytes(), b"fake executable")
        self.assertEqual((server.parent / "libtest.so").read_bytes(), b"fake shared library")
        with mock.patch.object(installer, "download", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.install(), server)
            self.assertEqual(self.install(verify=True), server)
        self.assertEqual(server.stat().st_mtime_ns, before)
        record = json.loads((server.parents[2] / "install-record.json").read_text())
        self.assertEqual(record["upstream"], self.lock)

    def test_hash_and_size_rejected_before_any_extraction(self):
        for changed in ("hash", "size"):
            with self.subTest(changed=changed):
                self.pack()
                if changed == "hash":
                    self.lock["archive"]["sha256"] = "0" * 64
                else:
                    self.lock["archive"]["bytes"] += 1
                with mock.patch.object(installer, "unpack_or_verify") as extract:
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        self.install()
                    extract.assert_not_called()
                self.assertFalse((self.root / self.lock["release_tag"]).exists())
                self.assertEqual(list(self.root.glob(".prism-stage-*")), [])

    def test_traversal_absolute_path_special_file_and_duplicates_precede_extraction(self):
        cases = [
            [("../outside", b"bad", None)],
            [("/tmp/absolute", b"bad", None)],
            [("pipe", None, None)],
            [("same", b"one", None), ("same", b"two", None)],
        ]
        for members in cases:
            with self.subTest(members=members):
                self.pack([("llama-server", b"first valid entry", None), *members])
                target = self.base / "extracted"
                target.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    installer.unpack_or_verify(self.archive, target)
                self.assertEqual(list(target.iterdir()), [])

    def test_escape_ancestor_dangling_and_cyclic_symlinks_rejected_before_extraction(self):
        cases = [
            [("lib.so", None, "../outside")],
            [("lib.so", None, "/etc/passwd")],
            [("lib.so", None, "missing")],
            [("a", None, "b"), ("b", None, "a")],
            [("lib", None, "llama-server"), ("lib/child", b"bad", None)],
        ]
        for members in cases:
            with self.subTest(members=members):
                self.pack([("llama-server", b"good", None), *members])
                target = self.base / "extracted"
                target.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    installer.unpack_or_verify(self.archive, target)
                self.assertEqual(list(target.iterdir()), [])

    def test_failure_mid_extract_leaves_no_published_destination(self):
        self.pack()
        with mock.patch.object(installer.shutil, "copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.install()
        self.assertFalse((self.root / self.lock["release_tag"]).exists())
        self.assertEqual(list(self.root.glob(".prism-stage-*")), [])

    def test_corrupt_existing_destination_is_never_replaced(self):
        self.pack()
        server = self.install()
        server.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash/mode"):
            self.install()
        self.assertEqual(server.read_bytes(), b"tampered")

    def test_reuse_does_not_trust_a_forged_record_over_pinned_archive(self):
        self.pack()
        server = self.install()
        server.write_bytes(b"tampered")
        record_path = server.parents[2] / "install-record.json"
        record = json.loads(record_path.read_text())
        record["inventory"]["files"]["package/llama-server"]["sha256"] = hashlib.sha256(b"tampered").hexdigest()
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "hash/mode"):
            self.install(verify=True)

    def test_extra_file_and_changed_symlink_rejected_on_reuse(self):
        self.pack()
        server = self.install()
        extra = server.parent / "extra"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "namespace"):
            self.install(verify=True)
        extra.unlink()
        link = server.parent / "libtest.so"
        link.unlink()
        link.symlink_to("llama-server")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install(verify=True)

    def test_dry_run_is_read_only_and_disk_preflight_blocks_before_mkdir(self):
        self.pack()
        result = self.install(dry_run=True)
        self.assertIn("required_free_bytes", result)
        self.assertFalse(self.root.exists())
        with mock.patch.object(installer.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaisesRegex(ValueError, "disk space"):
                self.install()
        self.assertFalse(self.root.exists())

    def test_reuse_rejects_runtime_directory_replaced_by_external_symlink(self):
        self.pack()
        server = self.install()
        runtime = server.parents[1]
        external = self.base / "external"
        runtime.rename(external)
        runtime.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular directory"):
            self.install(verify=True)

    def test_symlink_and_production_roots_are_rejected(self):
        self.pack()
        self.root.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.install()
        with self.assertRaisesRegex(ValueError, "production"):
            installer.validate_root(Path("/usr/lib/kiron/runtimes/prism"))

    def test_download_redirect_policy_rejects_non_github_targets(self):
        policy = installer.GitHubRedirect()
        for url in ("http://github.com/release", "https://evil.example/asset", "https://github.com:444/asset"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "redirect"):
                policy.redirect_request(None, None, 302, None, {}, url)

    def test_unpacked_size_limit_rejects_before_extraction(self):
        self.pack()
        target = self.base / "extract"
        target.mkdir()
        with mock.patch.object(installer, "MAX_UNPACKED", 2):
            with self.assertRaisesRegex(ValueError, "size limit"):
                installer.unpack_or_verify(self.archive, target)
        self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
