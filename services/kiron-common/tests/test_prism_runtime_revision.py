"""A bundle change invalidates evidence even when llama-server is unchanged."""
from dataclasses import replace
from pathlib import Path
import unittest

from kiron_common.prism_runtime_policy import Policy


class RuntimeRevisionTests(unittest.TestCase):
    def setUp(self):
        root = Path("/fixture/bundle")
        self.policy = Policy(root, root / "llama-server", {
            "llama-server": {"sha256": "a" * 64, "mode": 0o555},
            "libggml.so.0": {"link": "libggml.so.0.1"},
            "libggml.so.0.1": {"sha256": "b" * 64, "mode": 0o444},
        }, (root,), (), {})

    def test_canonical_order_and_relocation_preserve_revision(self):
        reordered = dict(reversed(tuple(self.policy.bundle_manifest.items())))
        root = Path("/another/bundle")
        moved = replace(self.policy, runtime_root=root, binary=root / "llama-server",
                        library_dirs=(root,), bundle_manifest=reordered)
        self.assertEqual(moved.runtime_revision, self.policy.runtime_revision)
        self.assertRegex(moved.runtime_revision, r"^sha256:[a-f0-9]{64}$")

    def test_library_bytes_link_mode_or_launch_selection_change_revision(self):
        for path, record in (
            ("libggml.so.0.1", {"sha256": "c" * 64, "mode": 0o444}),
            ("libggml.so.0", {"link": "other.so"}),
            ("libggml.so.0.1", {"sha256": "b" * 64, "mode": 0o555}),
        ):
            with self.subTest(path=path, record=record):
                manifest = {**self.policy.bundle_manifest, path: record}
                self.assertNotEqual(replace(self.policy, bundle_manifest=manifest).runtime_revision,
                                    self.policy.runtime_revision)
        for changed in (replace(self.policy, binary=self.policy.runtime_root / "other"),
                        replace(self.policy, library_dirs=(self.policy.runtime_root / "lib",))):
            self.assertNotEqual(changed.runtime_revision, self.policy.runtime_revision)
