"""Offline source-builder tests: no download, installed compiler or build executes."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location("prism_build", Path(__file__).with_name("build-runtime.py"))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patch = mock.patch.object(builder, "ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)
        self.lock = json.loads(builder.LOCK.read_text())
        self.lock["source"]["patches"] = []

    def archive(self, entries):
        path = self.root / "component.tar.xz"
        with tarfile.open(path, "w:xz") as archive:
            for name, target in entries:
                item = tarfile.TarInfo(name)
                if target is not None:
                    item.type, item.linkname = tarfile.SYMTYPE, target
                    archive.addfile(item)
                else:
                    item.size = 3
                    archive.addfile(item, io.BytesIO(b"abc"))
        return path, {"bytes": path.stat().st_size, "sha256": builder.digest(path),
                      "format": "tar.xz", "prefix": "component"}

    def test_fixed_root_rejects_production_relative_and_symlink_paths(self):
        for path in (Path("relative"), Path("/usr/lib/kiron/services"), self.root / ".." / "outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                builder.safe_path(path)
        (self.root / "linked").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            builder.safe_path(self.root / "linked" / "child")

    def test_host_drift_and_disk_preflight_reject_before_any_spawn(self):
        with mock.patch.object(builder, "module", return_value=types.SimpleNamespace(snapshot_host=lambda: {"different": True})), \
                mock.patch.object(builder.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20_000_000_000)), \
                mock.patch.object(builder.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "drift"):
                builder.preflight(self.lock)
            spawn.assert_not_called()
        with mock.patch.object(builder.shutil, "disk_usage", return_value=types.SimpleNamespace(free=builder.MIN_FREE - 1)):
            with self.assertRaisesRegex(ValueError, "13 GB"):
                builder.preflight(self.lock)

    def test_hash_failure_precedes_extract_and_does_not_write_destination(self):
        path, artifact = self.archive([("component/file", None)])
        artifact["sha256"] = "0" * 64
        with mock.patch.object(builder.tarfile, "open") as opened, self.assertRaisesRegex(ValueError, "SHA-256"):
            builder.extract(path, self.root / "unpacked", artifact)
        opened.assert_not_called()
        self.assertFalse((self.root / "unpacked").exists())

    def test_unmanaged_local_headers_reject_without_reading_their_contents(self):
        headers = self.root / "local-include"
        headers.mkdir()
        (headers / "override.h").touch()
        changed = {**self.lock, "empty_include_roots": [str(headers)]}
        with mock.patch.object(builder, "module") as module, self.assertRaisesRegex(ValueError, "local compiler headers"):
            builder.verify_host(changed)
        module.assert_not_called()

    def test_traversal_and_escaping_symlink_reject_entire_tar_before_writes(self):
        for entry in (("../outside", None), ("component/link", "../../outside")):
            with self.subTest(entry=entry):
                path, artifact = self.archive([("component/early-file", None), entry])
                with self.assertRaises(ValueError):
                    builder.extract(path, self.root / "unpacked", artifact)
                self.assertFalse((self.root / "unpacked").exists())

    def test_wheel_namespace_is_checked_before_first_output(self):
        path = self.root / "cmake.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("cmake/good", "ok")
            archive.writestr("../../outside", "bad")
        artifact = {"bytes": path.stat().st_size, "sha256": builder.digest(path), "format": "wheel"}
        with self.assertRaises(ValueError):
            builder.extract(path, self.root / "wheel", artifact)
        self.assertFalse((self.root / "wheel").exists())

    def test_merge_rejects_different_file_without_overwriting(self):
        source, target = self.root / "source", self.root / "target"
        source.mkdir()
        target.mkdir()
        (source / "header.h").write_text("new")
        (target / "header.h").write_text("old")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            builder.merge(source, target, "fixture")
        self.assertEqual((target / "header.h").read_text(), "old")

    def test_nvcc_lib64_search_resolves_to_packaged_libraries(self):
        cuda = self.root / "cuda"
        (cuda / "lib").mkdir(parents=True)
        for name in ("libcudadevrt.a", "libcudart_static.a"):
            (cuda / "lib" / name).write_bytes(b"fixture")
        builder.link_cuda_layout(cuda)
        self.assertEqual(builder.os.readlink(cuda / "lib64"), "lib")
        for name in ("libcudadevrt.a", "libcudart_static.a"):
            self.assertEqual((cuda / "lib64" / name).resolve(), cuda / "lib" / name)
        self.assertEqual(builder.inventory(cuda)["lib64"], {"link": "lib"})
        with self.assertRaises(FileExistsError):
            builder.link_cuda_layout(cuda)

    def test_prepare_publishes_lib64_layout_and_recipe_identity(self):
        lock = json.loads(json.dumps(self.lock))
        lock["artifacts"] = {"nvcc": {"format": "tar.xz", "prefix": "fixture"}}
        cublas = self.root / "cublas"
        (cublas / "include").mkdir(parents=True)
        (cublas / "lib").mkdir()
        lock["host"]["cublas"]["path"] = str(cublas)
        report = self.root / "report"
        report.mkdir()
        def fixture_extract(path, destination, artifact):
            component = destination / artifact["prefix"]
            (component / "include").mkdir(parents=True)
            (component / "lib").mkdir()
            for name in ("libcudadevrt.a", "libcudart_static.a"):
                (component / "lib" / name).write_bytes(b"fixture")
        with mock.patch.object(builder, "run"), mock.patch.object(builder, "apply_source_patches"), \
                mock.patch.object(builder, "extract", side_effect=fixture_extract), \
                mock.patch.object(builder, "git_output", return_value=lock["source"]["commit"]):
            prepared = builder.prepare(lock, report, 100)
        self.assertEqual(builder.os.readlink(prepared / "cuda" / "lib64"), "lib")
        self.assertEqual((prepared / "cuda" / "lib64" / "libcudadevrt.a").read_bytes(), b"fixture")
        record = json.loads((prepared / "prepared.json").read_text())
        self.assertEqual(record["inventory"]["cuda/lib64"], {"link": "lib"})
        self.assertEqual(record["recipe"], builder.recipe_identity())
        self.assertEqual(list(self.root.glob(".prepare-*")), [])

    def test_explicit_cache_reuse_verifies_inputs_and_never_downloads(self):
        lock = json.loads(json.dumps(self.lock))
        cached = self.root / "cached.archive"
        cached.write_bytes(b"verified archive")
        lock["artifacts"] = {"nvcc": {"format": "tar.xz", "prefix": "fixture",
            "bytes": cached.stat().st_size, "sha256": builder.digest(cached)}}
        cublas = self.root / "cublas"
        (cublas / "include").mkdir(parents=True)
        (cublas / "lib").mkdir()
        lock["host"]["cublas"]["path"] = str(cublas)
        report = self.root / "report"
        report.mkdir()
        def fixture_extract(path, destination, artifact):
            self.assertEqual(path.read_bytes(), cached.read_bytes())
            component = destination / artifact["prefix"]
            (component / "include").mkdir(parents=True)
            (component / "lib").mkdir()
        def cached_path(relative):
            return cached if relative == "nvcc.archive" else self.root / "cache/source/.git"
        with mock.patch.object(builder, "cache_input", side_effect=cached_path), \
                mock.patch.object(builder, "run") as run, \
                mock.patch.object(builder, "extract", side_effect=fixture_extract), \
                mock.patch.object(builder, "apply_source_patches"), \
                mock.patch.object(builder, "git_output", return_value=lock["source"]["commit"]):
            prepared = builder.prepare(lock, report, 100, reuse_verified_inputs=True)
        self.assertEqual(cached.read_bytes(), b"verified archive")
        commands = [call.args[0] for call in run.call_args_list]
        self.assertFalse(any("_download" in command for command in commands))
        fetch = next(command for command in commands if "fetch" in command)
        self.assertIn((self.root / "cache/source").as_uri(), fetch)
        self.assertEqual(json.loads((prepared / "prepared.json").read_text())["input_cache"], str(builder.INPUT_CACHE))
        shutil.rmtree(prepared)
        cached.write_bytes(b"changed archive")
        with mock.patch.object(builder, "cache_input", side_effect=cached_path), \
                mock.patch.object(builder, "run") as run:
            with self.assertRaises(ValueError):
                builder.prepare(lock, report, 100, reuse_verified_inputs=True)
            run.assert_not_called()
        self.assertFalse(prepared.exists())

    def test_environment_and_compiler_parallelism_are_explicit(self):
        with mock.patch.dict(builder.os.environ, {"NVCC_PREPEND_FLAGS": "--threads=0", "LLAMA_ARG_AGENT": "1", "HF_TOKEN": "secret"}):
            env = builder.clean_environment(self.lock)
        self.assertNotIn("NVCC_PREPEND_FLAGS", env)
        self.assertNotIn("HF_TOKEN", env)
        commands = builder.build_commands(self.lock, self.root / "prepared", self.root / "output")
        self.assertIn("-DCMAKE_CUDA_FLAGS=--threads=1 --split-compile=1", commands[0])
        self.assertIn("-DLLAMA_BUILD_UI=OFF", commands[0])
        self.assertIn("-DLLAMA_USE_PREBUILT_UI=OFF", commands[0])
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", commands[0])
        self.assertEqual(commands[1][-4:], ["--target", "llama-server", "--parallel", "2"])

    def test_final_executable_link_search_is_pinned_without_runtime_path_override(self):
        cuda = "/usr/lib/kiron/test-runtimes/prism/source-builds/cuda12.8-sm86-tokens-v1/prepared/cuda/lib"
        flag = f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{cuda}"
        commands = builder.build_commands(self.lock, self.root / "prepared", self.root / "output")
        self.assertIn(flag, commands[0])
        self.assertNotIn("LD_LIBRARY_PATH", builder.clean_environment(self.lock))
        self.assertFalse(any("-rpath," in arg for arg in commands[0]))

    def test_timeout_terminates_and_reaps_only_own_group(self):
        process = mock.Mock(pid=123456, returncode=None)
        with mock.patch.object(builder.subprocess, "Popen", return_value=process) as spawn, \
                mock.patch.object(builder.time, "monotonic", side_effect=[9, 11, 11]), \
                mock.patch.object(builder.os, "waitid", return_value=None), \
                mock.patch.object(builder, "group_running", return_value=False), \
                mock.patch.object(builder.os, "killpg") as kill:
            with self.assertRaises(TimeoutError):
                builder.run(["fake"], cwd=self.root, env={}, log=self.root / "build.log", deadline=10)
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(kill.call_args_list, [mock.call(process.pid, signal.SIGTERM), mock.call(process.pid, signal.SIGKILL)])
        process.wait.assert_called_once_with(timeout=5)

    def test_cleanup_kills_descendants_before_reaping_reserved_leader(self):
        process = mock.Mock(pid=123456, returncode=None)
        order = []
        process.wait.side_effect = lambda **kwargs: order.append("reap")
        with mock.patch.object(builder, "group_running", return_value=False), \
                mock.patch.object(builder.os, "killpg", side_effect=lambda pid, sig: order.append(sig)) as kill:
            builder.stop(process)
            self.assertEqual(kill.call_args_list, [mock.call(process.pid, signal.SIGTERM), mock.call(process.pid, signal.SIGKILL)])
            self.assertEqual(order, [signal.SIGTERM, signal.SIGKILL, "reap"])
            kill.reset_mock()
            process.returncode = 0
            builder.stop(process)
            kill.assert_not_called()

    def test_expired_deadline_never_spawns_and_ram_limit_stops_running_group(self):
        with mock.patch.object(builder.time, "monotonic", return_value=11), \
                mock.patch.object(builder.subprocess, "Popen") as spawn:
            with self.assertRaises(TimeoutError):
                builder.run(["fake"], cwd=self.root, env={}, log=self.root / "early.log", deadline=10)
            spawn.assert_not_called()
        process = mock.Mock(pid=123456, returncode=None)
        with mock.patch.object(builder.time, "monotonic", return_value=1), \
                mock.patch.object(builder.subprocess, "Popen", return_value=process), \
                mock.patch.object(builder.os, "waitid", return_value=None), \
                mock.patch.object(builder.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20_000_000_000)), \
                mock.patch.object(builder, "ram_available", return_value=builder.MIN_RAM - 1), \
                mock.patch.object(builder, "stop") as stop:
            with self.assertRaisesRegex(ValueError, "RAM"):
                builder.run(["fake"], cwd=self.root, env={}, log=self.root / "ram.log", deadline=10)
            stop.assert_called_once_with(process)

    def test_real_exited_leader_does_not_leave_term_resistant_child(self):
        # A tiny offline process fixture, never a compiler/runtime/network call.
        script = """import os, signal, time
r, w = os.pipe()
child = os.fork()
if child:
    os.close(w)
    os.read(r, 1)
    print(os.getpid(), child, flush=True)
    os._exit(0)
os.close(r)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.write(w, b'1')
os.close(w)
time.sleep(30)
"""
        log = self.root / "process-fixture.log"
        with mock.patch.object(builder.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20_000_000_000)), \
                mock.patch.object(builder, "ram_available", return_value=8 * 1024**3):
            builder.run([sys.executable, "-c", script], cwd=self.root, env={"PATH": "/usr/bin:/bin"},
                        log=log, deadline=builder.time.monotonic() + 20)
        pgid, child = map(int, log.read_text().splitlines()[-1].split())
        for unused in range(20):
            if not builder.group_running(pgid):
                break
            builder.time.sleep(0.05)
        self.assertFalse(builder.group_running(pgid), f"surviving fixture child {child}")

    def test_failed_prepare_cleans_stage_without_publishing(self):
        report = self.root / "report"
        report.mkdir()
        with mock.patch.object(builder, "run", side_effect=ValueError("fixture download failure")):
            with self.assertRaises(ValueError):
                builder.prepare(self.lock, report, 100)
        self.assertFalse((self.root / "prepared").exists())
        self.assertEqual(list(self.root.glob(".prepare-*")), [])

    def test_prepared_manifest_detects_modified_file_and_source_drift(self):
        prepared = self.root / "prepared"
        prepared.mkdir()
        (prepared / "binary").write_text("original")
        builder.write_json(prepared / "prepared.json", {"lock": self.lock, "recipe": builder.recipe_identity(),
                                                       "inventory": builder.inventory(prepared)})
        with mock.patch.object(builder, "git_output", side_effect=[self.lock["source"]["commit"], ""]):
            builder.verify_prepared(prepared, self.lock)
        with mock.patch.object(builder, "git_output", return_value="different"):
            with self.assertRaisesRegex(ValueError, "source checkout"):
                builder.verify_prepared(prepared, self.lock)
        (prepared / "binary").write_text("modified")
        with self.assertRaisesRegex(ValueError, "drift"):
            builder.verify_prepared(prepared, self.lock)

    def test_prepared_recipe_rejects_changed_runner_or_helper(self):
        prepared = self.root / "prepared"
        prepared.mkdir()
        recipe = builder.recipe_identity()
        self.assertEqual(set(recipe), {"build-runtime.py", "host-toolchain.py", "install-runtime.py"})
        builder.write_json(prepared / "prepared.json", {"lock": self.lock, "recipe": recipe, "inventory": {}})
        for name in recipe:
            with self.subTest(name=name), mock.patch.object(builder, "recipe_identity", return_value={**recipe, name: "0" * 64}):
                with self.assertRaisesRegex(ValueError, "recipe drift"):
                    builder.verify_prepared(prepared, self.lock)

    def test_download_redirects_never_escape_fixed_https_hosts(self):
        for url in ("http://developer.download.nvidia.com/x", "https://evil.invalid/x", "https://user@files.pythonhosted.org/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                builder.check_url(url)

    def test_checked_in_patch_is_pinned_and_changes_only_final_stream_token_retention(self):
        lock = json.loads(builder.LOCK.read_text())
        patches = builder.patchset(lock)
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual(patch["target"], "tools/server/server-context.cpp")
        value = (builder.HERE / patch["path"]).read_text()
        self.assertIn('+            res->tokens      = std::move(slot.generated_tokens);', value)
        altered = json.loads(json.dumps(lock))
        altered["source"]["patches"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, 'digest'):
            builder.patchset(altered)

    def test_real_git_patch_is_applied_once_and_all_other_source_changes_fail(self):
        source = self.root / 'source'
        source.mkdir()
        scripts = self.root / 'scripts'
        (scripts / 'patches').mkdir(parents=True)
        target = source / 'fixture.cpp'
        before, after = 'original\n', 'patched\n'
        target.write_text(before)
        env = builder.clean_environment(self.lock)
        def git(*args):
            return subprocess.check_output(['/usr/bin/git', '-C', str(source), *args],
                env=env, text=True, stderr=subprocess.DEVNULL).strip()
        git('init')
        git('add', 'fixture.cpp')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
        patch = scripts / 'patches' / 'fixture.patch'
        patch.write_text('--- a/fixture.cpp\n+++ b/fixture.cpp\n@@ -1 +1 @@\n-original\n+patched\n')
        lock = json.loads(json.dumps(self.lock))
        lock['source']['commit'] = git('rev-parse', 'HEAD')
        lock['source']['patches'] = [{'path': 'patches/fixture.patch', 'sha256': builder.digest(patch),
            'target': 'fixture.cpp', 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
            'after_sha256': hashlib.sha256(after.encode()).hexdigest()}]
        with mock.patch.object(builder, 'HERE', scripts):
            builder.apply_source_patches(source, lock, env, self.root / 'patch.log', builder.time.monotonic() + 15)
            self.assertEqual(target.read_text(), after)
            builder.verify_source(source, lock, env)
            with self.assertRaisesRegex(ValueError, 'clean'):
                builder.apply_source_patches(source, lock, env, self.root / 'patch.log', builder.time.monotonic() + 15)
            (source / 'unexpected.cpp').write_text('untracked')
            with self.assertRaisesRegex(ValueError, 'patchset'):
                builder.verify_source(source, lock, env)
            (source / 'unexpected.cpp').unlink()
            target.write_text('changed after patch\n')
            with self.assertRaisesRegex(ValueError, 'digest'):
                builder.verify_source(source, lock, env)

    def test_build_failure_preserves_cache_and_cleans_products_with_isolated_argv(self):
        report = self.root / "report"
        report.mkdir()
        def failed_command(argv, **kwargs):
            (report / "build" / "CMakeCache.txt").write_text("fixture cache")
            raise TimeoutError("fixture failure")
        with mock.patch.object(builder, "verify_prepared"), mock.patch.object(builder.os, "geteuid", return_value=0), \
                mock.patch.object(builder.pwd, "getpwnam", return_value=types.SimpleNamespace(pw_uid=65534)), \
                mock.patch.object(builder.grp, "getgrnam", return_value=types.SimpleNamespace(gr_gid=982)), \
                mock.patch.object(builder.os, "chown"), mock.patch.object(builder, "run", side_effect=failed_command):
            with self.assertRaises(TimeoutError):
                builder.build(self.lock, report, 100)
        self.assertFalse((report / "build").exists())
        self.assertEqual((report / "CMakeCache.txt").read_text(), "fixture cache")
        launch = json.loads((report / "launch.json").read_text())
        self.assertEqual(launch["commands"][0][:4], ["/usr/bin/unshare", "--net", "--", "/usr/bin/setpriv"])
        self.assertIn("--clear-groups", launch["commands"][0])
        self.assertIn("--no-new-privs", launch["commands"][0])
        self.assertEqual(launch["environment"]["TMPDIR"], str(report / "build"))
        self.assertEqual(launch["recipe"], builder.recipe_identity())

    def test_only_ordinary_build_failure_is_quarantined_and_never_marked_complete(self):
        cases = [("link", 1, 2, True), ("configure", 0, 1, False),
                 ("signal", 1, -signal.SIGKILL, False), ("low_ram", 1, 2, False)]
        for name, failed_index, code, preserved in cases:
            with self.subTest(name=name):
                report = self.root / name
                report.mkdir()
                invocations = []
                def command(argv, **kwargs):
                    (report / "build" / "CMakeCache.txt").write_text("fixture cache")
                    (report / "build" / "incomplete.o").write_bytes(b"incomplete")
                    (report / "build.log").write_text("fixture build diagnostic")
                    invocations.append(argv)
                    if len(invocations) - 1 == failed_index:
                        raise subprocess.CalledProcessError(code, argv)
                with mock.patch.object(builder, "verify_prepared"), \
                        mock.patch.object(builder.os, "geteuid", return_value=0), \
                        mock.patch.object(builder.pwd, "getpwnam", return_value=types.SimpleNamespace(pw_uid=65534)), \
                        mock.patch.object(builder.grp, "getgrnam", return_value=types.SimpleNamespace(gr_gid=982)), \
                        mock.patch.object(builder.os, "chown"), \
                        mock.patch.object(builder.time, "monotonic", return_value=1), \
                        mock.patch.object(builder.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20_000_000_000)), \
                        mock.patch.object(builder, "ram_available", return_value=0 if name == "low_ram" else builder.MIN_RAM), \
                        mock.patch.object(builder, "run", side_effect=command):
                    with self.assertRaises(subprocess.CalledProcessError):
                        builder.build(self.lock, report, 100)
                self.assertFalse((report / "build").exists())
                self.assertFalse((report / "binary-manifest.json").exists())
                self.assertEqual((report / "CMakeCache.txt").read_text(), "fixture cache")
                self.assertEqual((report / "failed-build").exists(), preserved)
                self.assertEqual((report / "failed-build.json").exists(), preserved)
                if preserved:
                    self.assertEqual((report / "failed-build" / "incomplete.o").read_bytes(), b"incomplete")
                    self.assertEqual((report / "failed-build").stat().st_mode & 0o777, 0o700)
                    record = json.loads((report / "failed-build.json").read_text())
                    self.assertEqual(record, {"status": "failed", "returncode": 2,
                                            "diagnostics": "failed-build", "reusable": False})


if __name__ == "__main__":
    unittest.main()
