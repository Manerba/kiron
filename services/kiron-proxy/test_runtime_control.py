from pathlib import Path
import tempfile
import unittest

from kiron_common.local_inference import LocalInferenceError, ProviderHealth
from runtime_control import HELPER, PrismServiceControl, UNIT


def state(active="inactive", sub="dead", pid="0", group="/system.slice/kiron-prism.service"):
    return f"ActiveState={active}\nSubState={sub}\nMainPID={pid}\nControlGroup={group}\n"


class RuntimeControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_start_waits_for_running_unit(self):
        calls = []
        responses = iter((state(), "", state("active", "running", "123")))
        async def run(argv, *, timeout):
            calls.append(argv)
            return next(responses)
        control = PrismServiceControl(run=run)
        self.assertTrue(await control.start())
        self.assertEqual(calls[1], ["/usr/bin/sudo", "-n", HELPER, "start", UNIT])

    async def test_successful_stop_command_with_living_cgroup_is_not_end_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "system.slice/kiron-prism.service/cgroup.events"
            events.parent.mkdir(parents=True)
            events.write_text("populated 1\nfrozen 0\n")
            responses = iter((state("active", "running", "123"), "", state()))
            async def run(argv, *, timeout):
                return next(responses)
            with self.assertRaisesRegex(LocalInferenceError, "still has processes"):
                await PrismServiceControl(run=run, cgroup_root=root).stop()

    async def test_empty_or_removed_cgroup_confirms_stopped_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            responses = iter((state("active", "running", "123"), "", state(group="")))
            async def run(argv, *, timeout):
                return next(responses)
            self.assertTrue(await PrismServiceControl(run=run, cgroup_root=Path(tmp)).stop())

    async def test_failed_state_is_not_stopped_confirmation(self):
        responses = iter((state("active", "running", "123"), "", state("failed", "failed")))
        async def run(argv, *, timeout):
            return next(responses)
        with self.assertRaises(LocalInferenceError):
            await PrismServiceControl(run=run).stop()

    async def test_foreign_cgroup_is_rejected(self):
        responses = iter((state(group="/other.service"), "", state(group="/other.service")))
        async def run(argv, *, timeout):
            return next(responses)
        with self.assertRaisesRegex(LocalInferenceError, "Unexpected"):
            await PrismServiceControl(run=run).stop()

    async def test_startable_status_is_read_only_and_requires_verified_empty_unit(self):
        calls = []
        async def run(argv, *, timeout):
            calls.append(argv)
            return state(group="")
        self.assertEqual(await PrismServiceControl(run=run).status(), ProviderHealth.STARTABLE)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ["/usr/bin/systemctl", "show"])
        async def failed(argv, *, timeout):
            return state("failed", "failed")
        self.assertEqual(await PrismServiceControl(run=failed).status(), ProviderHealth.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
