"""Fixed Prism unit operations with independently checked stop completion."""
from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import os

from kiron_common.local_inference import ErrorCode, ProviderHealth
from provider_transport import failure


UNIT = "kiron-prism.service"
HELPER = "/usr/local/sbin/kiron-service-control"


async def _command(argv, *, timeout):
    process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE, start_new_session=True,
                                                  env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
    task = asyncio.create_task(process.communicate())
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            raise failure(ErrorCode.TIMEOUT, "Prism service operation timed out")
        stdout, stderr = task.result()
        if len(stdout) + len(stderr) > 65536 or process.returncode:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism service operation failed")
        return stdout.decode("utf-8", errors="strict")
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait({task}, timeout=2)
            if not task.done():
                task.cancel()


class PrismServiceControl:
    def __init__(self, *, run=_command, cgroup_root=Path("/sys/fs/cgroup")):
        self.run, self.cgroup_root = run, cgroup_root

    async def _state(self):
        output = await self.run(["/usr/bin/systemctl", "show", UNIT,
            "--property=ActiveState,SubState,MainPID,ControlGroup"], timeout=5)
        values = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in values:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Invalid Prism unit observation")
            values[key] = value
        if set(values) != {"ActiveState", "SubState", "MainPID", "ControlGroup"}:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Incomplete Prism unit observation")
        return values

    async def start(self):
        before = await self._state()
        await self.run(["/usr/bin/sudo", "-n", HELPER, "start", UNIT], timeout=60)
        after = await self._state()
        if after["ActiveState"] != "active" or after["SubState"] != "running" or not after["MainPID"].isdigit() or int(after["MainPID"]) <= 0:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism unit start was not confirmed")
        return before["ActiveState"] != "active"

    def _confirm_empty(self, group):
        if not group:
            return
        if group != "/system.slice/kiron-prism.service":
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Unexpected Prism process group")
        events = self.cgroup_root / group.lstrip("/") / "cgroup.events"
        try:
            observed = dict(line.split() for line in events.read_text().splitlines())
        except FileNotFoundError:
            observed = {"populated": "0"}
        except (OSError, ValueError) as exc:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism process group end is unknown") from exc
        if observed.get("populated") != "0":
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism process group still has processes")

    async def status(self):
        """Read-only proof of an explicitly stopped, controllably startable unit."""
        state = await self._state()
        if state["ActiveState"] == "inactive" and state["SubState"] == "dead" and state["MainPID"] == "0":
            self._confirm_empty(state["ControlGroup"])
            return ProviderHealth.STARTABLE
        return ProviderHealth.UNAVAILABLE

    async def stop(self):
        before = await self._state()
        await self.run(["/usr/bin/sudo", "-n", HELPER, "stop", UNIT], timeout=90)
        after = await self._state()
        if after["ActiveState"] != "inactive" or after["SubState"] != "dead" or after["MainPID"] != "0":
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism unit stop was not confirmed")
        group = after["ControlGroup"] or before["ControlGroup"]
        self._confirm_empty(group)
        return before["ActiveState"] != "inactive"
