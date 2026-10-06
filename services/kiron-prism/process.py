"""One native child, loopback readiness and bounded process-group cleanup."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx


def require_free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match native restart semantics: TIME_WAIT is not a live listener.
        # SO_REUSEPORT stays disabled, so a foreign listener still blocks bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def group_running(pgid):
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


class NativeChild:
    def __init__(self, argv, env, descriptors, policy):
        if os.geteuid() == 0:
            raise PermissionError("Prism controller must run as its unprivileged service identity")
        self.policy = policy
        self._alias = None
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{policy.port}",
                                       trust_env=False, follow_redirects=False, timeout=policy.health_timeout)
        self.process = subprocess.Popen(argv, env=env, cwd="/", stdin=subprocess.DEVNULL,
                                        start_new_session=True, pass_fds=tuple(descriptors), umask=0o077)

    def alive(self):
        if self.process.returncode is not None:
            return False
        # Keep an exited leader unreaped until all group signals have completed.
        return os.waitid(os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None

    def owns_port(self):
        try:
            sockets = {os.readlink(path) for path in Path(f"/proc/{self.process.pid}/fd").iterdir()}
            for line in Path(f"/proc/{self.process.pid}/net/tcp").read_text().splitlines()[1:]:
                row = line.split()
                if (row[1] == f"0100007F:{self.policy.port:04X}" and row[3] == "0A"
                        and f"socket:[{row[9]}]" in sockets):
                    return True
        except (OSError, IndexError, ValueError):
            pass
        return False

    async def _json(self, path):
        if self._alias is None:
            return None
        async with self.client.stream("GET", path, headers={"Authorization": f"Bearer {self._alias}"}) as response:
            if response.status_code != 200:
                return None
            content = bytearray()
            async for block in response.aiter_bytes():
                content.extend(block)
                if len(content) > 65536:
                    return None
        return json.loads(content)

    async def ready(self, alias):
        self._alias = alias
        async def check():
            if not self.alive() or not self.owns_port():
                return False
            health = await self._json("/health")
            models = await self._json("/v1/models")
            return (self.alive() and self.owns_port() and isinstance(health, dict)
                    and health.get("status") == "ok" and isinstance(models, dict)
                    and any(isinstance(item, dict) and item.get("id") == alias
                            for item in models.get("data", [])))
        try:
            return await asyncio.wait_for(check(), self.policy.health_timeout)
        except (OSError, ValueError, TypeError, httpx.HTTPError, TimeoutError):
            return False

    async def slots(self):
        """Current slot evidence, not proof that an as-yet unassigned request ended."""
        async def check():
            if not self.alive() or not self.owns_port():
                return None
            slots = await self._json("/slots")
            if (not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict)
                    or slots[0].get("id") != 0 or type(slots[0].get("is_processing")) is not bool):
                return None
            task_id = slots[0].get("id_task")
            if task_id is not None and (type(task_id) is not int or task_id < 0):
                return None
            if slots[0]["is_processing"] and task_id is None:
                return None
            if not self.alive() or not self.owns_port():
                return None
            return {"active_requests": int(slots[0]["is_processing"]), "slot_task_id": task_id,
                    "slots_observed_at": time.time()}
        try:
            return await asyncio.wait_for(check(), self.policy.health_timeout)
        except (OSError, ValueError, TypeError, httpx.HTTPError, TimeoutError):
            return None

    async def stop(self):
        """Preserve PID/PGID ownership through TERM/KILL, then reap the direct child."""
        process = self.process
        try:
            if process.returncode is not None:
                return True
            for sig, budget in ((signal.SIGTERM, self.policy.term_timeout),
                                (signal.SIGKILL, self.policy.kill_timeout)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + budget
                while group_running(process.pid) and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            if group_running(process.pid):
                return False
            try:
                process.wait(timeout=0.1)
                return True
            except subprocess.TimeoutExpired:
                return False
        finally:
            await self.client.aclose()
