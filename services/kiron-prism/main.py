"""Bounded HTTP control over a protected Unix socket; importing starts nothing."""
import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import grp
import json
import logging
import os
from pathlib import Path
import socket
import stat

from controller import Command, ControlError
from kiron_common.prism_runtime_policy import Policy


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CONTROL_SOCKET = Path("/run/kiron/prism/control.sock")
logger = logging.getLogger("kiron.prism")


class ControlApp:
    def __init__(self, controller):
        self.controller = controller

    async def monitor(self):
        while True:
            try:
                await self.controller.health()
            except Exception:
                logger.exception("Prism health reconciliation failed")
            await asyncio.sleep(1)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            task = asyncio.create_task(self.monitor())
            await send({"type": "lifespan.startup.complete"})
            await receive()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                await self.controller.close()
            except Exception:
                logger.exception("Prism shutdown cleanup remains unconfirmed")
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            return
        status = 200
        try:
            if scope.get("query_string"):
                raise ControlError("invalid_request", 400)
            if scope["method"] == "GET" and scope["path"] == "/health":
                result = await self.controller.health()
            elif scope["method"] == "POST" and scope["path"] in {"/load", "/unload"}:
                body = bytearray()
                async with asyncio.timeout(5):
                    while True:
                        event = await receive()
                        if event["type"] != "http.request":
                            raise ControlError("invalid_request", 400)
                        body.extend(event.get("body", b""))
                        if len(body) > 4096:
                            raise ControlError("request_too_large", 413)
                        if not event.get("more_body"):
                            break
                try:
                    value = json.loads(body)
                except (ValueError, UnicodeError):
                    raise ControlError("invalid_request", 400) from None
                result = await self.controller.mutate(scope["path"][1:], Command.parse(value))
            else:
                raise ControlError("not_found", 404)
        except ControlError as error:
            status, result = error.status, {"error": {"code": error.code}}
        except TimeoutError:
            status, result = 408, {"error": {"code": "timeout"}}
        except Exception:
            logger.exception("Prism control operation failed")
            status, result = 503, {"error": {"code": "provider_unavailable"}}
        payload = json.dumps(result, separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": payload})


def create_app(policy, resolver, admission, **controller_options):
    from controller import Controller
    return ControlApp(Controller(policy, resolver, admission, **controller_options))


@contextmanager
def control_socket(path=CONTROL_SOCKET, *, control_gid=None):
    if os.geteuid() == 0:
        raise PermissionError("controller must use its unprivileged service account")
    directory = path.parent.lstat()
    if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid()
            or directory.st_mode & 0o027):
        raise PermissionError("unsafe control socket directory")
    gid = control_gid if control_gid is not None else grp.getgrnam("kiron-prism-control").gr_gid
    if directory.st_gid != gid:
        raise PermissionError("wrong control socket group")
    descriptor = os.open(path.parent / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    sock = None
    bound = False
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
            raise PermissionError("unsafe controller lock")
        # Only a singleton-controller lock; shared GPU admission is owned elsewhere.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                raise PermissionError("foreign control socket")
            path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        bound = True
        os.chown(path, -1, gid)
        path.chmod(0o660)
        sock.listen(16)
        sock.setblocking(False)
        yield sock
    finally:
        if sock is not None:
            sock.close()
        if bound:
            path.unlink(missing_ok=True)
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DATA_DIR / "prism-runtime-policy.json")
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("run as kiron-prism, never root")
    policy = Policy.load(args.policy)
    # Root integration supplies the fixed composition: existing registry + shared
    # GPU admission. No configurable module names and no permissive fallback.
    from composition import build_controller
    import uvicorn
    app = ControlApp(build_controller(policy))
    with control_socket() as sock:
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, workers=1,
                                proxy_headers=False, timeout_keep_alive=5,
                                timeout_graceful_shutdown=5))
        asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    main()
