"""Internal healthcheck for systemd ExecStartPost and operator smokes."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable
from urllib import error, request

import config


EXIT_HEALTHY = 0
EXIT_UNHEALTHY = 1
EXIT_INVALID_CONFIG = 2
EXIT_UNSAFE_TARGET = 3


RequestFn = Callable[[str, float], object]


def probe_once(url: str, *, timeout: float, request_fn: RequestFn | None = None) -> bool:
    if request_fn is None:
        request_fn = _urlopen_status
    try:
        response = request_fn(url, timeout)
    except (OSError, TimeoutError, error.URLError, error.HTTPError):
        return False
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    close = getattr(response, "close", None)
    if callable(close):
        close()
    return status == 200


def wait_for_health(
    cfg: config.WorkerConfig,
    *,
    timeout: float,
    interval: float,
    request_fn: RequestFn | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
) -> bool:
    url = config.health_url(cfg)
    deadline = now_fn() + timeout
    while True:
        remaining = max(0.0, deadline - now_fn())
        attempt_timeout = min(max(interval, 0.01), max(remaining, 0.01))
        if probe_once(url, timeout=attempt_timeout, request_fn=request_fn):
            return True
        remaining = deadline - now_fn()
        if remaining <= 0:
            return False
        sleep_fn(min(interval, remaining))


def run(
    argv: list[str] | None = None,
    *,
    request_fn: RequestFn | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    load_config_kwargs: dict | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Check the internal kitt-worker health endpoint")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.interval <= 0:
        return EXIT_INVALID_CONFIG

    try:
        kwargs = {} if load_config_kwargs is None else dict(load_config_kwargs)
        cfg = config.load_config(validate_runtime=True, **kwargs)
    except config.UnsafeTargetError:
        return EXIT_UNSAFE_TARGET
    except config.ConfigError:
        return EXIT_INVALID_CONFIG

    if wait_for_health(
        cfg,
        timeout=args.timeout,
        interval=args.interval,
        request_fn=request_fn,
        sleep_fn=sleep_fn,
        now_fn=now_fn,
    ):
        return EXIT_HEALTHY
    return EXIT_UNHEALTHY


def _urlopen_status(url: str, timeout: float):
    req = request.Request(url, method="GET")
    return request.urlopen(req, timeout=timeout)


if __name__ == "__main__":
    sys.exit(run())
