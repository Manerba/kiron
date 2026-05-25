#!/usr/bin/env python3
"""Validate the running kiron-ollama container before runtime handoff.

The Ollama compat report is only valid for the expected container contract,
not just for an image digest. Deploy calls this after compose up and before it
writes /usr/lib/kiron/data/ollama_compat_runtime.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


CONTAINER_NAME = "/kiron-ollama"
COMPOSE_SERVICE = "ollama"
EXPECTED_PORT = "11435/tcp"
EXPECTED_HOST_PORT = "11435"
EXPECTED_VOLUME = "botmin_ollama"
EXPECTED_VOLUME_DEST = "/root/.ollama"
EXPECTED_RESTART_POLICY = "unless-stopped"
EXPECTED_ENV = {
    "OLLAMA_HOST": "0.0.0.0:11435",
    "OLLAMA_KEEP_ALIVE": "24h",
    "OLLAMA_FLASH_ATTENTION": "1",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
}


def _env_map(raw: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(raw, list):
        return result
    for item in raw:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def _has_expected_port_binding(port_bindings: Any) -> bool:
    if not isinstance(port_bindings, dict):
        return False
    bindings = port_bindings.get(EXPECTED_PORT)
    if not isinstance(bindings, list):
        return False
    for item in bindings:
        if not isinstance(item, dict):
            continue
        host_ip = item.get("HostIp", "")
        host_port = item.get("HostPort")
        if host_port == EXPECTED_HOST_PORT and host_ip in ("", "0.0.0.0", "::"):
            return True
    return False


def _unexpected_bound_ports(port_bindings: Any) -> list[str]:
    if not isinstance(port_bindings, dict):
        return ["<invalid PortBindings>"]
    unexpected: list[str] = []
    for port, bindings in port_bindings.items():
        if port == EXPECTED_PORT:
            continue
        if bindings:
            unexpected.append(port)
    return unexpected


def _has_expected_volume(mounts: Any) -> bool:
    if not isinstance(mounts, list):
        return False
    for mount in mounts:
        if not isinstance(mount, dict):
            continue
        if (
            mount.get("Type") == "volume"
            and mount.get("Name") == EXPECTED_VOLUME
            and mount.get("Destination") == EXPECTED_VOLUME_DEST
            and mount.get("RW") is True
        ):
            return True
    return False


def _has_gpu_device_request(device_requests: Any) -> bool:
    if not isinstance(device_requests, list):
        return False
    for request in device_requests:
        if not isinstance(request, dict):
            continue
        if request.get("Driver") != "nvidia":
            continue
        if request.get("Count") != -1:
            continue
        capabilities = request.get("Capabilities")
        if not isinstance(capabilities, list):
            continue
        if any(isinstance(group, list) and "gpu" in group for group in capabilities):
            return True
    return False


def validate_contract(
    inspect_payload: Any,
    *,
    target_image_id: str,
    expected_image: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(inspect_payload, list) or len(inspect_payload) != 1:
        return ["docker inspect muss genau einen kiron-ollama Container liefern"]

    container = inspect_payload[0]
    if not isinstance(container, dict):
        return ["docker inspect Payload ist kein Objekt"]

    state = container.get("State") if isinstance(container.get("State"), dict) else {}
    config = container.get("Config") if isinstance(container.get("Config"), dict) else {}
    host_config = container.get("HostConfig") if isinstance(container.get("HostConfig"), dict) else {}

    if container.get("Name") != CONTAINER_NAME:
        errors.append(f"Container-Name mismatch: {container.get('Name')!r} != {CONTAINER_NAME!r}")
    if state.get("Running") is not True:
        errors.append(f"Container ist nicht running (State.Running={state.get('Running')!r})")
    if container.get("Image") != target_image_id:
        errors.append(f"ImageID mismatch: {container.get('Image')!r} != {target_image_id!r}")
    if config.get("Image") != expected_image:
        errors.append(f"Config.Image mismatch: {config.get('Image')!r} != {expected_image!r}")

    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    if labels.get("com.docker.compose.service") != COMPOSE_SERVICE:
        errors.append("Container ist nicht der erwartete Compose-Service ollama")

    restart_policy = host_config.get("RestartPolicy")
    if not isinstance(restart_policy, dict) or restart_policy.get("Name") != EXPECTED_RESTART_POLICY:
        errors.append(
            "RestartPolicy mismatch: "
            f"{restart_policy!r} != {{'Name': {EXPECTED_RESTART_POLICY!r}}}"
        )

    port_bindings = host_config.get("PortBindings")
    if not _has_expected_port_binding(port_bindings):
        errors.append("PortBinding 11435/tcp -> HostPort 11435 fehlt oder ist nicht compose-konform")
    unexpected_ports = _unexpected_bound_ports(port_bindings)
    if unexpected_ports:
        errors.append(f"Unerwartete Host-Portbindings: {', '.join(sorted(unexpected_ports))}")

    if not _has_expected_volume(container.get("Mounts")):
        errors.append("Volume botmin_ollama:/root/.ollama:rw fehlt")

    env = _env_map(config.get("Env"))
    for key, expected in EXPECTED_ENV.items():
        if env.get(key) != expected:
            errors.append(f"Env {key} mismatch: {env.get(key)!r} != {expected!r}")

    if not _has_gpu_device_request(host_config.get("DeviceRequests")):
        errors.append("GPU DeviceRequest nvidia/count=all/capabilities=gpu fehlt")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-image-id", required=True)
    parser.add_argument("--expected-image", required=True)
    args = parser.parse_args(argv)

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"FEHLER: docker inspect JSON unlesbar: {exc}", file=sys.stderr)
        return 1

    errors = validate_contract(
        payload,
        target_image_id=args.target_image_id,
        expected_image=args.expected_image,
    )
    if errors:
        print("FEHLER: kiron-ollama Runtime-Contract ungueltig:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("  Ollama Runtime-Contract validiert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
