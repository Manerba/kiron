"""Machine-readable local root CLI for the shared registration service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from .composition import build_model_registration_service
from .errors import ModelRegistrationError
from .service import ModelRegistrationService


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="kiron-model-registry", add_help=False)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", add_help=False)
    register.add_argument("--provider", required=True)
    register.add_argument("--reference", required=True)
    register.add_argument("--loader")
    commands.add_parser("list", add_help=False)
    return parser


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": {"code": code, "message": message},
    }


def run_cli(
    argv: Sequence[str],
    *,
    service: ModelRegistrationService,
    effective_uid: int,
) -> tuple[int, dict[str, Any]]:
    """Execute one command without transport or presentation side effects."""

    if effective_uid != 0:
        return 2, _error(
            "root_required",
            "local model registry CLI requires root",
        )
    try:
        arguments = _parser().parse_args(list(argv))
    except (TypeError, ValueError) as exc:
        return 2, _error("invalid_command", str(exc))

    try:
        if arguments.command == "register":
            entry = service.register_model(
                provider=arguments.provider,
                reference=arguments.reference,
                loader=arguments.loader,
            )
            return 0, {"status": "registered", "model": entry.to_dict()}
        entries = [entry.to_dict() for entry in service.list_models()]
        return 0, {"status": "ok", "models": entries}
    except ModelRegistrationError as exc:
        return 3, _error(exc.code, str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    effective_uid = os.geteuid()
    if effective_uid != 0:
        code, payload = 2, _error(
            "root_required",
            "local model registry CLI requires root",
        )
    else:
        service = build_model_registration_service()
        code, payload = run_cli(
            sys.argv[1:] if argv is None else argv,
            service=service,
            effective_uid=effective_uid,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return code


if __name__ == "__main__":  # pragma: no cover - exercised as a command
    raise SystemExit(main())
