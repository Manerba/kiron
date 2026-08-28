"""Trusted SFT command validation shared by config, capabilities, and runner."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys


TRUSTED_TRAINER_FILENAME = "sft_trainer.py"
TRUSTED_TRAINER_PATH = Path(__file__).resolve().with_name(TRUSTED_TRAINER_FILENAME)
TRUSTED_PYTHON_PATH = Path(sys.executable)

_DUMMY_COMMANDS = {
    ":",
    "[",
    "cat",
    "dash",
    "echo",
    "false",
    "head",
    "printf",
    "sh",
    "sleep",
    "tail",
    "test",
    "true",
}


def parse_trusted_sft_command(command: object) -> list[str] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not _trusted_sft_argv(argv):
        return None
    return argv


def is_trusted_sft_command(command: object) -> bool:
    return parse_trusted_sft_command(command) is not None


def _trusted_sft_argv(argv: list[str]) -> bool:
    if len(argv) != 2:
        return False
    command_name = Path(argv[0]).name.lower()
    if command_name in _DUMMY_COMMANDS or not command_name.startswith("python"):
        return False
    executable_path = Path(argv[0])
    if not executable_path.is_absolute() or not os.access(executable_path, os.X_OK):
        return False
    script_path = Path(argv[1])
    if not script_path.is_absolute() or not script_path.is_file():
        return False
    try:
        trusted_python = TRUSTED_PYTHON_PATH.resolve(strict=True)
        executable = executable_path.resolve(strict=True)
        trusted_python_parent = TRUSTED_PYTHON_PATH.parent.resolve(strict=True)
        executable_parent = executable_path.parent.resolve(strict=True)
        return (
            executable == trusted_python
            and executable_parent == trusted_python_parent
            and script_path.resolve(strict=True) == TRUSTED_TRAINER_PATH
        )
    except OSError:
        return False
