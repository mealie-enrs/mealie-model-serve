from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from typing import Any


def _run_command(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def collect_environment_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "git_sha": os.environ.get("GIT_SHA", _run_command(["git", "rev-parse", "HEAD"])),
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def flatten_dict(data: dict[str, Any], prefix: str | None = None) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=joined))
        else:
            flattened[joined] = str(value)
    return flattened
