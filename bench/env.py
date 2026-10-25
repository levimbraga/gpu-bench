# SPDX-License-Identifier: Apache-2.0
"""Collect the environment that produced a benchmark result.

Every result in this repository is stamped with this dictionary,
so a number can always be traced to the machine that produced it.
"""

import subprocess
from pathlib import Path


def _run(command: list[str]) -> tuple[str | None, str | None]:
    """Run a command; return (output, None) on success, (None, reason) on failure.

    Never raises: a missing, hung or failing command becomes a reason.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return None, f"{command[0]} not found"
    except subprocess.TimeoutExpired:
        return None, f"{command[0]} timed out"
    if result.returncode != 0:
        return None, f"{command[0]} exited with code {result.returncode}"
    return result.stdout, None


def _gpu_not_recorded(reason: str) -> dict[str, str]:
    """Return the GPU fields, each holding why it could not be recorded.

    Keeps the stamp's shape fixed: callers always get the same keys,
    whether the probe worked or not.
    """
    return {
        "gpu": f"not recorded: {reason}",
        "gpu_driver": f"not recorded: {reason}",
        "gpu_memory": f"not recorded: {reason}",
        "gpu_count": f"not recorded: {reason}",
    }


def _gpu() -> dict[str, str]:
    """Read GPU 0's name, driver and memory, and count the GPUs.

    Only the first GPU listed by nvidia-smi is described; the others
    are only counted. Never raises: on any failure, every field holds
    the reason instead.
    """
    output, reason = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if output is None:
        return _gpu_not_recorded(reason or "no output")

    lines = output.splitlines()
    gpu_count = len(lines)
    if not lines:
        return _gpu_not_recorded("nvidia-smi returned no GPUs")
    else:
        gpu_info = lines[0].split(",")
        if len(gpu_info) != 3:
            return _gpu_not_recorded("unexpected nvidia-smi output")
        else:
            return {
                "gpu": gpu_info[0].strip(),
                "gpu_driver": gpu_info[1].strip(),
                "gpu_memory": gpu_info[2].strip(),
                "gpu_count": str(gpu_count),
            }


def _cuda_toolkit() -> str:
    """Read the CUDA toolkit version from nvcc, e.g. "13.4".

    Never raises: on any failure, returns the reason instead.
    """
    output, reason = _run(["nvcc", "--version"])

    if output is None:
        return f"not recorded: {reason or 'no output'}"
    lines = output.splitlines()
    for line in lines:
        if "release " in line:
            return line.split("release ")[1].split(",")[0]
    return "not recorded: no release line in nvcc output"


def _cpu() -> str:
    """Read the CPU model name from /proc/cpuinfo.

    Never raises: on any failure, returns the reason instead.
    """
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return "not recorded: cannot read /proc/cpuinfo"
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return "not recorded: no model name in /proc/cpuinfo"
