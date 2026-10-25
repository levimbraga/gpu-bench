# SPDX-License-Identifier: Apache-2.0
"""Collect the environment that produced a benchmark result.

Every result in this repository is stamped with this dictionary,
so a number can always be traced to the machine that produced it.
"""

import subprocess


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
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _gpu_not_recorded("nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return _gpu_not_recorded("command timed out")
    else:
        if result.returncode != 0:
            return _gpu_not_recorded(f"nvidia-smi exited with code {result.returncode}")
        else:
            lines = result.stdout.splitlines()
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
