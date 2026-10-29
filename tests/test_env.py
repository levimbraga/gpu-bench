# SPDX-License-Identifier: Apache-2.0
"""Tests for the environment stamp."""

from bench.env import _run, collect_environment

EXPECTED_FIELDS = {
    "gpu",
    "gpu_driver",
    "gpu_memory",
    "gpu_count",
    "cuda_toolkit",
    "cpu",
    "python",
    "os",
    "is_wsl",
    "is_cloud",
    "numpy",
    "torch",
    "cupy-cuda13x",
    "cupy-cuda12x",
    "timestamp",
}


def test_stamp_has_every_field() -> None:
    assert set(collect_environment()) == EXPECTED_FIELDS


def test_every_value_is_text() -> None:
    for value in collect_environment().values():
        assert isinstance(value, str), f"{value!r} is not text"


def test_run_reports_a_missing_command() -> None:
    assert _run(["no-such-command-xyz"]) == (None, "no-such-command-xyz not found")


def test_run_reports_a_failing_command() -> None:
    assert _run(["false"]) == (None, "false exited with code 1")
