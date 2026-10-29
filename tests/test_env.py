# SPDX-License-Identifier: Apache-2.0
"""Tests for the environment stamp."""

from pathlib import Path

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
    "git_commit",
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


def test_run_reports_a_command_that_cannot_run(tmp_path: Path) -> None:
    # Exists, but is not executable: the OS refuses to run it.
    script = tmp_path / "not-executable"
    script.write_text("#!/bin/sh\necho hi\n")
    output, reason = _run([str(script)])
    assert output is None
    assert reason is not None and "could not run" in reason


def test_run_reports_output_that_is_not_text() -> None:
    # printf writes the byte 0xff, which is not valid UTF-8.
    output, reason = _run(["printf", "\\377"])
    assert output is None
    assert reason == "printf printed output that is not text"
