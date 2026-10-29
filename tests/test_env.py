# SPDX-License-Identifier: Apache-2.0
"""Tests for the environment stamp."""

import subprocess
from pathlib import Path

import pytest

from bench import env
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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_git_commit_marks_only_tracked_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path.resolve()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", "code.py")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(env, "_REPOSITORY", repo)

    clean = env._git_commit()
    assert len(clean) == 40 and not clean.endswith("+dirty")
    # A new, untracked file (like a results CSV being written) is not a change
    # to the code.
    (repo / "results.csv").write_text("a,b\n")
    assert env._git_commit() == clean
    # An edit to a tracked file is.
    (repo / "code.py").write_text("x = 2\n")
    assert env._git_commit() == clean + "+dirty"


def test_git_commit_refuses_an_enclosing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repository with a commit, and this code copied into a folder of it:
    # git would happily report the outer repository's commit.
    repo = tmp_path.resolve()
    _git(repo, "init", "-q")
    (repo / "other.py").write_text("y = 1\n")
    _git(repo, "add", "other.py")
    _git(repo, "commit", "-q", "-m", "init")
    inner = repo / "copied-code"
    inner.mkdir()
    monkeypatch.setattr(env, "_REPOSITORY", inner)
    assert env._git_commit().startswith("not recorded")
