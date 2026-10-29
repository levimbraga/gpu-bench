# SPDX-License-Identifier: Apache-2.0
"""Tests for the harness that run on any machine (NumPy backend only)."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from bench.__main__ import main
from bench.backends import (
    BACKENDS,
    Array,
    Call,
    FloatArray,
    NumpyBackend,
    _GpuBackend,
)
from bench.harness import _per_second, measure, summarize
from bench.ops import MATMUL, OPERATIONS, SAXPY, relative_error
from bench.table import load, render


@pytest.mark.parametrize("name", ["matmul", "saxpy", "sum"])
def test_numpy_results_pass_their_own_check(name: str) -> None:
    op = OPERATIONS[name]
    host = op.make_inputs(64, np.random.default_rng(0))
    rows = measure(
        NumpyBackend(), op, 64, host, op.reference(host), repetitions=3, warmup=1
    )
    assert rows and all(row.correct for row in rows)


def test_a_wrong_result_fails_the_check() -> None:
    reference = np.array([1.0, 2.0, 4.0])
    wrong = np.array([1.0, 2.0, 5.0])
    assert relative_error(wrong, reference) == pytest.approx(0.25)


def test_summary_of_known_samples() -> None:
    stats = summarize([1.0, 2.0, 3.0])
    assert stats["median_ms"] == 2.0
    assert stats["stdev_ms"] == pytest.approx(1.0)
    assert stats["rsd"] == pytest.approx(0.5)


def test_one_repetition_is_refused() -> None:
    with pytest.raises(ValueError):
        summarize([1.0])


def test_work_counts() -> None:
    # A 2x2 matmul: 4 outputs, each 2 multiplies and 2 adds.
    assert MATMUL.flops(2) == 16
    # SAXPY on 10 float32: read x and y, write the result.
    assert SAXPY.bytes_moved(10) == 120


class _FakeGpuBackend(_GpuBackend):
    """Runs on NumPy but goes through the GPU timing code paths.

    Lets the three GPU clocks and the CSV columns they produce be tested on a
    machine with no GPU. Its "events" are a CPU clock; only the plumbing is
    under test here, not the timing.
    """

    name = "fakegpu"

    def __init__(self) -> None:
        self.synchronize_calls = 0

    def asarray(self, host: FloatArray) -> Array:
        return host.copy()

    def to_numpy(self, x: Array) -> npt.NDArray[Any]:
        return np.asarray(x)

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def _events_ms(self, fn: Call) -> float:
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000


def test_gpu_backends_produce_one_row_per_clock() -> None:
    backend = _FakeGpuBackend()
    host = SAXPY.make_inputs(32, np.random.default_rng(0))
    rows = measure(
        backend, SAXPY, 32, host, SAXPY.reference(host), repetitions=2, warmup=1
    )
    assert [row.method for row in rows] == [
        "cuda_events",
        "synced_cpu_timer",
        "unsynced_cpu_timer",
    ]
    assert all(row.correct for row in rows)
    # The two CPU clocks wait for the device: before and after each call.
    assert backend.synchronize_calls > 0


class _SlowDevice(_FakeGpuBackend):
    """A device whose synchronize takes 20 ms, like a GPU still busy."""

    def synchronize(self) -> None:
        super().synchronize()
        time.sleep(0.02)


def test_only_the_synced_timer_counts_the_wait() -> None:
    backend = _SlowDevice()

    def call() -> None:
        return None

    before = backend.synchronize_calls
    synced = backend._synced_cpu_timer_ms(call)
    # One wait before starting the clock, one before stopping it.
    assert backend.synchronize_calls - before == 2
    # The closing wait (20 ms) is inside the synced timer, and outside the
    # unsynced one, which stops as soon as the call returns.
    unsynced = backend._unsynced_cpu_timer_ms(call)
    assert synced >= 20
    assert unsynced < 20


def test_throughput_from_a_known_time() -> None:
    # 12 GB in one second is 12 GB/s.
    assert _per_second(12e9, 1000.0) == pytest.approx(12.0)
    assert _per_second(None, 1000.0) is None
    host = SAXPY.make_inputs(32, np.random.default_rng(0))
    (row,) = measure(
        NumpyBackend(), SAXPY, 32, host, SAXPY.reference(host), 2, warmup=0
    )
    moved = SAXPY.bytes_moved(32)
    assert moved is not None
    assert row.gbps == pytest.approx(moved / (row.median_ms / 1000) / 1e9)


class _WrongNumpy(NumpyBackend):
    """Returns every result off by one, so the correctness check must fail."""

    def to_numpy(self, x: Array) -> npt.NDArray[Any]:
        return np.asarray(x) + 1.0


def test_a_wrong_result_is_recorded_and_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(BACKENDS, "numpy", _WrongNumpy)
    status = main(
        [
            "run",
            "--ops",
            "saxpy",
            "--backends",
            "numpy",
            "--sizes",
            "64",
            "--reps",
            "2",
            "--warmup",
            "0",
            "--out",
            str(tmp_path),
        ]
    )
    assert status == 1
    (csv_path,) = tmp_path.iterdir()
    assert {row["correct"] for row in load(csv_path)} == {"False"}


def test_transfer_is_measured_only_on_gpu_backends(tmp_path: Path) -> None:
    status = main(
        ["run", "--ops", "h2d", "--backends", "numpy", "--out", str(tmp_path)]
    )
    assert status == 1
    assert list(tmp_path.iterdir()) == []


def test_run_writes_a_csv_that_renders(tmp_path: Path) -> None:
    status = main(
        [
            "run",
            "--ops",
            "saxpy",
            "--backends",
            "numpy",
            "--sizes",
            "64",
            "--reps",
            "2",
            "--warmup",
            "0",
            "--out",
            str(tmp_path),
        ]
    )
    assert status == 0
    (csv_path,) = tmp_path.iterdir()
    rows = load(csv_path)
    assert {row["op"] for row in rows} == {"saxpy"}
    # Every row carries the environment stamp and the knobs.
    assert all(row["gpu"] and row["dtype"] == "float32" for row in rows)
    assert all(row["seed"] == "0" and row["git_commit"] for row in rows)
    table = render(rows)
    assert "### saxpy" in table


def test_sizes_with_several_operations_are_refused() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--ops", "matmul", "saxpy", "--sizes", "4096"])
