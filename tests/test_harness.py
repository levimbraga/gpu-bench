# SPDX-License-Identifier: Apache-2.0
"""Tests for the harness that run on any machine (NumPy backend only)."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from bench.__main__ import main
from bench.backends import Array, Call, FloatArray, NumpyBackend, _GpuBackend
from bench.harness import measure, summarize
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
    table = render(rows)
    assert "### saxpy" in table


def test_sizes_with_several_operations_are_refused() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--ops", "matmul", "saxpy", "--sizes", "4096"])
