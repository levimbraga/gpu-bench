# SPDX-License-Identifier: Apache-2.0
"""Measure one operation on one backend: correctness first, then time.

For every (operation, backend, size) the harness checks the result against a
float64 reference, then times the call with each of the backend's clocks:
warm-up calls first (discarded), then the timed repetitions, summarized by
the median and the relative standard deviation.
"""

import csv
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.backends import Backend, FloatArray, TorchBackend
from bench.ops import DTYPE, Operation, relative_error


@dataclass(frozen=True, slots=True)
class Result:
    """One row of a results CSV."""

    op: str
    backend: str
    method: str
    size: int
    repetitions: int
    warmup: int
    median_ms: float
    mean_ms: float
    stdev_ms: float
    # Standard deviation divided by the mean: the spread, independent of scale.
    rsd: float
    min_ms: float
    gflops: float | None
    gbps: float | None
    max_rel_error: float
    correct: bool


def summarize(samples_ms: list[float]) -> dict[str, float]:
    """Median, mean, standard deviation, relative deviation and minimum."""
    if len(samples_ms) < 2:
        raise ValueError("at least two repetitions are needed to measure spread")
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "rsd": stdev / mean if mean > 0 else 0.0,
        "min_ms": min(samples_ms),
    }


def _per_second(amount: float | None, median_ms: float) -> float | None:
    """Turn a count (FLOPs or bytes) and a time into giga-units per second."""
    if amount is None or median_ms <= 0:
        return None
    return amount / (median_ms / 1000) / 1e9


def measure(
    backend: Backend,
    op: Operation,
    size: int,
    host: list[FloatArray],
    expected: npt.NDArray[np.float64],
    repetitions: int,
    warmup: int,
) -> list[Result]:
    """Check one operation on one backend, then time it with every clock.

    The host inputs and the float64 reference (`expected`) come from the
    caller: every backend is measured on the same data, and the reference,
    which takes seconds on the CPU for a large matmul, is computed once
    instead of between the first call and the timing, where it would leave
    the GPU idle right before it is timed.
    """
    # Inputs go to the device before any timing: the transfer is measured on
    # its own, as the h2d operation.
    device = [backend.asarray(x) for x in host]
    backend.synchronize()

    def call() -> Any:
        return op.run(backend, host, device)

    # Correctness before speed. This first call also pays one-time costs
    # (library handles, kernel loading) outside every timed region.
    result = call()
    backend.synchronize()
    error = relative_error(backend.to_numpy(result), expected)
    correct = error <= op.tolerance

    rows = []
    for method, timer in backend.timers().items():
        for _ in range(warmup):
            timer(call)
        samples = [timer(call) for _ in range(repetitions)]
        stats = summarize(samples)
        rows.append(
            Result(
                op=op.name,
                backend=backend.name,
                method=method,
                size=size,
                repetitions=repetitions,
                warmup=warmup,
                median_ms=stats["median_ms"],
                mean_ms=stats["mean_ms"],
                stdev_ms=stats["stdev_ms"],
                rsd=stats["rsd"],
                min_ms=stats["min_ms"],
                gflops=_per_second(op.flops(size), stats["median_ms"]),
                gbps=_per_second(op.bytes_moved(size), stats["median_ms"]),
                max_rel_error=error,
                correct=correct,
            )
        )
    return rows


def knobs(backends: list[Backend]) -> dict[str, str]:
    """The settings that can change a result, recorded next to it."""
    recorded = {
        "dtype": np.dtype(DTYPE).name,
        "numpy_blas": _numpy_blas(),
        # The BLAS thread count follows these variables when they are set;
        # unset means the library picks, usually one thread per core.
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpu_threads": str(os.cpu_count()),
        "torch_float32_matmul_precision": "not recorded: torch backend not used",
    }
    for backend in backends:
        if isinstance(backend, TorchBackend):
            recorded["torch_float32_matmul_precision"] = (
                backend.float32_matmul_precision()
            )
    return recorded


def _numpy_blas() -> str:
    try:
        # show_config(mode="dicts") returns a dict since NumPy 1.25, but the
        # type stubs still say it returns None; typing it as Any says so.
        show_config: Any = np.show_config
        config = show_config(mode="dicts")
        blas = config["Build Dependencies"]["blas"]
        return f"{blas['name']} {blas.get('version', '')}".strip()
    except (KeyError, TypeError) as error:
        return f"not recorded: {error}"


def write_csv(rows: list[Result], stamp: dict[str, str], directory: Path) -> Path:
    """Write one CSV per run; every row carries the full environment stamp."""
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    gpu = stamp.get("gpu", "cpu")
    slug = "cpu-only" if gpu.startswith("not recorded") else _slug(gpu)
    path = directory / f"{now}_{slug}.csv"
    fieldnames = list(asdict(rows[0]).keys()) + list(stamp.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = {k: _cell(v) for k, v in asdict(row).items()}
            writer.writerow(values | stamp)
    return path


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    return "-".join(part for part in "".join(keep).split("-") if part)
