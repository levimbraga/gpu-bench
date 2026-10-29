# SPDX-License-Identifier: Apache-2.0
"""The operations the harness measures, and how to judge each one.

Each operation knows how to build its inputs, run on any backend, compute a
float64 reference on the CPU, and count the work it does, so the harness can
check correctness first and then turn a time into GFLOP/s or GB/s.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.backends import Array, Backend, FloatArray

# Every input is float32: it is what GPUs are fast at, and what the
# throughput numbers below assume (4 bytes per element).
DTYPE = np.float32
BYTES_PER_ELEMENT = 4

# The scalar in SAXPY (y = a*x + y). Its value does not affect the timing.
ALPHA = 2.0


@dataclass(frozen=True, slots=True)
class Operation:
    """One measured operation."""

    name: str
    # Builds the host inputs for a given size.
    make_inputs: Callable[[int, np.random.Generator], list[FloatArray]]
    # Runs once on a backend. Gets the host inputs and their device copies.
    run: Callable[[Backend, list[FloatArray], list[Array]], Array]
    # The same result in float64 on the CPU, to judge correctness.
    reference: Callable[[list[FloatArray]], npt.NDArray[np.float64]]
    # Floating-point operations and bytes moved for one run, or None when the
    # metric does not describe this operation.
    flops: Callable[[int], float | None]
    bytes_moved: Callable[[int], float | None]
    # Largest relative error accepted against the reference.
    tolerance: float
    default_sizes: tuple[int, ...]
    gpu_only: bool = False


def _vector(size: int, rng: np.random.Generator) -> FloatArray:
    # Values in [0, 1): no cancellation, so relative errors stay meaningful.
    return rng.random(size, dtype=DTYPE)


def _square(size: int, rng: np.random.Generator) -> FloatArray:
    return rng.random((size, size), dtype=DTYPE)


def _as64(x: FloatArray) -> npt.NDArray[np.float64]:
    return x.astype(np.float64)


MATMUL = Operation(
    name="matmul",
    make_inputs=lambda n, rng: [_square(n, rng), _square(n, rng)],
    run=lambda backend, host, dev: dev[0] @ dev[1],
    reference=lambda host: _as64(host[0]) @ _as64(host[1]),
    # Each of the n*n outputs is a dot product of length n: n multiplies and
    # n adds (counting the first add, as is conventional), so 2*n^3 in total.
    flops=lambda n: 2.0 * n**3,
    bytes_moved=lambda n: None,
    # float32 accumulates rounding error along each length-n dot product.
    tolerance=1e-3,
    default_sizes=(256, 512, 1024, 2048, 4096),
)

SAXPY = Operation(
    name="saxpy",
    make_inputs=lambda n, rng: [_vector(n, rng), _vector(n, rng)],
    run=lambda backend, host, dev: ALPHA * dev[0] + dev[1],
    reference=lambda host: ALPHA * _as64(host[0]) + _as64(host[1]),
    # One multiply and one add per element.
    flops=lambda n: 2.0 * n,
    # Reads x and y, writes the result: 3 arrays of n float32. This is the
    # minimum traffic; a library that keeps a temporary moves more, which
    # shows up as a lower effective bandwidth.
    bytes_moved=lambda n: 3.0 * n * BYTES_PER_ELEMENT,
    tolerance=1e-6,
    default_sizes=(2**16, 2**18, 2**20, 2**22, 2**24),
)

SUM = Operation(
    name="sum",
    make_inputs=lambda n, rng: [_vector(n, rng)],
    run=lambda backend, host, dev: dev[0].sum(),
    reference=lambda host: np.asarray(_as64(host[0]).sum()),
    # n - 1 additions.
    flops=lambda n: float(n - 1),
    # Reads x once; the single output is negligible.
    bytes_moved=lambda n: float(n * BYTES_PER_ELEMENT),
    # Summation order differs between libraries; float32 rounding follows it.
    tolerance=1e-4,
    default_sizes=(2**16, 2**18, 2**20, 2**22, 2**24),
)

H2D = Operation(
    name="h2d",
    make_inputs=lambda n, rng: [_vector(n, rng)],
    # The copy itself, from ordinary (pageable) host memory to the GPU. Kept
    # as its own row, so the other operations are timed with data already on
    # the device.
    run=lambda backend, host, dev: backend.asarray(host[0]),
    reference=lambda host: _as64(host[0]),
    flops=lambda n: None,
    bytes_moved=lambda n: float(n * BYTES_PER_ELEMENT),
    tolerance=0.0,
    default_sizes=(2**16, 2**18, 2**20, 2**22, 2**24),
    gpu_only=True,
)

OPERATIONS: dict[str, Operation] = {op.name: op for op in (MATMUL, SAXPY, SUM, H2D)}


def relative_error(
    result: npt.NDArray[Any], reference: npt.NDArray[np.float64]
) -> float:
    """Largest absolute difference, relative to the largest reference value."""
    difference = float(np.max(np.abs(result.astype(np.float64) - reference)))
    scale = float(np.max(np.abs(reference)))
    return difference / scale if scale > 0 else difference
