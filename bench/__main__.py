# SPDX-License-Identifier: Apache-2.0
"""Command line: run benchmarks into a CSV, or render a CSV as tables.

python -m bench run [--ops ...] [--backends ...] [--reps N] [--warmup N]
python -m bench table results/<file>.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from bench.backends import BACKENDS, Backend, BackendUnavailableError
from bench.env import collect_environment
from bench.harness import Result, knobs, measure, write_csv
from bench.ops import OPERATIONS
from bench.table import load, render

# Measured on the RTX 2060, in a short check whose CSV was not saved: with 2
# warm-up calls, CUDA events (the first clock to run) read 27% slower than
# the synchronized timer on a 1024 matmul; with 20, the gap was 2%. The GPU
# was probably still raising its clock. See docs/phase-1-notes.md.
DEFAULT_WARMUP = 20


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand."""
    parser = argparse.ArgumentParser(prog="python -m bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure and write a CSV to results/")
    run.add_argument(
        "--ops", nargs="+", choices=list(OPERATIONS), default=list(OPERATIONS)
    )
    run.add_argument(
        "--backends", nargs="+", choices=list(BACKENDS), default=list(BACKENDS)
    )
    run.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help="override the default sizes (needs exactly one --ops)",
    )
    run.add_argument("--reps", type=int, default=30, help="timed repetitions")
    run.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="discarded calls before timing",
    )
    run.add_argument("--seed", type=int, default=0, help="seed for the inputs")
    run.add_argument("--out", type=Path, default=Path("results"))

    table = sub.add_parser("table", help="render a results CSV as Markdown")
    table.add_argument("csv", type=Path)

    plot = sub.add_parser("plot", help="draw one chart per operation as PNG")
    plot.add_argument("csv", type=Path, nargs="+", help="one or more results CSVs")
    plot.add_argument("--out", type=Path, default=Path("docs/img"))

    args = parser.parse_args(argv)
    if args.command == "plot":
        # Imported here so running benchmarks never needs matplotlib.
        from bench.plot import plot_all

        for path in plot_all(args.csv, args.out):
            print(f"wrote {path}", file=sys.stderr)
        return 0
    if args.command == "table":
        print(render(load(args.csv)))
        return 0
    if args.reps < 2:
        # The spread (standard deviation) needs at least two samples.
        parser.error("--reps must be at least 2")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.sizes and len(args.ops) > 1:
        # Sizes mean different things per operation: 4096 is a 4096x4096
        # matmul but a 4096-element vector. One operation at a time avoids
        # running a vector size as a matrix size.
        parser.error("--sizes needs exactly one --ops")
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    backends: list[Backend] = []
    for name in args.backends:
        try:
            backends.append(BACKENDS[name]())
        except BackendUnavailableError as error:
            # A missing GPU library skips that backend; it is not a failure.
            print(f"skipping {name}: {error}", file=sys.stderr)
    if not backends:
        print("no backend available", file=sys.stderr)
        return 1

    # Collected once, at the start, and written on every row.
    # The seed is recorded too: with it, the commit, the same NumPy and the
    # same --ops and --sizes (one generator is used in that order), the
    # inputs can be generated again exactly.
    stamp = collect_environment() | knobs(backends) | {"seed": str(args.seed)}
    rng = np.random.default_rng(args.seed)
    rows: list[Result] = []
    for op_name in args.ops:
        op = OPERATIONS[op_name]
        for size in args.sizes or op.default_sizes:
            # One set of inputs per (operation, size), shared by every
            # backend: the comparison runs on the same data. The float64
            # reference is computed here, once, before any backend runs.
            host = op.make_inputs(size, rng)
            expected = op.reference(host)
            for backend in backends:
                if op.gpu_only and backend.name == "numpy":
                    continue
                print(f"{op.name} {backend.name} {size}", file=sys.stderr)
                rows += measure(
                    backend, op, size, host, expected, args.reps, args.warmup
                )

    if not rows:
        print("nothing was measured (h2d needs a GPU backend)", file=sys.stderr)
        return 1
    path = write_csv(rows, stamp, args.out)
    print(f"wrote {path}", file=sys.stderr)
    failed = [r for r in rows if not r.correct]
    for row in failed:
        print(
            f"INCORRECT: {row.op} {row.backend} {row.size} "
            f"(relative error {row.max_rel_error:.3g})",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
