# SPDX-License-Identifier: Apache-2.0
"""Turn results CSVs into one throughput chart per operation.

Each chart has one panel per GPU (small multiples on a shared y-axis), and
one line per library: NumPy on the host CPU, CuPy and PyTorch on the GPU,
timed with CUDA events. A last chart shows what the unsynchronized timer
reports against CUDA events on matmul.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only: no window, works over SSH and in CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatter  # noqa: E402

from bench.table import load  # noqa: E402

# Colors by library, fixed whatever the chart: the same library is always
# the same color. Validated categorical slots 1-3 for a light surface.
COLORS = {"numpy": "#2a78d6", "cupy": "#eb6834", "torch": "#1baf7a"}
LABELS = {"numpy": "NumPy (CPU)", "cupy": "CuPy", "torch": "PyTorch"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# The clock each library is read from: the CPU timer for NumPy, CUDA events
# for the GPU libraries.
CLOCK = {"numpy": "cpu_timer", "cupy": "cuda_events", "torch": "cuda_events"}

Row = dict[str, str]


def _metric(op: str) -> tuple[str, str]:
    """The CSV column to plot and its axis label."""
    if op == "matmul":
        return "gflops", "GFLOP/s"
    return "gbps", "effective GB/s"


def _machine(row: Row) -> str:
    return row["gpu"]


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plain_y(ax: Axes) -> None:
    # Log axes label ticks as powers of ten (10^2), and over a narrow range
    # as 6 x 10^0 on the minor ticks; plain numbers read better. Minor ticks
    # get labels only when the axis spans less than one power of ten, so a
    # wide axis does not stack 6000, 7000 and 8000 on top of each other.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(1, 0.4)))


def plot_operation(rows: list[Row], op: str, out: Path) -> Path:
    """One chart for one operation, one panel per GPU."""
    column, unit = _metric(op)
    op_rows = [r for r in rows if r["op"] == op and r[column]]
    machines = sorted({_machine(r) for r in op_rows})
    fig, axes = plt.subplots(
        1, len(machines), figsize=(5.2 * len(machines), 4), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    panels = [axes] if len(machines) == 1 else list(axes)
    for ax, machine in zip(panels, machines, strict=True):
        _style(ax)
        series: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in op_rows:
            if _machine(r) == machine and r["method"] == CLOCK[r["backend"]]:
                series[r["backend"]].append((int(r["size"]), float(r[column])))
        for backend in ("numpy", "cupy", "torch"):
            points = sorted(series.get(backend, []))
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(
                xs,
                ys,
                color=COLORS[backend],
                linewidth=2,
                marker="o",
                markersize=5,
                label=LABELS[backend],
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _plain_y(ax)
        ax.set_title(machine, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(
            "size" + (" (n for n×n)" if op == "matmul" else " (elements)"),
            color=MUTED,
            fontsize=9,
        )
    panels[0].set_ylabel(unit, color=MUTED, fontsize=9)
    panels[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(
        f"{op}: throughput by library (median of each run)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    path = out / f"{op}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_timers(rows: list[Row], out: Path) -> Path:
    """Matmul time by clock: what skipping synchronization reports."""
    # Clocks, not libraries: drawn in ink and gray, solid and dashed, so they
    # never borrow a library's color from the other charts.
    clocks = {
        "cuda_events": ("CUDA events", INK, "-"),
        "unsynced_cpu_timer": ("CPU timer, no synchronize", MUTED, "--"),
    }
    matmul = [r for r in rows if r["op"] == "matmul" and r["backend"] == "torch"]
    machines = sorted({_machine(r) for r in matmul})
    fig, axes = plt.subplots(
        1, len(machines), figsize=(5.2 * len(machines), 4), sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    panels = [axes] if len(machines) == 1 else list(axes)
    for ax, machine in zip(panels, machines, strict=True):
        _style(ax)
        for method, (label, color, style) in clocks.items():
            points = sorted(
                (int(r["size"]), float(r["median_ms"]))
                for r in matmul
                if _machine(r) == machine and r["method"] == method
            )
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=style,
                linewidth=2,
                marker="o",
                markersize=5,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _plain_y(ax)
        ax.set_title(machine, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("matmul size (n for n×n)", color=MUTED, fontsize=9)
    panels[0].set_ylabel("median time (ms)", color=MUTED, fontsize=9)
    panels[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(
        "PyTorch matmul: the unsynchronized timer stops at the launch",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    path = out / "timers.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_all(csv_paths: list[Path], out: Path) -> list[Path]:
    """Every chart, from every CSV given."""
    rows = [row for path in csv_paths for row in load(path)]
    out.mkdir(parents=True, exist_ok=True)
    ops = sorted({r["op"] for r in rows})
    return [plot_operation(rows, op, out) for op in ops] + [plot_timers(rows, out)]
