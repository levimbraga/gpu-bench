# SPDX-License-Identifier: Apache-2.0
"""Turn a results CSV into one Markdown table per operation."""

import csv
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    """Read every row of a results CSV."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render(rows: list[dict[str, str]]) -> str:
    """Markdown: a header with the machine, then one table per operation."""
    if not rows:
        return "No rows.\n"
    first = rows[0]
    lines = [
        f"**GPU:** {first['gpu']} · **driver** {first['gpu_driver']} · "
        f"**CUDA** {first['cuda_toolkit']}  ",
        f"**CPU:** {first['cpu']} · **Python** {first['python']} · "
        f"**OS** {first['os']} · **WSL** {first['is_wsl']} · "
        f"**cloud** {first['is_cloud']}  ",
        f"**Measured:** {first['timestamp']} · median of "
        f"{first['repetitions']} repetitions after {first['warmup']} warm-up "
        f"calls · {first['dtype']}",
        "",
    ]
    by_op: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_op[row["op"]].append(row)
    for op, op_rows in by_op.items():
        lines += [
            f"### {op}",
            "",
            "| backend | method | size | median (ms) | RSD | GFLOP/s | GB/s "
            "| correct |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in op_rows:
            lines.append(
                f"| {row['backend']} | {row['method']} | {row['size']} "
                f"| {_short(row['median_ms'])} | {_percent(row['rsd'])} "
                f"| {_short(row['gflops'])} | {_short(row['gbps'])} "
                f"| {'yes' if row['correct'] == 'True' else 'NO'} |"
            )
        lines.append("")
    return "\n".join(lines)


def _percent(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def _short(value: str) -> str:
    # The CSV keeps 6 significant digits; 4 are enough to read a table.
    return f"{float(value):.4g}" if value else "—"
