# SPDX-License-Identifier: Apache-2.0
"""Run the benchmark on a fresh Google Colab VM and print the new CSV.

`colab run` uploads and runs a single file, but the harness is a package.
This file clones the public repository on the VM, runs the harness with the
NumPy, CuPy and PyTorch that Colab already has, and prints each new results
CSV between two marker lines, so it can be saved on the local machine:

    colab run --gpu T4 --timeout 1800 scripts/colab_run.py > colab-output.txt

Arguments after the script path are passed to `python -m bench run`.
"""

import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/levimbraga/gpu-bench.git"
WORKDIR = Path("/content/gpu-bench")


def main(args: list[str]) -> int:
    """Clone, run the harness, and print every CSV it wrote."""
    if not WORKDIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORY, str(WORKDIR)], check=True
        )
    results = WORKDIR / "results"
    before = set(results.glob("*.csv"))

    # Output is captured and printed here because, inside a notebook kernel,
    # a child process writes to the kernel's own log, not to the cell output.
    run = subprocess.run(
        [sys.executable, "-m", "bench", "run", *args],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
    )
    print(run.stderr)

    for path in sorted(set(results.glob("*.csv")) - before):
        print(f"=== CSV results/{path.name} ===")
        print(path.read_text().replace("\r\n", "\n"), end="")
        print("=== END ===")
    return run.returncode


if __name__ == "__main__":
    status = main(sys.argv[1:])
    if status != 0:
        print(f"bench exited with code {status}")
