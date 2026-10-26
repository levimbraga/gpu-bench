# Phase 0 — Where the numbers come from

Before measuring anything, this phase answers one question: **which machine
produced each number?** A timing without its machine is meaningless — the
same operation can take very different times on different GPUs, drivers,
CUDA versions or CPUs. So every result in this repository will carry an
**environment stamp**, produced by [`bench/env.py`](../bench/env.py).

## The three environments

Each column below is the output of the same file, unchanged, on a
different machine, on 2026-10-26 (UTC):

| Field | Desktop (main) | Google Colab | Laptop (no NVIDIA GPU) |
|---|---|---|---|
| `gpu` | NVIDIA GeForce RTX 2060 | Tesla T4 | not recorded: nvidia-smi not found |
| `gpu_driver` | 621.01 | 580.82.07 | not recorded: nvidia-smi not found |
| `gpu_memory` | 6144 MiB | 15360 MiB | not recorded: nvidia-smi not found |
| `gpu_count` | 1 | 1 | not recorded: nvidia-smi not found |
| `cuda_toolkit` | 13.4 | 13.0 | not recorded: nvcc not found |
| `cpu` | AMD Ryzen 5 3400G with Radeon Vega Graphics | Intel(R) Xeon(R) CPU @ 2.00GHz | AMD Ryzen 7 7735HS with Radeon Graphics |
| `python` | 3.13.17 | 3.13.16 | 3.13.17 |
| `os` | Ubuntu 26.04 LTS | Ubuntu 24.04.4 LTS | Ubuntu 26.04.1 LTS |
| `is_wsl` | yes | no | no |
| `is_cloud` | no | yes | no |
| `numpy` | not installed | 2.1.3 | not installed |
| `torch` | not installed | 2.11.0+cu130 | not installed |

Produced with `uv run python -m bench.env` on the desktop and the laptop,
and with `colab run --gpu T4 bench/env.py` on Colab.

The laptop column is the test of the stamp's main rule: **it never
crashes**. With no NVIDIA GPU and no CUDA, the same code ran to the end,
and every field it could not read says why.

## How the stamp works

- **One small probe per source of truth:** `nvidia-smi` for the GPU, `nvcc`
  for the CUDA toolkit, `/proc/cpuinfo` for the CPU, `/etc/os-release` for
  the OS, `/proc/version` for WSL, the `google.colab` package for Colab,
  package metadata for library versions, and the clock in UTC.
- **Fail open:** a command that is missing, hangs (5-second timeout) or
  exits with an error becomes `"not recorded: <reason>"` instead of an
  exception. A benchmark must never die because of its own label.
- **Library versions are read from package metadata, not imported:**
  importing PyTorch takes seconds and loads its CUDA libraries, just to
  read a version number.
- **A flat dictionary of text:** each field becomes one column in the
  results of the next phases.
- **Tested:** [`tests/test_env.py`](../tests/test_env.py) checks that every
  field is present, that every value is text, and that a missing or failing
  command is reported instead of raised.

## Decisions

- **WSL2 instead of native Windows** on the desktop: the same Ubuntu
  userland as the laptop and Colab, and CUDA C, Docker and CI are
  Linux-native.
- **The driver is installed only on Windows.** Inside WSL, only the CUDA
  toolkit is installed (`cuda-toolkit-13-4`, from the Ubuntu repository):
  the `cuda` and `cuda-drivers` metapackages would try to install a Linux
  driver inside WSL and break the one Windows provides.
- **Colab through its command-line interface:** the same file from this
  repository runs on both GPUs, and `colab run` releases the VM when the
  script ends.
- **Python pinned to 3.13 with `uv`**, because Colab runs 3.13; the exact
  library versions live in `uv.lock`.
- **Colab's T4 is also Turing (`sm_75`)**, like the RTX 2060: comparisons
  here are a consumer card against a data-center card of the same
  architecture, not across generations.

## Found along the way

- **The RTX 2060 also drives the desktop display:** with no benchmark
  running, two readings showed 858 MiB and 1,349 MiB of its memory in use
  and 8–9% utilization.
- **NVIDIA documents GPU utilization through NVML as not supported under
  WSL2**, yet `nvidia-smi` shows a value there. To be tested in phase 2.
- **The RTX 2060 is at the floor of CUDA 13:** `compute_75` is the oldest
  architecture `nvcc` 13.4 still targets.
- **The machines differ in ways nobody chose:** Python 3.13.17 vs 3.13.16,
  Ubuntu 26.04 vs 26.04.1 vs 24.04.4, CUDA 13.4 vs 13.0. Without the
  stamp, none of this would be visible.
- **Documentation is not the machine:** Colab's release notes list Python
  3.13.15 and Ubuntu 24.04.1 for GPU images; the stamp read 3.13.16 and
  24.04.4. Only what was read on the machine goes into a result.
- **The CPUs are very different chips**, and Colab exposes only
  "Intel Xeon @ 2.00GHz", not the model. A CPU-vs-GPU speedup is therefore
  only meaningful within one machine, not across them.
- **Setup details that differ from the guides:** the WSL-Ubuntu option
  linked from NVIDIA's WSL guide did not appear on the download page, so
  the toolkit came from the regular Ubuntu 26.04 repository; and the
  guide's `apt-key` step fails on Ubuntu 26.04, where `apt-key` no longer
  exists.
- **The first `colab run` lost its connection** before the script started,
  inside the CLI's own setup step; the retry worked, and the VM was
  released both times.

## Next: phase 1

The timing harness: the same operation on CPU and GPU, with NumPy, CuPy and
PyTorch, timed correctly — and every result saved with this stamp.