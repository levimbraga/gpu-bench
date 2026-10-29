# gpu-bench

GPU performance benchmarks in Python and CUDA C: NumPy, CuPy, PyTorch and
cuBLAS, measured on Linux on an RTX 2060 and cloud GPUs.

Every result is stamped with the environment that produced it — GPU, driver,
CUDA version, CPU, Python, OS — so any number here can be traced and reproduced.

## Results

Phase 1 timed four operations with NumPy on the CPU and CuPy and PyTorch on
the GPU, on both machines. All 270 results passed a correctness check
against a float64 reference. Medians of 30 repetitions, GPU times from CUDA
events; the full write-up is in [docs/phase-1-timing.md](docs/phase-1-timing.md).

| Largest size | RTX 2060 | Tesla T4 |
|---|---:|---:|
| `matmul` 4096 × 4096, NumPy (CPU) | 787 ms · 175 GFLOP/s | 1,164 ms · 118 GFLOP/s |
| `matmul` 4096 × 4096, PyTorch (GPU) | 20.9 ms · 6.6 TFLOP/s | 35.0 ms · 3.9 TFLOP/s |
| `saxpy` 2²⁴ elements, NumPy (CPU) | 26.3 ms · 7.7 GB/s | 43.2 ms · 4.7 GB/s |
| `saxpy` 2²⁴ elements, PyTorch (GPU) | 1.23 ms · 164 GB/s | 1.42 ms · 142 GB/s |

CuPy and PyTorch tied on `matmul` (both call cuBLAS). Each GPU is compared
only with the CPU of its own machine: the two CPUs are very different chips.

![matmul throughput by library](docs/img/matmul.png)

**The main finding: timing a GPU call without synchronizing measures the
launch, not the work.** On PyTorch `matmul` 4096, a CPU timer without
`synchronize` reported 0.22 ms on the RTX 2060 and 0.06 ms on the T4,
against 20.9 and 35.0 ms with CUDA events: errors of about 94× and 560×.

![PyTorch matmul time by clock](docs/img/timers.png)

## Roadmap

| Phase | Question it answers | Status |
|---|---|---|
| 0 | Where do the numbers come from? (environments, repository) | [done](docs/phase-0-environments.md) |
| 1 | How fast is the same operation on CPU and GPU, timed correctly? | [done](docs/phase-1-timing.md) |
| 2 | What is the GPU doing while it runs? (telemetry) | not yet |
| 3 | How fast does a real model run inference? | not yet |
| 4 | How close can a hand-written CUDA C kernel get to cuBLAS? | not yet |
| 5 | Can anyone rerun this the same way? (Docker, CI) | not yet |
| 6 | What did all of it show? (write-up) | not yet |

## Environments

| Machine | GPU | Architecture | Memory | Notes |
|---|---|---|---|---|
| Desktop (main) | RTX 2060 | Turing (sm_75) | 6,144 MiB | Windows 11 + WSL2, Ubuntu 26.04; also drives the display |
| Google Colab | Tesla T4 | Turing (sm_75) | 15,360 MiB | cloud GPU, assigned per session, no display |

Both GPUs are Turing: comparisons here are consumer vs. data center on the
same architecture, not across generations.

## Not yet

- **The RTX 2060 also drives the desktop display**, which uses about 1.3 GB of
  its memory and some of its time. Results on it carry that load; the T4 does not.
- **Telemetry under WSL2 may be limited**: the NVML documentation lists GPU
  utilization as not supported there. To be tested in phase 2.
- **Colab does not guarantee which GPU a session gets**: every Colab result
  records the GPU it actually ran on.
- **Multi-GPU machines are not fully described**: the environment stamp
  records the first GPU that `nvidia-smi` lists and only counts the others.
- **Warm-up is a fixed number of calls (20)**, but the GPU needs *time* to
  raise its clock after being idle: after seconds of CPU-only work, 20 calls
  were not enough (see phase 1). A time-based warm-up is not implemented yet.
- **The RTX 2060's results are noisier** (RSD above 50% in many rows, against
  under 10% on the T4), probably from the display and WSL2; not isolated yet.
- **The CUDA libraries installed by pip are 13.0** (PyTorch is built for
  CUDA 13.0), while the desktop's `nvcc` is 13.4: the `cuda_toolkit` field
  records the compiler, not the libraries the Python benchmarks ran on.
- **Host-to-device copies come from ordinary (pageable) memory**; pinned
  memory is not measured yet.
- **Some results are not explained yet**, and are listed as such in the
  phase 1 write-up.

## License

Apache 2.0 — see [LICENSE](LICENSE).