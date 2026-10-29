# gpu-bench

GPU performance benchmarks: NumPy, CuPy and PyTorch today, a hand-written
CUDA C kernel against cuBLAS later (phase 4), measured on Linux on an
RTX 2060 and a cloud Tesla T4.

Every result is stamped with the environment that produced it — GPU, driver,
CUDA version, CPU, Python, OS, the git commit and the seed — so any number
here can be traced to its machine and its code. Rerunning it the same way
anywhere is phase 5.

## Results

Phase 1 timed four operations with NumPy on the CPU and CuPy and PyTorch on
the GPU, on both machines. Every result passed a correctness check against
a float64 reference. Medians of 30 repetitions, GPU times from CUDA events,
from Run 5 on the RTX 2060
([CSV](results/20261029T175052Z_nvidia-geforce-rtx-2060.csv)) and Run 4 on
the T4 ([CSV](results/20261029T173230Z_tesla-t4.csv)); the full write-up is
in [docs/phase-1-timing.md](docs/phase-1-timing.md).

| Largest size | RTX 2060 | Tesla T4 |
|---|---:|---:|
| `matmul` 4096 × 4096, NumPy (CPU) | 796 ms · 173 GFLOP/s | 1,182 ms · 116 GFLOP/s |
| `matmul` 4096 × 4096, PyTorch (GPU) | 21.0 ms · 6.5 TFLOP/s | 31.2 ms · 4.4 TFLOP/s |
| `saxpy` 2²⁴ elements, NumPy (CPU) | 25.8 ms · 7.8 GB/s | 44.0 ms · 4.6 GB/s |
| `saxpy` 2²⁴ elements, PyTorch (GPU) | 1.22 ms · 166 GB/s | 1.43 ms · 141 GB/s |

CuPy and PyTorch tie on `matmul` at 4096 (both call cuBLAS). Each GPU is
compared only with the CPU of its own machine: the two CPUs are very
different chips. T4 numbers are uncertain by at least 12%: two Colab
sessions differed by that much on the same operation.

![matmul throughput by library](docs/img/matmul.png)

**The main finding: timing a GPU call without synchronizing measures the
launch, not the work.** On PyTorch `matmul` 4096, a CPU timer without
`synchronize` reported 0.21 ms on the RTX 2060 and 0.07 ms on the T4,
against 21.0 and 31.2 ms with CUDA events: errors of about 100× and 425×.

![PyTorch matmul time by clock](docs/img/timers.png)

**An independent audit found a bug in the reference clock** (PyTorch's
CUDA events were created inside the timed region). It was fixed, both
machines were measured again, and the before and after are in the
write-up.

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

- **The RTX 2060 also drives the desktop display**, which used 858 and
  1,349 MiB of its memory in two readings, and some of its time. Results on
  it carry that load; the T4 does not.
- **Telemetry under WSL2 may be limited**: the NVML documentation lists GPU
  utilization as not supported there. To be tested in phase 2.
- **Colab does not guarantee which GPU a session gets**: every Colab result
  records the GPU it actually ran on.
- **Multi-GPU machines are not fully described**: the environment stamp
  records the first GPU that `nvidia-smi` lists and only counts the others.
- **Warm-up is a fixed number of calls (20)**, but the GPU needs *time* to
  raise its clock after being idle: after seconds of CPU-only work, 20 calls
  were not enough (see phase 1). A time-based warm-up is not implemented yet.
- **The RTX 2060's results are noisier** (median RSD on the reference clocks
  15.8% against 5.1% on the T4), probably from the display and WSL2; not
  isolated yet.
- **Which CUDA libraries ran is only partly known**: on the desktop,
  PyTorch is built for CUDA 13.0 and its pip packages are 13.0–13.1, while
  `nvcc` is 13.4; Colab runs its own builds (CuPy 14.0.1 there, against
  14.2.0 here); and which CUDA CuPy loads was not checked. The `cuda_toolkit`
  field records the compiler, not the libraries the benchmarks ran on.
- **The `matmul` tolerance (1e-3) is loose**: the largest error measured
  was 4.4e-6, so a reduced-precision matmul could still pass. PyTorch's
  float32 matmul precision is recorded on every row.
- **One counted run per machine**: run-to-run variation is only seen in two
  T4 sessions that differed by 12%, with a harness change between them.
- **Host-to-device copies come from ordinary (pageable) memory**; pinned
  memory is not measured yet.
- **Some results are not explained yet**, and are listed as such in the
  phase 1 write-up.

## License

Apache 2.0 — see [LICENSE](LICENSE).