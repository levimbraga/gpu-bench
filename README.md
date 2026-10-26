# gpu-bench

GPU performance benchmarks in Python and CUDA C: NumPy, CuPy, PyTorch and
cuBLAS, measured on Linux on an RTX 2060 and cloud GPUs.

Every result is stamped with the environment that produced it — GPU, driver,
CUDA version, CPU, Python, OS — so any number here can be traced and reproduced.

## Results

No results yet. The first numbers arrive with phase 1, the timing harness.

## Roadmap

| Phase | Question it answers | Status |
|---|---|---|
| 0 | Where do the numbers come from? (environments, repository) | [done](docs/phase-0-environments.md) |
| 1 | How fast is the same operation on CPU and GPU, timed correctly? | not yet |
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

## License

Apache 2.0 — see [LICENSE](LICENSE).