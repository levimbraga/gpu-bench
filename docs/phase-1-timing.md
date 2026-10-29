# Phase 1 — Timing the GPU correctly

This phase answers one question: **how fast is the same operation on the
CPU and on the GPU, when it is timed correctly?** The hard part is the word
*correctly*. A GPU call in Python returns before the GPU has done the work,
so a stopwatch around it can report a time that has nothing to do with the
work. The harness in [`bench/`](../bench/) times every GPU call three ways,
including the wrong one, so the error is measured instead of assumed.

The working log, with every expectation written before its run, is in
[`phase-1-notes.md`](phase-1-notes.md).

## What was measured

| Operation | What it does | Sizes | Metric |
|---|---|---|---|
| `matmul` | `C = A @ B`, square float32 matrices | n = 256 … 4096 | GFLOP/s (2n³ operations) |
| `saxpy` | `y = a*x + y`, float32 vectors | 2¹⁶ … 2²⁴ elements | effective GB/s (12 bytes per element) |
| `sum` | sum of a float32 vector | 2¹⁶ … 2²⁴ elements | effective GB/s (4 bytes per element) |
| `h2d` | copy a vector from CPU memory to the GPU | 2¹⁶ … 2²⁴ elements | effective GB/s (4 bytes per element) |

Each with NumPy on the CPU, and CuPy and PyTorch on the GPU, on two
machines: the desktop's RTX 2060 (under WSL2) and a Tesla T4 on Google
Colab. Both GPUs are Turing, so the comparison is a consumer card against a
data-center card of the same architecture.

## How the harness times a call

- **Three clocks for every GPU call:**
  - **CUDA events**, the reference: two markers placed in the GPU's own
    queue, around the call; the GPU records the time when it reaches each
    one.
  - **A CPU timer with synchronization** before starting and before
    stopping the clock: correct, but it also counts the cost of
    synchronizing.
  - **A CPU timer without synchronization**, kept on purpose as the wrong
    way: it shows what a naive benchmark would report.
- **20 warm-up calls** are discarded before timing, then **30 repetitions**
  are timed, and the **median** is reported, with the relative standard
  deviation (RSD) beside it.
- **Every result is checked** against a float64 reference computed with
  NumPy, with a tolerance per operation. A fast wrong answer is not a
  result.
- **Every row of the CSV carries the environment stamp** from phase 0, plus
  the settings that change the numbers: the BLAS NumPy uses, the thread
  counts, and PyTorch's float32 matmul precision.
- **Tested:** [`tests/test_harness.py`](../tests/test_harness.py) uses a
  fake GPU backend to check the statistics, the correctness check and the
  CSV, without a GPU.

## Results

All 270 rows (135 per machine) passed the correctness check. Medians of 30
repetitions after 20 warm-up calls; GPU times from CUDA events.

**matmul, 4096 × 4096**

| | RTX 2060 | Tesla T4 |
|---|---:|---:|
| NumPy (CPU) | 787 ms · 175 GFLOP/s | 1,164 ms · 118 GFLOP/s |
| CuPy | 20.9 ms · 6,563 GFLOP/s | 34.7 ms · 3,962 GFLOP/s |
| PyTorch | 20.9 ms · 6,589 GFLOP/s | 35.0 ms · 3,924 GFLOP/s |
| GPU vs CPU | **38×** | **34×** |

**saxpy, 2²⁴ elements**

| | RTX 2060 | Tesla T4 |
|---|---:|---:|
| NumPy (CPU) | 26.3 ms · 7.7 GB/s | 43.2 ms · 4.7 GB/s |
| CuPy | 1.31 ms · 154 GB/s | 1.38 ms · 146 GB/s |
| PyTorch | 1.23 ms · 164 GB/s | 1.42 ms · 142 GB/s |
| GPU vs CPU | **21×** | **31×** |

**sum, 2²⁴ elements**

| | RTX 2060 | Tesla T4 |
|---|---:|---:|
| NumPy (CPU) | 4.57 ms · 15 GB/s | 5.60 ms · 12 GB/s |
| CuPy | 0.29 ms · 233 GB/s | 0.27 ms · 249 GB/s |
| PyTorch | 0.37 ms · 182 GB/s | 0.26 ms · 254 GB/s |

**h2d, 2²⁴ elements (64 MiB)**

| | RTX 2060 | Tesla T4 |
|---|---:|---:|
| CuPy | 16.3 ms · 4.1 GB/s | 18.2 ms · 3.7 GB/s |
| PyTorch | 10.9 ms · 6.2 GB/s | 14.2 ms · 4.7 GB/s |

![matmul throughput by library](img/matmul.png)
![saxpy throughput by library](img/saxpy.png)
![sum throughput by library](img/sum.png)
![h2d throughput by library](img/h2d.png)

The CPU-vs-GPU ratios compare each GPU with the CPU **of the same
machine**; the two CPUs are very different chips (phase 0), so the ratios
are not comparable across machines.

## The main finding: the unsynchronized timer measures the launch

![PyTorch matmul time by clock](img/timers.png)

Without synchronization, the CPU timer stops when the call has been
*queued*, not when it has *finished*. On PyTorch `matmul` 4096:

| | CUDA events | CPU timer, no synchronize | Implied throughput | Error |
|---|---:|---:|---:|---:|
| RTX 2060 | 20.9 ms | 0.22 ms | 621 TFLOP/s | ~94× |
| Tesla T4 | 35.0 ms | 0.06 ms | 2,203 TFLOP/s | ~560× |

Both "results" are impossible: they are hundreds of times what either card
measured with events (6.6 and 3.9 TFLOP/s). At the smallest sizes, where
launching is most of the cost, the wrong timer and the events read close to
each other — which is exactly what makes the mistake easy to miss when
testing with small inputs.

## Decisions

- **CUDA events as the reference clock**, because they are recorded by the
  GPU itself; the synchronized CPU timer is kept as a cross-check.
- **The wrong clock stays in the harness and in every CSV**, as a control:
  the error is part of the result.
- **Median, not mean**, with the RSD beside it: one slow repetition (the
  display, the OS) moves the mean and barely moves the median.
- **Warm-up raised from 2 to 20 calls**, after a measurement (below).
- **One set of inputs per operation and size, shared by every library**, so
  the comparison runs on the same data.
- **Sizes overridden only for one operation at a time**: 4096 is a
  4096 × 4096 matmul but a 4096-element vector.
- **Colab through a small script**, [`scripts/colab_run.py`](../scripts/colab_run.py):
  `colab run` uploads one file, so the script clones this repository on the
  VM, runs the harness, and prints the CSV for saving locally.
- **Charts as small multiples**, one panel per GPU on a shared axis, with
  one fixed color per library on every chart.

## Found along the way

- **Warm-up is a matter of time, not of calls.** With 2 warm-up calls,
  CUDA events (the first clock to run) read 27% slower than the
  synchronized timer on a 1024 matmul on the RTX 2060; with 20, 2%. The GPU
  was still raising its clock. In the full run, CuPy `matmul` 2048 came
  right after about 5 s of NumPy work with the GPU idle, and 20 calls of
  ~3 ms (60 ms) were not enough: the events read 4.12 ms against 2.88 ms on
  the synchronized timer.
- **The RTX 2060 ran above its rated clock:** 6.59 TFLOP/s measured,
  against 6.45 TFLOP/s computed from its rated boost clock
  (1,920 cores × 1.68 GHz × 2). A rated number is not a ceiling.
- **The T4 was far more stable:** almost every RSD under 10%, where the
  RTX 2060 went above 50% in many rows. The 2060 also drives the display
  and runs under WSL2; the T4 has neither.
- **Synchronizing cost more on the desktop:** on CuPy `matmul` 1024, the
  synchronized timer read 0.923 ms against 0.735 ms with events on the
  RTX 2060, and 0.877 against 0.880 ms on the T4. This supports WSL2 as the
  cause, but does not prove it: the GPU is different too.
- **The fixed cost of a GPU call was about half on the T4:** `saxpy` 2¹⁶
  took ~0.034 ms there against ~0.07 ms on the RTX 2060, enough for the T4
  to tie with NumPy at that size, where the 2060 lost to it.
- **Memory-bound operations reached about half the bandwidth on paper:**
  164 GB/s on `saxpy` against 336 GB/s for the RTX 2060's memory, since
  `a*x + y` runs as two kernels with a temporary array in between.

## Not explained yet

- **NumPy `matmul` dropped to 33–82 GFLOP/s at 512 and 1024** on the
  desktop (RSD ~40%), against ~170 at 256, 2048 and 4096.
- **On PyTorch `matmul` 1024 on the T4, the synchronized timer (0.537 ms)
  read below the events (0.866 ms).**
- **`h2d` without synchronization:** on PyTorch, the wrong timer read about
  the same as the events (10.8 against 10.9 ms on the RTX 2060), so the
  copy returned only when done; on CuPy it read far less (6.3 against
  16.3 ms).
- **Host-to-device copies reached only 4–6 GB/s.** The copies come from
  ordinary (pageable) CPU memory; pinned memory is not measured yet.

## Reproducing

```bash
uv sync --extra gpu
uv run python -m bench run                     # writes results/<UTC>_<gpu>.csv
uv run python -m bench table results/<file>.csv
uv run python -m bench plot results/*.csv      # writes docs/img/*.png

# Colab T4, from the local machine:
colab run --gpu T4 --timeout 1800 scripts/colab_run.py > colab-output.txt
```

## Next: phase 2

Telemetry: what the GPU is doing while it runs — clock, power, temperature
and utilization — sampled during the same benchmarks, to explain numbers
like the warm-up gap above instead of inferring them.