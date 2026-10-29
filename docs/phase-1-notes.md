# Phase 1 — notes

Working log for phase 1. Each run gets its expectation written **before**
it starts, then what happened. The phase write-up is built from this file.

## Rules found so far

- A GPU call returns before the GPU finishes: Python only queues the work.
- Synchronize **before** starting the clock (so earlier queued work is not
  counted) and **before** stopping it (so this call is counted in full).
- CUDA events are markers in the GPU's own queue: the GPU records the time
  when it reaches each one, without stopping the queue.
- On the RTX 2060, 2 warm-up calls were not enough: CUDA events (always the
  first clock) read 27% slower than the synchronized timer on a 1024 matmul.
  With 20 warm-up calls the gap fell to 2%. The GPU was still raising its
  clock. Default warm-up: 20.
- The GPU slows down while idle and needs time, not a number of calls, to
  speed up again: a warm-up of 20 calls is enough after another GPU run,
  but not after seconds of CPU-only work.

## Checks before the first run

### Check 1 — RTX 2060, PyTorch, matmul 1024, 5 repetitions

**Expectation:** the unsynchronized timer far below the other two; CUDA
events and the synchronized timer close to each other.

**What happened (2026-10-29 UTC):**

| warm-up | cuda_events | synced_cpu_timer | unsynced_cpu_timer |
|---:|---:|---:|---:|
| 2 | 0.7086 ms | 0.5598 ms | 0.0770 ms |
| 20 | 0.6209 ms | 0.6082 ms | 0.1461 ms |

All six results passed the correctness check against the float64
reference. The unsynchronized timer measured the launch, not the work. The
gap between events and the synchronized timer came from a warm-up too
short for the GPU to reach its clock.

### Check 2 — RTX 2060, all backends, matmul 256 and 1024, 5 repetitions

**What happened (2026-10-29 UTC):** at 1024, CUDA events measured 0.56 ms
(CuPy) and 0.58 ms (PyTorch) against 13.61 ms for NumPy, about 24×. The
unsynchronized timer reported ~11,500 GFLOP/s, almost twice the most the
card ever measured with events (6.6 TFLOP/s, Run 1), which proves it
measured the launch. The synchronized timer read
~0.17 ms above the events at 1024, likely the cost of the synchronize calls
themselves. At 256, the spread was 13–56% with only 5 repetitions.

## Runs

### Run 1 — RTX 2060, all operations, default sizes

**Expectation (written before running):**

- matmul, smallest size (256): CuPy and PyTorch about tied and around 2×
  faster than NumPy, because both call cuBLAS and 33 MFLOP is already more
  work than the fixed cost of a kernel launch, but too little to fill the GPU
  (check 2 showed ~0.1 ms against ~0.21 ms).
- matmul, largest size (4096): CuPy and PyTorch tied again (same cuBLAS
  underneath), and 30–60× faster than NumPy, because the work grows as n³
  and only the GPU has the throughput to absorb it: a few TFLOP/s against
  ~150 GFLOP/s on the CPU.
- saxpy, smallest size (2^16): NumPy fastest, because 768 KB fits in the
  CPU's cache and the GPU spends most of its time on launch overhead, not on
  the 65 thousand multiply-adds.
- saxpy, largest size (2^24): the GPU backends fastest, around 10× ahead of
  NumPy, because SAXPY is limited by memory, not math: the RTX 2060's memory
  (~336 GB/s on paper) is roughly 10× the CPU's DDR4-2400 (~19–38 GB/s,
  single or dual channel). Both GPU libraries will fall short of 336 GB/s,
  since `a*x + y` runs as two kernels with a temporary array in between.
- the `unsynced_cpu_timer` row compared with `cuda_events`: close at the
  smallest sizes, where the launch is most of the cost; far apart at the
  largest, where the unsynchronized timer stays near the launch time while
  the events grow with the work.

**What happened (2026-10-29 13:49 UTC, `results/20261029T135052Z_nvidia-geforce-rtx-2060.csv`):**

All 135 rows passed the correctness check. Medians of 30 repetitions after
20 warm-up calls; GPU times from CUDA events.

- matmul 256: CuPy 0.093 ms against NumPy 0.198 ms, about 2.1×, as
  expected. PyTorch was too noisy to rank (RSD 35–303%).
- matmul 4096: 20.9 ms on both GPU libraries against 787 ms on NumPy, about
  38×, inside the 30–60× expected. CuPy and PyTorch tied, as expected.
- saxpy 2^16: NumPy fastest (0.023 ms against ~0.07 ms), as expected.
- saxpy 2^24: the GPU was 21× faster (1.23 ms against 26.3 ms), not 10×.
  The GPU reached 164 GB/s, about half the 336 GB/s on paper, as expected;
  NumPy reached only 7.7 GB/s, well below the expected 19–38 GB/s.
- unsynced timer: close to the events at the smallest sizes (0.062 against
  0.068 ms on saxpy 2^16) and far apart at the largest (0.22 against
  20.9 ms on matmul 4096), as expected. At matmul 4096 it implied
  621 TFLOP/s against the 6.6 TFLOP/s measured with events: an error of
  about 94×.

**Not expected:**

- CUDA events measured 6.59 TFLOP/s on matmul 4096, above the 6.45 TFLOP/s
  computed from the card's rated boost clock (1,920 cores × 1.68 GHz × 2).
  The card ran above its rated clock, so that number is not a ceiling.
- On CuPy matmul 2048, CUDA events (4.12 ms) read above the synchronized
  timer (2.88 ms). NumPy had just spent about 5 s on the same size, with the
  GPU idle. 20 warm-up calls of ~3 ms are 60 ms: not enough for the GPU to
  raise its clock again. Warm-up should be measured in time, not in calls.
- NumPy matmul dropped to 33–82 GFLOP/s at 512 and 1024 (RSD ~40%), against
  ~170 at 256, 2048 and 4096. Cause not known yet.

### Run 2 — Colab T4, same command

**Expectation (written before running):**

- matmul 4096: the T4 slower than the RTX 2060, around 4–6 TFLOP/s against
  6.6, even though it has more cores on paper (2,560 against 1,920),
  because the T4 is capped at 70 W against the 2060's 170 W and has to keep
  its clock low under sustained load.
- saxpy 2^24: about the same bandwidth as the RTX 2060 (~150–170 GB/s),
  because the two cards have similar memory bandwidth on paper (320 against
  336 GB/s), and SAXPY is limited by memory.
- NumPy: slower than on the desktop, because a free Colab VM gives only a
  couple of shared CPU threads.
- the synchronized timer closer to CUDA events than on the desktop, if the
  extra ~0.17 ms seen at matmul 1024 came from synchronizing through WSL2;
  Colab runs native Linux.

**What happened (2026-10-29 14:16 UTC, `results/20261029T141820Z_tesla-t4.csv`):**

All 135 rows passed the correctness check. Medians of 30 repetitions after
20 warm-up calls; GPU times from CUDA events.

- matmul 4096: 34.7 ms on CuPy and 35.0 ms on PyTorch, 3.9 TFLOP/s, against
  6.6 TFLOP/s on the RTX 2060: slower, as expected, and a little below the
  4–6 expected.
- saxpy 2^24: 146 GB/s on CuPy and 142 GB/s on PyTorch, against 164 GB/s on
  the RTX 2060: about the same, as expected.
- NumPy: slower than on the desktop, as expected (matmul 4096 in 1,164 ms
  against 787 ms).
- the synchronized timer stayed close to CUDA events: 0.877 against
  0.880 ms on CuPy matmul 1024, where the desktop read 0.923 against
  0.735 ms. This supports the WSL2 explanation, but does not prove it: the
  GPU is different too.

**Not expected:**

- The T4 was far more stable: almost every RSD under 10%, where the RTX 2060
  went above 50% in many rows. The 2060 shares the card with the display
  and runs under WSL2; the T4 has neither.
- Smaller fixed cost per GPU call: saxpy 2^16 took ~0.034 ms on the T4
  against ~0.07 ms on the RTX 2060, enough for the GPU to tie with NumPy at
  that size.
- At matmul 4096, the unsynchronized timer implied 2,203 TFLOP/s against
  the 3.9 TFLOP/s measured with events: an error of about 560×.
- On PyTorch matmul 1024, the synchronized timer (0.537 ms) read below the
  events (0.866 ms). Cause not known yet.
## The audit

An independent review of the repository (2026-10-29) checked every number
in the docs against the CSVs and every claim against the code. It found,
among other things, a bug in the reference clock: PyTorch creates a CUDA
event lazily, on its first `record()`, so the end event was created inside
the timed region, and its creation time was counted whenever the GPU
finished first. The synchronized timer brackets the events plus two
synchronize calls, so it should never read below them; on PyTorch it did in
11 of 20 (operation, size) pairs on each machine, against 2 (RTX 2060) and
5 (T4) on CuPy, which creates its events up front.

Fixed in code before runs 3 and 4 (`d443e23` to `0bdc633`): one PyTorch
event pair created outside the timed region; the float64 reference computed
once per size, before any backend runs; the git commit and the seed
recorded on every row; plus tests and the stamp's fail-open paths. Runs 1
and 2 stay in `results/` as the before.

### Run 3 — RTX 2060, all operations, default sizes, after the fixes

**Expectation (written before running):**

- PyTorch: the synchronized timer reads below the events in at most 5 of
  20 (operation, size) pairs, about CuPy's level in Run 1, down from 11.
  Not zero: on this card, noise alone can flip two close readings.
- PyTorch at the smallest sizes: events closer to CuPy's than in Run 1,
  for example `matmul` 256 within 30% of CuPy (Run 1: 0.214 against
  0.093 ms).
- The largest sizes unchanged within 5%: `matmul` 4096 at about 20.9 ms
  (6.6 TFLOP/s) on both GPU libraries, since an event created once per
  call is negligible next to 20 ms of work.
- CuPy `matmul` 2048 events still above the synchronized timer, as in
  Run 1: the fix removed the reference computation, but NumPy still runs
  about 5 s on the same size right before CuPy, with the GPU idle. Only a
  time-based warm-up would fix that, and it is not implemented.
- `git_commit` on every row equal to the commit that adds these
  expectations, without `+dirty`.

### Run 4 — Colab T4, same command, after the fixes

**Expectation (written before running):**

- PyTorch: the synchronized timer reads below the events in at most 5 of
  20 pairs, down from 11.
- PyTorch `matmul` 1024: events and synchronized timer within 10% of each
  other (Run 2: 0.866 against 0.537 ms).
- `matmul` 4096 unchanged within 5%: about 35 ms, 3.9 TFLOP/s.
- RSD as low as in Run 2: a median around 8%.