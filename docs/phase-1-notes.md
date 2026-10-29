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

## Runs

### Run 1 — RTX 2060, all operations, default sizes

**Expectation (written before running):**

- matmul, smallest size (256): fastest backend is ___, because ___
- matmul, largest size (4096): fastest backend is ___, because ___
- saxpy, smallest size (2^16): fastest backend is ___, because ___
- saxpy, largest size (2^24): fastest backend is ___, because ___
- the `unsynced_cpu_timer` row compared with `cuda_events`: ___

**What happened:**

### Run 2 — Colab T4, same command

**Expectation (written before running):**

- compared with the RTX 2060: ___

**What happened:**