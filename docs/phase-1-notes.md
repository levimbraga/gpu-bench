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

**What happened (2026-10-29 17:29 UTC, `results/20261029T172926Z_nvidia-geforce-rtx-2060.csv`, commit `e4357c2`):**

All 135 rows passed the correctness check; `git_commit` was `e4357c2`
on every row, without `+dirty`, as expected. The other expectations
failed, because the machine was busier than in Run 1: VS Code, Spotify and
several Firefox tabs were open.

- PyTorch: synchronized timer below the events in 8 of 20 pairs, not 5.
  CuPy, whose code did not change, went from 2 to 7.
- PyTorch `matmul` 256: 0.244 ms against CuPy's 0.178 ms, 37% apart, not
  within 30%. CuPy itself went from 0.093 to 0.178 ms.
- `matmul` 4096: 22.0 ms on CuPy (+5%) and 22.9 ms on PyTorch (+10%), not
  within 5%.
- CuPy `matmul` 2048: events still above the synchronized timer, as
  expected, but only 4% above (3.05 against 2.94 ms; Run 1: 43%).

**Not expected:** the whole run was slower and noisier. Median time
against Run 1, on the reference clocks: NumPy 1.16×, CuPy 1.09×, PyTorch
1.01×. NumPy `matmul` 4096 took 1,016 ms against 787 ms, with no change to
its code. The median RSD rose from 21% to 37%, with 51 rows above 50%
(Run 1: 22). This run cannot show the effect of the fix: the noise is
larger than the effect.

### Run 4 — Colab T4, same command, after the fixes

**Expectation (written before running):**

- PyTorch: the synchronized timer reads below the events in at most 5 of
  20 pairs, down from 11.
- PyTorch `matmul` 1024: events and synchronized timer within 10% of each
  other (Run 2: 0.866 against 0.537 ms).
- `matmul` 4096 unchanged within 5%: about 35 ms, 3.9 TFLOP/s.
- RSD as low as in Run 2: a median around 8%.

**What happened (2026-10-29 17:30 UTC, `results/20261029T173230Z_tesla-t4.csv`, commit `a724354`):**

All 135 rows passed the correctness check.

- PyTorch: synchronized timer below the events in 11 of 20 pairs, not 5:
  the same count as in Run 2. Expectation failed. But the gaps are a few
  microseconds: the median of events minus synchronized timer on PyTorch
  fell from 2.5 µs (Run 2) to 0.8 µs. At sizes where the GPU finishes
  before Python has queued the end marker, both clocks mostly measure
  CPU-side work, and which reads higher is not a law. The count was a
  poor test.
- PyTorch `matmul` 1024: events 0.529 ms against 0.528 ms on the
  synchronized timer, within 10% as expected (Run 2: 0.866 against
  0.537 ms). The gap the bug was suspected of causing is gone.
- `matmul` 4096: 30.7 ms on CuPy and 31.2 ms on PyTorch, about 12% faster
  than Run 2 (34.7 and 35.0 ms), not within 5%. Both libraries moved the
  same way, CuPy included, whose code did not change: this Colab session's
  T4 was faster. 4.5 TFLOP/s against 3.9.
- RSD median 6.7%, as low as in Run 2 (7.9%), as expected.

**Not expected:**

- Two Colab sessions of the same T4 model differed by 12% on the same
  operation, with the same code on CuPy. One run per machine is not
  enough to state a T4 number to better than about that.
- CuPy `matmul` 2048 read 6.07 ms on events against 3.80 ms on the
  synchronized timer, after about 7.5 s of NumPy on the same size: the
  warm-up gap seen on the RTX 2060 in Run 1, now on the T4.

### Run 5 — RTX 2060, all operations, nothing else open

**Expectation (written before running):** the same command as Run 3, with
VS Code, Spotify and Firefox closed, from a plain terminal.

- NumPy back to Run 1 within 5%: `matmul` 4096 at about 787 ms. This is
  the check that the extra load is gone; if it fails, the run says
  nothing about the fix either.
- RSD median at or below Run 1's 21%.
- `matmul` 4096 at about 20.9 ms on both GPU libraries, within 5%.
- PyTorch `matmul` 256 events within 30% of CuPy's.
- CuPy `matmul` 2048 events still above the synchronized timer: the
  warm-up is still counted in calls.

**What happened (2026-10-29 17:50 UTC, `results/20261029T175052Z_nvidia-geforce-rtx-2060.csv`, commit `cdb7ab0`):**

All 135 rows passed the correctness check, and every expectation held.

- NumPy `matmul` 4096: 796 ms against 787 ms in Run 1, within 5%: the
  control passed, so the run counts.
- RSD median 16.7%, below Run 1's 21%; 10 rows above 50% (Run 1: 22).
- `matmul` 4096: 20.8 ms on CuPy and 21.0 ms on PyTorch, within 5% of
  Run 1's 20.9 ms.
- PyTorch `matmul` 256: 0.079 ms on events against CuPy's 0.096 ms, within
  30% (Run 1: 0.214 against 0.093 ms).
- CuPy `matmul` 2048: events 4.11 ms against 2.99 ms on the synchronized
  timer, still above, as expected.

**Not expected:** NumPy `matmul` at 512 and 1024 reached 170 and
152 GFLOP/s, with RSD 21% and 8%: the drop to 33–82 GFLOP/s seen in Run 1
did not reappear on a quiet machine.

Runs 5 (RTX 2060) and 4 (T4) are the ones the write-up and the README use.

## Corrections to earlier entries

Entries above are kept as written; these correct them.

- Check 2's paragraph was rewritten after Run 2 (`eaab5c3`), to drop a
  claim about the card's theoretical peak. Checks 1 and 2 were short runs
  whose CSVs were not saved: their numbers cannot be traced to a file.
- Run 1: PyTorch `matmul` 256's RSD ranged 30–303%, not 35–303%.
- Run 1: the `saxpy` 2¹⁶ GPU time of ~0.07 ms was CuPy's; PyTorch's was
  0.164 ms.
- Run 1: "close at the smallest sizes" held for CuPy (0.062 against
  0.068 ms); PyTorch's unsynchronized timer read 3.4× below its events at
  `matmul` 256.
- Run 2's expectation gave the RTX 2060's power limit as 170 W, from
  memory and not checked.
- Run 2: "almost every RSD under 10%" on the T4 was wrong: 74 of 135 rows,
  with a median of 7.9%. The RTX 2060's rows above 50% were 22 of 135.
- Run 2: the synchronized timer closer to the events on the T4 held for
  CuPy only; on the RTX 2060, PyTorch read 0.616 against 0.642 ms.
- Run 1: the entry first said "All 150 rows" and was changed to 135 in
  `eaab5c3`, without a note; 135 is the count in the CSV.
- Run 2's expectation: the RTX 2060's power is 160 W, not 170 W
  ([HotHardware](https://hothardware.com/reviews/nvidia-geforce-rtx-2060-review)).
- Runs 3 and 5: the times in their headers (17:29 and 17:50) are when the
  CSV was written; the runs started at 17:27:58 and 17:49:38, as the stamp
  records. The other headers use the start time.
- Run 4: "0.528 ms" for the synchronized timer on PyTorch `matmul` 1024 is
  0.527 ms (0.527478).
- Run 5: the count of pairs with the synchronized timer below the events
  was not given: 11 of 20 on PyTorch and 6 on CuPy, so the count did not
  fall after the fix on either machine. The median of events minus
  synchronized timer on PyTorch did: from 4.1 to 0.4 µs (RTX 2060) and from
  2.5 to 0.8 µs (T4).
- Run 5: "did not reappear on a quiet machine" implies Run 1's machine was
  not quiet, which was not recorded. Run 1's NumPy dip did not reappear in
  Run 5; why, is not known.
- Runs 3 and 5: the two NumPy controls used different measures. On the
  same one, `matmul` 4096, NumPy took 29% longer than Run 1 in Run 3 and
  1% longer in Run 5.
- Run 4: "with the same CuPy code" was not quite true: the reference moved
  before the timing between Runs 2 and 4 (`fe568ad`), so the 12% between
  them mixes the session and that change.