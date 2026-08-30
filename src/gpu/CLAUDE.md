# src/gpu — Shrestha

- **Fixed-point, never float atomics.** int32 accumulators in 1 cm units.
  Float atomic adds are non-associative: the map changes run to run and bugs
  move when you look at them. `make test-determinism` is CI-blocking (math §3.4).
- **Structure-of-arrays.** One array per field, for coalesced access. Never
  array-of-structs.
- **No allocation in the frame loop.** Grid arrays, refinement pool (512 × 16
  × 12 B = 98 KB), transient layer and tracked-object list are all preallocated
  at startup. One allocation in the loop makes the memory claim false.
- **Timing is p50 and p99, not mean**, and percentiles are nearest-rank —
  numpy's default interpolation invents a latency no frame ever took and
  rounds the tail down. A 10 Hz claim is about the tail.
- **Scatter has two paths, and they must stay bit-identical.** `scatter_sorted`
  (default, scratch sized by points) and `scatter_atomic` (dense accumulator,
  the literal reading of master v4 §3.5). `tests/test_kernels.py` asserts they
  agree field-for-field; if they diverge, the optimisation is the bug.
- **`annulus_index()` returns -1 for the hole.** Scatter drops those. A -1 used
  as an index piles the far field into cell 0 and still looks plausible.
- **Rings are stored as full squares, toroidally.** Absolute lattice cell
  (ix, iy) lives permanently at slot (iy mod W, ix mod W); a shift moves the
  origin and clears only the newly visible columns and rows. Annulus storage
  saves 1.98 MB but makes the shift a gather over the whole ring — measured
  15.2 ms against 0.04 ms, on a 100 ms budget. `RingBuffer.slot()` returns -1
  out of view: that slot belongs to a live cell now.
- **745,000 is the LOGICAL cell count** and the only number report ratios use.
  910,000 is what we allocate. Never conflate them on a slide.
- **No OptiX / RT cores.** Unsupported on Jetson; visibility cleanup is already
  O(1) per cell by range-image comparison. Future-work line only.

Day 0–1 your work blocks both other devs, so the allocator and timing harness
are non-negotiable and come first. From Day 2 it reverses and you are
optimising what they built — which is also why you are the right person to
pull onto split/merge if Aakash is behind at the Day 2 gate.
