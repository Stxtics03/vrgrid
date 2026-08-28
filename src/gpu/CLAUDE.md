# src/gpu — Shrestha

- **Fixed-point, never float atomics.** int32 accumulators in 1 cm units.
  Float atomic adds are non-associative: the map changes run to run and bugs
  move when you look at them. `make test-determinism` is CI-blocking (math §3.4).
- **Structure-of-arrays.** One array per field, for coalesced access. Never
  array-of-structs.
- **No allocation in the frame loop.** Grid arrays, refinement pool (512 × 16
  × 12 B = 98 KB), transient layer and tracked-object list are all preallocated
  at startup. One allocation in the loop makes the memory claim false.
- **`scatter_sorted` must be passed the scratch from `allocate()`.** Omitting it
  is legal and allocates a private one per call — fine in a test, wrong in the
  loop. It cost 19 MB a frame, more than twice the whole grid, behind a
  docstring that said the opposite; caught with a profiler, not by reading.
  Measured now: 1.8 MB against 19 MB, p50 6.0 ms against 10.0 ms.
- **The aggregate `scatter_sorted` returns is a set of VIEWS into that
  scratch**, valid only until the next scatter on the same scratch. `fuse()`
  consumes it inside the frame. Copy it if it must outlive the frame.
- **The packed sort key's bottom field is POSITION, not `point_id`.** They are
  equal in the default case, which is exactly why conflating them survives
  casual testing: `point_id` orders the points, `position` says where the
  payload columns live. `test_point_order_does_not_change_the_map` is the only
  thing that catches it, and the wrong version still produces a plausible map.
- **Timing is p50 and p99, not mean**, and percentiles are nearest-rank —
  numpy's default interpolation invents a latency no frame ever took and
  rounds the tail down. A 10 Hz claim is about the tail.
- **Scatter has two paths, and they must stay bit-identical.** `scatter_sorted`
  (default, scratch sized by points) and `scatter_atomic` (dense accumulator,
  the literal reading of master v4 §3.5). `tests/test_kernels.py` asserts they
  agree field-for-field; if they diverge, the optimisation is the bug. At
  120,000 returns into 745,000 cells: sorted p50 5.8 / p99 8.9 ms, 10.6 MB
  scratch; atomic p50 15.3 / p99 24.2 ms, 19.4 MB. 11.2× headroom at 10 Hz.
- **Scratch is now the largest line item in the budget** — 13.2 MB at the
  150,000-point cap, against 8.94 MB of grid, for a preallocated total of
  27.86 MB. That is the honest price of a zero-allocation scatter: the same
  memory was being spent per frame before, just undeclared and with the GC
  paying for it. `scatter.max_points_per_frame` in `configs/thresholds.yaml`
  is the knob — a real HDL-64E sweep is ~120,000, so 150,000 has headroom to
  spare. **The report's ratios are unaffected: they compare map memory, and
  scratch is working memory.** Do not let the two meet on a slide.
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
- **A baseline that does not fault its pages in is not a baseline.** `np.zeros`
  gets copy-on-write zero pages: `np.zeros(2_560_000_000, np.uint8)` moves RSS
  by **0.0 MB**. Built the obvious way, the dense-3D baseline would show 0 MB
  beside our counter and we would be claiming 286x over something visibly
  free, on stage. `baseline.commit()` touches one byte per page; 2.56 GB costs
  0.23 s once. `allocate()` commits ours for the same reason, plus a second
  one: pages faulted in at startup are pages not faulted in during frame 1,
  where the spike would land in the p99 the 10 Hz claim rests on.
- **Show RESIDENT next to CLAIMED, never claimed alone.** Claimed is the slide;
  resident is what the machine gave up. Measured: ours 27.86 claimed / 27.98
  resident, uniform 192.00 / 192.03, dense 2.56 GB / 2.56 GB.
- **Two ratios, and know which one is being asked for.** Against our *map*
  memory (8.94 MB): 21.5x uniform, 286x dense — these are the report's, and
  they are pure cell-count ratios. Against our *total preallocated* footprint
  (27.86 MB, scratch included): 6.9x and 91.9x. The dashboard counter shows
  the total, so be ready for the second pair; quoting the first while the
  screen shows the second is how a good claim gets called cherry-picking.
- **No OptiX / RT cores.** Unsupported on Jetson; visibility cleanup is already
  O(1) per cell by range-image comparison. Future-work line only.

Day 0–1 your work blocks both other devs, so the allocator and timing harness
are non-negotiable and come first. From Day 2 it reverses and you are
optimising what they built — which is also why you are the right person to
pull onto split/merge if Aakash is behind at the Day 2 gate.
