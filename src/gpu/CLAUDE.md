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
  120,000 returns into 745,000 cells: sorted p50 5.9 / p99 6.1 ms; atomic p50
  15.3 / p99 24.2 ms. 16× headroom at 10 Hz.
- **Scratch is now the largest line item in the budget** — 15.0 MB at the
  150,000-point cap, against 8.94 MB of grid, for a preallocated total of
  29.66 MB. That is the honest price of a zero-allocation scatter: the same
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
- **`np.take` allocates unless you ask it not to.** With int32 indices and the
  default `mode="raise"` it copies the index array twice — once to widen it to
  intp, once to bounds-check it. Measured at 200,000 elements: 3.2 MB a frame
  int32/raise, 1.6 MB intp/raise, **1 KB intp/clip**. Every gather on a frame
  path uses `intp` indices and `mode="clip"`, and the index buffers are built
  in range so the check cannot fire. This alone took scatter's p99 from 8.9 to
  6.1 ms and visibility's from 17.5 to 11.4 ms. It is worth 0.6 MB of extra
  declared scratch to not narrow `order` to int32.
- **Visibility cleanup is eq (32) plus the guard, and it produces a MASK.**
  Log-odds and the three-state decision are fusion (§10.1, Aakash). At 200,000
  candidate cells: p50 8.2 / p99 11.4 ms, 0.07 MB transient, 13.6 MB scratch.
- **⚑ The visibility scratch is NOT in `allocate()` yet, on purpose.** Sizing it
  means choosing a cap on candidate cells per frame, which moves the headline
  total — the same kind of decision as the transient-layer line, so it goes to
  the room rather than into my directory quietly. `visibility_scratch_bytes()`
  computes it for whatever cap gets chosen.
- **⚑ delta for eq (32): the math doc and the config disagree.** §10.4 says
  `delta = 3*sigma(r)` precisely so the band widens with range; the config says
  a flat 0.30 m, which is 26.9σ at 5 m and 1.7σ at 100 m — so it barely clears
  in the near field where ghosts matter most, and it clears real structure at
  range, which is what the guard exists to prevent. Implemented as 3σ(r) with
  the config value as a floor for pose error. No threshold was changed; one was
  given a new job. Raise it at a gate review.
- **Height sums are GROUND returns only, and `w_sum == 0` is a real state.**
  §3 estimates the elevation of the ground, so a canopy return is not weak
  evidence about it -- it is evidence about something else. Both scatter paths
  zero the weight where `is_ground` is false; every other column (count,
  reflectivity, class, ceiling) stays over all returns. The consequence is the
  part to remember: a wall, a car flank or a tree trunk now yields `w_sum = 0`,
  `mean_height_cm()` returns 0 there, and **0 is not a neutral height** -- the
  road sits near -173 cm in the vehicle frame, so writing it stands every wall
  1.7 m above the road at `meas_var` 1024 cm^2, confident enough to hold
  against the next few real returns. `has_ground_evidence()` is the predicate;
  `fuse()` leaves those cells' height and variance alone and ages them. The
  mask is `np.multiply(w, g, out=w)` and not `w[~g] = 0` because `~g` is a
  per-frame temporary: measured, the mask adds **0 B** per frame.
- **`mean_height_cm()` rounded every NEGATIVE mean 1 cm low**, and the ground
  plane is almost entirely negative. `(2*wz + sign(wz)*w) // (2*w)` is right
  for positive sums and one short for negative ones -- `//` floors, so after
  the away-from-zero nudge a negative quotient takes an extra step down. 540
  of 600 negative probes were wrong; only the exact half-centimetres agreed,
  and all 203 positive probes were fine, which is why it read as correct. A
  systematic 1 cm sag over the whole ground plane against a §3.2 noise floor
  of 0.8 cm at 5 m, and per-ring RMSE is the only place it would ever have
  shown. Round on the magnitude and put the sign back.
- **`EMPTY_CELL` is the one definition of a never-observed cell**, used by
  `allocate()` and by the strip `shift()` clears. Only `ceiling_height` differs
  from zero, and it is not a small difference: 0 cm reads as solid ground at
  the datum, so `ceiling - ground < h_vehicle` holds everywhere, TRAV_CLEARANCE
  marks the whole world untraversable, and it never recovers because `fuse()`
  only ever lowers a ceiling. Fixing it in `allocate()` alone gives a map that
  is correct exactly until the vehicle moves, which is why it is a shared dict
  and not two literals. The refinement pool needs it too -- a block handed out
  by `acquire()` is brand-new cells, so missing it there makes precisely the
  cells we chose to look at more closely go untraversable.
- **Residency is `mincore(2)` on the array, not a process-RSS delta.** The
  delta is the right instrument for "what did this cost the machine" -- it is
  the dashboard counter and it stays -- and the wrong one for "are these pages
  in core". glibc raises its mmap threshold once it has seen a large block
  freed, so a later allocation of about that size comes off the heap and
  reuses pages the process already holds: measured, a correctly committed
  64 MB baseline reported a 42 MB delta after the allocator tests had run and
  the full 64 MB when it ran alone. `resident_fraction()` asks the pages
  themselves and returns None off POSIX so callers can fall back rather than
  silently report 0.
- **The conservative pyramid is OFF by default** (`allocate(with_pyramid=True)`).
  It is a stretch item and it takes the preallocated total from 29.06 MB to
  32.17 MB — a number already on a slide does not get to move because a
  default did. 2.73 MB of nodes plus 0.38 MB of shared reduction scratch;
  rebuild p50 2.45 / p99 3.10 ms over 910,000 slots, 32× headroom, zero
  allocation per frame. `scripts/bench_pyramid.py` produces all of it.
- **⚑ §7.2's 1.24 MB is low by about half, for two compounding reasons.** A
  node stores the REDUCTIONS, not the source fields: ground contributes both
  `H_max` and `H_min`, so it is 4 bytes not 2, and `n_min` adds a fifth —
  8 B/node by §7.2's own list, 9 with `OR_mask`. And `N` is the ring WINDOWS,
  910,000 allocated slots, not 745,000 logical cells. Corrected: 2.73 MB, and
  the `N/3` claim itself is exactly right (measured 3.00). Ratify at a gate
  review before it reaches a slide.
- **Three pyramid predicates, and only one of them means "drivable".**
  `theorem3_safe()` is §7.3 verbatim and covers bits 0, 2 and 5 — three of
  six. A uniformly steep bank is SAFE by Theorem 3 and fails bit 1 in every
  cell, so a planner reading it as drivable drives onto the bank. `all_clear()`
  (`OR_mask == 0`) is the one that answers the question, and it is what
  `classify()` returns as `QueryLOD.SAFE`. `certainly_blocked()`
  (`AND_mask != 0`) needs a COMMON reason: a block where every cell is blocked
  differently comes back MIXED, which costs a descent and never safety.
- **Pyramid levels halve by CEILING.** Ring windows are 400 and 500 across,
  neither a power of two, and floor-halving 500 gives 250, 125, **62** —
  dropping a 125-wide level's last row and column silently, at the map edge.
  Edge blocks are therefore 1 cell wide, which is why the reduction is two
  pairwise passes and not `reshape(h, 2, h, 2)`: the reshape spelling needs an
  even side, so it would need `np.pad` per field per level per ring per frame.
- **The theorem test is mutation-checked.** Six deliberate breakages — each
  min written as a max, and/or swapped, floor-halving, the odd leftover column
  dropped — and all six fail the suite. §7.3's test passes trivially on an
  implementation where SAFE is never true, and on a uniform-random map SAFE
  fires ~150 times in 120 maps, so the test asserts a floor on how often it
  saw its own antecedent and generates terrain-shaped maps to get there.
- **`spherical_project` ran the azimuth axis backwards for three days.** JP's
  `perception/range_image.py` is authoritative -- column 0 is azimuth -pi and u
  INCREASES with `atan2(y, x)` -- and this kernel gathers out of his image, so
  a mirrored axis meant eq (32) compared a cell in front of the vehicle against
  the beam behind it. `u_here + u_JP == W - 1` for every point, exactly. Every
  other test in `test_visibility.py` builds its own image and is therefore
  blind to the convention; `test_columns_match_jp_projection` is the only one
  that can see it, and it was written after the fact. Two projections in one
  system is what `docs/frames.md` exists to prevent, and the second one was
  mine.
- **A cell's visibility height is `ceiling_height`, not `ground_height`.** A
  cell whose returns are all non-ground has `w_sum == 0` and reads
  `ground_height == 0`, and 0 cm is the datum, not "no information". Projecting
  a parked car at 0 aims the ray 1.73 m below the sensor and lands it on a
  different image row. Measured on the Gate 3 scene: all 379 car cells read
  ground 0 while the ceiling carried the real 34 cm, and not one ghost cleared.
  `ceiling_height` is the lowest thing overhead, which is the surface that
  actually stops the beam.
- **⚑ `occupancy_state()` allocates 8.19 MB per call** -- full-grid temporaries
  over 910,000 slots -- and the frame loop needs it every frame to find the
  cleanup's candidates. That is `src/grid/fusion.py`, not mine, and it is the
  largest single per-frame allocation in the system, ahead of `ring_of`'s 6.96
  MB. Both are in the Gate 3 review.
- **No OptiX / RT cores.** Unsupported on Jetson; visibility cleanup is already
  O(1) per cell by range-image comparison. Future-work line only.

Day 0–1 your work blocks both other devs, so the allocator and timing harness
are non-negotiable and come first. From Day 2 it reverses and you are
optimising what they built — which is also why you are the right person to
pull onto split/merge if Aakash is behind at the Day 2 gate.
