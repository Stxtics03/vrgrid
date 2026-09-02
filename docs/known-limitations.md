# Known limitations

Three boundaries of the current system, stated precisely. All are understood,
bounded, and scoped deliberately for this submission — none is a surprise
waiting in the demo.

---

## 1. Ghost removal is inert above ~11.7 m vehicle world-z

**What breaks.** The visibility cleanup (math §10.4) — the stage that erases the
trailing "ghost" cells a moving object leaves behind — stops clearing cells once
the vehicle's own elevation in the world frame rises past about **11.7 m**.
Between roughly +2.4 m and +11.7 m it still runs but clears ~15× fewer cells per
frame; above +11.7 m it clears nothing.

**Why.** Heights enter the map on a **world-absolute** vertical band of
`[−2.0, +6.0] m` — `quantise_height` in `src/gpu/kernels.py` clamps every point
to that range, matching `vertical_extent_m` in the schedule config, which
scopes out overpasses and multi-storey structures by design. The cleanup is
then handed cell heights in that same world-absolute frame, without the
vehicle's own elevation subtracted. On a sequence that climbs, the near-field
cell heights saturate at the +6 m ceiling while the sensor sits tens of metres
higher, so every cleanup candidate projects outside the sensor's vertical field
of view and is skipped.

**Measured** (SemanticKITTI seq 08, which climbs +45.7 m over its loop):

| vehicle world-z | cells cleared / frame |
|---|---|
| −6.7 … +2.4 m | 15,580 |
| +2.4 … +11.6 m | 1,022 |
| +11.6 … +20.7 m | 35 |
| +20.7 … +39.0 m | 0 |

2,304 of seq 08's 4,071 frames (57 %) fall in the inert region. Seq 07 (flat,
world-z −5.8 … −1.0 m) is unaffected — zero inert frames. Seq 00 has two inert
stretches (frames 2071–3244) on its mid-sequence hill.

**Scope, not a defect.** The limitation is confined to the map's height layer
and the ghost-removal cleanup. The perception front-end, the point cloud, and
every dashboard `--color-by` mode are untouched — none of them pass through
`quantise_height`. The variable-resolution grid, its memory bound, the
foveation, and the per-ring accuracy claims all hold regardless of elevation.
The submission's mapping and memory results are reported on the portions of each
sequence where the assumption holds, and `docs/demo-safe-ranges.md` lists the
exact safe frame windows per sequence for the live demo.

**The fix** is a vehicle-relative vertical window (or subtracting ego-z in the
cleanup's cell-centre inverse) rather than a world-absolute clamp. It is a
change to the mapping-engine layer, not the perception front-end, and is
tracked for that team. It was found in our own soak testing this week, not by a
reviewer — the diagnosis, the measured degradation curve above, and the
demo-safe windows are all documented.

---

## 2. Plan-regret evaluation: synthetic complete, real-sequence in progress

**Status.** The offline plan-regret pipeline — reference map M* (§9.1), cost-map
construction, A\* planning, and the regret / Fréchet metrics against the
reference (§8) — is **fully implemented and unit-tested** (`src/eval/`,
`tests/test_plan_regret.py`, `tests/test_reference_map.py`,
`tests/test_metrics.py`). It has been **run end-to-end on the synthetic
sequence** and produces the memory-vs-regret sweep the script
`scripts/regret_plot.py` draws.

**What is not yet done.** The regret sweep has **not** been run on real
SemanticKITTI sequences. Two things gate that:

1. **The reference map M* has not been built for seq 07 / 08.** The builder
   (`reference_map.build`) reads through the perception loader, which only
   landed this week; building the real M* is the next step and had not been run
   at the time of writing.
2. **The synthetic planning query needs revising before it transfers.** On the
   synthetic terrain the path runs down a fixed lane that misses every hazard
   the terrain was built to contain, so the synthetic regret curve measures
   tie-breaking between equal-cost paths rather than the cost of coarsening.
   Posing the query properly — a start and goal that force a decision about a
   kerb or a pothole — is a change to the evaluation harness and is being made
   deliberately rather than tuned until the curve looks right.

**Scope.** The memory reduction and the per-ring geometric accuracy against M*
are the load-bearing quantitative claims and are independent of this. The
plan-regret result — "the compression does not change the plan a robot would
make" — is the strongest single claim in the project, and it is being produced
on real data with the same rigour as the rest, not rushed. Its status is stated
plainly here so a reviewer knows exactly what has run against what.

*(Verified current as of the latest `src/eval/` state — no reference-map or
regret artifacts exist in the repo, and `src/eval/` has had no new commits since
the class-byte re-split.)*

---

## 3. Per-ring RMSE (§9.2): a fixed confound, and an unverified sign

**What was wrong.** §9.2 eq. (26) does not say what `C_L` is, and the
implementation read it as every cell in ring L's buffer. That buffer is a
square of half-width `R_L`, so it covers the region the finer rings serve,
while only its annulus `[R_{L-1}, R_L)` is ever written — no cell moves between
rings; what moves is the vehicle, and with it which ring answers for a place.
The interior is never cleared (a toroidal shift clears only the edge coming
into view, §2.4) and never read (`query()` routes those places to a finer
ring), so it holds values written when that ground was far away. Those cells
were being scored against an `M*` that went on accumulating the close-range
returns they never received.

**Size.** 19% of ring 1 and 21% of ring 2 of the scored population after 22 m
driven; 20% and 26% after 46 m. It scales with distance driven, not frame
count. An independent measurement on longer runs reports 13–38% per ring at 40
frames and near half of ring 2 by 80 frames.

**Fixed.** `C_L` is now the cells ring L still *serves*, decided by `ring_of`
on the cell centre — the same function `query()` routes with, pinned against
`slot_of` rather than asserted. It lives in `metrics._ring_cells`, so all four
§9.2/§9.3 metrics inherit it.

**⚑ What is NOT settled: the direction, and no figure for it is quoted here.**
On the synthetic sequence the fix *lowers* RMSE (ring 1 0.40 → 0.37 cm, ring 2
0.37 → 0.32 cm) — there the stale value was written at grazing incidence on the
terrain's 6% ramp and is worse than the live annulus.

An external measurement against seq 07/08 was reported on 3 Sep and revised the
same day: first as a consistent **3–12% understatement**, then withdrawn for
**"no consistent bias, −40% to +21%"** depending on ring, sequence and frame
count. Neither is reproducible in this repo — there is no data root on the
machine and no `M*` artefact for either sequence (see limitation 2). The
earlier figure was briefly written into this file and has been removed rather
than replaced, because a withdrawn number quoted as if it stood is worse than
no number.

**So the direction on real data is unknown, and no per-ring RMSE figure should
be quoted as improved or worsened by this fix until it is re-measured on
07/08.** The mechanism, the population size, the schedule asymmetry and the fix
are settled; the sign is not.

**An inconsistent sign argues *for* the fix.** A bias with a known direction
could be corrected for in the write-up without touching the metric; one that
swings with where a sequence's rough terrain falls relative to the stale region
cannot be. The fix is already in and tested, so there is no timeline trade to
make here.

**Why it mattered more than its size.** The confound was asymmetric across the
schedules §8.2 compares. A uniform baseline has one ring, `ring_of` always
answers 0, and nothing can migrate out from under it — so the money plot
charged the foveated schedules for stale memory and the uniform grids for
none. Worst-ring RMSE before → after: 5/10/20/40 0.40 → 0.37, 5/10/50
0.46 → 0.37, uniform 10 cm 0.35 → 0.35, uniform 20 cm 0.41 → 0.41. Only our own
schedules move.

**ρ moves, but not enough to change what it says.** The coarsening ratio shifts
by up to 0.06 per ring on the synthetic sequence (ring 1 1.50 → 1.42, ring 2
1.37 → 1.29, ring 3 unchanged). The external measurement reports up to −12.5%,
unreproducible here for the reason above. On either reading ρ stays in the
"coarsening cost about what the terrain itself costs" band, so the foveation
finding survives in shape — but **it is not "unaffected", and no ρ figure
should be quoted to two decimals until it is re-measured on 07/08.**

⚑ **Attribution, so it is not repeated: there is no median ρ of 1.45 in this
project.** The only 1.450 on record is a **plan-regret** `R(S)` at 24 frames
(`scripts/regret_plot.py`; research log, 2 Sep) — and it sits in the paragraph
concluding that the money plot does not yet show what it was built to show,
because `PLAN_LANE_CELLS` runs the path off the hazards. ρ and `R(S)` are
different metrics and must not be reconciled against one another.

**⚑ A separate caveat on ρ's denominator.** `spread` is estimated from the 5 cm
reference cells of `F(c)` that `M*` observed. Median coverage on the 12-frame
synthetic sequence is 1.00 at ring 0, 0.25 at ring 1, 0.06 at ring 2 and 0.02
at ring 3 — ring 3's sub-cell variability comes from roughly one reference cell
in sixty-four. A spread estimated from two points is biased low and ρ divides
by it, so ρ on the coarse rings is biased **high**: the conservative direction
for a number we want near 1. Disclosed rather than corrected, and coverage is
now a column in the per-ring table so ρ cannot be read without it.

---

## What is not on this list

For the avoidance of doubt, the following are **settled**, not open questions:

- **FRNet is not used, on purpose.** Semantic and motion labels are ground truth
  from the SemanticKITTI `.label` files. This isolates the mapping contribution
  from segmentation error and is disclosed everywhere it matters. The one
  standalone FRNet port available does not reproduce the trained network; it is
  kept, flagged non-functional, for a possible future `mmdet3d` swap.
- **The KITTI reflectivity path is deliberate.** KITTI intensity is already
  firmware range/incidence-compensated, so the raw-power `·r²/cos` normalisation
  is not applied to it (it saturated 62 % of near-field road at the byte rail).
  The eq-(31) path is retained for sensors that need it.
- **Full-sequence robustness is verified.** Perception + mapping engine, every
  frame of seq 00 / 07 / 08: 0 crashes, 0 NaN/Inf across ~3.2 × 10⁹ field
  values, no memory leak (`scratchpad/soak_grid_0708_out.txt`).
