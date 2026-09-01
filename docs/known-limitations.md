# Known limitations

Two boundaries of the current system, stated precisely. Both are understood,
bounded, and scoped deliberately for this submission — neither is a surprise
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
