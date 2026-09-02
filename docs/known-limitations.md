# Known limitations

One issue was found in our own testing this week, root-caused, and fixed — it
is documented below (§1) as a resolved item because a reviewer may still see it
referenced in older notes. One genuinely open item remains (§2), and its status
is stated precisely. Everything else the team considers settled is listed at the
end.

---

## 1. Elevation / ghost-removal — FOUND, ROOT-CAUSED, FIXED

**Status: fixed on 2026-09-01** (`51bff0f`, "gpu/run: make the vertical band
follow the vehicle, not the world datum"; `337bb30` records its per-frame cost).
Independently verified three ways — see the bottom of this section.

### What the bug was

The visibility cleanup (math §10.4) — the stage that erases the trailing
"ghost" cells a moving object leaves behind — stopped clearing cells once the
vehicle's own elevation in the world frame rose past ~11.7 m. Heights entered
the map on a **world-absolute** vertical band of `[−2.0, +6.0] m`
(`quantise_height`, matching `vertical_extent_m` in the schedule config, which
scopes out overpasses and multi-storey structures by design), and the cleanup
was handed cell heights in that same world-absolute frame. On a climbing
sequence the near-field cell heights saturated at the +6 m ceiling while the
sensor sat tens of metres higher, so every cleanup candidate projected outside
the sensor's vertical FOV and was skipped.

*Pre-fix measurement, kept for context (SemanticKITTI seq 08, +45.7 m climb,
grid-wired soak, ghost removal ON):*

| vehicle world-z | cells cleared / frame (**pre-fix**) |
|---|---|
| −6.7 … +2.4 m | 15,580 |
| +2.4 … +11.6 m | 1,022 |
| +11.6 … +20.7 m | 35 |
| +20.7 … +39.0 m | 0 |

Pre-fix, **2,304 of seq 08's 4,071 frames (57 %) were fully inert**, and — a
finding from the re-soak below that the pre-fix notes missed — **seq 07 was
also degraded**: at its −4 to −6 m elevation it cleared only ~42–61 cells/frame
with ~70 % of candidates out of FOV, because −5.8 m saturates the *floor* of
the same world-absolute band.

### The fix

`quantise_height` now takes a `datum_m` the heights are measured from
(defaulting to 0.0 — bit-for-bit the old behaviour). `MapEngine._track_datum`
slides the 8 m band in whole 1 m steps to follow the vehicle's world-z, and
re-bases the stored ground/ceiling heights when it moves — the vertical
counterpart of the toroidal horizontal shift, and rare for the same reason.
`_centres` takes a 2- or 3-vector ego so the cleanup receives the vehicle-frame
z its contract asks for. **The band stays 8 m wide**, so the dense-3D baseline
in `dashboard/_config.py` counts the same voxels and the headline 286× memory
ratio is untouched — which is why moving the band was the right fix and
widening the clamp would have been the wrong one.

### Verification

1. **`tests/test_engine.py`** —
   `test_the_ghost_clears_at_any_vehicle_elevation`, parametrised at 0, −5.8,
   6.0, 12.0 and 39.0 m (seq 07's floor and seq 08's hill); and
   `test_the_band_follows_the_vehicle_rather_than_the_world_datum`.
2. **Aakash re-ran the Gate 3 scene through the real engine, kernels and
   projection** at −5.8, 0, 6, 12 and 39 m — the departed car's ghost now
   clears at every one; every one except 0 m failed on the pre-fix code.
3. **Full-sequence re-soak against the fixed engine** (`scratchpad/soak_elev_postfix_out.txt`):

| veh_z band (m) | frames | cleared/frame (**post-fix**) | protected/frame | prot_frac |
|---|---|---|---|---|
| −6.7 … 2.4 | ~1090 | 18,958 | 12,116 | 0.390 |
| 2.4 … 11.6 | ~848 | 20,524 | 11,499 | 0.359 |
| 11.6 … 20.7 | ~926 | 15,695 | 8,763 | 0.358 |
| 20.7 … 29.9 | ~760 | 17,775 | 10,614 | 0.374 |
| 29.9 … 39.0 | ~433 | 17,054 | 10,287 | 0.376 |

   Seq 08: **0 of 4,071 frames inert** (was 57 %), clearing flat at ~15–20 k
   cells/frame across every elevation band, `prot_frac` flat at ~0.36–0.39
   (was 0.37 → 0.22 → 0.13 → 0.00 → 0.00). Occupied-cell heights now track the
   vehicle — at frame 3000, veh_z 38.4 m, `occ_z` is `[36.0, 44.0] m` (was
   pinned at `[−2.0, 6.0]`). Seq 07 total cleared 373,846 → 15,731,026, all
   bands healthy. 0 NaN in the readout. Datum re-base fires 147 times in
   seq 08's 4,071 frames, median 3.17 ms each, 0.20 ms/frame amortised — the
   frames it fires on sit near 52 ms against the 100 ms budget.

**Consequence for the demo:** the ghost toggle is safe to demonstrate at any
vehicle elevation, including seq 08's full 39 m climb. `docs/demo-safe-ranges.md`
has been updated to drop the elevation-based frame restrictions.

---

## 2. Plan-regret evaluation — the two defects are closed; the scene is the limit

**The offline plan-regret pipeline is implemented and unit-tested** (`src/eval/`,
`tests/test_plan_regret.py`, `test_reference_map.py`, `test_metrics.py`) and has
been **run end-to-end on the synthetic sequence**. Two things stand between that
and a real §8.2 result, and the second is larger than a build step.

### The build path is fixed and works on real data

`reference_map.build()` — the only path from SemanticKITTI to the reference map
M\* — **had never been executed by anything** until this week: it raised
`ValueError: too many values to unpack` on its first line, behind two more
latent frame-convention bugs. Aakash fixed all three and added
`scripts/build_reference_map.py` (`74f555d`), with the whole real path now
exercised against the synthetic writer's KITTI layout so heights can be
asserted rather than eyeballed. `harness.FrameGuard` now checks frame 0 **and**
the first frame ≥ 10 m from the start (a `poses.txt` begins at the identity,
where the right and wrong compositions agree).

Verified on JP's machine: `python scripts/build_reference_map.py 00
--max-frames 5` builds a real M\* (226,485 observed cells, 621,510 returns,
median height −1.55 m), writes and reloads the cache. **Building M\* for 07/08
is one tested command; the ~40 GB SemanticKITTI download is the only thing left
on that path** — and it is not on the shared/CI infrastructure, only on JP's
dev box (`data/README.md` assigns it, execution-plan decision B).

### Both metric-semantics defects are fixed — 2 September

`docs/memo-shrestha-day5-plan-regret-query.md` (`bf03b8c`) named two defects
underneath the regret figure that no start/goal placement fixes. Both are now
closed, and a third was found while closing them.

1. **The fill-rate confound.** `common_support()` restricted on
   `CostMap.unknown` (never-observed, ~0.9% of the window) while `w_unknown`
   was charged on bit 5 (`n < n_min`, ~91.9%), and the diagnostic built to
   expose that read the array that hid it. Aakash root-caused it in `baa44b4`:
   `costmap_from_gridmap` was OR-ing the confidence bit over the sub-cells of a
   planning cell, so **the handicap grew with resolution**. Counts are now
   summed over the footprint's distinct cells and compared against `n_min`
   once. Inside the common support the frozen schedules went from **100.0% to
   0.9%**; uniform 20 cm from 4.2% to 0.0%.

2. **The two lattices.** M\_S set §7.1 bits at ring resolution while M\* set
   them at 25 cm. Both sides are now evaluated at `plan.cell_m` from the same
   summed statistics, and clearance is dropped from both — M\* is 2.5D ground
   and cannot set it. **0 invented walls, 0 missed**, both frozen schedules.
   Separately, §7.1 eq. (22a) now differences bits 1 and 2 over a fixed
   physical baseline rather than one cell, so a 12 cm kerb reads the same at
   5 cm and at 25 cm instead of being a wall on the fine rings only.

3. **Bit 4 was on one side only** — found while tracing the residue.
   `ReferenceMap` carries `class_id`, but `costmap_from_reference` built from
   `block_stats`, heights alone, so M\* charged 0 class penalties against the
   schedules' 18. Both paths are scored on M\*, so a schedule paid pure regret
   for routing around ground it had correctly labelled non-drivable. Symmetric
   now, via `ReferenceMap.block_class()`.

### What remains is the scene, not the metric

After all three, the frozen schedules report **R(S) = 0.207** and every uniform
baseline **0.000** on the synthetic sequence. That 0.207 is not a knee and must
not be drawn as one: it is

```
2 · (√2 − 1) · plan.cell_m  =  0.2071
```

— two diagonal steps, i.e. a path that jogs one 25 cm cell sideways and back.
**It is the smallest non-zero value the planning lattice can express.** Traced
to its cause it is a single cell: column 16 of the window holds one cell with
bit 5 set and column 17 holds none, so the fine schedules sidestep for the
length of the corridor. The uniform maps pool more observations per cell,
nothing falls under `n_min`, and they go straight.

So Shrestha's original reading survives the fixes: **the synthetic scene cannot
draw a knee.** M\* over the planning window has one passable cost value — 1,924
cells at 1.00× — and a graded curve needs a graded cost field. The real §8.2
plot needs sequence 08, which is now on disk. See
`docs/decisions-2026-09-02.md` for the query-design question this leaves open.

### Scope

The memory reduction and the per-ring geometric accuracy against M\* are the
load-bearing quantitative claims and do not depend on the regret figure. The
regret result — "the compression does not change the plan a robot would make" —
is the project's strongest single claim, and its status is stated plainly here
so a reviewer knows exactly what has and has not run.

*(Current as of `origin/main` `38edfb5`, 2 Sep. No M\* / regret artifacts are
committed; `.gitignore` excludes them by design.)*

---

## 3. Curb and pothole detection — real numbers, no ground truth to score against

`src/grid/features.py` answers the problem statement's own sentence about
curbs and potholes directly (§7.4). Measured on **sequence 08**, 40 frames,
schedule 5/10/20/40, through the real loader → transforms → Patchwork++ →
`run_sequence` → `features.detect` path:

| ring | cell | curbs | median | potholes | median |
|---|---|---|---|---|---|
| 0 | 5 cm | 11,869 | 9.1 cm | 143 | 9.0 cm |
| 1 | 10 cm | 18,321 | 11.3 cm | 363 | 11.5 cm |
| 2 | 20 cm | 1,724 | 14.2 cm | 70 | 15.0 cm |

Real urban kerbs are 10–15 cm, and on the synthetic scene — where the answer is
known — it returns **12.0 cm against a built 12 cm kerb** and **40.0 cm against
a built 40 cm hole**.

**The limitation: SemanticKITTI has no ground truth for curb or pothole
geometry.** There is no detection rate to quote, only counts and a plausibility
check on the height distribution. The `road`/`sidewalk` label boundary is the
only cross-check available, and it locates curbs without measuring them. Quote
these as counts with that caveat attached, never as accuracy.

Two further honest notes. The median rises with cell size (9.1 → 11.3 →
14.2 cm) because a coarser cell averages across the kerb face and admits more
sloped ground — so it is reported per ring rather than pooled. And potholes are
bounded above at 50 cm (`pothole.max_depth_m`): ring 2 first reported 156
detections at a median 71.5 cm and a p90 of 200 cm, which are ditches, kerb-line
drop-offs and the space under parked cars. Those are real hazards and they need
a separate detector; this one does not claim them.

## 4. Per-cell confidence is a margin, not a probability

`src/grid/confidence.py` (§7.5) reports how far to trust each drivability
verdict, on four derived channels — nothing is stored, `CELL_BYTES` stays 12.

**It is not calibrated.** Nothing has been fitted against outcomes, so 0.6 is
not a 60% chance of anything; each channel is a margin with a stated meaning
and they are combined by taking the weakest. The `label` channel is
additionally a **floor, not an estimate**: the Boyer-Moore counter saturates at
7, so a cell observed 200 times unanimously reports a *lower* share than one
observed 8 times. `saturated()` flags that regime.

On sequence 08 the binding channel is **geometry** for rings 0–2, where the
synthetic scene binds on `label` and `evidence` — real terrain sits near the
slope and step thresholds and the analytic scene does not. That is the clearest
single argument in the project for not reporting synthetic numbers.

## 5. The visibility candidate cap moves a memory figure

`visibility.max_candidate_cells` is now `null`, meaning the grid's own slot
count — 910,000 at 5/10/20/40, **58.24 MB** of scratch, up from 9.60 MB at the
retired placeholder of 150,000. That placeholder dropped 52.3% of sequence 07's
peak occupied set and 67.1% of 08's, untested and in silence.

This is **working memory, not map memory**, so the report's cell-count ratios
are unaffected, and `with_visibility` is off by default so the 29.06 MB headline
does not move unless the cleanup's scratch is switched on. If a smaller declared
total is wanted, an explicit `600000` (38.40 MB) covers every measured sequence
at 1.32× the observed max.

Measured peaks: **314,442** (07, 1,101 frames), **455,714** (08, 4,071),
**278,226** (00, 4,541). The peak does **not** scale with sequence length — 00
is the longest and the lowest, which refuted the first version of this argument.
It tracks scene density instead, varies 1.64× across three sequences with no
available predictor, and 19 sequences remain unmeasured. That unpredictability,
not growth, is the case for the structural bound. See
`docs/decisions-2026-09-02.md`, Decision 2.

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
  frame of seq 00 / 07 / 08, on both the pre- and post-elevation-fix engine:
  0 crashes, 0 NaN/Inf across ~3.2 × 10⁹ field values, no memory leak
  (`scratchpad/soak_grid_0708_out.txt`, `soak_elev_postfix_out.txt`).
