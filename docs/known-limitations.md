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

### The money plot on real data — a curve now, and what it does and does not show

`regret_plot.py --seq` exists (it had no `--seq`), R(S) is averaged over 64
seeded planning queries instead of one, and `common_support` equalises evidence
rather than only coverage. Sequence 08, four window lengths:

| schedule | MB | @20 | @40 | @80 | @160 |
|---|---|---|---|---|---|
| 5_10_20_40 | 29.06 | 0.488 | 0.171 | 0.714 | 0.758 |
| 5_10_50 | 23.62 | 0.488 | 0.171 | 0.714 | 0.758 |
| uniform 10 cm | 18.19 | 0.231 | 0.084 | 0.759 | 0.597 |
| uniform 20 cm | 10.71 | 0.251 | 0.104 | 0.827 | 0.791 |
| uniform 40 cm | 7.82 | 0.402 | 0.182 | 0.918 | 0.838 |
| uniform 80 cm | 7.09 | 0.798 | 0.130 | 1.165 | 0.819 |

**What holds.** The uniform series rises with cell size at every window —
strictly at 20 and 80 frames, and at 40 and 160 except for the 80 cm point
dipping below 40 cm. Fréchet distance tracks it. Before averaging, the same
runs gave multiples of the 0.207 lattice quantum and an ordering that inverted
with the frame count; that is gone.

**What does not.** The magnitude still moves with the window (5_10_20_40 reads
0.171 at 40 frames and 0.758 at 160), so **R(S) is comparable across schedules
at a fixed window and not across windows.** Any quoted number must state its
frame count.

**And the frozen schedules are not winning.** 5_10_20_40 scores worse than
uniform 10 cm at three of four windows. Before reading that as a result, see
the extent mismatch below — it is very likely an artifact of the x-axis.

### ⚑ The money plot's memory axis compares maps of different extent

The uniform baselines are built at `half_width_m=24.0`; the frozen schedules
reach 100 m:

| schedule | half-width | cells | MB | area |
|---|---|---|---|---|
| 5/10/20/40 | **100 m** | 745,000 | 29.06 | 0.0400 km² |
| uniform 10 cm | **24 m** | 230,400 | 18.19 | 0.0023 km² |

**5/10/20/40 maps 17× the area for 1.6× the memory**, and the figure's x-axis
puts those side by side as though they were comparable. Every "we cost more
than uniform 10 cm" reading in the project, including several made while
investigating this, ignored it.

Fixed properly this is an argument *for* foveation rather than against it, but
it changes what every point on the plot means, so it has not been changed here.
The figure should either match the extents or state the ratio on its face.

### What the plot says at MATCHED extent — measured, not applied

Rebuilding the uniform baselines at 100 m half-width so every map covers the
same ground, seq 08, 20 frames, same 64-query set, nothing else changed:

| schedule | MB | cells | R(S) | Fréchet |
|---|---|---|---|---|
| uniform 10 cm | **78.50** | 4,000,000 | 0.231 | 0.18 m |
| uniform 20 cm | **30.50** | 1,000,000 | 0.251 | 0.23 m |
| **5/10/20/40** | **29.06** | 745,000 | 0.488 | 0.33 m |
| 5/10/50 | 23.62 | 520,000 | 0.488 | 0.33 m |
| uniform 40 cm | 18.50 | 250,000 | 0.402 | 0.32 m |
| uniform 80 cm | 11.04 | 62,500 | 0.798 | 0.55 m |

**The memory claim holds, and is larger than the committed figure shows.**
Matched to the same ground, uniform 10 cm costs **78.50 MB against 29.06** — a
2.7× saving. The current plot instead shows us *more expensive* than an
18.19 MB baseline covering a seventeenth of the area.

**The regret claim does not hold on this query.** `uniform_20cm` at 30.50 MB —
essentially the same memory as 5/10/20/40's 29.06 — scores **0.251 against
0.488**. At equal memory and equal coverage, a plain uniform grid plans about
twice as well. §8.2 claims foveation is free in decision terms; measured
fairly for the first time, on this query it costs roughly 2× the regret of
spending the same bytes uniformly.

**How much weight this carries.** One sequence, one window, one query family.
R(S) magnitude still moves with the window, and the query design itself is
parked (`docs/decisions-2026-09-02.md`, Decision 4 — the single longitudinal
lane). It is **not** proof the thesis is wrong. It is the first fair run of the
comparison, and it does not go our way.

**Not applied.** This was a read-only measurement; no committed figure or
number was changed by it. Rebuilding the plot at matched extent moves every
point and is a §8.2 owner's decision.

### Scope

The memory reduction and the per-ring geometric accuracy against M\* are the
load-bearing quantitative claims and do not depend on the regret figure. The
regret result — "the compression does not change the plan a robot would make" —
is the project's strongest single claim, and its status is stated plainly here
so a reviewer knows exactly what has and has not run.

*(Current as of `origin/main` `38edfb5`, 2 Sep. No M\* / regret artifacts are
committed; `.gitignore` excludes them by design.)*

---

## 2b. Accuracy across ALL eleven labelled sequences — the headline result

Everything in this project was measured on 07 and 08 until 2 Sep, and the
honest reason for those two is that they downloaded first. All eleven labelled
sequences, 40 frames each, schedule 5/10/20/40, with the per-sequence pose
source and Patchwork++:

| seq | r0 RMSE | r1 RMSE | r1 ρ | r2 RMSE | r2 ρ |
|---|---|---|---|---|---|
| 00 | 2.73 | 6.54 | 1.52 | 28.09 | 2.32 |
| 01 | 1.59 | 2.28 | 1.59 | 4.39 | 1.46 |
| 02 | 1.34 | 7.15 | 1.45 | 15.53 | 1.36 |
| 03 | 5.25 | 12.16 | 1.48 | 24.26 | 1.76 |
| 04 | 0.91 | 3.65 | 1.26 | 11.50 | 1.31 |
| 05 | 1.26 | 3.53 | 1.55 | 11.17 | 1.50 |
| 06 | 4.18 | 3.02 | 1.43 | 9.48 | 1.52 |
| 07 | 1.76 | 3.48 | 1.32 | 6.13 | 1.18 |
| 08 | 1.16 | 2.55 | 1.30 | 6.46 | 1.22 |
| 09 | 1.85 | 3.67 | 1.50 | 6.37 | 1.43 |
| 10 | 1.61 | 3.51 | 1.29 | 8.57 | 1.56 |

```
ring 1:  rho median 1.45 [1.26-1.59]     RMSE median  3.53 cm [2.28-12.16]
ring 2:  rho median 1.46 [1.18-2.32]     RMSE median  9.48 cm [4.39-28.09]
```

**Lead with ρ, not RMSE.** At ring 1, ρ spans **1.26×** across the whole
dataset while RMSE spans **5.3×**. That is §9.3's decomposition doing exactly
what it is for: RMSE tracks how rough each road happens to be, ρ divides that
out and leaves what the coarsening cost. "ρ = 1.45, range 1.26–1.59, n = 11" is
both more defensible and closer to the actual claim than any single sequence's
RMSE.

⚑ **07 and 08 are at the good end, not typical.** Their ring-1 ρ of 1.32 and
  1.30 sit near the bottom of the range against a median of 1.45. Anyone who
  checks a third sequence gets a slightly worse number than the one we quoted
  first, so quote the distribution.

⚑ **Sequence 00 is the only ρ outlier** at 2.32 on ring 2, against 1.18–1.76
  everywhere else. Its systematic bias was a pose artifact and is fixed (see
  §6), but that only moved ρ from 2.40 to 2.32 — the rest is dispersion
  (spread 14.41 cm at ring 2) and is **unexplained**. 00 is a long urban loop
  and ring 2 spans 20–50 m where ground segmentation is hardest; that is a
  hypothesis, not a finding.

⚑ **Ring 0 has no ρ on any sequence.** `coarsening_ratio_per_ring` excludes
  footprints holding a single reference return, and at 5 cm essentially every
  footprint holds one. So the finest ring — the one the foveation argument is
  actually about — has RMSE (0.91–5.25 cm) and no ρ anywhere. That is a real
  gap in the evidence, not an oversight in the run.

## 3. Curb and pothole detection — real numbers, no ground truth to score against

`src/grid/features.py` answers the problem statement's own sentence about
curbs and potholes directly (§7.4). Measured on **all eleven labelled sequences**, 40 frames, schedule 5/10/20/40,
through the real loader → transforms → Patchwork++ → `run_sequence` →
`features.detect` path. Curb median by ring:

| seq | r0 | r1 | r2 | r3 | curb cells | pothole cells |
|---|---|---|---|---|---|---|
| 00 | 8.2 | 8.1 | 9.9 | 13.1 | 6,964 | 180 |
| 01 | 8.2 | 8.2 | 8.9 | 15.0 | 14,581 | 192 |
| 02 | 9.1 | 8.1 | 12.0 | 12.0 | 9,055 | 292 |
| 03 | 8.1 | 10.8 | 10.0 | 18.1 | 7,783 | 345 |
| 04 | 8.1 | 7.6 | 10.0 | 9.9 | 5,827 | 404 |
| 05 | 9.1 | 8.5 | 9.5 | 9.0 | 9,414 | 132 |
| 06 | 8.1 | 8.2 | 10.3 | 11.8 | 5,388 | 107 |
| 07 | 8.5 | 9.2 | 9.1 | — | 4,041 | 257 |
| 08 | 8.6 | 8.1 | 8.5 | 10.4 | 9,499 | 166 |
| 09 | 8.5 | 8.9 | 9.1 | 11.5 | 9,140 | 56 |
| 10 | 8.2 | 9.0 | 9.8 | 14.8 | 23,527 | 551 |

**Ring 0 returns 8.1–9.1 cm on every one of eleven sequences** — different
recording dates, different calibrations, a one-centimetre band. That
consistency is the evidence the detector measures a physical feature rather
than an artifact, and it is a stronger claim than any single number.

The rise with ring is systematic and physical — 8–9 cm at 5 cm cells, 9–12 at
20 cm, 9–18 at 40 cm — because a coarser cell straddles the kerb face and
averages in sloped ground. Report per ring; ring 3's spread (9.0–18.1) is where
it stops being reliable.

⚑ **"Reads 1–2 cm low" — WITHDRAWN, it was an assumption not a measurement.**
  I attributed the 8–9 cm reading to `curb.baseline_m` of 0.20 m sampling
  partway up the kerb face. Tested by sweeping the baseline, ring-0 median on
  three sequences:

  | baseline | 07 | 08 | 05 |
  |---|---|---|---|
  | 0.20 m | 8.5 | 8.6 | 9.1 |
  | 0.30 m | **9.1** | **9.8** | **9.5** |
  | 0.40 m | 9.1 | 9.0 | 10.3 |
  | 0.50 m | 9.1 | 7.6 | 11.0 |
  | 0.60 m | 8.5 | 7.1 | 11.0 |

  A longer baseline does **not** systematically recover height — past 0.30 m
  the sequences diverge, 05 rising and 08 falling, while cell counts grow
  throughout. So a longer baseline admits more and different features rather
  than measuring the same one better, and the kerbs in this data genuinely
  measure ~9 cm. Karlsruhe kerbs including dropped crossings at 9 cm is
  entirely ordinary.

  I then moved `curb.baseline_m` 0.20 → 0.30 on that three-sequence evidence
  and **reverted it** after running all eleven. It raises the ring-0 median
  0.7 cm and nearly triples the cross-sequence spread:

  | baseline | median | range | sd |
  |---|---|---|---|
  | **0.20 m** | 8.2 cm | **8.1–9.1** | **0.36** |
  | 0.30 m | 8.9 cm | 7.3–10.2 | 0.89 |

  The tight band *is* the claim — eleven independent recordings, different
  dates and calibrations, agreeing to 1.0 cm. Trading it for 0.7 cm of median
  trades the result for the number. Three sequences were not enough to choose
  this; eleven were.

  ⚑ It also only ever moved ring 0. `baseline_k` is an integer and
  `round(0.30/(2×0.10))` is `round(1.4999999999999998)` = 1, so at 10 cm cells
  and coarser the span stays `2×cell_m` whatever the config says. **Below
  `baseline_m`/2 the knob does nothing** — worth knowing before anyone tunes
  it.

⚑ **Potholes are a demonstration, not a claim.** 56 to 551 cells per sequence
  is a 10× spread with no pattern, and 00's ring 0 reports five cells at
  48.5 cm. The detector fires occasionally and correctly; there is no rate to
  quote.

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

## 6. The eval harness had no height datum — FOUND, FIXED for 07, OPEN for 08

### What the bug was

`kernels.quantise_height` clips to an **8 m band, world-absolute at datum 0**:
[−2.00, +6.00] m. `MapEngine` tracks a moving datum and adds it back on
readout. **`harness.run_sequence` had none**, so every height it stored was
world-absolute.

Sequence 07's ground sits at world z ≈ −1.61 m. Anything more than 39 cm below
it — ditches, kerbside drops, the low side of the road camber — clipped
*upward* to −2.00 m. Measured: **69,470 of 2,158,949 ground returns, 3.2%**,
all clamped in the same direction.

That produced a positive height bias, concentrated in cells whose returns
spread lowest, which is disproportionately the far field. It looked exactly
like a range-dependent measurement error.

### The fix, and what it is worth

`run_sequence` now sets `gm.z_datum_m` from the first pose's elevation and
stores heights relative to it; `metrics._compared` adds it back before
comparing against the world-absolute M\*. **One** datum for the run, not a
moving one, deliberately: a constant offset cancels in every DIFFERENCE the map
computes — slope, step, curb height, pothole depth — so §7.1 and §7.4 are
untouched. A moving datum would not cancel and would put a spurious step
between any two cells last seen at different times.

Sequence 07, 40 frames, clipping falls from 3.2% to **1 return in 2.16 million**:

| ring | cell | cells | mean bias | sd | before |
|---|---|---|---|---|---|
| 0 | 5 cm | 102,988 | **−0.33 cm** | 2.17 | +1.24 |
| 1 | 10 cm | 54,320 | **−0.17 cm** | 3.23 | +3.98 |
| 2 | 20 cm | 11,231 | **−0.41 cm** | 5.67 | +23.46 |
| 3 | 40 cm | 47 | −18.52 cm | 94.08 | noise, 47 cells |

RMSE 22.02 → **3.23 cm** at ring 1; ρ 19.21 → **2.95**. Sub-centimetre
systematic bias at every ring that carries cells, with dispersion growing
sensibly with cell size. **This is the accuracy claim, and it is a good one.**

### ⚑ Corrections to what this document previously said

Two earlier conclusions here were wrong and are withdrawn.

- **"The bias is ring migration, structural, not a fusion bug."** It is not.
  That rested on a controlled experiment — one-ring uniform 20 cm reading
  +7.83 cm against the four-ring schedule's ring 2 at +23.46 cm — and that
  experiment was itself confounded by the clipping, which hit the two
  schedules differently. Re-run with the datum they agree: **−0.26 cm and
  −0.41 cm**. Migration costs nothing measurable.
- **"Grazing incidence and inverse-variance weighting are ruled out."** Those
  remain correctly ruled out, but for a better reason than given: neither could
  have mattered, because the clipping happens in `quantise_height` *before* any
  weight is applied.

### The moving datum, and where 08 actually fails

`gpu.shift.track_datum` now holds the one implementation and both callers use
it — `MapEngine._track_datum` delegates, and `run_sequence` calls it per frame.
The band slides in whole 1 m steps and **re-bases every stored height** as it
moves, so all cells stay relative to the same current datum and every
difference the map computes is unaffected.

Two metric defects surfaced while proving it, and both are fixed:

1. **`obs_count > 0` was not the right predicate.** It counts every return,
   and a cell whose returns were all NON-ground has no measured ground height
   — `fuse` leaves it at its initial 0, and 0 cm is not a neutral height, it is
   *the datum*. So the metric's answer moved with the datum: on seq 07, shifting
   it from −1.64 m to −2.00 m took ring 1's RMSE from 3.19 to 6.15 cm without
   changing a single measurement. `height_variance > 0` is the predicate — the
   codec maps code 0 to maximum variance exactly so "never fused" is
   distinguishable. The metric is now datum-independent, verified by A/B.
2. **Band-saturated cells were being scored.** A cell clamped at the band edge
   holds the edge, not a measurement. Excluded, and reported by
   `saturated_fraction_per_ring` so a ring that loses most of itself says so.

**Sequence 07 after all of it** — 40 frames, 5/10/20/40:

| ring | cell | cells | RMSE | mean bias | spread | rho |
|---|---|---|---|---|---|---|
| 0 | 5 cm | — | **1.16 cm** | — | — | — |
| 1 | 10 cm | 47,059 | **1.69 cm** | −0.08 | 1.07 | **1.95** |
| 2 | 20 cm | 8,323 | **1.02 cm** | +0.14 | 2.40 | **1.11** |

ρ near 1 is the thesis stated numerically: the coarsening cost only what the
terrain's own sub-cell variability costs.

### Sequence 08 — RESOLVED: it was the pose file

08's official KITTI ground-truth poses put the same patch of road **16.6 cm**
apart from one frame to the next, consistently (16.1–17.6 cm across every pair,
so a systematic offset rather than drift). A cell seen over N frames
accumulated about N × 16.6 cm, which made M\* itself carry a **64.5 cm median
standard deviation inside a 10 cm footprint** and put 08's per-ring RMSE at
162 cm.

Median absolute ground-height disagreement between consecutive frames, in
20 cm cells both frames saw:

| sequence | official GT poses | SemanticKITTI SLAM poses |
|---|---|---|
| 07 | **0.49 cm** | 0.66 cm |
| 08 | 16.63 cm | **1.04 cm** |

The two pose files are not interchangeable. KITTI's GT is a GPS/IMU solution
optimised for **trajectory** evaluation; SemanticKITTI computed its own SLAM
poses so that scans **register into a consistent map**. `README.md:21` chose GT
on Day 0 — right for most sequences, wrong for 08.

`loader.pose_source()` now decides per sequence: 08 reads SLAM, everything else
GT, and `VRGRID_POSE_SOURCE=gt|slam` forces one globally so the table above
stays reproducible.

**Checked across every labelled sequence**, same measure, 2 Sep:

| seq | GT | SLAM | | seq | GT | SLAM |
|---|---|---|---|---|---|---|
| 00 | 2.27 | 1.05 | | 06 | 1.32 | 1.23 |
| 01 | 1.43 | 1.38 | | 07 | **0.47** | 0.64 |
| 02 | 1.20 | 1.20 | | 08 | **16.53** | **1.04** |
| 03 | 1.97 | 1.24 | | 09 | 1.21 | 1.26 |
| 04 | 1.25 | 1.11 | | 10 | 1.19 | 1.20 |
| 05 | 1.02 | 1.00 | | | | |

**08 is the only pathological sequence on that measure.** But per-frame
agreement turned out to be a **weak predictor**, and choosing the override list
from it alone was wrong. What matters is the bias that ACCUMULATES, measured
per ring against M\*:

| seq | per-frame | mean_b r1 | mean_b r2 | mean_b r3 | |
|---|---|---|---|---|---|
| 00 | 2.27 cm | −2.86 | −9.85 | **−13.95** | → SLAM |
| 06 | 1.32 cm | −0.34 | −3.32 | −5.80 | wash, stays GT |
| 03 | 1.97 cm | −1.91 | +2.45 | −0.80 | fine |
| others | 1.0–1.4 cm | < \|0.8\| | < \|2.3\| | < \|2.9\| | fine |

Sequence 00 disagrees by only 2.27 cm per frame yet accumulates **−13.95 cm**
by ring 3, while 03 at a comparable 1.97 cm/frame accumulates −0.80. Switching
00 to SLAM takes ring 2's bias from −9.85 to **−0.44 cm**, ring 3's from −13.95
to −1.18, and ring 3 RMSE from 26.03 to **13.57**.

Sequence 06, the next worst accumulator, was tested the same way and is a
**wash** — GT better at ring 1, SLAM marginally better at rings 2–3 — so it
stays on GT. Only sequences with a measured win are overridden.

So the list is `{"00": "slam", "08": "slam"}`, and
`test_only_08_needs_the_slam_poses` pins it so it cannot quietly widen.

**Ruled out along the way, each by measurement:** the height datum (07 is clean
on the same code), band saturation, the ground mask, frame alignment (08 is
4,071/4,071/4,071), the calibration (07 and 08 have *byte-identical* `Tr`), and
`real_scans`'s own composition (bit-identical to `frames.md`'s textbook
`sensor_to_world` chain, 0.49 / 16.63 either way).

### Both sequences, 40 frames, 5/10/20/40, with Patchwork++ and the right poses

| | ring 0 (5 cm) | ring 1 (10 cm) | ring 2 (20 cm) | ring 3 (40 cm) |
|---|---|---|---|---|
| **07** RMSE | 1.76 cm | 3.48 cm | 6.13 cm | 16.34 cm |
| **07** rho | — | 1.32 | 1.18 | 1.25 |
| **08** RMSE | 1.16 cm | 2.55 cm | 6.46 cm | 52.12 cm |
| **08** rho | — | 1.30 | 1.22 | 1.84 |

**ρ between 1.18 and 1.84 on both sequences** is the thesis stated numerically:
the coarsening cost only what the terrain's own sub-cell variability costs.
Mean bias is under 2.4 mm everywhere except 08's ring 3.

⚑ 07's figures moved from an earlier 1.69 cm at ring 1 because **Patchwork++
  replaced the semantic-class ground fallback**, not because of the pose
  change — 07 still reads GT poses. The geometric segmenter admits genuine
  terrain the class mask missed, so `spread` rises (1.07 → 3.95 cm at ring 1)
  and more cells are scored (47,059 → 51,975; ring 3 goes from 0 cells to 949).
  RMSE rises and ρ *falls*, which is the more honest reading: the earlier
  number was over a narrower, flatter subset.

### Patchwork++ is now installed

`pip install pypatchworkpp` fails at every published version: the sdist's
`python/CMakeLists.txt` falls into an out-of-tree branch that fetches
`.../refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz`, the variable is empty under
scikit-build-core, and GitHub returns 404 for `tags/v.tar.gz`. Building from a
git clone takes the `if(EXISTS ../cpp/)` branch instead and works:

```
git clone --depth 1 https://github.com/url-kaist/patchwork-plusplus.git
.venv/bin/pip install ./patchwork-plusplus/python
```

`ground.segment_ground` is now the geometric segmenter rather than the
`ground_from_semantics` fallback, on this machine.

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
