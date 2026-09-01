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

## 2. Plan-regret evaluation on real data — genuinely open

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

### The blocker is two metric-semantics defects, not "needs a run"

`docs/memo-shrestha-day5-plan-regret-query.md` (`bf03b8c`, 2026-09-02) is a
read-only diagnostic — it computes no R(S) — and its conclusion is **"do not
reposition the query yet, it is not what is wrong."** Two defects underneath the
regret figure that no start/goal placement fixes:

1. **The fill-rate confound is not actually closed** (it was documented as
   closed on 1 Sep). `common_support()` restricts on `CostMap.unknown`
   (never-observed, ~0.9 % of the window), but `w_unknown` is charged on bit 5
   (`n < n_min`, ~91.9 %). After restriction the frozen schedules still pay
   `w_unknown` on **91.9 %** of the surviving window against uniform 20 cm's
   **4.1 %** — every R(S) in the current table is dominated by that ratio. The
   diagnostic meant to catch this (`PlanResult.unknown_fraction`,
   `eval_synthetic`'s "low-confidence" column) reads `np.mean(costmap.unknown)`
   — the 0.9 % number, not the 91.9 % one.
2. **The two sides of eq. (23) apply §7.1 at different lattices**, and §7.1's
   step/slope thresholds are not scale-invariant. The frozen schedules **invent
   154 walls M\* does not have and miss 10 of the 12 real ones** (the kerb, read
   as impassable at 5–10 cm but not at M\*'s 25 cm planning lattice). At
   `regret_plot.py`'s default `--frames 16`, **M\* contains zero impassable
   cells** while the schedules wall 160.

Shrestha's honest reading is that even with both defects fixed, the synthetic
scene — one 60 cm pothole in an otherwise empty field — cannot draw a knee, and
the real §8.2 plot needs sequence 08. The fixes are metric semantics, in
`src/eval/eval_synthetic.py` / `plan_regret.py`, and are **Aakash's and
Pratyushi's** — Shrestha has produced the measurement and deliberately not
touched those files.

### Scope

The memory reduction and the per-ring geometric accuracy against M\* are the
load-bearing quantitative claims and do not depend on the regret figure. The
regret result — "the compression does not change the plan a robot would make" —
is the project's strongest single claim, and its status is stated plainly here
so a reviewer knows exactly what has and has not run.

*(Current as of `origin/main` `aacd8a4`. `src/eval/` has had four commits since
the class-byte re-split — `d93a2b9`, `74f555d`, `52983e3`, `bf03b8c`. No M\* /
regret artifacts are committed; `.gitignore` excludes them by design.)*

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
