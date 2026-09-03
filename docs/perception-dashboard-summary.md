# Perception front-end & dashboard — summary

*For a reviewer seeing the project cold. SIH26053: an adaptive
variable-resolution 2.5D LiDAR map on SemanticKITTI. This document covers the
perception input stage and the Rerun dashboard; the mapping engine, split/merge
and metrics are covered in `docs/master-v4.md` and `docs/sih-math.md`.*

Every number below is from a script or a full-sequence soak run this week, not
an estimate. Sources are named inline.

*Status note: everything described here is merged to `main` (PRs #17–#27). The
elevation/vertical-extent issue in §5 was fixed on 2026-09-01.*

---

## 1. What the perception front-end does

One `PerceptionFrame` per LiDAR scan, produced by `vrgrid.run.iter_pipeline`
(`src/run/__main__.py`). Each stage is a module under `src/perception/`:

| stage | module | what it produces |
|---|---|---|
| **load** | `loader.py` | raw HDL-64E points `(N, 4)` = x,y,z,intensity; raw SemanticKITTI `.label` word; official GT pose (Camera-0 → world) |
| **transform** | `transforms.py` | points in a **z-up world frame** (x forward, y left) via `R_flip · pose · Tr · T_vehicle←sensor`; vehicle origin in world |
| **range image** | `range_image.py` | a 64 × 512 × 5 spherical image `[range, x, y, z, intensity]` + an inverse index (pixel → source point), byte-exact reversible |
| **semantics** | `semantics.py` | 19-class SemanticKITTI label per point (**ground truth**, from the `.label` file — see §3) |
| **motion** | `semantics.py` | `is_moving()` per point — the `moving-*` raw ids 250–259 (**ground truth**) |
| **ground** | `ground.py` | ground / non-ground mask from **Patchwork++** (`pypatchworkpp`), or a semantic-class proxy fallback when the C++ extension is absent — `PerceptionFrame.ground_method` records which ran, and the fallback warns once per run (§6) |
| **reflectivity** | `reflectivity.py` | one normalised reflectivity byte per point (see §4) |

Downstream, `run.engine.MapEngine.step(frame)` folds each frame into the grid
(bin → scatter → fuse → visibility cleanup → shift). That code is the mapping
team's; the front-end's contract is the `PerceptionFrame`.

**Instance clustering** (`instances.py`, Day-4 addition, PR #20): connected
components on the range image group the `is_moving()` mask into discrete objects
(centroid, bounding box, point set per instance) rather than one mask. Verified
on seq 07 frame 30 — two moving pedestrians 30 m apart resolve to two instances
with centroids within 0.1 m of ground truth.

### Verified robustness (full-sequence soaks)

Every frame of every sequence, perception + mapping engine, tracking
crash / NaN / timing / memory
(`scratchpad/soak_grid_0708_out.txt`, `soak_0708_out.txt`):

| | seq 00 | seq 07 | seq 08 |
|---|---|---|---|
| frames | 4541 / 4541 | 1101 / 1101 | 4071 / 4071 |
| crashes | 0 | 0 | 0 |
| NaN / Inf | 0 | 0 of 6.7 × 10⁸ values | 0 of 2.5 × 10⁹ values |
| perception / frame | ~95 ms | ~60 ms | ~80 ms |
| `MapEngine.step` / frame | ~35 ms | ~36 ms | ~39 ms |
| peak RSS | ~150 MB | 158 MB | 163 MB |
| memory leak | none | none (warm-up only) | +0.28 MB / 1000 frames — flat |

---

## 2. The dashboard

`python -m vrgrid.dash --seq 07` (live viewer) or `--save run.rrd` (headless →
open with `rerun run.rrd`). Runs as a **separate process** from the pipeline, so
if it falls over the framework still produces numbers.

### Point cloud — `--color-by`

The `world/points` entity, one interpretation layer at a time:

| mode | shows |
|---|---|
| `intensity` | raw Velodyne intensity, greyscale — the "no semantics" baseline |
| `class` | the 19 SemanticKITTI classes |
| `motion` | static dim grey, moving in reddish-purple (`#CC79A7`) |
| `ground` | Patchwork++ ground tan vs non-ground steel-blue |
| `reflectivity` | the normalised reflectivity byte, greyscale |

`--palette groups` collapses the 19 classes into 7 colourblind-safe
super-groups (drivable-ground, structure, vegetation, vehicle,
vulnerable-road-user, pole-signage, unknown). All palettes are checked in CI by
`dashboard/cvd.py` / `tests/test_cvd.py` — Machado et al. (2009) CVD simulation
at severity 1.0, CIELAB ΔE ≥ 12 required between every pair; the groups palette
clears at min ΔE 16.2.

### Ghost toggle

`world/ghosts` holds the moving returns on their own entity; toggling its
visibility (eye icon) is the ghost toggle. `--show-ghosts` **also** stops the
mapping engine running the §10.4 visibility cleanup, so the "off" state leaves
the ghost trails *in the map cells*, not just in the point cloud.

Verified (seq 00, Gate 3 run, 60 frames): with the cleanup on, **176,482 map
cells cleared**, 144,918 spared by the never-clear-a-current-return guard.
`--show-ghosts` vs default at the end of the run: **+17,127 occupied cells** —
the ghost trails the cleanup removes.

### Occupancy layers — three distinct states

"Unknown ≠ free" is a hard project invariant (`docs/sih-math.md` §10.1). The map
view keeps all three occupancy states on separate entities:

| entity | look | meaning |
|---|---|---|
| `world/map/occupied` | raised solid boxes, coloured by elevation (fixed −3…15 m blue→vermillion ramp), **sized to the cell** | the 2.5D surface |
| `world/map/free` | flat translucent slate tiles at the ground datum | "the sensor looked here and it is clear" |
| `world/map/unknown` | flat muted-violet tiles | cells observed but still below the observation-count threshold |

Never-observed cells (the bulk of the allocation) are deliberately **not drawn** —
on seq 00 that is 705,065 of 910,000 slots, every one with zero observations;
drawing them would bury the 17,127 free cells and 187,808 occupied ones. The
blind cone is the standing marker for the "unknown, never observed" region.

### Ring cell counts — the foveation, verified

Cell size grows with range. On seq 00, 60 frames, cleanup on:

| ring | cell | occupied cells | 95th-pct distance from vehicle | ring half-width |
|---|---|---|---|---|
| 0 | 5 cm | 91,160 | 11.0 m | 10 m |
| 1 | 10 cm | 57,602 | 25.3 m | 25 m |
| 2 | 20 cm | 31,266 | 51.6 m | 50 m |
| 3 | 40 cm | 7,780 | 123.1 m | 100 m |

Each cell size stays inside its ring; boxes visibly quadruple in edge length at
each boundary. The full grid is 745,000 cells = **8.94 MB** at 12 B/cell
(schedule `5_10_20_40`); the alternative `5_10_50` schedule is 520,000 cells =
6.24 MB.

### Blind cone

A red circle of radius **3.74 m** under the vehicle — `h_sensor / tan|φ_min|` =
1.73 / tan(24.8°), the ground the HDL-64E physically cannot see in one sweep
(`docs/sih-math.md` §1.4). Read from `configs/thresholds.yaml`
(`sensor.blind_cone_m`), never hardcoded. It is `unknown`, never `free`.

### Schedule selector

A `schedules` text panel in every recording lists each schedule in
`configs/schedule_*.yaml` (ring boundaries, cell count, MB) and marks the active
one. Display-only — `--schedule <name>` picks one at startup; the ring
boundaries drawn on screen come straight from the chosen `Schedule` object, not
a second hardcoded copy.

---

## 3. FRNet — the port works; the map still uses ground truth, by choice

**Status changed on 2 September.** The standalone port was non-functional for
five days at ~15 % point accuracy, and this section used to say so. It now
reproduces the pretrained checkpoint: **98.3 %** point accuracy on seq 00
frame 43, **69.8 %** mIoU over 200 frames of seq 08 against the paper's
**73.3 %** (`scripts/frnet_eval.py`). Fine-tuning was tried and **rejected**,
−0.5 mIoU (`docs/handover-2026-09-02.md`).

### The three divergences, and the two the original header got wrong

| # | what the old header said | what was actually true |
|---|---|---|
| 1 | "backbone activation is `nn.LeakyReLU` here; the checkpoint was trained with HSwish" | **Correct.** 7 sites in `frnet_backbone.py`. mmcv's HSwish is `x·relu6(x+3)/6`, which is exactly torch's `nn.Hardswish`. |
| 2 | "FOV is fed as `fov_up=2.0` / `fov_down=-24.8`; training used 3.0 / −25.0" | **Wrong place.** `frnet.py` always defaulted to the correct 3.0 / −25.0. `perception/semantics.py` was overriding them with the HDL-64E's *physical* vertical FOV out of `configs/frnet.yaml`. Different quantities — the checkpoint learned a **fixed** spherical projection, so points must land in the grid the weights were trained on whatever sensor produced them. |
| 3 | "the test-time RangeInterpolation densification (H=64, W=2048) is missing" | **Understated.** It was missing, but it is a *test-pipeline transform to be reproduced verbatim*, not a resolution setting — and it projects with the divergence-2 FOV, so the two were coupled. Upstream's `proj_mask = (proj_idx > 0)` off-by-one (point index 0 reads invalid against a −1 sentinel) is **reproduced deliberately**: the paper's 73.3 % was measured with it, and "fixing" it makes our numbers incomparable. |

Two further corrections to the same header: it recorded "413/413 tensors, no
shape mismatch", where the checkpoint actually loads **421 tensors, 0 missing
and 8 unexpected** — all `auxiliary_head.*`, training-only heads correctly
unused at inference. And it listed the manual `scatter_max` / `scatter_mean` in
`frustum_encoder.py` as a suspected fourth cause; they were not one, and that
file is unchanged.

### The map still reads `.label` files — and that is now a decision, not a fallback

Both the 19-class semantic label and the `moving-*` motion flag come straight
from SemanticKITTI's raw files. Zero training, zero inference on the map path.
The reason is no longer "the port is broken" but that ground-truth labels
**isolate the mapping contribution from segmentation error**, which is what §9's
evaluation is for. The model is reported **alongside** the map, never swapped
into it, so no mapping number in this project depends on it.
`semantics.get_frnet` / `segment*` still raise for exactly that reason — the
message says "disabled", not "broken".

**This is disclosed plainly, not hidden.** It is also the *right* call for the
evaluation: the project's contribution is the variable-resolution mapping
engine, and feeding it ground-truth labels **isolates the mapping quality from
segmentation error** — which is exactly what a careful evaluator wants to
measure. A future real-FRNet swap lines up with no relabelling (the 19-class map
is the canonical SemanticKITTI one).

---

## 4. The reflectivity fix (and why)

Math appendix eq (31) normalises raw LiDAR power as
`ρ̂ = I · r² / max(cos θ_incidence, 0.1)` — undoing the `1/r²` fall-off and the
incidence-angle foreshortening to recover a surface property.

Applying that to **KITTI** was wrong. KITTI's intensity channel is **already
range- and incidence-compensated by the sensor firmware**: on a known-uniform
surface (flat road) the log(I)-vs-log(r) slope is ~0.01, i.e. no range trend.
Re-applying `· r²` on top pinned **62 % of ring-1 road reflectivity at the
saturated byte value 255** — the lane-vs-road contrast the map uses for
refinement was being destroyed by range.

**Fix** (`reflectivity.normalise(..., range_compensated=True,
incidence_compensated=True)`, the KITTI default): `ρ̂ = I`, byte = `round(I · 255)`.
The eq-(31) raw-power path is kept for non-firmware-compensated sensors and for
the incidence-angle geometry (`incidence_cos()`, finite-difference surface
normal, still used elsewhere).

**Verified:** across seq 00 / 07 / 08 full soaks the reflectivity byte ranges
0–252 with **no ring saturating past 15 %** (`tests/test_reflectivity.py`); ring-1
road median dropped from 255 to ~64. Lane paint stays ~1.25–1.55× brighter than
asphalt, range-stable.

---

## 5. Elevation / vertical-extent issue — found and fixed

**This was an open limitation earlier in the week; it is now fixed** (`51bff0f`,
2026-09-01). Recorded here because older notes still reference it.

**The bug.** Heights entered the map on a **world-absolute** band of
`[−2.0, +6.0] m` (`quantise_height`, matching `vertical_extent_m` — the project
scopes out "overpasses and multi-storey"), and the visibility cleanup was fed
cell heights in that same world-absolute frame. On a climbing sequence the
near-field heights saturated at the +6 m ceiling while the sensor sat metres
higher, so every cleanup candidate projected outside the sensor's vertical FOV
and **ghost removal stopped working**. Pre-fix on seq 08: 2,304 of 4,071 frames
(57 %) fully inert; seq 07's −5.8 m elevation similarly saturated the floor of
the band and degraded it there.

**The fix.** `quantise_height` takes a `datum_m`; `MapEngine._track_datum`
slides the 8 m band in whole 1 m steps to follow the vehicle and re-bases the
stored heights when it moves; `_centres` hands the cleanup vehicle-frame z. The
band stays 8 m wide, so the dense-3D baseline count and the 286× memory ratio
are unchanged.

**Verified.** `tests/test_engine.py::test_the_ghost_clears_at_any_vehicle_elevation`
(parametrised at 0, −5.8, 6.0, 12.0, 39.0 m); a re-run of the Gate 3 scene
through the real engine at each of those elevations; and a full-sequence
re-soak — seq 08 now **0 of 4,071 frames inert**, clearing flat at ~15–20 k
cells/frame across every elevation band, occupied-cell heights tracking the
vehicle (`occ_z [36.0, 44.0] m` at veh_z 38.4). The ghost toggle demos at any
elevation, including seq 08's full climb. See `docs/known-limitations.md` §1 for
the full numbers.

**Never affected:** the point cloud and every `--color-by` mode — they do not
touch `quantise_height`.

---

## 6. Ground segmentation — Patchwork++ vs the semantic fallback

The ground stage is **Patchwork++** (`pypatchworkpp`, wired in via
`ground.segment_ground`, not reimplemented). If the C++ extension is not
importable, `ground.segment_ground_or_fallback` uses `ground_from_semantics`
instead — a pure class-membership test (`road / parking / sidewalk /
other-ground / terrain` → ground).

**The two are not equivalent.** The fallback marks *every* point of a ground
*class* as ground regardless of geometry. On a sloped verge it therefore admits
the raised part of an embankment — measured at ~4–12 % of the `terrain` points
on sequences 07/08 — that Patchwork++'s local plane fit rejects. On flat
road/sidewalk the two agree to ~1–2 %. So a per-ring accuracy, curb/pothole, or
plan-regret number computed on the fallback is **not comparable** to one on
Patchwork++ (e.g. seq 07 ring-1 RMSE moved 1.69 → 3.48 cm when the real
segmenter replaced the fallback — `known-limitations.md` §2b).

**How you know which one ran:**

- `PerceptionFrame.ground_method` is `"patchworkpp"` or `"semantic_fallback"`,
  set per frame at the branch point.
- The first time the fallback stands in within a process, `ground.py` emits a
  `RuntimeWarning` (once, not per frame) naming the reason — missing extension
  vs. deliberate `--no-patchworkpp`.
- `python -m vrgrid.run` / `python -m vrgrid.dash` print a `[!] ground:
  SEMANTIC-CLASS FALLBACK …` line in their end-of-run summary.

**Installing the real segmenter.** `pip install -e ".[perception]"`. PyPI ships
prebuilt wheels for CPython 3.8–3.13 on `win_amd64` / `win32`,
`manylinux`/`musllinux` `x86_64` + `i686`, and macOS `arm64`. There is **no
wheel for Linux `aarch64` or Intel macOS**; on those, pip falls back to the
sdist, whose CMake fetches an empty version tag (`.../tags/v.tar.gz`) and 404s.
Build from a git checkout instead:

```
git clone --depth 1 https://github.com/url-kaist/patchwork-plusplus.git
pip install ./patchwork-plusplus/python
```

CI does not install it, so `tests/test_ground.py`'s Patchwork++ cases skip
there and the semantic-proxy cases always run.

---

## 7. File map

```
src/perception/
  loader.py        scans, GT poses, calib, .label            [JP]
  transforms.py    sensor -> vehicle -> z-up world           [JP]
  range_image.py   64x512 spherical projection + inverse     [JP]
  semantics.py     19-class + is_moving, both GT              [JP]
  ground.py        Patchwork++ wire-in + loud fallback (§6)  [JP]
  reflectivity.py  eq (31) + the KITTI firmware-compensated path  [JP]
  instances.py     range-image connected-components clustering    [JP]
  frnet/           standalone port, WORKING (not wired in)   [JP]
dashboard/
  __main__.py      `python -m vrgrid.dash` entry point       [JP]
  pipeline_view.py the real per-frame Rerun view             [JP]
  _config.py       blind cone / schedule helpers from configs [JP]
  palettes.py      rerun-free colour tables (CVD-checked)     [JP + Shrestha]
  cvd.py           colourblindness simulation / audit         [JP]
configs/
  thresholds.yaml  frozen; blind cone, occupancy, fusion, sensor
  schedule_*.yaml  ring schedules
  frnet.yaml       range-image sensor geometry
docs/
  demo-safe-ranges.md          which frames to drive live
  perception-dashboard-summary.md   this file
```
