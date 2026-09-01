# KITTI 07 / 08 — download status, verification, soak test

*JP overnight task, 2026-08-31. Written by an unattended session. Nothing
committed or pushed — review before deciding what to do with this file.*

---

## 1. Download status — NO DOWNLOAD NEEDED, both sequences already present and complete

Checked `VRGRID_DATA_ROOT = C:/KITTI/dataset` against every file
`loader.verify_sequence_exists()` needs (`poses/<seq>.txt` + `sequences/<seq>/velodyne/`)
and every file the pipeline actually reads, replicated manually for 07 and 08:

| file | seq 00 (reference) | seq 07 | seq 08 |
|---|---|---|---|
| `poses/<seq>.txt` (official GT) | 4541 lines | **1101 lines** ✓ | **4071 lines** ✓ |
| `sequences/<seq>/calib.txt` | 1171 B | **1172 B** ✓ | **1172 B** ✓ |
| `sequences/<seq>/poses.txt` (SLAM, unused) | present | present | present |
| `sequences/<seq>/times.txt` | 4541 lines | 1101 lines ✓ | 4071 lines ✓ |
| `sequences/<seq>/velodyne/*.bin` | 4541 | **1101** ✓ | **4071** ✓ |
| `sequences/<seq>/labels/*.label` | 4541 | **1101** ✓ | **4071** ✓ |

Frame counts match the KITTI-odometry canonical counts exactly (07 = 1101,
08 = 4071) and match `poses/<seq>.txt` / `times.txt` line counts. Byte totals
are consistent with seq 00 (~1.94 MB/frame velodyne, ~0.49 MB/frame labels):

```
seq 07 velodyne  2,137,363,408 B    labels    534,340,852 B
seq 08 velodyne  7,985,272,992 B    labels  1,996,318,248 B
```

**aria2c was not invoked** — there is nothing to fetch. No file was created,
fabricated, or placeholder-filled.

---

## 2. Small-file verification (calib.txt, GT poses) — well-formed

### calib.txt — 07 and 08 are byte-identical (correct: same recording rig / day)

Every key line has **12 float values** (not 4):

```
P0: -> 12   P1: -> 12   P2: -> 12   P3: -> 12   Tr: -> 12
```

Full `Tr` (Velodyne -> Camera-0 extrinsic), 07 and 08:
```
Tr: -1.857739385241e-03 -9.999659513510e-01 -8.039975204516e-03 -4.784029760483e-03
    -6.481465826011e-03  8.051860151134e-03 -9.999466081774e-01 -7.337429464231e-02
     9.999773098287e-01 -1.805528627661e-03 -6.496203536139e-03 -3.339968064433e-01
```
**Note:** this differs from seq 00's `Tr` (00 was a different recording day).
The pipeline reads calib per-sequence — `iter_pipeline` passes `sequence=`, so
07/08 get their own `Tr`. Correct.

### GT poses — real varying rotation + translation, not placeholders

**`poses/07.txt`** (1101 lines, all 12 values):
```
first 3:
1.000000e+00 1.197625e-11 1.704638e-10 5.551115e-17 1.197625e-11 1.000000e+00 3.562503e-10 0.000000e+00 1.704638e-10 3.562503e-10 1.000000e+00 2.220446e-16
9.999795e-01 5.025123e-04 -6.380358e-03 -4.596714e-03 -5.005160e-04 9.999998e-01 3.144878e-04 -2.001524e-03 6.380515e-03 -3.112871e-04 9.999796e-01 9.154274e-02
9.999096e-01 1.061516e-03 -1.340599e-02 -1.001116e-02 -1.058762e-03 9.999994e-01 2.126022e-04 -4.359704e-03 1.340621e-02 -1.983884e-04 9.999101e-01 1.857373e-01
last 3:
9.821755e-01 2.517525e-02 -1.862725e-01 -1.643540e+00 -2.371988e-02 9.996682e-01 1.003805e-02 -1.932834e-01 1.864634e-01 -5.440765e-03 9.824468e-01 9.371050e+00
9.821916e-01 2.562504e-02 -1.861266e-01 -1.643726e+00 -2.404424e-02 9.996531e-01 1.074592e-02 -1.924916e-01 1.863374e-01 -6.079276e-03 9.824670e-01 9.370281e+00
9.821853e-01 2.567392e-02 -1.861530e-01 -1.643555e+00 -2.411462e-02 9.996526e-01 1.063629e-02 -1.910780e-01 1.863614e-01 -5.957800e-03 9.824632e-01 9.367453e+00
```
Frame 0 = identity + ~0 translation (KITTI convention). By the end the rotation
matrix has drifted well off identity (R00 = 0.982, R02 = -0.186) and the
translation has grown to `(-1.64, -0.19, 9.37)` m. Genuine trajectory.

**`poses/08.txt`** (4071 lines, all 12 values):
```
first 3:
1.000000e+00 1.197624e-11 1.704639e-10 3.214096e-14 1.197625e-11 1.000000e+00 3.562503e-10 -1.998401e-15 1.704639e-10 3.562503e-10 1.000000e+00 -4.041212e-14
9.999956e-01 2.872219e-03 6.839952e-04 3.879738e-03 -2.873712e-03 9.999934e-01 2.190302e-03 -1.849906e-01 -6.776995e-04 -2.192257e-03 9.999974e-01 8.085136e-01
9.999832e-01 5.587658e-03 1.562738e-03 7.192398e-03 -5.592623e-03 9.999792e-01 3.190365e-03 -3.682362e-01 -1.544879e-03 -3.199050e-03 9.999937e-01 1.630418e+00
last 3:
9.974814e-01 3.238182e-02 -6.310489e-02 -1.372250e+01 -3.388247e-02 9.991644e-01 -2.285676e-02 -1.756629e+01 6.231202e-02 2.493734e-02 9.977451e-01 3.094837e+02
9.977645e-01 3.148385e-02 -5.894678e-02 -1.376440e+01 -3.285963e-02 9.992063e-01 -2.251707e-02 -1.759602e+01 5.819108e-02 2.440370e-02 9.980072e-01 3.103057e+02
9.979596e-01 3.123463e-02 -5.568625e-02 -1.380839e+01 -3.249705e-02 9.992316e-01 -2.191037e-02 -1.762495e+01 5.495911e-02 2.367530e-02 9.982079e-01 3.111493e+02
```
Frame 0 = identity. By the end the vehicle is ~311 m forward, 17.6 m lateral,
13.8 m down, with a real rotation matrix. Genuine trajectory. (Seq 08 is the
long "loop" sequence.)

**No identity/incrementing placeholders in either file.**

---

## 3. Soak test — seq 07 and 08, full pipeline, every frame

`scratchpad/soak_0708.py` — same approach as the seq-00 soak:
transforms -> range_image -> semantics -> is_moving -> ground (Patchwork++) ->
reflectivity, every frame, tracking crash / NaN-Inf / per-frame timing /
peak RSS + leak slope / danger zones (sharp turns, sparse-class frames,
boundary frames). Full stdout in `scratchpad/soak_0708_out.txt`.

### Headline: both clean — 0 crashes, 0 NaN/Inf, no memory leak

| | seq 07 | seq 08 |
|---|---|---|
| frames processed | **1101 / 1101, 0 failed** | **4071 / 4071, 0 failed** |
| wall time | 66.8 s (1.1 min) | 254.4 s (4.2 min) |
| per-frame ms (mean / p50 / p99 / max) | 61 / 61 / 75 / 167 | 62 / 61 / 84 / 116 |
| frames > 5x median | **0** | **0** |
| peak RSS | 80 MB | 92 MB |
| RSS leak slope | 13.8 MB/1000f (warmup-dominated, 1101f run) | **1.5 MB/1000f — flat, no leak** |
| NaN / Inf found | **0** of 714,796,253 values | **0** of 2,671,563,932 values |

### Field ranges (full run)

| field | seq 07 | seq 08 | note |
|---|---|---|---|
| `semantic` | -1 .. 18 | -1 .. 18 | exactly the valid range |
| `rho8` | 0 .. 252 | 0 .. 252 | **no saturation** (matches seq 00 — reflectivity fix holds) |
| `rho_hat` | 0 .. 0.99 | 0 .. 0.99 | = raw intensity, [0,1] |
| `cos_inc` | 1.0e-4 .. 1.0 | 4.2e-5 .. 1.0 | extreme-grazing pixels; no NaN |
| `ground_frac` (per frame) | 0.19 .. 0.62 | 0.17 .. 0.74 | sane |
| `world_z` | -32.2 .. 7.6 | -29.0 .. **49.9** | seq 08 has real terrain — see below |
| `veh_z` (trajectory) | -5.8 .. -1.0 | -6.7 .. **39.0** | seq 08 climbs ~39 m |

### Seq 08 — two things that look alarming but aren't

1. **`world_z` to +49.9 m, `veh_z` to +39 m.** Seq 08's GT trajectory climbs
   ~39 m of elevation over its 3.2 km loop (verified in `poses/08.txt` — the
   rotation matrix and z-translation both vary smoothly, it's terrain, not INS
   drift). Far-field points seen from up on the hill reach +50 m world-z.
   0 NaN, transform is rigid. Same phenomenon as seq 00's +20 m hill, more
   extreme. **Aakash: seq 08 is the steepest of the three — any world-absolute
   z assumption in the reference map breaks here worst.**

2. **`range_image.project` warned "15.0 / 15.7 / 15.2 % of points outside the
   vertical FOV, clamped" on a few frames.** The warn threshold is 15%
   (`OUT_OF_FOV_WARN_FRAC`); seq 00 sat at ~4-7%. On a steep slope more of the
   ground is at extreme elevation angles relative to the sensor, so a handful
   of 08 frames just cross the line. This is the clamp-to-edge-ring behaviour
   working exactly as designed (points kept, counted, warned) — **not a bug**,
   and not corrupted data. Only a few frames, not systematic.

### Danger zones

| | seq 07 | seq 08 |
|---|---|---|
| sharp turns (>2 deg/frame) | 131, max 3.5 deg | 418, max 3.5 deg |
| frames with 0 building points | **0** | **0** |
| frames with <60k points | **0** | **0** |
| frames with 0 lane-marking points | **1101 / 1101** | 4054 / 4071 |
| boundary frame 0 | ok (122,626 pts) | ok (123,389 pts) |
| boundary frame last | ok (115,982 pts, 7,672 moving) | ok (122,346 pts) |
| field flags | none | none |

**Neither 07 nor 08 has usable lane markings** (07: zero; 08: 17 frames). The
reflectivity lane-vs-road test is correctly pinned to seq-00 frame 4431 — do
not try to generalise it to 07/08.

Seq 07's last frame has **7,672 moving points** (heavy end-of-sequence traffic)
and seq 08 frame 0 has only **2,126 building points** (opens in a wide area) —
both processed fine, noted so a future test that assumes "every frame is
building-dense" or "moving is a small fraction" knows these exist.

---

## Bottom line

- **Nothing to download.** 07 and 08 are fully present and verified — small
  files well-formed, velodyne + labels complete and frame-count-matched.
- **No file was created or fabricated.**
- **The full pipeline runs cleanly end-to-end on all three sequences** (00, 07,
  08): 9,713 frames total, 0 crashes, 0 NaN/Inf across ~3.9 billion field
  values, no memory leak, no pathological frames.
- Seq 08 is the steep one (+39 m elevation) and pushes the range-image
  out-of-FOV fraction to ~15% on a few frames — handled by the clamp, flagged
  here for Aakash's reference-map z handling.

*Files touched: none committed. `docs/soak-0708-status.md` (this file) written
and left for review. Scratch artifacts in the session scratchpad:
`soak_0708.py`, `soak_0708_out.txt`.*
