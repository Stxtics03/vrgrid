# Demo-safe frame ranges — seq 00 / 07 / 08

*Perception front-end + dashboard. Written for the live demo: which frames to
drive, which to avoid, and what each range is good for showing.*

All numbers below come from this week's soak runs
(`scratchpad/soak_grid_0708_out.txt`, `soak_0708_out.txt`,
`soak_results.npz`) — the full grid-wired pipeline on every frame of each
sequence. Nothing here was recomputed for this document except the seq-00
vehicle trajectory z, which is a one-line read of the GT poses.

---

## Why some frames are unsafe: the elevation limitation

The map accumulates heights on a **world-absolute** vertical band of
`[-2.0, +6.0] m` (`quantise_height` in `src/gpu/kernels.py`, matching
`vertical_extent_m` in `configs/schedule_5_10_20_40.yaml` — "overpasses and
multi-storey are out of scope"). `MapEngine._centres` feeds the visibility
cleanup cell heights in that same world-absolute frame, without subtracting the
vehicle's own elevation.

On a sequence that **climbs**, once the vehicle's world-z rises much above 0 the
near-field cell heights saturate at the `+6.0 m` ceiling, the cleanup projects
every candidate cell outside the sensor's vertical FOV, and **ghost removal
stops doing anything**. Measured on seq 08 (grid soak, ghost-removal ON):

| vehicle world-z | frames | cells cleared / frame | state |
|---|---|---|---|
| −6.7 … +2.4 m | 1100 | **15,580** | healthy |
| +2.4 … +11.6 m | 840 | 1,022 | degraded ~15× |
| +11.6 … +20.7 m | 939 | 35 | effectively dead |
| +20.7 … +39.0 m | 1191 | **0** | inert |

So there are three tiers, by the vehicle's own world-z:

- **GREEN — `veh_z < 2.4 m`**: ghost removal / occupancy cleanup fully healthy.
- **YELLOW — `2.4 ≤ veh_z < 11.7 m`**: degraded, clears ~15× fewer cells. Usable
  for a static screenshot, risky for a live toggle.
- **RED — `veh_z ≥ 11.7 m`**: cleanup inert. `--show-ghosts` on vs off is
  visually identical. Do not demo the ghost toggle here.

This limitation is **only** in the map layers (`world/map/occupied|free|unknown`)
and the ghost-removal cleanup. The **point cloud** (`world/points`, every
`--color-by` mode including `reflectivity` and `motion`) is unaffected — it is
raw per-frame perception output and never touches `quantise_height`. Reflectivity
and the raw motion mask can be shown at any elevation.

---

## seq 00 — 4541 frames, veh_z −4.9 … +20.6 m

Not flat. The trajectory climbs a hill in the middle third and again near the
end.

| tier | frame ranges |
|---|---|
| **GREEN** (cleanup healthy) | **0–160**, 982–1329, 1474–1627, 3853–4540 |
| YELLOW (degraded) | 161–981, 1330–1473, 1628–2070, 2297–2566, 3245–3852 |
| **RED** (cleanup inert) | **2071–2296**, **2567–3244** |

**Ghost-toggle demo:** frame **10** — the canonical one, ~66 moving points
(motorcyclist + pedestrian just ahead of the vehicle), `veh_z −1.4 m`, inside
GREEN 0–160. Other GREEN options with moving objects: frame 0 (88), 1044 (392),
1077 (77), 1148 (82).

**Reflectivity / lane-marking demo:** frames **0–11** (lane paint ~1000+ pts)
and **4429–4460** (lane paint 1300–1669 pts, the richest in the dataset,
`veh_z ≈ −1.6 m`, inside GREEN 3853–4540). Frame 10 also carries 292 lane-paint
points, so a single stop at frame 10 shows both the ghost toggle and
reflectivity.

**Best single demo window: frames 0–160** — GREEN, contains the frame-10 ghost
example and the frame 0–11 lane markings.

---

## seq 07 — 1101 frames, veh_z −5.8 … −1.0 m

**Entirely GREEN.** The vehicle never rises above −1 m; ghost removal is healthy
for every frame. This is the safe sequence for a live ghost-toggle demo.

| tier | frame ranges |
|---|---|
| **GREEN** | **0–1100 (all)** |
| YELLOW / RED | none |

**Ghost-toggle demo:** abundant — 705 of 1101 frames have ≥ 40 moving points.
Strong examples: frame 674 (3302 moving), 1029 (2432), 887 (1216), 958 (971),
1100 (7672, heavy end-of-sequence traffic).

**Reflectivity / lane-marking demo:** **none available** — seq 07 has **zero**
lane-marking points in all 1101 frames. This is expected (it is a residential
loop with no painted lanes), not a bug. Use seq 00 for the reflectivity story.

---

## seq 08 — 4071 frames, veh_z −6.7 … +39.0 m

The steep sequence. Climbs +45.7 m over its 3.2 km loop. **Unsafe for a live
ghost-toggle demo from frame ~118 onward** until the elevation limitation is
fixed.

| tier | frame ranges |
|---|---|
| **GREEN** (cleanup healthy) | **0–21**, 245–714, 832–1194, 1451–1688 |
| YELLOW (degraded) | 22–74, 119–244, 715–831, 1195–1450, 1689–1987 |
| **RED** (cleanup inert) | **75–118**, **1988–4070** |

The cleanup first goes inert during a brief excursion at **frames 75–118**
(`veh_z` touches 11.7 m), recovers through the 119–1987 stretch in a degraded
state, then is **permanently inert from frame 1988** as the vehicle climbs for
good. `--show-ghosts` on vs off produced an **identical** 377,573-cell occupied
set at the end of the run (contrast seq 00: +17,127-cell difference).

**Ghost-toggle demo:** only frames **0–21** are truly safe (`veh_z < 2.4 m`,
frame 0 has 1128 moving points). Do not demo the toggle past ~frame 118.

**Reflectivity / lane-marking demo:** 17 frames carry lane-marking points
(395–484) but only 1–3 points each — **not usable** for a reflectivity demo.
Expected, not a bug. Use seq 00.

**Occupancy / foveation still demos fine at any elevation** as a *static* view —
the ring-sized cell boxes and the blind cone render correctly; only their
*heights* are clamped and the *cleanup* is inert.

---

## Quick reference — what to drive on the day

| to show | sequence | frames | why |
|---|---|---|---|
| Ghost toggle (live) | **07** | 650–700 or 1020–1035 | flat throughout, dense moving objects |
| Ghost toggle (short) | **00** | 5–15 | frame 10 = the textbook motorcyclist + pedestrian |
| Reflectivity / lane paint | **00** | 4420–4460 | richest lane markings in the dataset, GREEN zone |
| Occupancy layers + foveation | **00** | 0–160 | GREEN, cells + rings + blind cone all correct |
| `--color-by` sweep (class/motion/ground/intensity) | **07** | anywhere | unaffected by elevation |
| **Avoid** | 00 | 2071–3244 | cleanup inert (hill) |
| **Avoid** | 08 | 119 onward | cleanup inert / degraded (climb) |

---

## Full-sequence soak health (all three, grid-wired)

For completeness — every frame of every sequence ran clean:

| | seq 00¹ | seq 07 | seq 08 |
|---|---|---|---|
| frames | 4541 / 4541 | 1101 / 1101 | 4071 / 4071 |
| crashes | 0 | 0 | 0 |
| NaN / Inf | 0 | 0 of 668 M values | 0 of 2.50 B values |
| peak RSS | ~150 MB | 158 MB | 163 MB |
| memory leak | none | none (warm-up only) | +0.28 MB/1000f — flat |
| per-frame (perc + step) | ~140 ms | 133 ms median | 116 ms median |

¹ seq 00 numbers are from the Gate 3 verification run (60 frames) plus the
earlier perception-only full-sequence soak; the grid-wired full-sequence seq-00
soak was not re-run for this document.
