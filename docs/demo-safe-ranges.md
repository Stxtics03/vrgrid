# Demo frame ranges — seq 00 / 07 / 08

*Perception front-end + dashboard. A shot list for the live demo: which frames
to drive for each thing worth showing.*

Numbers come from this week's soak runs (`scratchpad/soak_grid_0708_out.txt`,
`soak_0708_out.txt`, `soak_results.npz`, `soak_elev_postfix_out.txt`) — the
full grid-wired pipeline on every frame of each sequence.

---

## There are no unsafe frames any more

An earlier version of this document flagged frames above ~11.7 m vehicle
world-z as unsafe for the ghost toggle, on all three sequences. **That
limitation was fixed on 2026-09-01** (`51bff0f`; see
`docs/known-limitations.md` §1). The vertical band now follows the vehicle
instead of the world datum, so:

- **The ghost toggle demos correctly at any vehicle elevation**, including
  seq 08's full 39 m climb. Post-fix re-soak: seq 08 clears ~15–20 k cells/frame
  across every elevation band (was 0 above 20 m), **0 of 4,071 frames inert**
  (was 57 %).
- Seq 07 — which the earlier doc called "entirely healthy" — was in fact also
  degraded before the fix (its −5.8 m elevation saturates the *floor* of the
  old band; it cleared ~50 cells/frame at low elevation). It is now healthy
  too: total cleared 373,846 → 15,731,026 over the run.
- **The point cloud was never affected** by any of this. Every `--color-by`
  mode (`intensity` / `class` / `motion` / `ground` / `reflectivity`), the
  `--palette groups` view, the ring boundaries and the blind cone render
  correctly at any elevation and always did — they never touch
  `quantise_height`.

So the frame choice below is only about *where the interesting things are*, not
about avoiding anything.

---

## Ghost toggle

Two levels, both worth knowing:

- **Point-cloud level** — toggle the `world/ghosts` entity's visibility (the
  eye icon). Shows / hides the moving returns in the rendered cloud. Works on
  any frame of any sequence.
- **Map level** — `--show-ghosts` also stops the engine running §10.4, so the
  ghost *trails* stay in the occupied cells. The on/off difference in the map
  needs a few frames of trail to accumulate, so it reads best over a short
  continuous run rather than a single frame.

**seq 00, frame 10** — the textbook shot: ~66 moving points, a motorcyclist and
a pedestrian just ahead of the vehicle. Run frames ~0–20 so the trail is
visible. Frame 10 also carries 292 lane-marking points, so one stop shows the
ghost toggle and reflectivity together.

**seq 00, frames 0–60** — the strongest *map-level* `--show-ghosts` demo: the
departed motorcyclist leaves a trail that stays in-window, and on-vs-off
differs by **+17,127 occupied cells** at frame 60.

**seq 07** — abundant moving objects (705 of 1,101 frames have ≥ 40 moving
points). Strong single frames: **674** (3,302 moving), 1029 (2,432), 887
(1,216), 958 (971), 1100 (7,672 — heavy end-of-sequence traffic). A run of
~650–700 is a good continuous clip.

**seq 08** — frame 0 has 1,128 moving points; the sequence has moving traffic
throughout the climb. Any stretch works now.

Other seq-00 frames with moving objects: 0 (88), 1044 (392), 1077 (77),
1148 (82).

---

## Reflectivity / lane markings

**seq 00, frames 4429–4460** — the richest lane paint in the dataset,
1,300–1,669 painted points per frame. Also **frames 0–11** (~1,000+ points).
`--color-by reflectivity`.

**seq 07 has zero lane-marking points** in all 1,101 frames — it is a
residential loop with no painted lanes. Expected, not a bug. **seq 08** has 17
frames with lane markings but only 1–3 points each — not usable. Use seq 00 for
the reflectivity story.

---

## Occupancy layers + foveation

**seq 00, frames 0–160** — the occupied-cell surface fills in, ring boundaries
and the blind cone track the vehicle, and the 5 / 10 / 20 / 40 cm cell-size
stepping across the rings is visible. `--color-by class` (or `groups`). The
`world/map/{occupied,free,unknown}` entities and the live memory panel are all
active.

**Dense-3D comparison** — `python -m vrgrid.dash.dense3d_comparison --seq 00
--frames 60` renders the variable grid beside a uniform 5 cm dense voxel grid
(reduced 20 m footprint), box counts in the overlay.

`--color-by` sweep (any mode) — seq **07**, anywhere. Flat, clean, dense scene.

---

## Quick reference — what to drive on the day

| to show | sequence | frames | notes |
|---|---|---|---|
| Ghost toggle (short, textbook) | 00 | ~5–20 | frame 10 = motorcyclist + pedestrian |
| Ghost toggle (map-level, strong) | 00 | 0–60 | `--show-ghosts` on/off differs by ~17 k cells |
| Ghost toggle (live, dense traffic) | 07 | ~650–700 | frame 674 has 3,302 moving points |
| Ghost toggle at elevation | 08 | any climbed stretch | now works — was the fixed bug |
| Reflectivity / lane paint | 00 | 4420–4460 | richest lane markings; also 0–11 |
| Occupancy layers + foveation | 00 | 0–160 | cells + rings + blind cone + memory panel |
| Dense-3D comparison | 00 | 0–60 | `vrgrid.dash.dense3d_comparison` |
| `--color-by` sweep | 07 | anywhere | |

---

## Full-sequence soak health

Every frame of every sequence, grid-wired pipeline, pre- and post-elevation-fix:

| | seq 00¹ | seq 07 | seq 08 |
|---|---|---|---|
| frames | 4541 / 4541 | 1101 / 1101 | 4071 / 4071 |
| crashes | 0 | 0 | 0 |
| NaN / Inf | 0 | 0 of 668 M values | 0 of 2.50 B values |
| peak RSS | ~150 MB | 158 MB | 145–163 MB |
| memory leak | none | none (warm-up only) | +0.28 MB/1000f — flat |
| per-frame (perc + step) | ~140 ms | 34–133 ms | 39–116 ms² |
| frames with cleanup inert (post-fix) | — | **0** | **0** (was 2,304 / 57 %) |

¹ seq 00 grid-wired full-sequence soak was not re-run for this document; the
number is from the Gate 3 verification run (60 frames) plus the perception-only
full-sequence soak.
² the wide seq-08 range is machine memory pressure during the post-fix re-soak,
not the engine — the same run's clean seq-07 pass sat at 34–43 ms.
