# Research Log

Append-only. Newest entries at the bottom. One entry per finding, with who and when.

Format:

```
## YYYY-MM-DD — <name>
**Module:** <research module>
**Finding:**
**Source:**
**So what:** what this changes in the build, if anything.
```

---

## 2026-08-31 — JP
**Module:** β — dynamics & segmentation
**Finding:** The standalone FRNet implementation in `src/perception/frnet/` (a
hand port with no mmcv/mmdet3d dependency) does not reproduce the trained
network. The published checkpoint loads cleanly — 413/413 tensors, no shape
mismatch — but the forward pass is wrong: the backbone uses `nn.LeakyReLU`
where `configs/_base_/models/frnet.py` in the FRNet repo trains with
`act_cfg=dict(type='HSwish')`; the FOV is fed as `fov_up=2.0 / fov_down=-24.8`
where training used `3.0 / -25.0`; and the test-time `RangeInterpolation`
densification (H=64, W=2048) is absent. Measured point accuracy vs GT on
SemanticKITTI seq 00: **16.3% (frame 43), 15.3% (frame 100)** — worse than
predicting "all road". Both frames collapse to ~50–58% *other-ground* where GT
has ~0%. Not frame-specific; systemic.
**Source:** FRNet-master repo configs (`configs/_base_/models/frnet.py`,
`configs/_base_/datasets/semantickitti_seg.py`); direct inference runs on the
pretrained checkpoint `frnet-semantickitti_seg.pth`.
**So what:** FRNet is dropped from the pipeline. Semantic class (19-class) now
comes straight from the SemanticKITTI raw `.label` files via
`semantics.semantic_labels()`, the same source and 19-class scheme as the
`moving-*` motion flag. Both semantic and motion labels are therefore ground
truth — disclose it plainly; it isolates the mapping contribution from
segmentation error, which is what a careful evaluator wants (master v4 §3.6,
risk register). The broken port is kept, flagged non-functional, so a proper
`mmengine`/`mmcv`/`mmdet`/`mmdet3d` install can swap real FRNet back in later if
time allows.

## 2026-08-31 — JP
**Module:** β — segmentation / perception front-end
**Finding:** Reflectivity normalisation eq (31), `rho_hat = I * r^2 / max(cos
theta_inc, 0.1)`, assumes the sensor reports raw received power. **KITTI's
Velodyne does not** -- its firmware already delivers a range- and
incidence-normalised reflectance-like quantity. Measured on label `road` (flat
asphalt) across sequence 00: `log(I)` vs `log(r)` slope = **0.01** (no range
trend), and median `I` stays ~0.25 while `cos(theta_inc)` falls 3x from 6 m to
17 m (no incidence trend). Applying eq (31)'s geometric terms to KITTI
re-injects a range trend that is not in the data: the `* r^2` term saturated
**62%** of ring-1 road pixels at byte 255 (ring 0: 0%), median rho8 255 vs
ring 0's 37 at the same raw intensity -- i.e. reflectivity carried zero
information past ~10 m. Caught by Aakash via a ring-by-ring per-cell analysis of
`test_lane_marking_reflectivity_separates_from_road`, which also showed the
original "1.46x median gap" was a pooled-all-pixels statistic, not the per-cell
quantity `fusion.py` aggregates (single-frame ground returns are n=1-2 per cell
at 5-10 cm, so there is no per-cell median/mean effect at all).
**Source:** direct measurement on `data/sequences/00` velodyne + labels;
`vrgrid.grid.lattice` ring/cell assignment.
**So what:** `reflectivity.normalise()` now defaults to `range_compensated=True,
incidence_compensated=True` for KITTI -> `rho_hat = I`, `rho8 = round(I * 255)`.
A raw-power sensor passes both False and gets eq (31) verbatim (still
implemented and tested). Result: 0% saturation in every ring, ring-0/1 road
median 64/64 (range-stable), lane-vs-road per-point ratio 1.36 (median) / 1.61
(mean) in ring 0. `incidence_cos()` (finite-difference range-image normal,
verified to 1e-4 against analytic planes) is still computed and returned for
the elevation-variance model (§3.2). **`docs/sih-math.md` §10.3 eq (31) should
note the raw-power assumption -- flag for Aakash.** `fusion.scatter()` is still
a stub on this branch and does not consume reflectivity yet.

## 2026-08-31 — JP (decision)
**Module:** β — dynamics & segmentation
**Decision:** Given the FRNet finding above, three options were on the table —
(A) use GT 19-class semantic labels from the `.label` files, disclosed;
(B) install the real `mmengine`/`mmcv`/`mmdet`/`mmdet3d` stack and run the
published checkpoint properly; (C) fix the standalone port (swap HSwish in, fix
FOV, add RangeInterpolation, audit the fusion wiring). **Option A is chosen and
FRNet is closed for the rest of the project.** Reasoning: FRNet is not one of
the five spine items (grid engine, reference map, ghost toggle, dashboard,
plan-regret curve); the risk register already pre-approved "ship with GT
semantic labels" as the fallback for this exact failure mode; and the mmdet3d
install chain is a known multi-hour Windows risk for something that is not
demo-critical. Option C is a reimplementation-vs-paper validation task that
could take a full day and still be subtly wrong — the brief's named failure
mode. This decision is not to be revisited mid-project. If a real FRNet is ever
wanted, Option B against the kept port scaffolding is the route, as a
post-submission or slack-time task only.
