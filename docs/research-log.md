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


## 2026-08-29 — Pratyushi
**Module:** R3 (Traversability, Evaluation & Novelty Claim)
**Finding:** Formulated and frozen the formal evaluation metric specifications and algorithmic pseudocode for Plan Regret $R(S)$, Discrete Fréchet Distance $d_F$, Coarsening-Justification Ratio $\rho = IL/\text{spread}$, and Dynamic Removal rates ($DR, SP, F$). Handed over `docs/eval-metric-specs.md` to Aakash to unblock `src/eval/plan_regret.py` and `src/eval/metrics.py`. Established the core invariant: both optimal reference path $\pi^*$ and candidate schedule path $\pi_S$ must be evaluated strictly on the 5 cm reference map $M^*$ so that unobserved obstacles / blurred kerbs result in infinite regret rather than false safety.
**Source:** `docs/sih-math.md` §8, §9, §10; `docs/eval-metric-specs.md`
**So what:** Unblocks Aakash (D1) to implement the evaluation harness and A* path regret scorer. Establishes the exact testable invariant that $R(S) \ge 0$ for all schedules and $R(S) \approx 0$ at our 8.94 MB operating point.

## 2026-08-29 — Pratyushi
**Module:** R3 (Traversability, Evaluation & Novelty Claim)
**Finding:** Delivered Day-3 Hard Gate Novelty Verdict: Evaluated Psomiadis et al. (ICRA 2024, arXiv:2309.13451) and Larsson et al. (RA-L 2021). Verdict: NO PREEMPTION. Their work targets communication bandwidth reduction across multi-robot systems via information-bottleneck (IB) abstractions and convex optimization decoders on generic 2D grids. In contrast, vrgrid addresses real-time 2.5D automotive LiDAR elevation mapping under a strict compile-time 8.94 MB memory bound ($O(1)$ per cell, zero runtime allocations), sensor-physics ring geometry ($s_{\text{rad}} \propto r^2$), exact Law of Total Variance split/merge round-trip idempotence, and decision-lossless evaluation using Plan Regret $R(S)$ scored against ground-truth 5 cm reference maps $M^*$ on real SemanticKITTI scans.
**Source:** Psomiadis et al. (ICRA 2024); Larsson et al. (RA-L 2021); `docs/novelty-verdict-psomiadis.md`
**So what:** Day 3 Hard Gate passed ahead of schedule. The novelty claim stands firm. Positioning established for Related Work section and panel defense.

## 2026-08-30 — Pratyushi
**Module:** R3 (Traversability, Evaluation & Novelty Claim)
**Finding:** Validated the 6-condition traversability bitfield against SALON (Sivaprakasam et al., ICRA 2025) and EVORA (Cai et al., IEEE T-RO). Confirmed that geometry must decide while semantics filters (e.g. potholes on drivable classes), and evidential confidence (`TRAV_CONFIDENCE`) must enforce a strict fail-safe against blind areas. Synthesized the one-sentence defence against DOGMa particle grids (Nuss et al. 2018): DOGMa requires millions of particles and tens of GB/s bandwidth vs our deterministic $O(1)$ range-image visibility cleanup under 8.94 MB. Extracted baseline multi-resolution metrics from RoadRunner M&M (Patel et al., RA-L 2024), establishing our 21.5× memory compression vs uniform 2.5D and 286× vs dense 3D voxels.
**Source:** SALON (ICRA 2025); EVORA (IEEE T-RO); RoadRunner M&M (RA-L 2024); `docs/traversability-and-baselines.md`
**So what:** Solidifies the theoretical ground for Section 3 (Traversability) and Section 5 (Experimental Baseline Comparison Table) of the final report.

## 2026-08-28 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Recursive 1D Kalman filter elevation updates per cell with range-dependent measurement variance $\sigma_m^2(r) = \sigma_0^2 + c \cdot r^2$ provide statistically optimal terrain height tracking for mobile robots.
**Source:** Fankhauser, P., Bloesch, M., Gehring, C., Hutter, M., & Siegwart, R. (2014). "Robot-Centric Elevation Mapping with Uncertainty Estimates." *International Conference on Climbing and Walking Robots (CLAWAR)*.
**So what:** Confirms that the Kalman elevation measurement variance model in `docs/sih-math.md` §3 matches canonical robotics literature.

## 2026-08-29 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Radial ground beam spacing grows quadratically with range ($s_{\text{rad}}(r) = \frac{r^2 \Delta\phi}{h_s}$), reaching $10.8\text{ m}$ at $50\text{ m}$. Consequently, a uniform $5\text{ cm}$ grid at $50\text{ m}$ is $99.87\%$ empty in a single frame. Far rings are populated over time via vehicle ego-motion ("Ring-Sweep Filling"). Potholes ($30\text{ cm}$) are physically undetectable beyond $r_{\max} \approx 8.3\text{ m}$.
**Source:** Derivation from LiDAR beam trigonometry on KITTI HDL-64E parameters; formalized in `docs/memo-r1-sensor-physics-and-ring-justification.md`.
**So what:** Provides the physical proof that coarsening far rings is not merely a memory optimization, but a physical necessity matching LiDAR sampling density. Sets the hard $8.3\text{ m}$ scope limit for negative obstacles.

## 2026-08-29 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Compiled an 8-way comparative taxonomy contrasting `vrgrid` against OctoMap (2013), MLS (2006), Droeschel (2014), Elevation Mapping (2014), Adaptive Patched Grid (2023), PCT (2024), and Wavemap (2023).
**Source:** `docs/prior-art-taxonomy-matrix.md`.
**So what:** Formally isolates `vrgrid`'s three defensible novelty claims: (1) joint range+semantic foveation under compile-time 8.94 MB SoA bounds, (2) variance-honest split/merge via Law of Total Variance, and (3) validation via downstream Plan Regret $R(S)$.

## 2026-08-30 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Point Cloud Tomography (PCT) slices 3D point clouds into parallel 2.5D elevation layers for GPU-accelerated traversability planning. While PCT represents a modern revival of MLS maps, its core contribution is parallel GPU planning rather than foveated spatial compression.
**Source:** Yang, T., Cheng, K., Xue, J., Jiao, J., & Liu, M. (2024). "Efficient Global Navigational Planning in 3D Structures based on Point Cloud Tomography." *IEEE/ASME Transactions on Mechatronics*, arXiv:2403.07631.
**So what:** Validates our critique that PCT is MLS reframed. Positions `vrgrid` as solving the orthogonal problem: foveated spatial compression under hard memory bounds rather than uniform tensor slicing.

## 2026-08-30 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Adaptive Patched Grid Mapping dynamically alters 2.5D cell patch sizes for automotive LiDAR, but merges child cells using naive inverse-variance averaging ($1/\sigma_p^2 = \sum 1/\sigma_i^2$). This drops the spatial between-cell variance term ($\sum w_i (\mu_i - \mu_p)^2$), creating artificial high confidence where cells straddle elevation steps (e.g., curbs).
**Source:** Wodtko, T., Griebel, M., & Buchholz, M. (2023). "Adaptive Patched Grid Mapping." *arXiv:2308.03416*, Ulm University.
**So what:** Identifies the critical mathematical error in modern adaptive grids. Proves why Aakash's (D1) implementation of the Law of Total Variance in `sih-math.md` §4 is mathematically necessary for obstacle safety.

## 2026-08-30 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Wavemap implements 3D multi-resolution volumetric mapping via Haar wavelets. While memory-efficient for 3D aerial robots, tree traversal creates irregular memory lookups and branch divergence on GPUs. For ground vehicles, $O(1)$ flat 2.5D ring buffers maximize memory bandwidth and provide planner-native queries.
**Source:** Reijgwart, V., Cadena, C., Siegwart, R., & Ott, L. (2023). "wavemap: Efficient Volumetric Hierarchical Occupancy Mapping." *Robotics: Science and Systems (RSS)*, arXiv:2306.01279.
**So what:** Supplies the formal justification for why `vrgrid` deliberately uses 2.5D foveated rings rather than 3D wavelet trees for autonomous driving.

## 2026-08-30 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Maximum Mipmaps build hierarchical max-reduction pyramids over height fields for fast ray-stepping in terrain rendering.
**Source:** Tevs, A., Ihrke, I., & Seidel, H.-P. (2008). "Maximum Mipmaps for Fast, Accurate, and Scalable Dynamic Height Field Rendering." *ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games (I3D)*.
**So what:** Confirms the graphics lineage for Shrestha's (D3) conservative pyramid (§7.2), providing guaranteed zero-false-negative traversability ray-stepping for safety.

## 2026-09-01 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Formulated the definitive architectural comparison between 2.5D foveated ring grids and 3D volumetric hierarchies (OctoMap, Wavemap). Proved that for ground robots operating on 2D surface manifolds, 3D trees waste memory on empty air ($>98\%$), introduce GPU warp divergence via tree pointer chasing, and lack deterministic compile-time memory bounds. Flat 2.5D ring arrays deliver $O(1)$ indexing, 100% coalesced GPU memory access, and native 2D traversability bitfields under an 8.94 MB compile-time bound (~286x smaller than 3D voxels).
**Source:** `docs/memo-r1-day4-rings-vs-octree.md`.
**So what:** Unblocks the Day 4 justification milestone and provides the submission-ready defense text for the report.

## 2026-09-03 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Synthesized and finalized the complete Related Work section (~720 words) with verified academic citations across volumetric 3D grids, elevation/MLS mapping, foveated clipmaps, and plan-sensitivity evaluation.
**Source:** `docs/related-work-final-section.md`.
**So what:** Fully completes the Day 6 writing milestone. Ready for direct inclusion in the final SIH submission report without late-stage drafting.

## 2026-09-04 — Srinivas
**Module:** R1 — Representation & prior art
**Finding:** Formulated the master defense Q&A playbook addressing the top 6 potential panel counter-arguments (Droeschel/clipmap heritage, uncompression myth, 3D voxel alternatives, near-field resolution gain, pothole range limits, and dataset separation).
**Source:** `docs/defense-rehearsal-playbook.md`.
**So what:** Fully arms the presentation team and R1 lead with mathematically derived, unified answers for the live jury defense.
---
---
## 2026-08-28 - Hriday
**Module:** R2 (Dynamics & Segmentation)
**Finding:** FRNet (19-class SemanticKITTI checkpoint) requires a 64x512 spherical projection. Standard model is 10M params (73.3% mIoU). Fast-FRNet fallback is 7.5M params. Dynamic classes are explicitly mapped to IDs 252 (moving-car), 253 (moving-bicyclist), 254 (moving-person), 255 (moving-motorcyclist).
**Source:** FRNet GitHub (Xiangxu-0103/FRNet) & paper (arXiv:2312.04484).
**So what:** D2 (JP) must build a projection with an inverse index using exact spherical math ($r, \theta, \phi \rightarrow u, v$). **Use Velodyne FOV bounds for the projection: +2° (top) to -24.9° (bottom).** If the main pipeline OOMs (runs out of memory), swap to the 7.5M Fast-FRNet checkpoint immediately. These four moving IDs are the sole triggers for ghost removal.
---
---

## 2026-08-30 - Hriday
**Module:** R2 (Dynamics & Segmentation)
**Finding:** DynamicMap Benchmark (Zhang et al., ITSC 2023) defines the primary evaluation metrics for ghost removal:
1. Preservation Rate (PR / SP): $\text{PR} = \frac{|\mathcal{S}_{\text{preserved}}|}{|\mathcal{S}_{\text{groundtruth}}|}$. Measures what fraction of true static points are retained (guards against erasing walls/fences).
2. Dynamic Removal (DR): $\text{DR} = \frac{|\mathcal{D}_{\text{removed}}|}{|\mathcal{D}_{\text{groundtruth}}|}$. Measures what fraction of dynamic points are successfully filtered out.
**Source:** Zhang et al., "A Dynamic Points Removal Benchmark in Point Cloud Maps," ITSC 2023 (KTH-RPL).
**So what:** D3 (Shrestha) must implement PR and DR exactly as defined above for the evaluation scripts and dashboard. **Critical defense note:** Zhang et al. evaluate offline global map cleaning, whereas our engine runs an online rolling local map. Our metrics must be explicitly contextualized against this constraint during the presentation.

---

## 2026-09-01 - Hriday
**Module:** R2 (Dynamics & Segmentation)
**Finding:** Sub-cloud range representations drastically outperform full-sweep views in memory-constrained settings. FLARES (Bosch, 2025) demonstrates that lower azimuth resolution with sub-clouds (64×512) improves both runtime and segmentation accuracy compared to full 64×2048 scans. Furthermore, BeautyMap (RA-L 2024) shows that range-visibility filtering causes over-clearing of thin geometry unless stabilized by static restoration / ground encoding.
**Source:** FLARES (arXiv:2502.09274, Feb 2025); BeautyMap (RA-L 2024).
**So what:** Locks D2's range image input to 64×512 sub-clouds. If our visibility cleanup produces over-clearing on thin static structures, we will implement BeautyMap's coarse-to-fine restoration mechanism as a mitigation.

---

## 2026-09-03 - Hriday
**Module:** R2 (Dynamics & Segmentation)
**Finding:** Synthesis of baseline benchmarks and related work for the final submission report:
* *Segmentation:* Modern architectures shift from 3D voxelization to efficient 2D frustum-range representations. FRNet (Xu et al., IEEE TIP 2025) achieves state-of-the-art efficiency (~5× faster than voxel baselines) while maintaining 73.3% mIoU on SemanticKITTI (19 classes), optimized by FLARES sub-cloud processing.
* *Dynamic Removal:* Conventional LiDAR MOS relies on multi-scan residual subtraction (LMNet). Geometric approaches like ERASOR (pseudo-occupancy) and DUFOMap (single-parameter ray-casting free space) provide non-learning alternatives. Our architecture combines lightweight semantic masking with localized rolling-map visibility cleanup, balancing latency against offline cleaners (Removert, Dynablox, BeautyMap).
**Source:** Synthesized from Tier 1 & Tier 2 R2 bibliography (FRNet, LMNet, ERASOR, DUFOMap, BeautyMap, DynamicMap).
**So what:** Completes R2 Day 5 (baseline numbers) and Day 6 (related work synthesis) deliverables for direct integration into the SIH submission documentation.

---
## 2026-09-02 — Pratyushi
**Module:** R3 (Traversability, Evaluation & Novelty Claim)
**Finding:** Drafted and delivered publication-grade Sections for the final SIH26053 submission: Section 2 (Related Work covering foveated clipmaps, 2.5D elevation fields, and decision-sensitive abstractions) and Section 4 (Decision-Theoretic Evaluation covering Plan Regret $R(S)$, Coarsening Ratio $\rho$, and Dynamic Removal $F$-score). Verified all 12 literature citations against peer-reviewed proceedings (ICRA, RA-L, IROS, TOG, IJRR, SIGGRAPH).
**Source:** `docs/draft-report-sections.md`
**So what:** Fulfills the Days 5–6 writing deliverable for Track γ. Ready for direct inclusion in the final technical report and submission slide deck.

---
## 2026-09-02 — Aakash
**Module:** D1 — Evaluation harness, reference map

**Finding:** Triage items #1 and #2, both landed on me. #1 is already closed and #2 was not what it looked like.

**#1, elevation / ghost removal — no scope decision needed, it is fixed.** The triage list is timestamped 04:37 on 1 Sep; Shrestha's fix landed at 10:52 the same day (`337bb30`) and the list has not caught up. `quantise_height` takes a `datum_m`, `MapEngine._track_datum` slides the 8 m band in whole 1 m steps to follow the vehicle, heights are stored relative to that datum, and `_centres` takes a 3-vector ego to hand `visibility_cleanup` the vehicle-frame z its contract asks for. **My answer to the scope question is that it does not arise:** the fix keeps the band 8 m wide, so the dense-3D baseline in `dashboard/_config.py` counts the same voxels and the headline memory ratio is untouched — which is exactly why widening the clamp would have been the wrong fix and moving it was the right one. `test_the_ghost_clears_at_any_vehicle_elevation` is parametrised at 0, −5.8, 6.0, 12.0 and 39.0 m: seq 07's floor and seq 08's hill. **No demo routing is needed and seq 08's climbed section is safe on stage.** Verified rather than taken on trust — I ran it.

Two other items on that list are also stale: the "two 15 MB per-frame allocations in `src/grid`" were closed on 1 Sep (`983f2fb`, 8.15 → 1.31 MB/frame), and the visibility scratch cap is the same commit.

**#2, plan regret on real data — the blocker is not plan regret.** `reference_map.build()` is the only path from SemanticKITTI to M*, and **it had never been executed by anything.** It raised `ValueError: too many values to unpack` on its own first line: `loader.scans` yields three values and it unpacked two. Behind that were two more, both of which would have produced a plausible map rather than an error — it passed `poses[i]` straight through (a Camera-0 → World_cam row, not vehicle → world: the 90° permutation `docs/frames.md` exists to prevent, applied to the artefact every metric is measured against) and it handed the loader's (N, 4) **sensor**-frame array to a function wanting (N, 3) in the **vehicle** frame (an intensity column read as a coordinate, every return 1.73 m under the road). Nothing caught it because every test and every script calls `build_from_scans` directly, and the only caller of `build` needs the download.

It does not need the download. Since 1 Sep `eval/synthetic.write_sequence` writes the layout `perception.loader` reads, so the whole real path now runs against a scene whose surface is known analytically — a stronger check than the real data gives on its own, because the heights can be asserted rather than eyeballed. `scripts/build_reference_map.py` builds and caches M* for a sequence, and exits 2 with the path it looked in when there is no data.

**⚑ A guard that could not fire.** Writing the test that proves `build` rejects a camera-convention pose, it did not raise — and the reason generalises past this bug. **A KITTI `poses.txt` starts at the identity by construction**, so on frame 0 the wrong composition and the right one agree: raw sensor points through an identity pose stay in the sensor frame, which is x-forward, y-left, z-up with the ground at −1.73 m, inside the guard's 2 m tolerance. `assert_world_is_z_up` on frame 0 — which is what `run_sequence` did, and what I added on 1 Sep — would have passed every real sequence regardless of convention. `FrameGuard` now looks twice: frame 0, and the first frame at least 10 m from the start, where the conventions are unmistakably apart. Then it stops costing anything.

Also generalised `costmaps_for` to place the planning window at the vehicle's final `(x, y)` rather than at `(x, PLAN_Y0_M)` about the world origin. That is only the vehicle's lane while the trajectory is a straight line along +x, which is true of the synthetic sequence and of no real one: on a sequence that turns, the window would have stayed near the origin while the vehicle drove away, and the regret would have been measured over ground neither map ever saw — coming out as a confident zero. Worth flagging because that failure produces a *better*-looking number than the truth.

**Source:** `src/eval/reference_map.py` (`build`), `src/eval/harness.py` (`FrameGuard`), `scripts/build_reference_map.py`, `scripts/eval_synthetic.py` (`costmaps_for`), `tests/test_build_reference_map.py` (5), `tests/test_reference_map.py` (+3), `tests/test_frame_convention.py` (+4).

**So what:** M* for 07/08 is now one command and the command is tested, so the download is the only thing left on that path — it is not an architecture gap. What remains genuinely open for a real regret number is the planning query itself: `PLAN_LANE_CELLS` runs the path where none of the scene's hazards are (Shrestha's `regret_plot.py` finding, still unresolved), and on a real sequence the start/goal need to come from the trajectory rather than from two constants. That is the Pratyushi half of item #2 and it is a smaller thing than "no real evidence exists" suggests. 508 passed, 27 skipped.

---

## 2026-09-01 — Aakash
**Module:** D1 — Evaluation harness, synthetic sequence

**Finding:** Closed Shrestha's ask that `eval/synthetic.write_sequence` write the layout `perception.loader` reads, so his duplicate writer `tests/kitti_layout.py` could be deleted. He estimated two lines — poses to `poses/<seq>.txt`, add a `calib.txt`. It was five conventions, and the three he did not see are each a silent wrong answer rather than a crash:

* **frame** — a `.bin` holds SENSOR-frame points and the vehicle origin is 1.73 m below the laser. This wrote vehicle-frame points into one, i.e. every road return 1.73 m underground.
* **labels** — a `.label` holds RAW SemanticKITTI ids. This wrote 19-class learning ids, which are read back as raw and collide *inside* the valid range: 9 is unmapped, 10 is `car`, 11 is `bicycle`. The synthetic road arrived as ignore and the parking as car. It had been invisible because `harness.learning_ids` auto-detects, and on the default path the moving car is stripped before it looks, dropping the max under `CLASS_MAX` so the ids passed through untouched. Correct by coincidence.
* **poses** — a `poses/<seq>.txt` row is Camera-0 → World_cam. This wrote vehicle→world rows, which through `vehicle_to_world` come out permuted by 90°: exactly the failure `assert_world_is_z_up` was added on 1 Sep to catch.

`read_sequence` now composes through JP's `vehicle_to_world` rather than assuming, which needed two additive seams in `perception/`: `loader.read_calib(path)` (the parser without the DATA_ROOT lookup) and a `tr=` override on `vehicle_to_world`. One frame convention, two callers. `tests/kitti_layout.py` is deleted and `test_loader_path.py` runs on the analytic scene, which is a beam-model sweep rather than uniform samples — so it now exercises the sampling density the ring schedule was actually derived from.

**⚑ The correctness finding, and it moves the §8.2 figure.** Chasing an FOV assertion — 788 returns a sweep outside the sensor's own −24.8° floor — turned up a sign error in the beam-ground intersection. The sensor's height above a surface at elevation `z` is `(h_s − z)`; the sampler used `(h_s + z)`. **The two agree exactly on flat ground, which is why five days of tests passed over it**, and on a feature they disagree by about `2z/tan|φ|` — 1.7 m radially at the steepest beam. Fixing the sign then exposed that the "one correction step" is a fixed-point iteration with multiplier `(dz/dr)/tan|φ|`, which is 2.0 on the 6% ramp: it diverges, and returns came back at elevations down to −85°. `_beam_range` now bisects, which has no such condition.

The consequence is bigger than the residual: **across a whole sequence the old sampler returned not one point below −30 cm.** The 40 cm pothole at (18, 0) — the scene's only negative obstacle, and the reason §1.4 is in the document — had never once been observed as a hole at any range. R(S) was being read off a lane with no hazard in it, and it read 0.000. It now resolves the hole from 8 m and not from 14 or 16 m, where beams land inside the footprint and come back at rim height: eq. (6) doing what it says, and now a named test rather than a derivation.

Ring 1 RMSE without the transient layer re-measures 0.41 → 12.72 cm. The confound table in `plan_regret.py` no longer reproduces at all — the window is 99.1% common support now — so it is re-measured, kept, and reproducible from `scripts/eval_synthetic.py --confound`.

**⚑ Correction, same day, to the paragraph this entry originally carried here.** I recorded that the pothole fix moved plan regret from 0.000 to 2.389 and wrote that into the commit message, `scripts/regret_plot.py` and §1.4. **It did not.** R(S) is measured down a lane six cells off the centreline and the pothole is on the centreline, so putting it into the map changes no decision there. What actually moved R(S) was the second bug below, which the label correction in this same commit made reachable. Both fixes in, R(S) is back to Shrestha's original pattern to the third decimal. The pothole finding stands on its own terms — the old sampler genuinely never observed the scene's only negative obstacle as a hole — it is simply not what moved the figure, and the two arrived in the same hour.

**Source:** `src/eval/synthetic.py`, `src/perception/{loader,transforms}.py`, `tests/test_synthetic_layout.py` (10 tests), `tests/test_loader_path.py`, `docs/sih-math.md` §1.4 and §12, `scripts/eval_synthetic.py --confound`.

**So what:** One sequence writer, one frame convention, and the real-data path is exercised end to end on a scene with a negative obstacle actually in it. Gate 4's curve is unchanged and Shrestha's diagnosis of it is unresolved: R(S) is 0.000 for both frozen schedules and for uniform 20/40/80 cm, with a single spike at uniform 10 cm (1.389 → 1.536), and the frame-count dependence he flagged is larger rather than smaller — at 24 frames the frozen schedules go to 1.450 where they used to go to 0.207. `PLAN_LANE_CELLS` is deliberately untouched. **Until the query is posed so it has to decide about the kerb or the pothole, the honest statement is that the money plot does not yet show what it was built to show**, and that is a Gate 4 item rather than a Day 6 one.

---

## 2026-09-01 — Aakash
**Module:** D1 — Traversability (§7.1), Day 4

**Finding:** The §7.1 bitfield's six conditions, the central-difference gradient, live schedule selection from `configs/schedule_*.yaml`, and `validate()`'s two checks were all already in place. What was not was bit 4. **`grid/traversability.py` held a hand-written `CLASS_IDS` table that was off by one for every class** — it began `unlabeled: 0, car: 1, …` where SemanticKITTI's learning map begins `car: 0` and puts `unlabeled` at 19.

So `drivable_classes: [road, parking, sidewalk, other-ground, terrain]` resolved to the ids of **{parking, sidewalk, other-ground, building, pole}**. `road` and `terrain` were not drivable; a building facade and a lamp post were. Bit 4 costs `w_class = 3.0` rather than blocking, so nothing crashed, no path failed and no test noticed — the entire road surface just quietly carried a penalty and the planner preferred the kerb line.

**It survived five days because two errors cancelled.** The synthetic scene wrote learning ids 9/10/11 directly, and those three fall inside the *wrong* table's drivable set by coincidence. Correcting that scene to raw ids earlier the same day put `road` = 8 into the map and made this reachable — which is why R(S) jumped to 2.389, and I initially and wrongly attributed that to the pothole fix in the same commit. With both fixed, R(S) returns to 0.000 / 0.000 / 1.536 / 0.000 / 0.000 / 0.000 at 12 and 14 frames.

There were **three** copies of the 19-class ordering — `configs/frnet.yaml`, `perception.semantics.FRNET_CLASS_NAMES`, and this one. There are now two, and the surviving source is the config: `schedule.load_class_names()` reads it, so `grid/` gets a dataset fact without importing `perception/` (CLAUDE.md keeps the core free of that dependency). `traversability.class_ids()` and `gate.py` both go through it.

**Source:** `src/grid/traversability.py`, `src/grid/schedule.py` (`load_class_names`), `src/grid/gate.py`, `tests/test_traversability.py::test_the_class_table_is_the_one_the_labels_use`, `::test_the_road_is_drivable_and_a_building_is_not`.

**So what:** Every §7.1 number measured before today was computed with the road penalised and buildings drivable, so the traversability layer and anything downstream of it — plan regret, the corridor rule, the dashboard's traversability view — need re-reading. The new test asserts against `semantics.semantic_labels` end to end rather than against a literal list, so the next table shift fails loudly. **The general lesson is the one from the `>> 4` sweep two commits ago:** a constant copied into a second file is not a duplicate, it is a future disagreement, and both times it was found only because something unrelated forced the two copies into contact.

---
## 2026-09-01 — Aakash
**Module:** D1 — Grid, fusion, evaluation
**Finding:** Ratified and applied the §10.2 class-byte re-split (Gate 3, item 3): **5-bit candidate | 3-bit counter**, replacing 4 | 4. The learning set is 20 ids (0–19) and a 4-bit candidate held 16. The shortfall reached three independent places, each of which read as working: the semantic gate could never match `pole` (18) or `traffic-sign` (19) — the two classes semantic refinement exists for — so it fired on nothing; `terrain` (17) is one of five `drivable_classes` the §7.1 predicate consults on every cell; and the first real frame raised, around which two *disagreeing* stand-ins had grown — `% 16` in the eval harness (mapping `terrain`→`car`, so drivable verges arrived as blocked cells) and `clip(0,15)` in the frame loop (everything >15 → `vegetation`). Both are now deleted. Applying a two-constant change surfaced **six further copies of the field width** — `>> 4` in `traversability.py`, `gate.py`, `query.py`, and hand-rolled `(id << 4) | 5` in two test fixtures and a script — each of which would have read a 4-bit field out of a 5-bit byte and marked every drivable cell untraversable. All now route through `pack_class`/`unpack_class`.

**⚑ Correction to §10.2, and it changes a report sentence.** The section claimed the majority guarantee "does not depend on where the counter saturates". That is false, and it was false at 4 | 4 as well — Boyer–Moore's proof assumes an *unbounded* counter, and a saturating one discards the evidence the proof rests on. Counterexamples at both widths: cap 15 loses a 32-of-63 majority; cap 7 loses a 16-of-31 majority. What the cell provides is a **time constant, not a theorem** — the one-byte version is exactly textbook Boyer–Moore on any sequence whose running excess stays within `C`, and a cell changes its mind after more than `C` net contradicting observations. `C = 7` re-labels about twice as fast as `C = 15`, which for a rolling local map with dynamics is arguably the better default, but it is a tuning claim.

**Source:** `src/grid/fusion.py` (§10.2 header), `docs/sih-math.md` §10.2 and §7.1, `tests/test_fusion.py::test_saturation_is_what_bounds_the_guarantee`.

**So what:** Unblocks the first end-to-end frame on JP's GT labels — 19-class frames now fuse without clipping (`test_a_realistic_label_set_fuses_without_clipping`, on the real loader path). Revives the semantic gate's class criterion and makes `terrain` storable for §7.1. **The report must not say "guaranteed majority" without the running-excess condition attached**; the honest sentence is the time-constant one. Cell struct unchanged at 12 bytes, so no memory figure moves.

---
## 2026-09-01 — Aakash
**Module:** D1 — Grid, lattice, fusion
**Finding:** Gate 3 item 2 closed. Point→slot binning now has one owner, `grid.lattice.bin_points`, and the four hand-rolled copies (`fusion.scatter`, `grid/transient.py`, `run/engine.py`, `scripts/timing_table.py` — four spellings across three directories) are deleted. **The rewrite has no ring loop at all.** The obvious shape is a pass per ring over the points falling in it, which is what all four copies did; it needs the selected world coordinates compacted into a preallocated buffer, and numpy will not do that without allocating — `np.compress(..., out=)` still built 1.54 MB of internal index at 96,000 selected points. So the ring became a per-*point* attribute: `k`, `side`, `x0`, `y0`, `offset` are gathered by ring index and the whole sweep is binned in one pass of full-length ufuncs.

Measured, 120,000 returns, same machine: **6.962 MB/frame → 0.002 MB/frame, and 13.64 → 12.35 ms p50** (9% faster — one pass beats four plus four compacting copies). `fusion.occupancy_state` gained an `out=`/`scratch=` path: **8.19 MB/call → 0**, and even the unpreallocated path fell to 2.73 MB, because `np.where` picking between two Python ints chooses int64 and one int64 array over 910,000 slots is 7.28 MB — for a uint8 answer.

Whole frame, from `scripts/timing_table.py --alloc`: **8.15 → 1.31 MB/frame**, p99 **74.7 → 49.4 ms**. `engine.step()` peaks at 1.15 MB and is now under a flat cap rather than one that subtracted the two grid allocations.

**⚑ Two allocations that no profile names.** `np.take(table, idx, out=)` builds a full-length bounds-check array under its default `mode="raise"` — 0.96 MB a call, six calls a frame; `mode="clip"` is allocation-free and 5× faster, and is safe here only because the ring index is explicitly clamped first. And `int64 += bool` casts through numpy's fixed 64 kB internal buffer; a masked scalar increment avoids it.

**Source:** `src/grid/lattice.py`, `src/grid/fusion.py`, `tests/test_bin_points.py` (32 tests), `tests/test_engine.py::test_the_two_grid_allocations_stay_fixed`.

**So what:** "No allocation inside the frame loop" is a hard invariant in CLAUDE.md and a sentence in the report; it was false by 15.15 MB a frame and is now true to within ~1 MB of per-call bookkeeping. `bin_points` is pinned bit-identical to `ring_of` + `i_ring` + `RingBuffer.flat_slot` over both frozen schedules and four speeds — including `5/10/50`, whose ratios are 2 and 5 and which is the only thing that catches a power-of-two assumption. The new declared footprint is 3.75 MB of binning scratch, replacing scratch the frame loop already held.
