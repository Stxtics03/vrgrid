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
## 2026-09-03 — Aakash
**Module:** D1 — Information-loss metrics (§9.2, §9.3), follow-up

**Finding:** A second, independent trace of the §9.2 ring-scoring bug came back the same day the fix landed. Same mechanism, confirmed from the other direction, plus one concrete cell — a ring-2 slot scored at 9.5 m from the vehicle on data written when that ground was 25–50 m out, charging 350.7 cm of error to ring 2. Stale share on longer runs: 13–38% per ring at 40 frames, near half of ring 2 by 80 frames, scaling with distance driven. All consistent with what was measured here (19%/21% at 22 m, 20%/26% at 46 m) and with the fix already in `_ring_cells`.

**⚑ Two corrections to yesterday's entry, one of them mine and wrong.**

**1. The "hard forward edge near x = 54 m" was not the reason ring 3 shows 13 migrated cells, and there is no such edge.** The synthetic scene follows the vehicle: world forward reach is 51.4 m at frame 0, 71.5 m by frame 24, 101.5 m by frame 36. The real cause is the terrain itself — flat to x = 30 m and then a 6% ramp — and a rising surface closes the forward horizon. Forward returns past 50 m appear at frame 0 and never again, so ring 3's forward band is written exactly once. The conclusion is unchanged (ring 3's annulus is lateral and rear, where a straight drive leaves nothing behind) but the stated cause was wrong and is corrected in §9.2, `metrics.py` and PR #31.

**2. "Migration" was read as cells moving between ring buffers.** They do not: the buffers are static and world-anchored and a cell never changes ring. What moves is the vehicle, and with it which ring is *responsible* for a place. Reworded everywhere it appears, because the misreading cost a reviewer a full re-derivation.

**⚑ THE SIGN OF THE CORRECTION IS UNVERIFIED AND NO FIGURE FOR IT IS RECORDED.** The synthetic measurement has the fix *lowering* RMSE (ring 1 0.40 → 0.37, ring 2 0.37 → 0.32). **It could not be checked against real data here**: `VRGRID_DATA_ROOT` is unset, `data/` holds only its README, and no `M*` artefact exists for either sequence — which is what known-limitation 2 already says. Until it is reproduced, **no per-ring RMSE figure may be quoted as improved or worsened by this fix**. The mechanism, the population size and the schedule asymmetry are settled; the sign is not. ρ moves by up to 0.06 per ring here, which is real movement rather than none, but not enough to change what ρ says.

*(⚑ Correction, same day, to the paragraph this entry originally carried here. It recorded the external seq 07/08 measurement as "RMSE understated by 3–12% across rings 1–3" and ρ moving 0.034, and I wrote both into `metrics.py`, §9.2 and `known-limitations.md` on that basis. **Both were withdrawn by their author within hours** — restated as "no consistent bias, −40% to +21% depending on ring, sequence and frame count" and up to −12.5% on ρ, with the traced 350.7 cm example retracted as a separate `M*` defect and restated as 169.5 cm. None of it is reproducible in this repo. Removed rather than swapped for the replacement: a withdrawn number quoted as if it stood is worse than no number, and the replacement comes from the same unreproducible source. **The lesson is mine, not theirs — I put an outside figure into four project files without being able to run it.** What went in should have been "an external measurement disagrees, direction unknown, unreproducible here", which is what all four now say.)*

**⚑ One claim in the incoming report is checkably wrong, and it is the one its recommendation rests on.** "Every RMSE number in known-limitations.md right now is flattering us" — that file contained no RMSE numbers at all, and no §9.2 or §9.3 numbers. Its two items were the ghost-removal elevation limit and the plan-regret status. The underlying point was still right (the per-ring accuracy claim is load-bearing and was undisclosed there), so §9.2 is now limitation 3 — but written as "fixed, sign unverified" rather than as a standing bias, because that is what the evidence supports.

**⚑ Separate finding, confirmed and acted on: ρ's denominator is estimated from very few cells.** `block_stats`'s `n` counts the 5 cm cells of `F(c)` that `M*` observed, capped at `k²`. Median coverage, 12-frame synthetic: **1.00 / 0.25 / 0.06 / 0.02** for rings 0–3 — ring 3's sub-cell terrain variability comes from roughly one reference cell in sixty-four. A spread estimated from two points is biased low and ρ divides by it, so ρ on the coarse rings is biased **high**, the conservative direction for a number we want near 1. `coarsening_ratio_per_ring` already drops `n_ref ≤ 1`; at `k = 8` that guard admits a spread from two cells of sixty-four. Disclosed rather than corrected, and **coverage is now a column in the per-ring table, printed next to ρ**, so the two cannot be read apart.

**⚑ Two numbers were attributed to us that we have never published, and both are worth killing before they spread.** The revised report reconciles against "your headline 1.45 median ρ" and against a "published §2b". **There is no §2b in this project**, and **no median ρ of 1.45**: the only 1.450 on record is a **plan-regret `R(S)`** at 24 frames (`scripts/regret_plot.py`; this log, 2 Sep), sitting in the paragraph that concludes the money plot does not yet show what it was built to show because `PLAN_LANE_CELLS` runs the path six cells off the hazards. ρ and `R(S)` are different metrics on different scales and must not be reconciled against one another. Our actual ρ is 1.42 / 1.29 / 1.20 per ring on the synthetic sequence, and `master-v4.md` states the claim qualitatively as "ρ ≈ 1 is optimal" — there is no published ρ figure to shift.

**Source:** `src/eval/metrics.py` (`footprint_coverage_per_ring`, corrected module note), `src/eval/harness.py` (`Result.coverage`, `format_result`), `docs/sih-math.md` §9.2, `docs/known-limitations.md` item 3, `tests/test_metrics.py::test_coverage_says_how_little_of_a_coarse_footprint_M_star_saw`.

**So what:** The fix stands and is already in PR #31; what changed today is what may be *said* about it. The report must not claim the correction improves per-ring RMSE until it is re-run on 07/08, and ρ must not be quoted without its coverage column. Building `M*` for 07/08 (known-limitation 2, item 1) is now blocking a second claim, not one.

---
## 2026-09-03 — Aakash
**Module:** D1 — Information-loss metrics (§9.2, §9.3)

**Finding:** §9.2 scored each ring against a reference holding observations that ring never received. `C_L` was read as "every cell in ring L's buffer", and that buffer is a square of half-width `R_L` — it physically covers the hole the finer rings serve. `ring_of` hands a place to the *finest* ring containing it, so ring L only ever receives returns from the annulus `[R_{L-1}, R_L)`. The vehicle drives, that annulus sweeps outward, and every cell it leaves behind keeps its last far-range value for as long as it stays in the window: nothing clears it (a toroidal shift clears only the edge coming into view, §2.4) and nothing reads it, because `query()` routes that place to a finer ring now. So the metric asked a height frozen at 60 m to match an M* that went on accumulating the close-range returns the cell never got.

Stale share of each ring's scored population, 5/10/20/40 on the synthetic sequence: **19% of ring 1 and 21% of ring 2 after 22 m driven, 20% and 26% after 46 m.** Dropping them, before → after:

| ring | RMSE_L cm | IoU | fill |
|---|---|---|---|
| 1 | 0.40 → **0.37** | 0.59 → **0.68** | 0.59 → **0.68** |
| 2 | 0.37 → **0.32** | 0.39 → **0.72** | 0.39 → **0.72** |
| 3 | 0.33 → 0.33 | 0.23 → **0.83** | 0.23 → **0.83** |

Occupancy and fill move furthest because the hole is full of ground M* knows about and the ring never wrote — **ring 3's far-field fill rate was being reported as 0.23 when the ring fills 0.83 of what it actually answers for**, which is the §1.3 ring-sweep claim being understated by a factor of three and a half. `fill_rate_per_ring`'s own docstring already warned about "the hole covered by the finer ring" and thought `n_ref > 0` handled it; it does not, and cannot — the reference has plenty of returns under the hole, which is the whole problem.

**⚑ The confound was asymmetric across the schedules §8.2 compares, which is why it mattered more than its size.** A uniform baseline has one ring, `ring_of` always answers 0, and nothing can migrate out from under it. The money plot was charging the foveated schedules for stale memory and the uniform grids for none. Worst-ring RMSE before → after: 5/10/20/40 **0.40 → 0.37**, 5/10/50 **0.46 → 0.37**, uniform 10 cm 0.35 → 0.35, uniform 20 cm 0.41 → 0.41. Only our own schedules move, and 5/10/50 stops reading worse than uniform 20 cm on a difference that was never real.

**⚑ Ring 3 shows 13 migrated cells, not a fraction, and that is this scene rather than the effect.** The synthetic terrain has a hard forward edge near x = 54 m, so after the first frames nothing enters ring 3's forward band and its 50–100 m annulus is populated laterally and to the rear, where a straight-line drive migrates nothing. On a real sequence the far band is fed continuously for kilometres and ring 3 is the ring that carries most of this. **The synthetic numbers above are a floor on the effect, not an estimate of it.**

**⚑ What is deliberately not fixed.** A cell the ring still serves is scored against every return in its footprint, including returns fired from outside the ring's band — ground behind the vehicle was driven over at 2 m before it fell back to 40 m, and M* kept all of it. Fixing that needs a range-stratified M*: (n, Σh, Σh²) per band per 5 cm cell, ~4× its memory and 4× its summed-area tables, on an array already at 205 MB for a 12-frame synthetic scene. Measured against a reference rebuilt from band-restricted returns, the difference is **at most 0.05 cm** (ring 1: 0.37 → 0.32, ring 2: 0.32 → 0.30, ring 3: 0.33 → 0.32) — under the 0.29 cm quantisation floor §9.3 already puts on these numbers. Second order to the migration confound, and the memory is not worth spending on it *on this sequence*; it wants re-measuring on real data, where the rear band has a kerb in it and this one has smooth analytic terrain.

**Source:** `src/eval/metrics.py` (`_ring_cells`, `_cell_centres_m`), `docs/sih-math.md` §9.2 correction, `tests/test_metrics.py::test_a_ring_is_scored_only_where_it_still_answers`, `::test_the_scored_set_is_the_set_query_routes_to`, `::test_cell_centres_agree_with_the_frame_path`, `::test_a_single_ring_schedule_gives_up_nothing_to_the_band_filter`.

**So what:** §8.2's money plot is comparable across schedules for the first time — the axis it varies is now the schedule and not how much stale memory each schedule happens to carry. Every per-ring number measured before today needs re-reading, and the far-field fill rate needs re-reading hardest. The predicate is `ring_of` on the cell centre, the same function `query()` routes with, so the scored set cannot drift from the set the map answers with; that identity is pinned against `slot_of` rather than asserted.

---
## 2026-09-02 — Aakash
**Module:** D1 — Plan regret (§8.1 eq. 23)

**Finding:** Two independent defects in eq. (23), both reported from outside and both confirmed. `R(S) = J_M*(π_S) − J_M*(π*)` scores both paths on M*, so the only thing M_S may contribute is the *path* — and a path is meaningful only if the two costmaps describe the same world at the same scale. They did not.

**1. The fill-rate confound, and a diagnostic that hid it.** `_cost_from_bits` charges `w_unknown` for `unknown | TRAV_CONFIDENCE`. The `--confound` diagnostic I added on 1 Sep read `CostMap.unknown` alone — the smaller term by two orders of magnitude, because a cell can be observed and still sit below `n_min`. It printed **0.0% where the real figure was 100.0%**. The tool added specifically to make this visible reported that the problem was absent, which is worse than having no tool.

The confound itself was in `costmap_from_gridmap`, which OR-ed the confidence bit over the sub-cells of each planning cell. A 25 cm planning cell covers 25 map cells of a 5 cm ring; at ring 0's fill rate most are thin, so the OR fired essentially always and **the handicap grew with the resolution** — a 4-unit penalty on every cell, for resolving finely. Cells paying `w_unknown` inside the common support, at 14 frames:

    schedule        before    after
    5/10/20/40      100.0%     0.9%
    uniform 20 cm     4.2%     0.0%
    M* reference        --      0.0%

Confidence is evidence and evidence adds up, so the observation counts over the footprint's *distinct* cells are now summed and compared against `n_min` once — which is exactly what this file's own reference side had always done with `block_stats`. "Distinct" matters: 25 samples over a 40 cm ring cell all land in one cell, and summing per sample would multiply its evidence by 25.

**2. The two sides were on different lattices.** M_S OR-ed bitfields computed on the **ring** lattice — a step over a 5 cm neighbourhood in ring 0, 40 cm in ring 3 — while M* computed them at the 25 cm planning cell from block means. The synthetic scene's 12 cm kerb is a step at 5 cm and smooth at 25 cm, so:

    5/10/20/40, 14 frames:  invented 148 impassable cells M* did not have,
                            missed 8 that it did, against 12 real ones.

A path planned around 148 phantom walls and then scored on a map without them is not a regret, it is two different problems subtracted. The predicate is now evaluated **once, on the planning lattice**, from the same summed statistics on both sides — heights combined by observation count, variances by the law of total variance (§4.2). Clearance is evaluated on neither side, because M* is 2.5D ground and structurally cannot set that bit; keeping it on the M_S side alone scored a difference in map *contents* as a cost of coarsening. After: **0 invented, 0 missed** for both frozen schedules.

**⚑ The direction of the result inverted, and the new direction is the physical one.** It is now the *coarse uniform* grid that misses M*'s hazards — uniform 20 cm misses all 12, uniform 10 cm misses 9 at 12 frames — because it averages a 12 cm kerb into its cells until the step falls under `s_max`. The fine map used to be the one reported as disagreeing with the reference, for the sole reason that it was the only one resolving the kerb at all.

**⚑ One claim did not reproduce.** "At the default frame count the reference map has zero impassable cells at all" — at 12 frames M* has 12 impassable cells inside the planning window, and at 14 it has 12. I could not reproduce a zero, and I would guess it predates 1 Sep's traversability class-table fix (which had `road` marked non-drivable) or the beam-model fix, both of which change M*.

**What is still open, and it is the same thing as yesterday.** The money plot still reads 0.207 for both frozen schedules against 0.000 for every uniform. That is no longer a lattice or a confound artefact — M_S and M* now agree cell for cell about the walls — it is that `PLAN_LANE_CELLS` runs the path six cells off the centreline, where none of the scene's hazards are. A map that cannot see a wall scores identically to one that can, because the path never goes near it. Shrestha's `regret_plot.py` finding, unresolved, and `PLAN_LANE_CELLS` remains untouched by me.

**Source:** `src/eval/plan_regret.py` (`costmap_from_gridmap`, `CostMap.low_confidence`), `scripts/eval_synthetic.py` (`--confound`), `tests/test_regret_lattice.py` (6).

**So what:** Eq. (23) now subtracts like from like, and the ablation's headline is no longer carrying a 4-unit-per-cell handicap against its own contribution. **Neither fix moves the money plot's ordering**, and that is worth saying plainly before Gate 4: the remaining gap is the planning query, not the metric. 520 passed, 27 skipped.

---

## 2026-09-02 — Aakash
**Module:** D1 — Memory bound under load (Day 6 D1, brought forward)

**Finding:** "Confirm the memory bound holds under a worst-case dense-crowd scene" had no artefact behind it. The existing frame-loop allocation test runs on the quiet wall-and-car scene, which is the easy case for a bound.

`synthetic.scan(crowd=N)` adds N pedestrians on the raw `moving-person` id (254 → learning id 5, `person`). That is the worst case in four ways at once, and they are four *different* caps: every return is dynamic, so the transient layer takes all of them and the tracked-object list is pushed at `max_tracks`; `person` is a refine class, so the semantic gate fires on all of them and the pool is pushed at its 512 blocks; they are small and close, so they occupy many fine cells rather than a few coarse ones, pushing `max_candidate_cells`; and they are separate objects a metre apart, which is the clustering worst case — one blob is far cheaper than two hundred.

Measured, peak transient allocation over the harness path:

    crowd     0    47,579 returns    21.01 MB
    crowd    50    48,779 returns    21.04 MB
    crowd   200    52,379 returns    21.13 MB
    crowd   400    57,179 returns    21.26 MB

**20% more returns for 1.2% more peak.** The bound is flat in the scene, which is the claim the report actually makes — not that the loop allocates nothing (this is the eval harness, which composes world coordinates per frame), but that what it allocates does not scale with how much is happening.

**⚑ Two things I got wrong first, both worth recording.** My first threshold was an absolute cap of 12 MB against a measured 21 MB, and the honest fix was not to raise the number but to change what is asserted: an absolute figure here measures the *harness* and would drift with every unrelated change to it, so the test now compares crowded against quiet on the same code. And a test asserting "the crowd is fully mapped" would have been asserting the opposite of the design — under a crowd the correct behaviour of a fixed pool is refusal and eviction, so that is what is asserted.

**Source:** `src/eval/synthetic.py` (`_crowd`), `tests/test_memory_crowd.py` (5).

**So what:** Day 6 D1's memory item now has a CI-enforced answer rather than a plan, and it exercises the caps that the E1 fix earlier today changed the behaviour of. Nothing in the memory table moves — the whole point is that it cannot.

---

## 2026-09-02 — Aakash
**Module:** D1 — Refinement pool, lattice (Day 5 stretch)

**Finding:** Day 5's D1 item is "conservative pyramid (§7.2) + the exhaustive no-false-negative test, then anisotropic foveation with hysteresis". The pyramid half was already done and tested (21 tests, `test_theorem3_has_no_false_negatives`). The hysteresis half was implemented and **had no caller**: `lattice.migrate_ring` appeared nowhere outside its own unit test. §6.3's own specified unit test — *"drive a synthetic trajectory with sinusoidal speed across a ring boundary; assert the number of split/merge events per cell is bounded and that variance does not grow monotonically over 1,000 frames"* — did not exist either, which is a CLAUDE.md rule ("every formula in `sih-math.md` has a named unit test").

Writing that test turned up the larger thing. **Flaw E1's fix was inert, and the symptom was printed in every run we have ever done.** `gate.apply` calls `pool.release_overtaken(lambda ring, slot: ring_of_slot(gm, slot))`, and `ring_of_slot` answers which ring a flat **slot** is *stored* in — a property of the allocation, fixed at startup, which cannot change because the vehicle moved. So `now` was always exactly `ring`, the release test `now <= ring - levels` was unsatisfiable for `levels >= 1`, and **nothing was ever released**. Fourteen frames of the synthetic sequence ended `released 0 ... pool 512/512 blocks` with 15,791 refusals: the pool filled once, early, then refused every later request — flaw E1 precisely, with the fix for it sitting in the same function.

The existing test did not catch it because it monkeypatched `gate.ring_of_slot` to return 0 for every slot. That monkeypatch was the tell: the only way to observe a release was to replace the function with one that lies.

`release_overtaken` now gets `migrate_ring(*_cell_centre(gm, ring, slot), schedule, ring, speed)` — where the cell *is*, from its already-vehicle-relative centre — which both makes the E1 fix live and puts §6.3's hysteresis on the frame path for the first time. Measured, same 14 frames:

    fired    116,684 -> 395        refused  15,791 -> 0
    acquired   2,954 -> 395        released      0 -> 324
    pool     512/512 -> 62/512

`fired` collapses because a refused cell never gets `FLAG_REFINED` and so re-fires every frame; 116,684 was a few hundred cells asking again and again against a full pool. **No memory figure moves** — the pool is 512 x 16 x 12 B whether full or empty.

**⚑ A measurement trap worth writing down.** My first version of §6.3's test counted `out["acquired"] + out["released"]` and reported 200 events in 200 frames, which reads as total thrash. It was not: `pool.acquire` is idempotent for a `(ring, slot)` it already holds, so `acquired` counts the gate *re-affirming* a block it already has. `free_blocks` never moved. The metric §6.3 actually bounds is pool **occupancy** change, and by that measure the band holds a boundary cell for 1,000 frames of sinusoidal speed with the variance byte unchanged — §5.4's `merge(split(c)) == c` doing its job. I nearly recorded a bug that was not there.

**Source:** `src/grid/gate.py`, `tests/test_gate.py::test_release_happens_when_the_vehicle_drives_up_to_the_cell`, `::test_hysteresis_keeps_a_boundary_cell_from_thrashing_the_pool`.

**So what:** The refinement pool is the mechanism behind the "bounded memory, spend it where it matters" claim, and until today it spent everything in the first few frames and then refused. Gate 5 asks that every stretch item either works and is integrated or is reverted cleanly; the anisotropic-foveation-with-hysteresis item is now integrated rather than present. 509 passed, 27 skipped.

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
