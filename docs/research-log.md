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
