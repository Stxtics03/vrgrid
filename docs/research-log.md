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

## 2026-09-02 — Pratyushi
**Module:** R3 (Traversability, Evaluation & Novelty Claim)
**Finding:** Drafted and delivered publication-grade Sections for the final SIH26053 submission: Section 2 (Related Work covering foveated clipmaps, 2.5D elevation fields, and decision-sensitive abstractions) and Section 4 (Decision-Theoretic Evaluation covering Plan Regret $R(S)$, Coarsening Ratio $\rho$, and Dynamic Removal $F$-score). Verified all 12 literature citations against peer-reviewed proceedings (ICRA, RA-L, IROS, TOG, IJRR, SIGGRAPH).
**Source:** `docs/draft-report-sections.md`
**So what:** Fulfills the Days 5–6 writing deliverable for Track γ. Ready for direct inclusion in the final technical report and submission slide deck.
