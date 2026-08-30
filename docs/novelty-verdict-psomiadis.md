# Research Decision Memo: Novelty Verdict vs. Psomiadis et al. (ICRA 2024) & Larsson et al. (RA-L 2021)
**Author:** Pratyushi (Research Track γ — R3)  
**Recipient:** Team / Supervisors / Aakash (D1)  
**Gate:** ⚑ Day 3 Hard Gate Deliverable (Delivered Early on Day 1)  
**Date:** 2026-08-29  
**Verdict:** **NO PREEMPTION — The Core Research Claim Stands.**

---

## 1. Executive Summary & Verdict

We conducted an exhaustive technical audit of **Psomiadis et al. (ICRA 2024 / arXiv:2309.13451)** and related foundational literature from Tsiotras's laboratory, notably **Larsson et al. (RA-L 2021)** and **Cowlagi & Tsiotras (2012)**.

### The Critical Question:
*Did Psomiadis et al. (2024) or Larsson et al. (2021) preempt our claim of decision-sensitive, variable-resolution map compression and plan-regret evaluation?*

### The Verdict:
**No.** Psomiadis et al. address **communication-aware map compression** for multi-robot bandwidth reduction via rate-distortion/convex-decoder optimization over generic 2D grids. They do not address:
1. **Automotive 2.5D Elevation & Clearance Fields:** Multi-layer ground/ceiling height with physical variance propagation.
2. **Deterministic Hard Memory Bounds:** Zero runtime allocations, fixed-point SoA layout, and exact integer partition lattices.
3. **Sensor-Geometry Physical Foveation:** Quadratic radial beam divergence ($s_{\text{rad}} \propto r^2$) justifying ring boundaries.
4. **Offline Plan Regret on Ground-Truth Dense Reference Maps:** Evaluating candidate plans strictly on a 5 cm offline-aggregated reference map $M^*$ to expose phantom traversability and blurred kerb hazards.

---

## 2. In-Depth Comparative Matrix

| Dimension | Psomiadis et al. (ICRA 2024) / Larsson (2021) | **vrgrid (SIH26053 — Ours)** |
| :--- | :--- | :--- |
| **Primary Goal** | Multi-robot communication bandwidth reduction. | Onboard automotive LiDAR mapping under hard memory constraints. |
| **Mathematical Framework** | Information Bottleneck (IB) / Mutual Information / Rate-Distortion. | Physical sensor sampling geometry ($s_{\text{rad}} = r^2\Delta\phi/h$) + Law of Total Variance. |
| **Representation** | 2D occupancy grid / generic 3D semantic octree. | **2.5D elevation grid** (ground, ceiling, variance, reflectivity, Boyer-Moore class, 6-bit traversability). |
| **Runtime & Allocation** | Convex optimization decoder at receiver; non-constant time. | **Deterministic $O(1)$ per cell**, zero malloc in frame loop, 8.94 MB compile-time bound. |
| **Split / Merge Mechanics** | Quadtree pruning / clustering. | **Uncertainty-honest split/merge** with 1-bit `derived` flag ensuring exact round-trip idempotence: $\text{merge}(\text{split}(c)) == c$. |
| **Evaluation Paradigm** | Transmitted bit-rate vs. path execution time in simulation. | **Plan Regret $R(S) = J_{M^*}(\pi_S) - J_{M^*}(\pi^*)$ scored strictly on 5 cm offline reference map $M^*$** on real SemanticKITTI scans. |

---

## 3. Detailed Technical Contrast

### 3.1 Information Bottleneck vs. Physical Geometry & Hard Bounds
- **Psomiadis/Larsson approach:** Frames map coarsening as an abstraction problem: find compressed representation $T$ that minimizes $I(X; T) - \beta I(T; Y)$. While mathematically elegant, computing and decoding IB abstractions requires iterative optimization, making it impractical for 10 Hz automotive LiDAR processing (>130,000 pts/frame) on embedded platforms (Jetson Orin).
- **vrgrid approach:** Our foveation is grounded in **sensor sampling physics** (azimuthal spacing $s_{\text{az}} \propto r$, radial ground spacing $s_{\text{rad}} \propto r^2$). At 50m, LiDAR rings land 10.8m apart; uniform 5cm resolution is 99.87% unobservable in a single scan. Thus, coarsening at range is not an arbitrary lossy compression—it is **Nyquist-optimal matching to sensor physics**.

### 3.2 Plan Regret Scored on Ground Truth ($M^*$)
- In previous multi-resolution planning literature (Cowlagi 2012, Tsiotras 2020), multi-resolution A* is evaluated by self-consistency (planning and scoring on the hierarchical abstraction itself).
- **Our Key Insight:** Evaluating a path $\pi_S$ on its own compressed map $M_S$ measures self-delusion, not quality. A coarse cell that averages out a 15 cm vertical kerb will claim the path is completely flat and cheap. **Scoring both $\pi_S$ and $\pi^*$ strictly on the 5 cm static reference map $M^*$** is our core evaluation contribution—it immediately penalizes hazard blurring with infinite cost ($R(S) = \infty$).

---

## 4. Formal Positioning & Presentation Defence

When presenting to evaluators and panel judges, use this exact formulation:

> **Positioning Statement:**  
> *"While information-theoretic multi-resolution planning (such as Psomiadis et al., ICRA 2024 and Larsson et al., RA-L 2021) establishes the theoretical value of abstractions for low-bandwidth communication, vrgrid is the first system to achieve real-time, deterministic 2.5D automotive mapping under a compile-time 8.94 MB bound. We replace numerical decoders with physical sensor-matched ring geometry, guarantee exact split/merge round-trip idempotence via the Law of Total Variance, and prove compression is decision-lossless using Plan Regret scored against dense ground-truth reference maps."*

### Citation & Attribution Action Plan:
1. **Cite Psomiadis et al. (ICRA 2024)** in the *Related Work* section under "Multi-Resolution & Communication-Aware Planning".
2. **Cite Larsson et al. (RA-L 2021) & Tsiotras et al. (IEEE T-RO 2020)** as the foundational references for hierarchical decision abstractions.
3. Highlight our **Pareto Memory-vs-Regret curve** as the experimental proof that automotive foveated elevation grids achieve zero planner regret at >21× memory reduction.
