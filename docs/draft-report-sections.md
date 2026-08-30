# Final Report Draft: Related Work & Decision-Theoretic Evaluation
**Author:** Pratyushi (Research Track γ — R3)  
**Deliverable:** Days 5–6 Research Report Contribution (~700 words, publication-grade)  
**Date:** 2026-08-29  
**Target:** SIH26053 Final Submission Document & Slide Deck  

---

## Section 2: Related Work

### 2.1 Multi-Resolution & Foveated Grid Representations
Adaptive spatial representations in mobile robotics have historically balanced sensor sample density against computational footprint. Nested multi-resolution egocentric ring buffers were introduced by Droeschel et al. [1] for aerial micro-vehicles, adapting the computer graphics geometry clipmaps of Losasso & Hoppe [2] to 3D point cloud filtering. In autonomous driving, Patel et al. (*RoadRunner M&M*, 2024) [3] recently demonstrated learned multi-range, multi-resolution elevation mapping across 50 m and 100 m horizons. However, existing multi-resolution grids either rely on expensive neural network decoders [3], suffer from non-deterministic pointer-chasing hashing overheads in 3D (such as OpenVDB/VDB-Mapping [4]), or fail to maintain strict, preallocated memory guarantees required for safety-critical embedded deployment. `vrgrid` differs fundamentally by enforcing an exact integer partition lattice with toroidal $O(\text{perimeter})$ updates and a compile-time fixed 8.94 MB memory bound.

### 2.2 2.5D Elevation & Traversability Fields
Multi-Level Surface (MLS) maps were pioneered by Triebel, Pfaff & Burgard (2006) [5] to represent overhanging obstacles and multi-level structures within 2D cells. Modern robot traversability pipelines, such as Fankhauser et al. [6], typically compute slope and step height over continuous elevation grids. Recent state-of-the-art off-road frameworks—notably SALON (Sivaprakasam et al., ICRA 2025) [7] and EVORA (Cai et al., IEEE T-RO) [8]—highlight two critical principles: geometric obstacles must strictly override semantic classifications, and epistemic uncertainty in unobserved regions must enforce a fail-safe constraint. Building upon these principles, `vrgrid` encodes traversability into a deterministic 6-bit bitfield, integrating a graphics-inspired conservative pyramid (Greene et al. [9]) that mathematically guarantees zero false-positive traversability assertions across coarse queries.

### 2.3 Information-Theoretic vs. Decision-Sensitive Compression
Recent theoretical work from Tsiotras's laboratory, including Larsson et al. (RA-L 2021) [10] and Psomiadis et al. (ICRA 2024) [11], explores Information Bottleneck (IB) abstractions and rate-distortion theory for compressing 2D occupancy grids to minimize inter-robot communication bandwidth. While effective for multi-agent telemetry, these approaches require iterative convex optimization decoders and evaluate compression quality purely via mutual information or self-contained simulation costs. In contrast, `vrgrid` introduces *Plan Regret* evaluated against dense ground-truth reference maps, directly measuring whether memory compression alters the navigation decisions of high-speed autonomous ground vehicles.

---

## Section 4: Decision-Theoretic Evaluation & Plan Regret

### 4.1 The Plan Regret Metric
Standard spatial mapping benchmarks evaluate compression using reconstruction metrics such as Root Mean Square Error (RMSE) or Occupancy IoU. However, geometric reconstruction error is an imperfect proxy for navigation efficacy: a 5 cm vertical error on an open highway is operationally irrelevant, whereas the same 5 cm error on a 15 cm kerb can lead to fatal vehicle destabilization.

To measure compression loss directly in decision space, we formulate **Plan Regret** $R(S)$. Let $M^*$ denote the static, high-resolution (5 cm) reference map aggregated offline with ground-truth poses. Let $\pi^* = P(M^*)$ be the optimal trajectory produced by deterministic planner $P$ on $M^*$, and let $\pi_S = P(M_S)$ be the trajectory planned on the candidate adaptive grid under resolution schedule $S$. The plan regret is defined as:
$$R(S) = J_{M^*}(\pi_S) - J_{M^*}(\pi^*) \ge 0$$
where $J_M(\pi)$ evaluates the cumulative path cost functional.

> **The Methodological Invariant:** Crucially, both candidate path $\pi_S$ and optimal path $\pi^*$ are **scored strictly on the reference map $M^*$**. Evaluating $\pi_S$ on its own compressed representation $M_S$ measures self-consistency rather than true safety; scoring on $M^*$ ensures that any hazard blurred by spatial coarsening immediately incurs an infinite cost penalty ($J_{M^*}(\pi_S) = \infty \implies R(S) = \infty$). To capture spatial deviation alongside cost, we simultaneously report the discrete Fréchet distance $d_F(\pi_S, \pi^*)$.

### 4.2 Coarsening-Justification Ratio ($\rho$)
To isolate algorithm estimation bias from intrinsic terrain roughness, we decompose information loss $IL(c)$ over the constituent fine cells $F(c)$ of coarse cell $c$:
$$IL(c)^2 = \frac{1}{|F(c)|} \sum_{f \in F(c)} (\mu_c - h^*_f)^2 = \underbrace{(\mu_c - \bar{h}^*(c))^2}_{\text{bias}^2} + \underbrace{\text{Var}_{f}(h^*_f)}_{\text{spread}^2}$$
The dimensionless coarsening ratio is defined as:
$$\rho(c) = \frac{IL(c)}{\max(\text{spread}(c), \epsilon)}$$
where $\rho(c) \approx 1.0$ mathematically proves that coarsening paid only the irreducible sub-cell terrain variability, demonstrating optimal compression without algorithmic distortion.

### 4.3 Dynamic Ghost Removal ($DR, SP, F$-Score)
To evaluate the elimination of transient moving objects without eroding static infrastructure (such as fences and poles), we compute the dynamic removal rate ($DR$), static preservation rate ($SP$), and their harmonic mean:
$$F = 2 \cdot \frac{DR \cdot SP}{DR + SP}$$
This dual-direction metric prevents degenerate policies (e.g. over-aggressive map clearing) from artificially inflating benchmark scores.

---

## References

1. **Droeschel, D., Stückler, J., & Behnke, S.** (2014). Local multi-resolution representation for 6D motion estimation and mapping with a 3D laser scanner. *IEEE International Conference on Robotics and Automation (ICRA)*, 5133–5140.
2. **Losasso, F., & Hoppe, H.** (2004). Geometry clipmaps: terrain rendering using nested regular grids. *ACM Transactions on Graphics (TOG)*, 23(3), 769–776.
3. **Patel, M., Frey, J., Atha, D., Spieler, P., Hutter, M., & Khattak, S.** (2024). RoadRunner M&M -- Learning Multi-range Multi-resolution Traversability Maps for Autonomous Off-road Navigation. *IEEE Robotics and Automation Letters (RA-L)*, 9(12), 11425–11432.
4. **Museth, K.** (2013). VDB: High-resolution sparse volumes with run-time topology changes. *ACM Transactions on Graphics (TOG)*, 32(3), 1–22.
5. **Triebel, R., Pfaff, P., & Burgard, W.** (2006). Multi-level surface maps for outdoor terrain mapping and loop closing. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2276–2282.
6. **Fankhauser, P., Bloesch, M., & Hutter, M.** (2018). Probabilistic terrain mapping for mobile robots with uncertain localization. *IEEE Robotics and Automation Letters (RA-L)*, 3(4), 3019–3026.
7. **Sivaprakasam, M., Triest, S., Ho, C., et al.** (2025). SALON: Self-Supervised Adaptive Learning for Off-Road Navigation. *IEEE International Conference on Robotics and Automation (ICRA 2025)*.
8. **Cai, X., How, J. P., et al.** (2024). EVORA: Deep Evidential Traversability Learning for Risk-Aware Off-Road Autonomy. *IEEE Transactions on Robotics (T-RO)*.
9. **Greene, N., Kass, M., & Miller, G.** (1993). Hierarchical Z-buffer visibility. *ACM SIGGRAPH*, 231–238.
10. **Larsson, D. T., Maity, D., & Tsiotras, P.** (2021). Information-theoretic abstractions for planning in agents with computational constraints. *IEEE Robotics and Automation Letters (RA-L)*, 6(4), 7651–7658.
11. **Psomiadis, E., Maity, D., & Tsiotras, P.** (2024). Communication-Aware Map Compression for Online Path-Planning. *IEEE International Conference on Robotics and Automation (ICRA)*.
12. **Nuss, P., Yuan, S., Krehl, F., et al.** (2018). A random finite set approach for dynamic occupancy grid maps with velocity distribution. *The International Journal of Robotics Research (IJRR)*, 37(8), 841–866.
