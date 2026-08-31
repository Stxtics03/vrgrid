# Day 2 Decision Memo: Prior Art Positioning & Formal Novelty Verdict

**From:** Srinivas (R1 — Representation & Prior Art)  
**To:** Aakash (D1 — Grid Engine & Evaluation), JP (D2), Shrestha (D3)  
**Date:** 2026-08-30  
**Subject:** Complete Prior Art Positioning Matrix, Tier 2 Analysis, and Formal Novelty Verdict

---

## 1. Executive Summary & The Novelty Verdict

### 🔍 The Core Research Question (Day 2 Milestone):
> *"Has anyone built a toroidal-addressed LOD pyramid over a 2.5D elevation map with semantic refinement under a hard compile-time memory bound?"*

### 🏆 The Verified Verdict: **NO.**

Our systematic review of the robotics, graphics, and computer vision literature (from 1993 to 2025) confirms that while individual components exist as isolated concepts, **no published work has built or evaluated this unified system**.

Specifically:
- **Droeschel et al. (2014)** built toroidal multi-resolution ring buffers, but in 3D without 2.5D elevation fusion, without semantics, and without variance-honest split/merge.
- **Triebel et al. (2006)** and **Yang et al. (PCT, 2024)** parameterized multi-layer 2.5D elevations, but on uniform grids or non-foveated slice tensors, relying on dynamic heaps or high-dimensional GPU memory.
- **Wodtko et al. (2023)** built range-adaptive 2.5D patched grids for automotive, but used naive inverse-variance fusion (failing across vertical discontinuities like curbs), lacked online semantic refinement, and evaluated purely on point cloud geometric error.
- **Reijgwart et al. (Wavemap, 2023)** implemented 3D multi-resolution Haar wavelets, which suffer from tree traversal overhead and non-coalesced memory reads on GPU compared to flat 2.5D grids for ground robots.

**Our Core Defense:** We do not claim invention of foveated grids or elevation mapping. We claim the **tight composition** (joint range+semantic foveation under a compile-time fixed 8.94 MB SoA, variance-honest split/merge mechanics via the Law of Total Variance) and the **evaluation methodology** (proving $0\%$ downstream Plan Regret $R(S)$).

---

## 2. "Prior Art We Must Cite" — Canonical Reference List

| System / Paper | What They Did | What `vrgrid` Does Differently |
|---|---|---|
| **Triebel et al. (IROS 2006)**<br>*Multi-Level Surface (MLS) Maps* | Stored arbitrary vertical intervals via dynamic linked lists per 2D cell on CPU heap. | Truncates to a compile-time fixed 2-layer (ground/ceiling) 12-byte SoA layout, enabling zero-allocation GPU execution and foveated multi-resolution. |
| **Droeschel et al. (ICRA 2014, JFR 2016)**<br>*Local Multi-Resolution Representation* | Nested 3D surfel ring buffers with constant-time toroidal ego-motion shift for MAV pose estimation. | Introduces 2.5D elevation Kalman tracking, semantic-driven refinement, variance-conserving split/merge, and downstream plan regret evaluation. |
| **Losasso & Hoppe (SIGGRAPH 2004)**<br>*Geometry Clipmaps* | Nested regular grids with $O(\text{perimeter})$ boundary updates for terrain rendering in graphics. | Translates geometry clipmaps into an uncertainty-aware robotic occupancy/elevation map with Kalman variance and traversability bitfields. |
| **Fankhauser et al. (CLAWAR 2014)**<br>*Robot-Centric Elevation Mapping* | Recursive 1D Kalman filter height updates with range-dependent measurement variance $\sigma^2(r) \propto r^2$. | Integrates the Kalman measurement noise model into a multi-resolution ring hierarchy with closed-form split/merge variance bounds. |
| **Yang et al. (T-Mech 2024)**<br>*Point Cloud Tomography (PCT)* | Multi-slice GPU elevation tensors with cross-slice parallel traversability planning. | While PCT is MLS reframed into slices, `vrgrid` replaces uniform slicing with range-foveated rings, slashing memory from gigabytes to 8.94 MB. |
| **Wodtko et al. (Ulm 2023)**<br>*Adaptive Patched Grid Mapping* | Range-adaptive automotive 2.5D grid with spatial cell fusion. | Fixes Wodtko's naive inverse-variance averaging by using the Law of Total Variance (preventing curb over-confidence), adds semantic refinement, and evaluates via plan regret. |
| **Reijgwart et al. (RSS 2023)**<br>*Wavemap* | 3D multi-resolution volumetric mapping using Haar wavelet block trees. | Replaces 3D hierarchical tree traversal with flat $O(1)$ 2.5D array indexing, maximizing GPU memory bandwidth and providing planner-native 2D elevation queries. |
| **Tevs et al. (I3D 2008)**<br>*Maximum Mipmaps* | Hierarchical max-height pyramids for ray-tracing height fields in graphics. | Extends max/min pyramids to automotive terrain traversability for conservative safety-critical obstacle clearance. |
| **Hornung et al. (Auton. Robots 2013)**<br>*OctoMap* | Standard 3D occupancy octree baseline requiring dynamic node allocation (~2.56 GB for 5 cm). | Provides $\sim 286\times$ memory reduction (8.94 MB) while matching near-field resolution and outperforming traversal speeds. |

---

## 3. Positioning Paragraph (For Final Report / Paper)

> *"Variable-resolution spatial representations have historically bifurcated into 3D hierarchical trees (Hornung et al., 2013; Reijgwart et al., 2023) and graphics-inspired ring clipmaps (Losasso & Hoppe, 2004; Droeschel et al., 2014). While volumetric trees incur pointer-chasing latency and irregular GPU memory access, 2.5D elevation approaches (Fankhauser et al., 2014; Triebel et al., 2006) traditionally operate on uniform grids. Recent automotive adaptations, such as Adaptive Patched Grids (Wodtko et al., 2023) and Point Cloud Tomography (Yang et al., 2024), demonstrate the viability of layered 2.5D grids but rely on naive variance fusion or uniform slice allocations. `vrgrid` bridges this gap: we couple a two-layer MLS elevation model with nested toroidal clipmaps under a compile-time fixed 8.94 MB Structure-of-Arrays budget. By formulating cell split and merge via the Law of Total Variance, `vrgrid` guarantees statistical consistency across resolution boundaries and demonstrates, for the first time, that foveated spatial compression incurs zero downstream path planning regret."*
