# Related Work (Final Section for Submission Report)

**Author:** Srinivas (R1 — Representation & Prior Art)  
**Date:** 2026-09-03  
**Status:** Complete & Submission-Ready  
**Word Count:** ~720 words  

---

## Section 2: Related Work

Spatial mapping for autonomous mobile robots has historically balanced a fundamental trade-off between geometric fidelity, memory consumption, and query latency. Existing representations broadly fall into three categories: volumetric 3D grids, 2.5D elevation models, and multi-resolution foveated structures.

### 2.1 Volumetric 3D Representations
Volumetric occupancy mapping extends 2D occupancy grids to full three-dimensional space. OctoMap (Hornung et al., 2013) established the standard tree-based formulation, using hierarchical octrees to compress unoccupied space. While OctoMap models arbitrary 3D geometry, its dynamic node allocation on the CPU heap creates memory fragmentation and lacks deterministic bounds, consuming over 2.5 GB for a 5 cm resolution outdoor scene. Recent advances, such as Wavemap (Reijgwart et al., 2023), apply multi-resolution Haar wavelets over hierarchical block trees to achieve tighter memory compression and continuous likelihood updates. However, hierarchical 3D trees introduce irregular pointer-chasing and warp divergence on modern SIMD/GPU architectures. Furthermore, because ground vehicles operate primarily on 2D surface manifolds, full 3D voxelization incurs severe computational overhead by allocating resources to uninformative volumetric air.

### 2.2 Elevation and Multi-Level Surface (MLS) Mapping
To eliminate the $O(L \times W \times H)$ explosion of 3D grids, 2.5D elevation mapping models the environment as a 2D lattice storing scalar terrain heights (Herbert et al., 1989). Fankhauser et al. (2014) formalized robot-centric elevation mapping using recursive 1D Kalman filters with range-dependent measurement variance $\sigma_m^2(r) \propto r^2$. While computationally efficient, standard 2.5D grids fail in environments with multi-level geometry, such as overhanging bridges, tree canopies, and tunnels. To resolve this, Triebel et al. (2006) introduced Multi-Level Surface (MLS) maps, storing discrete lists of vertical surface intervals per cell. Modern extensions like Point Cloud Tomography (PCT; Yang et al., 2024) reformulate MLS concepts into parallel 2.5D elevation slices for GPU traversability planning. However, existing MLS and elevation approaches operate on uniform spatial discretizations, resulting in severe data sparsity at long ranges due to LiDAR beam divergence.

### 2.3 Multi-Resolution and Foveated Grid Structures
Multi-resolution representations reduce computational and storage burdens by varying spatial resolution across the observation space. Originating in computer graphics with geometry clipmaps (Losasso & Hoppe, 2004), nested toroidal regular grids enable constant-time $O(1)$ scrolling upon observer motion by updating only newly exposed boundary perimeters in $O(\text{perimeter})$ time. Droeschel et al. (2014, 2016) translated this concept to mobile robotics by designing egocentric multi-resolution ring buffers for micro aerial vehicle (MAV) 3D laser motion estimation. Recently, Adaptive Patched Grid Mapping (Wodtko et al., 2023) explored dynamic patch refinement for automotive perception. However, existing automotive multi-resolution implementations apply naive inverse-variance averaging ($1/\sigma_p^2 = \sum 1/\sigma_i^2$) when coarsening cells; this critically discards the spatial spread term between child cells, causing the map to report high confidence over elevation step discontinuities (e.g., curbs). Moreover, none of these systems incorporate online semantic refinement under a strict, compile-time bounded memory envelope.

### 2.4 Downstream Planning and Evaluation
Mapping systems are traditionally evaluated via geometric reconstruction error (RMSE) or volumetric Intersection-over-Union (IoU) against dense offline point clouds. While geometrically informative, these metrics fail to quantify the operational impact of map compression on autonomous navigation. Path planners operating on costmaps derived from 2.5D elevations require conservative traversability bounds to ensure safety. Graphics-inspired hierarchical maximum pyramids (Tevs et al., 2008) have demonstrated the utility of conservative reduction for ray-stepping. In this work, we bridge mapping and navigation by introducing *Plan Regret*—measuring map quality directly by the optimality gap of downstream path planning on the compressed representation compared to a ground-truth uniform baseline.

---

## References
- Droeschel, D., Stückler, J., & Behnke, S. (2014). Local multi-resolution representation for 6D motion estimation and mapping with a continuously rotating 3D laser scanner. *IEEE ICRA*, pp. 4921–4926.
- Fankhauser, P., Bloesch, M., Gehring, C., Hutter, M., & Siegwart, R. (2014). Robot-centric elevation mapping with uncertainty estimates. *CLAWAR*, pp. 433–440.
- Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: An efficient probabilistic 3D mapping framework based on octrees. *Autonomous Robots*, 34(3), 189–206.
- Losasso, F., & Hoppe, H. (2004). Geometry clipmaps: Terrain rendering using nested regular grids. *ACM SIGGRAPH*, pp. 769–776.
- Reijgwart, V., Cadena, C., Siegwart, R., & Ott, L. (2023). wavemap: Efficient volumetric hierarchical occupancy mapping. *Robotics: Science and Systems (RSS)*.
- Tevs, A., Ihrke, I., & Seidel, H.-P. (2008). Maximum mipmaps for fast, accurate, and scalable dynamic height field rendering. *ACM SIGGRAPH I3D*, pp. 183–190.
- Triebel, R., Pfaff, P., & Burgard, W. (2006). Multi-level surface maps for outdoor terrain mapping and loop closing. *IEEE/RSJ IROS*, pp. 2276–2282.
- Wodtko, T., Griebel, M., & Buchholz, M. (2023). Adaptive patched grid mapping. *arXiv preprint arXiv:2308.03416*.
- Yang, T., Cheng, K., Xue, J., Jiao, J., & Liu, M. (2024). Efficient global navigational planning in 3D structures based on point cloud tomography. *IEEE/ASME Transactions on Mechatronics*.
