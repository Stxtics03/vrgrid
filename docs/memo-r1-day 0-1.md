# Decision Memo: Tier 1 Prior Art & Novelty Boundary (Day 0–1)

**From:** Srinivas (R1 — Representation & Prior Art)  
**To:** Aakash (D1 — Grid Engine & Evaluation)  
**Date:** 2026-08-28  
**Subject:** Tier 1 Literature Review & Positioning Before Cell Struct / Grid Engine Freeze

---

## 1. Executive Summary & Actionable Decisions

Before we freeze the cell struct and finalize comments in `include/vrgrid/` and `src/grid/`, here are the four prior art findings and the exact decisions we must enforce:

1. **Two-Layer Cells (Ground + Overhead Clearance) $\rightarrow$ MLS Heritage:**
   - **Prior Art:** Triebel, Pfaff & Burgard (*IROS 2006*) introduced Multi-Level Surface (MLS) maps for storing multiple vertical intervals per 2D column.
   - **Decision:** In our code comments, docstrings, and report, explicitly describe our representation as **"MLS-style two-layer cells"**. Do *not* claim we invented multi-level/two-layer elevation representation.

2. **Nested Toroidal Ring Grid $\rightarrow$ Droeschel (2014) & Geometry Clipmaps (2004):**
   - **Prior Art:** Droeschel et al. (*ICRA 2014*, *JFR 2016*) used egocentric multi-resolution ring buffers with constant-time toroidal shifts for 3D laser scanners. Losasso & Hoppe (*SIGGRAPH 2004*) introduced nested geometry clipmaps with $O(\text{perimeter})$ boundary updates.
   - **Takeaway for D1/D3:** We borrow the $O(\text{perimeter})$ toroidal shift boundary clear.
   - **Novelty Boundary:** Droeschel did *not* have 2.5D elevation Kalman fusion, semantic-driven refinement, variance-honest split/merge under a fixed memory bound, or plan-regret validation. **That is our contribution.**

3. **Kalman Elevation Variance Model $\rightarrow$ Fankhauser (2014) Alignment:**
   - **Prior Art:** Fankhauser et al. (*CLAWAR 2014*) established the standard range-dependent measurement variance $\sigma_m^2(r) = \sigma_0^2 + c \cdot r^2$ for robot-centric elevation mapping.
   - **Decision:** Confirm `docs/sih-math.md` §3 uses this exact variance formulation.

---

## 2. Our Formal Novelty Statement (Use in Presentations & Reports)

> *"Foveated ring buffers, multi-level surface maps, and elevation mapping are established in literature. Our contribution is threefold:*  
> *1. **Adaptive Resolution Schedule:** Dynamic coarsening driven jointly by sensor divergence and semantics under a strict compile-time preallocated memory bound (8.94 MB).*  
> *2. **Uncertainty-Honest Split/Merge:** Closed-form variance propagation using the Law of Total Variance that guarantees $\text{merge}(\text{split}(c)) \equiv c$ via the derived bit.*  
> *3. **Plan-Sensitivity Evaluation:** Proving that geometric compression is free where it matters by evaluating mapping quality in units of **Plan Regret** rather than purely reconstruction RMSE."*

---

## 3. Positioning Paragraph (Submission-Ready Related Work)

> Nested multi-resolution grid representations have a rich history in computer graphics (Losasso & Hoppe, 2004) and mobile robotics (Droeschel et al., 2014), where ring buffers address LiDAR beam sparsity. Similarly, representing multi-level vertical structures via discrete surface intervals was formalized in Multi-Level Surface (MLS) maps by Triebel et al. (2006). While recursive 1D Kalman elevation filtering has become standard for robot-centric mapping (Fankhauser et al., 2014), existing representations either allocate dynamic memory hierarchies (such as octrees) or apply uniform spatial discretizations. In contrast, `vrgrid` unifies MLS two-layer cells with foveated ring clipmaps under a rigid compile-time memory bound, introducing uncertainty-conserving split/merge mechanics and validating compression via downstream path planning regret.

---

## 4. Checklist for D1 (Aakash)
- [x] Reference Triebel et al. (2006) in the cell struct header.
- [x] Use $O(\text{perimeter})$ toroidal clear based on Losasso & Hoppe (2004).
- [x] Ensure variance model matches Fankhauser et al. (2014).
