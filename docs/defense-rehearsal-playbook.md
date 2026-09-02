# Presentation & Panel Defense Master Playbook

**Author:** Srinivas (R1 — Representation & Prior Art)  
**Date:** 2026-09-04  
**Scope:** The authoritative defense manual for responding to tough panel questions during the SIH presentation.

---

## 🎯 The Golden Rule of Defense
> **Acknowledge the literature immediately, state the exact distinction with precision, and back it up with a reproducibly derived number.** Never claim you invented adaptive resolution, foveation, or multi-level cells. Claim the **composition**, the **closed-form variance split/merge guarantees**, the **fixed 8.94 MB compile-time bound**, and the **Plan Regret proof**.

---

## ❓ Question 1: *"Isn't this just a clipmap or Droeschel's 2014 paper?"*

### 🎙️ Your Answer (Assigned to Srinivas / R1):
> *"Yes, our toroidal scrolling mechanism builds directly on the geometry clipmaps of Losasso & Hoppe (SIGGRAPH 2004) and the egocentric multi-resolution ring buffers developed by Droeschel et al. (ICRA 2014).  
> 
> However, Droeschel's work was designed for 3D MAV pose estimation and lacked 2.5D elevation Kalman fusion, online semantic refinement, and variance-conserving split/merge mathematics under a strict compile-time memory bound. We adapt the toroidal ring concept into an MLS-style 2.5D elevation grid where cell size adapts jointly to distance and semantic class, and we validate it through downstream path planning regret."*

---

## ❓ Question 2: *"Planners operate on uniform grids, so don't you lose your memory savings when the planner decompresses your map?"*

### 🎙️ Your Answer:
> *"No, for three concrete reasons:  
> 1. **Native Multi-Resolution Querying:** Our grid provides an $O(1)$ query API (`query(x, y)`). Planners like Lattice $A^*$ or hierarchical planners query the multi-resolution grid directly without uncompressing to a dense 5 cm grid in memory.  
> 2. **Local Corridor Unrolling:** If a legacy planner strictly demands a uniform grid, we only instantiate a small rolling local corridor ($10\text{ m} \times 10\text{ m}$) around the robot's immediate path, which requires $< 1\text{ MB}$, preserving our $>20\times$ global savings.  
> 3. **Pre-Baked Traversability Bitfields:** Traversability is computed on GPU during the map update and stored in a 1-byte bitfield per cell, eliminating runtime costmap computation for the planner."*

---

## ❓ Question 3: *"Why not use full 3D voxels (OctoMap) or Wavemap (RSS 2023)?"*

### 🎙️ Your Answer:
> *"Ground vehicles navigate 2D surface manifolds, not 3D volumetric space. In an outdoor driving scene, over $98\%$ of 3D voxel space represents empty air.  
> 
> OctoMap requires dynamic heap allocation (~2.56 GB for a 5 cm grid), which causes memory fragmentation and unpredictable $p99$ latency spikes violating ISO 26262 automotive safety. Tree-based structures also induce thread divergence and uncoalesced memory reads on GPUs. `vrgrid`'s 2.5D Structure-of-Arrays delivers $100\%$ coalesced GPU memory throughput under a strict, compile-time preallocated 8.94 MB bound."*

---

## ❓ Question 4: *"Your Ring 0 shows no resolution improvement over a standard grid (both are 5 cm). Where is the gain?"*

### 🎙️ Your Answer:
> *"The metric to evaluate is **Accuracy-per-Megabyte**.  
> 
> In the safety-critical near field (0–10 m), `vrgrid` delivers the exact same 5 cm ground-truth fidelity as a dense map. But instead of paying 192 MB to keep that 5 cm resolution out to 100 meters—where laser beams are physically 10.8 meters apart radially—`vrgrid` reallocates resolution where sensor measurements physically exist, reducing the overall footprint to 8.94 MB without losing a single millimeter of near-field accuracy."*

---

## ❓ Question 5: *"Can you detect a 30 cm pothole at 50 meters?"*

### 🎙️ Your Answer:
> *"No, and neither can a uniform 5 cm grid with a real sensor.  
> 
> Due to vertical beam divergence ($\Delta\phi = 0.427^\circ$) from a sensor at height $1.73\text{ m}$, radial ground beam spacing grows quadratically as $s_{\text{rad}}(r) = \frac{r^2 \Delta\phi}{h_s}$. At $50\text{ m}$, consecutive laser rings strike the asphalt $10.8\text{ meters}$ apart. A $30\text{ cm}$ negative obstacle is physically invisible beyond $r_{\max} = \sqrt{\frac{W \cdot h_s}{\Delta\phi}} \approx 8.3\text{ m}$ on a single scan.  
> 
> Rather than hallucinating a free surface, `vrgrid` explicitly marks unsampled cells beyond $8.3\text{ m}$ as `UNKNOWN`, never `FREE`.
>
> What we **do** detect, measured on sequence 07: **170 pothole cells** at ring medians of 11.0, 30.0 and 8.0 cm, and **4,002 curb cells** at ring medians of 9.0, 13.0 and 8.2 cm. On the synthetic scene, where the answer is known, the detector returns **12.0 cm against a built 12 cm kerb** and **40.0 cm against a built 40 cm hole**."*

**⚠️ Quote 07, not 08.** Sequence 08 climbs 45.7 m and the eval harness clips 16.91% of its ground returns against the 8 m height band — see known-limitations §6. An earlier version of this answer quoted 08 numbers; they are withdrawn.

**⚠️ If pressed on accuracy — do not claim a detection rate.** SemanticKITTI has no ground truth for curb or pothole geometry. These are counts plus a plausibility check on the height distribution against the 10–15 cm a real urban kerb actually is. Say that before you are asked.

---

## ❓ Question 6: *"Did you tune your mapping thresholds on your test dataset?"*

### 🎙️ Your Answer:
> *"No. We strictly separated our datasets across development, tuning, and evaluation:  
> - **Sequence 00:** Used exclusively for scaffold engineering and unit test validation.  
> - **Sequence 07:** Used for mapping hyperparameter tuning and held out from final reporting.  
> - **Sequence 08:** The completely unseen test sequence on which all final benchmark metrics, RMSE curves, and Plan Regret plots were evaluated after freezing all thresholds in `configs/thresholds.yaml`."*


---

## ❓ Question 7: *"Your traversability predicate has a scale problem — a 12 cm kerb is a 67% gradient at 5 cm cells and 24% at 25 cm. Which is it?"*

### 🎙️ Your Answer:
> *"It was both, and that was a real defect we found and fixed on 2 September. Differenced over one cell, eq. (22) measures height change per metre **at the cell scale**, so a step discontinuity reads steeper the finer the lattice: the same 12 cm kerb is a gradient of 1.200 at 5 cm, 0.600 at 10 cm and 0.240 at 25 cm, against one frozen $\tan\theta_{max} = 0.364$. It was a wall on our fine rings and flat ground on the coarse ones — which is one of the two ways the sides of our plan-regret equation ended up on different geometry.
>
> Eq. (22a) now differences both geometric bits over a **fixed physical baseline** of 0.50 m. The kerb reads passable at every lattice, and the baseline is bounded by the scene rather than chosen: it must exceed $0.12/\tan 20° = 0.33$ m so the kerb reads passable everywhere, and stay under $0.40/\tan 20° = 1.10$ m so a 40 cm pothole rim still fails."*

---

## ❓ Question 8: *"You claim a compile-time memory bound. Is it actually a bound, or a number you measured once?"*

### 🎙️ Your Answer:
> *"For the map itself it is structural — every array is allocated at startup and nothing in the frame loop grows. For the ghost-removal scratch it **was** a measured guess until 2 September, and we changed it because the guess was wrong by a factor of three.
>
> The cap on candidate cells was a provisional 150,000. Measured on whole sequences it drops **52.3% of sequence 07's peak occupied set and 67.1% of 08's** — and the failure is silent: dropped cells keep their occupancy, are never tested, and cannot appear in the cleared count, so a truncating run prints a healthy ghost number while the map keeps its ghosts.
>
> We did not simply fit a bigger number, because the peak scales with sequence length — frames ×3.70 from 07 to 08, peak ×1.45, and three sequences are longer than 08. The cap is now the grid's own slot count, because the occupied set cannot exceed the grid: **truncation is impossible by construction rather than unlikely by measurement.** And we count and print truncation now, so if anyone ever sets a tighter cap, the trade is visible."*

**⚠️ Follow-up you should expect:** *"Doesn't that cost you memory?"* — Yes: 58.24 MB of scratch instead of 9.60 MB. It is **working memory, not map memory**, so the cell-count ratios in the report are unaffected, and it is off by default. Say the number; do not let them find it.
