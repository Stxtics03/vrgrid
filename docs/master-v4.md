# SIH26053 — Adaptive Variable-Resolution 2.5D LiDAR Mapping
## Master Plan v4 — the single source of truth

*Supersedes plan v2, the gaps-and-corrections addendum, and final-additions. Everything still true from those is folded in here. Where v4 overturns something, it is marked ⛔ with the reason.*

*Deadline: 5 September, morning. Software only. Public datasets only.*

---

## Part 0 — The pitch, in five sentences

> **A drop-in replacement for a uniform 2.5D occupancy grid.** Cell size adapts to distance, semantics, and direction of travel, under a memory bound fixed at compile time. It uses ~21× less memory than a uniform 5 cm 2.5D grid and ~286× less than a dense 5 cm 3D voxel grid, at the same near-field accuracy. It removes dynamic ghosts. And — the part nobody else will have — **we prove the compression is free by showing it does not change the plan a robot would make.**

That last clause is the differentiator. Everything before it is engineering; that clause is a research claim.

### The honest novelty statement

Say exactly this, and no more:

> Foveated grids, elevation maps, and dynamic-point removal are all published. What we contribute is (a) a resolution schedule driven by **both range and semantics** under a **hard preallocated memory bound**, (b) **uncertainty-honest split/merge** with a provable round-trip property, and (c) a **plan-sensitivity evaluation** — measuring coarsening in units of *planner regret* rather than reconstruction error.

Do not claim you invented adaptive resolution, foveation, or ghost removal. People who know the field will notice, and one of them will be on your panel.

---

## Part 1 — ⛔ What v4 overturns

Four changes. Two are corrections to physics we got wrong, one is a prior-art landmine, one is a claim that does not survive contact.

### ⛔ 1.1 — The sensor sampling argument was half-right, and the missing half is enormous

v2 justified the ring schedule with **azimuthal** point spacing: `s_az = r · Δθ`. At 0.2° that gives 3.5 cm at 10 m and 35 cm at 100 m, which tracks the 5/10/20/40 schedule beautifully. That argument is correct and you should keep it.

But it is the *easy* axis. The **radial** spacing — the gap between where consecutive laser rings strike the ground — behaves completely differently. For a sensor at height `h` with vertical beam spacing `Δφ`:

```
s_rad(r) = r² · Δφ / h        ← grows as r², not r
```

For KITTI's HDL-64E (h = 1.73 m, Δφ = 0.427°):

| Range | Azimuthal spacing | **Radial ground spacing** |
|---|---|---|
| 10 m | 3.5 cm | **43 cm** |
| 25 m | 8.7 cm | **2.7 m** |
| 50 m | 17.5 cm | **10.8 m** |
| 100 m | 34.9 cm | **43 m** |

At 50 metres, consecutive laser rings land **ten metres apart** on the road surface. This is not a small correction. It reframes three things:

**It makes your central argument far stronger.** A uniform 5 cm grid at 50 m is not merely wasteful — the ground returns are 10.8 m apart radially, so **99.5% of those cells can never receive a ground return in a single frame.** Uniform high resolution at range is not high resolution. It is an empty array with a confident-looking axis label. Now you can say that with a number.

**It means the far rings are filled by *ego-motion*, not by the sensor.** As the vehicle drives forward, the ring pattern sweeps across the ground and progressively fills cells that no single frame could reach. This is a genuine property of your system and it has consequences: temporal accumulation is not an optimisation in Rings 2–3, it is the *only* fill mechanism. Single-frame evaluation of the far field is meaningless. Report far-ring metrics as a function of frames-since-first-observation.

**It sets the real pothole limit, and it is much tighter than we assumed.** See 1.2.

Call this **ring-sweep filling** and give it a figure in the report. It is the most defensible original analysis in the project and it costs nothing to produce — it is one equation and one plot.

### ⛔ 1.2 — The negative-obstacle range limit is ~8 m, not "Rings 0–1"

The final-additions doc said potholes are detectable "within Rings 0–1" (i.e. to 25 m). The radial sampling equation says otherwise. A pothole of width `W` is only sampled at all if `W > s_rad(r)`:

```
r_max(W) = √(W · h / Δφ)
```

| Pothole width | Max detection range (first observation) |
|---|---|
| 30 cm | **8.3 m** |
| 50 cm | **10.8 m** |
| 1.0 m | **15.2 m** |

Rewrite B6 as:

> Negative obstacles are detectable on first observation only within ~8 m for a 30 cm defect, set by the radial ground sampling limit `s_rad = r²Δφ/h`. Beyond that a pothole falls entirely between two laser rings and produces no signal whatsoever. Larger defects and closer approach extend this; temporal accumulation detects a distant pothole only once the vehicle has closed to within `r_max`. Cells beyond that range are marked **unknown**, never free.

This is a *better* answer than the vague one, because it is a derived bound rather than a hedge. Judges reward a precise limitation over a fuzzy capability.

### ⛔ 1.3 — Prior art you must cite or get caught by

Two landmines in the current positioning.

**Multi-Level Surface maps predate this by twenty years.** Triebel, Pfaff & Burgard (IROS 2006) already stored multiple vertical surface patches per 2D cell — ground plus overhang — for exactly the bridges-and-multi-storey reason. Your ground+ceiling scheme is a two-layer MLS map. Point Cloud Tomography (2024), which v2 cites as prior art, is itself substantially an MLS map with a better evaluation pipeline; PCT's real contribution is GPU-parallel traversability evaluation and cross-slice planning, not the representation. **Cite Triebel 2006 as the origin of your two-layer scheme.** Presenting ground+ceiling as novel is the single easiest way to lose credibility with an informed panel, and it costs you nothing to attribute — your contribution was never the representation.

**Nested ego-centric multi-resolution rolling grids already exist in robotics.** Droeschel, Stückler & Behnke (ICRA 2014; *Journal of Field Robotics* 33(4), 2016) built local multiresolution grid maps with **interlaced ring buffers**, high resolution near the robot and coarser with distance, shifting in constant time on ego-motion, explicitly justified by sensor measurement density. That is your foveated grid, in 2014, on a MAV. It is also the geometry-clipmap idea (Losasso & Hoppe, SIGGRAPH 2004) imported to robotics.

**This does not sink you.** What is genuinely thin in the literature is the combination: per-level toroidal addressing over a *2.5D elevation* map, with *semantic* refinement under a hard bound, with *uncertainty-honest* split/merge, evaluated by *planner regret*. Claim the composition and the evaluation. Cite Droeschel and Losasso in the same breath as your ring diagram. A panel that sees you cite the thing you resemble concludes you know the field; a panel that finds it themselves concludes you don't.

### ⛔ 1.4 — Merging by inverse-variance fusion is mathematically wrong

v2 called merging "variance-weighted average, standard, uncontroversial." It is not uncontroversial, and the naive version is wrong.

Inverse-variance fusion (`1/σ² = Σ 1/σᵢ²`) is the correct rule for **repeated measurements of the same quantity**. Four child cells are not four measurements of one height — they are measurements of **four different places**. Merging them is *marginalisation over a footprint*, which obeys the law of total variance:

```
σ²_parent = Σ wᵢ σᵢ²          (within-cell, the average uncertainty)
          + Σ wᵢ (μᵢ − μ_p)²   (between-cell, the spread you just erased)
```

Naive fusion drops the second term entirely. The consequence is that merging four cells that *disagree* — precisely the case where merging is dangerous, e.g. a kerb running through the block — produces a merged cell that reports **lower** variance than any of its children. Your map becomes most confident exactly where it is least justified.

Fixing this gives you a pleasing symmetry to put on a slide: **variance increases on split (we assert detail we never measured) and increases on merge whenever children disagree (we hide detail we did measure).** Both directions are honest. Full derivation in `sih-math.md` §4.

### ⛔ 1.5 — Drop the RT-core / hardware-raytracing idea entirely

If anyone has floated using GPU ray-tracing cores to accelerate visibility cleanup: don't. OptiX is not supported on Jetson (NVIDIA confirmed, most recently March 2026), the hardware is reachable only through Vulkan `VK_KHR_ray_query`, and it is not even publicly confirmed that path is hardware-accelerated rather than a compute fallback. Your visibility cleanup is already O(1) per cell by range-image comparison and needs no ray casting at all. This is a research rabbit hole with an eight-day deadline attached to it. One line in "future work," nothing more.

---

## Part 2 — The three things that make this project unique

Everything in Part 3 is table stakes. These three are the reason you win.

### 2.1 — Plan-sensitivity: measure coarsening in units of *decision*, not *geometry*

**The gap.** Every adaptive-mapping paper reports memory, latency, and reconstruction error. None of them answer the question that actually matters: *did the compression change what the robot would do?* Reconstruction error is a proxy. Planner regret is the thing itself.

**The metric.** Let `M*` be the reference map (5 cm, no LOD, aggregated offline — you are building it anyway for Level 7). Let `π_S` be the path a planner produces on your map under schedule `S`, and `π*` the path it produces on `M*`. Then:

```
Plan regret   R(S) = J_{M*}(π_S) − J_{M*}(π*)
```

The crucial detail: **both paths are scored on the reference map.** Never evaluate a plan on the map that produced it — that measures self-consistency, not quality. Also report discrete Fréchet distance between `π_S` and `π*` for a geometric read.

**The money plot.** Sweep the schedule, plot memory on x and plan regret on y. You get a curve with a knee. The finding is: *"below 8.9 MB regret is zero — the coarse map produces an identical plan — and above the knee it degrades in a way we can quantify."* That single figure justifies the entire project in a way no memory bar chart can.

**It also becomes the online policy, for free.** Run one forward Dijkstra (cost-to-come `f`) and one backward Dijkstra (cost-to-go `g`) on the coarse traversability grid. For any cell, the best path *through* it costs `T(c) = f(c) + g(c)`. Refine only where `T(c) − J(π*) < τ` — the corridor of near-optimal alternatives. Cells outside that band cannot change the decision no matter how finely you resolve them, so refining them is provably wasted compute. One extra O(N) pass. Details in `sih-math.md` §8.

**Why this is safe to attempt in eight days:** the offline metric needs only a grid A\* and the reference map you already have. That is one dev-day. The online policy is a stretch goal. Even the offline curve alone is the strongest slide in the deck.

### 2.2 — Conservative multi-resolution queries with a proved no-false-negative property

Borrowed from graphics — maximum mipmaps (Tevs, Ihrke & Seidel, I3D 2008) and hierarchical Z-buffers (Greene et al., SIGGRAPH 1993) — and, as far as I can find, never applied to 2.5D traversability.

**The problem it solves.** Standard mip pyramids average. Averaging heights is actively dangerous: a coarse cell straddling a kerb reports the mean and *hides the hazard*. So coarse queries are cheap but unsafe, and you can't use them.

**The fix.** Store per-block `H_max`, `H_min`, `C_min` (ceiling) instead of means, and an AND-mask of the traversability bitfield. Then a coarse query is **conservative**:

```
SAFE(B)  ⟺  H_max(B) − H_min(B) < step_max
         ∧  C_min(B) − H_max(B) > vehicle_height
         ∧  n_min(B) ≥ n_threshold
```

If `SAFE(B)` holds, **every** cell in `B` is traversable — provable in two lines (§7). If the AND-mask shows a hard failure, every cell fails. Otherwise descend one level. You pay fine resolution only where the coarse answer is genuinely ambiguous, and you can never get a false "traversable."

Cost: a 4-ary pyramid adds 1/3 to the storage of whatever layers you build it over. Build it over ground, ceiling, and the traversability byte only — about 1.7 MB. Cheap, provable, and it gives you a theorem for the report.

### 2.3 — Uncertainty-honest split/merge with a round-trip theorem

v2 identified variance-inflation-on-split as the mathematically interesting core. It is. v4 makes it provable and adds the missing half (merge, per 1.4).

Two theorems, both testable as unit tests, both go on a slide:

**Theorem 1 (Variance monotonicity).** Splitting a cell with non-zero local gradient strictly increases child variance. `σ²_child = σ²_parent + κ‖∇z‖²(c_p² − c_c²) > σ²_parent`. Splitting a perfectly flat road costs nothing, which is the correct limiting behaviour.

**Theorem 2 (Round-trip idempotence).** With a one-bit `derived` flag, `merge(split(c)) = c` **exactly**, in both mean and variance, when no measurement intervenes. Without the flag it is not — inflation on split would not be undone by merge, and the map would drift toward uncertainty every time the vehicle changed speed. This is a real bug that the flag prevents, not a formality.

That flag costs one bit and turns "we handle split and merge" into "our split and merge form an exact inverse pair, and here is the test." Derivations in §4–5.

---

## Part 3 — The system, consolidated

### 3.1 Ring schedule — settled

**Default: 5 / 10 / 20 / 40 cm.** Ablation: 5 / 10 / 50 cm. This is Enigma's call from the group chat and it is right: 50 cm was only broken *next to* 20 cm (50 ÷ 20 = 2.5), not on its own (50 ÷ 10 = 5, integer).

| Ring | Half-width | Cell | Cells | Justification |
|---|---|---|---|---|
| 0 | 0–10 m | 5 cm | 160,000 | s_az = 3.5 cm |
| 1 | 10–25 m | 10 cm | 210,000 | s_az = 8.7 cm |
| 2 | 25–50 m | 20 cm | 187,500 | s_az = 17.5 cm |
| 3 | 50–100 m | 40 cm | 187,500 | s_az = 34.9 cm |
| | | **Total** | **745,000** | |

Ablation schedule: 160,000 + 210,000 + 150,000 = **520,000** cells.

**Generalise the rule, don't hard-code it.** The requirement is *integer ratio between consecutive rings*, not powers of two. Powers of two make indexing a bit-shift; other integers make it an exact integer divide. Both are drift-free. Ship a `validate()` that rejects non-integer ratios, and a **second** check that warns when cell size diverges by more than 2× from `s_az(r)` at that range — because 5 cm cells out to 100 m passes the integer test and is nonsense (this was flaw E4, and the sampling table is the fix).

### 3.2 Anisotropic foveation — settled

Circular foveation spends equal resolution 80 m behind and 80 m ahead. Elongate along the direction of travel, scaled by speed, in the vehicle frame:

```
d_aniso = max( x⁺/a_f(v),  x⁻/a_r,  |y|/a_s(v) )
a_f(v) = clamp(1 + κ_f · v/v_ref, 1, 2)      forward stretch
a_s(v) = 1 / (1 + κ_s · v/v_ref)              lateral squeeze
a_r    = 1                                     rear: never stretched
```

**Take anisotropy from the sides, not the back.** Rear resolution floor: never coarser than 20 cm within 50 m behind. Closing traffic is exactly where coarse cells hurt.

**Two correctness notes that are easy to get wrong:**

- Anisotropy changes ring *membership* only. Every cell stays on the same base 5 cm lattice, so nesting and alignment are untouched. Say this explicitly — it looks like it should break alignment, and a judge may assume it does.
- **Add hysteresis.** Split at boundary `R_L`, merge only at `R_L(1+ε)` with ε ≈ 0.1. Without it, a cell sitting on a boundary while the vehicle changes speed will split/merge every frame, burning the refinement pool and — because of Theorem 1 — inflating variance on every cycle.

### 3.3 The cell — 12 bytes, settled

| Field | Type | B | Note |
|---|---|---|---|
| ground_height | int16, 1 cm | 2 | |
| ceiling_height | int16, 1 cm | 2 | lowest thing overhead |
| height_variance | uint8, log-quantised | 1 | |
| log_odds | int8 | 1 | occupancy |
| class | uint8 | 1 | **Boyer–Moore majority**: 4-bit candidate + 4-bit counter |
| reflectivity | uint8 | 1 | range-normalised — lane markings, wet road |
| obs_count | uint8 saturating | 1 | |
| frames_since_seen | uint8 saturating | 1 | |
| traversability | uint8 | 1 | **bitfield, not scalar** — one bit per failing condition |
| flags | uint8 | 1 | derived / refined / blind / dynamic |
| *(reserved)* | | 1 | alignment |

**Class fusion in one byte** is the neat trick here. You cannot store a K-vector histogram. Boyer–Moore streaming majority gives you the true majority class in constant memory: match → increment, mismatch → decrement, zero → adopt new candidate. Never average softmax vectors across frames.

**Traversability as a bitfield, not a scalar** (flaw E5). Same byte, one bit per failing condition — overhead clearance, slope, step, roughness, class, confidence. A planner that knows a cell failed on *clearance* behaves differently from one facing a *slope*. It also strengthens the framework claim: consumers apply their own policy instead of inheriting yours.

**1 cm height quantisation — justified, not defaulted** (answering the group-chat challenge). Quantisation noise is `q²/12`, so σ = 2.9 mm. Sensor height uncertainty is ~8 mm at 5 m and ~87 mm at 50 m. Quantisation is ≤ 1/3 of sensor noise at the closest range and negligible everywhere else, and a standard 12 cm kerb resolves into 12 distinct levels. int16 at 1 cm covers ±327 m. **Keep 1 cm; state these three numbers.**

**Memory, with explicit assumptions** (recompute if the cell changes):

| Representation | Size | Ratio |
|---|---|---|
| Dense 3D voxels, 5 cm, 200×200×8 m, 1 B/voxel (2.56 G voxels) | 2.56 GB | **286×** |
| Sparse/hashed 3D, realistic surface occupancy, 8 B/voxel | ~130–240 MB | ~15–27× |
| Uniform 5 cm 2.5D, same 12 B cell | 192 MB | **21.5×** |
| **Ours (4-ring)** | **8.94 MB** | — |
| Ours (3-ring ablation) | 6.24 MB | 30.8× vs uniform |

Report all four, in that order. Leading with 286× alone looks like cherry-picking; volunteering the sparse-3D number before someone else raises it reads as good faith. And note: **the 21.5× ratio is invariant to bytes-per-cell** — it is a pure cell-count ratio — which is why you can afford to add fields without weakening the headline.

### 3.4 Memory bound — now actually true

Flaw E2 was correct: "compile-time bounded" was false. Three things to preallocate:

- **Grid:** fixed arrays per ring, allocated once. ✓ already bounded
- **Refinement pool:** 512 blocks × 16 cells × 12 B = **98 KB**, fixed. Eviction by priority = closeness × dynamism × time-to-collision.
- **Transient layer:** preallocate as a fixed grid, *not* scaled to dynamic point count.
- **Tracked object list:** cap at N tracks with the same priority eviction.

Also fix E1 — the pool and ring migration fought each other. **Define refinement as "levels finer than the current ring," never as an absolute cell size**, and release a block back to the pool automatically when the schedule overtakes it. Otherwise blocks near the vehicle hold budget for refinement the schedule provides free, while being priority-protected from eviction, and the pool degrades to useless exactly as you approach things.

Slide phrasing: **"bounded at 9.1 MB, degrading gracefully by dropping the least relevant tracks."** A bound you can hold beats a bound you can't.

### 3.5 Pipeline levels — unchanged from v2 except where noted

| Level | What | v4 change |
|---|---|---|
| 0 | Load, transform, cache | Run **GT poses and KISS-ICP in parallel from day one**, report the gap |
| 1 | Range image | 64×512 per FLARES, sweep as config; keep inverse index |
| 2 | Semantic segmentation | **GT 19-class labels from the raw `.label` files** — FRNet dropped, see 3.6 |
| 3 | Motion | GT `moving-*` labels for the map demo; residual images as the learned path |
| 4 | Scatter to variable grid | Kalman height, log-odds ×3-state, Boyer–Moore class, **fixed-point integer atomics** |
| 5 | Persist and forget | Semantic gate → transient layer; visibility cleanup; decay |
| 6 | Dashboard | Rerun; decoupled from pipeline |
| 7 | Evaluate | Reference map, per-ring metrics, **plan regret** |

### 3.6 ⛔ The training decision that saves your deadline

v2 and the additions doc both assume retraining FRNet on the 25-class multi-scan config to get `moving-*` labels. **That is several days of GPU time you do not have, and it is not your contribution.**

You do not need it. **The `moving-*` labels already exist in SemanticKITTI's raw `.label` files** (IDs 250–259). The 19-class collapse happens in the `learning_map` config, not in the data. So:

- **Semantics:** the plan was pretrained FRNet 19-class off the shelf. **Dropped** — the only standalone implementation available (no mmdet3d) does not reproduce the trained network (LeakyReLU where the checkpoint trained with HSwish, FOV 2.0/−24.8 vs 3.0/−25.0, missing RangeInterpolation), giving ~15% point accuracy on every frame. Instead: read the **19-class semantic label straight from the raw `.label` files**, the same source as the motion flag. A real mmengine/mmcv/mmdet/mmdet3d install can swap back in later if time allows; the port code is kept, flagged non-functional.
- **Motion, for the mapping demo and all map metrics:** read `moving-*` straight from the GT label files.
- **Disclose both plainly** — *"semantic and motion labels are both ground truth; the mapping contribution is evaluated independently of segmentation quality."* This is a **feature**, not a compromise: it isolates your contribution from segmentation error, which is exactly what a careful evaluator wants. Zero training, zero inference in the label path.
- **Motion, learned:** residual-image MOS (LMNet approach) as a parallel track. If it lands, you show both and report the gap. If it doesn't, you lost nothing.

This single decision converts the project from "will not finish" to "will finish with slack." Take it.

### 3.7 Output interface — frozen, day one

Non-negotiable, per B1/B2 and the group-chat ruling. **Plain C++/Python core, no ROS dependency.** ROS 2 adapter optional.

```cpp
struct CellQuery {
    float ground_height, ceiling_height;
    uint8 semantic_class;
    uint8 traversability;   // bitfield: 0 = traversable
    uint8 confidence;
    enum { OCCUPIED, FREE, UNKNOWN } occupancy;
    bool  dynamic;
};

CellQuery  query(float x, float y);          // resolution-agnostic
bool       is_traversable(float x, float y);
QueryLOD   query_conservative(AABB region);  // SAFE / BLOCKED / MIXED  (§2.2)
ObjectList dynamic_objects();                // tracked, with velocity
GridMap    export_gridmap();                 // lossy, near-field only, interop
```

**`grid_map::GridMap` is uniform-resolution and cannot hold your map.** Export to it is a lossy near-field convenience, clearly labelled, never the primary output.

**Transient layer interface — decide now** (the group-chat item that was still open): the transient layer **shares the foveated grid geometry** (same lattice, same rings, preallocated) and `query()` returns the **union**: `occupancy = OCCUPIED if persistent OR transient`, with `dynamic = true` when the transient layer supplied it. One combined query, one merge rule, defined in one place — otherwise every downstream consumer inherits your merge problem.

**Do not wipe the transient layer's memory, only its grid.** The grid is frame-fresh; the *tracked object list* persists for ~1 s with constant-velocity prediction. Otherwise a pedestrian briefly hidden by a parked car vanishes entirely, which is the failure mode that matters most.

### 3.8 Evaluation — the harness is the product

Build the reference map in week one. You are blind without it.

1. Aggregate every scan of a held-out sequence into one dense static cloud using GT poses and GT labels.
2. Rasterise at 5 cm, no LOD, no dynamics, no time limit. **This is `M*`.**
3. Score the online map against it.

**Tune on one sequence, report on another.** The group chat caught this: do not tune mapping hyperparameters on sequence 08 and then report on 08. Use 08 for validation as the community does, and hold out a *different* sequence (e.g. 07) purely for mapping-parameter selection. Freeze all thresholds before comparing ring schedules (flaw E6) — otherwise you are comparing tuning effort, not schedules.

| Claim | Metric |
|---|---|
| Near accuracy held, far degrades gracefully | Per-ring mIoU (0–10 / 10–25 / 25–50 / 50–100) |
| Small classes not ignored | Per-class IoU, pedestrian and cyclist called out |
| No ghosts | Dynamic removal rate **and** static preservation rate |
| Map is correct | Per-ring height RMSE vs `M*` |
| Coarsening is justified | **ρ = IL / spread** — information loss over intrinsic sub-cell spread; ρ ≈ 1 is optimal (§9) |
| **Coarsening is free** | **Plan regret R(S), memory-vs-regret curve** |
| It's small | Cells and MB vs all three baselines |
| It's fast | Per-stage latency, mean **and p99** |
| Migration is honest | Round-trip test; variance strictly increases on split |
| Localisation isn't hiding the error | Per-ring RMSE under GT poses vs KISS-ICP |
| Anisotropy isn't starving the rear | All of the above split by azimuth sector (front/side/rear) |

**Ring 0 will tie with the uniform baseline** (flaw E3) — because in Ring 0 they *are* the same map. That is the correct result and presented naively it reads as "no benefit." Report **accuracy retention per megabyte**, and state the finding as *"identical near-field quality at 21× less memory."*

**Build a 3D baseline that actually allocates.** The problem statement says *demonstrating* memory reduction, not calculating it. A stub dense-voxel grid that really allocates 2.56 GB, with both counters ticking live on the dashboard, is worth ten times a number in a table.

### 3.9 Dashboard

```
┌──────────────────────────────────┬─────────────────────────┐
│  TOP-DOWN MAP                    │  MEMORY (live)          │
│  · colour by class               │   Ours         8.9 MB   │
│  · height as contour/brightness  │   Uniform 2.5D  192 MB  │
│  · VISIBLE CELL BOUNDARIES       │   Dense 3D     2.56 GB  │
│  · unknown ≠ free (distinct)     │   Ratio          286×   │
│  · blind cone shown as unknown   ├─────────────────────────┤
│  · tracked boxes + velocity      │  FRAME TIME    p50  p99 │
│                                  │   range image   2ms  3  │
│  [ GHOST REMOVAL:  ON / OFF ]    │   segmentation  8ms 11  │
│  [ SCHEDULE: 5/10/20/40 ▾ ]      │   scatter       3ms  5  │
│  [ PLAN OVERLAY:   ON / OFF ]    │   cleanup       4ms  6  │
│                                  │   total        17ms 25  │
└──────────────────────────────────┴─────────────────────────┘
```

Three controls carry the demo. The **ghost toggle** is the five most persuasive seconds. The **schedule dropdown** switching live with the memory counter jumping proves the config API in one gesture. The **plan overlay** — showing the path on your map and the path on the reference map lying on top of each other — is the visual form of your headline claim.

If a judge cannot *see* the cells growing with distance, you have hidden your own contribution. Draw the boundaries.

### 3.10 "Real-time" means naming the hardware

> "Benchmarked on ▢ [GPU], processing at ▢ FPS against the 10 Hz sensor rate — ▢× headroom."

40 FPS against 10 Hz is **4×** headroom, not 3×. Compute it. Do not quote a Jetson number you have not measured; say "desktop GPU, embedded deployment is future work."

---

## Part 4 — Scope boundaries, stated before someone finds them

State these in the report. Naming a limit reads as understanding; being caught by one reads as an oversight.

- **Vertical extent −2 m to +6 m.** Overpasses and multi-storey structures are out of scope.
- **Blind cone: 3.74 m radius**, not the 1–2 m assumed earlier (`r = h/tan|φ_min|`, h = 1.73, φ = −24.8°). That is **11% of Ring 0** unobservable in any single frame. Mark it **unknown**, never free. Report both the instantaneous blind fraction and the persistent-unknown fraction after ego-motion fills it.
- **Slow motion is geometrically undetectable beyond ~25 m.** A pedestrian moves 14 cm between scans at 10 Hz; that is smaller than a Ring-2 cell. Beyond 25 m, motion detection is a semantic prior ("that shape is a person, people move"), not a measurement. Say this explicitly.
- **Negative obstacles: ~8 m for a 30 cm defect** (§1.2).
- **Ground-truth poses are an assumption**, mitigated by running KISS-ICP in parallel and reporting the gap.
- **Camera fusion, planning, full 3D voxels: out of scope**, deliberately.
- **Open-set / anomaly detection is not in the problem statement.** It is a self-imposed stretch. Do not let it block anything.

**The counter-argument to have ready:** *"standard planners want uniform grids, so your variable-resolution map needs resampling and you give the savings back."* Your answer: the query API is resolution-agnostic (§3.7), the conservative pyramid lets a planner query coarse safely (§2.2), and the plan-regret result shows the planner reaches the same decision anyway (§2.1). That is three independent answers to the hardest question you will be asked — which is why 2.1 and 2.2 are worth the days they cost.

---

## Part 5 — The seven things most likely to sink you

Ranked by how often they actually happen.

1. **Coordinate frame confusion.** Write every transform down day one. Build the static-wall test: map a known wall across 100 frames, assert it doesn't move.
2. **Disk throughput starving the GPU.** SemanticKITTI is ~200 GB. Pre-process to a cached compact format *before* anything else. This kills more projects than algorithms do.
3. **No reference map until week two.** You tune the ring schedule by eye and have no defence when asked whether coarsening lost the kerb.
4. **Class imbalance.** A satisfying 90% accuracy that never predicts pedestrian. Per-class IoU from epoch one. *(Mitigated to near-zero by 3.6 — you aren't training.)*
5. **Odometry drift misdiagnosed as a fusion bug.** A blurred map with 30 cm-thick walls is a localisation bug. GT poses first, always, to isolate.
6. **Demo slower than the pipeline.** An excellent 40 FPS system presenting at 6 FPS. Decouple rendering from processing on day one.
7. **Non-deterministic float atomics.** Float addition is not associative, so two identical runs give different maps and you cannot bisect a bug that moves. You are already quantising to 1 cm — accumulate in fixed-point integers. Integer atomics are exactly associative and bit-identical run to run.

---

## Part 6 — Where the risk now sits

After v4 the architecture is closed. What remains is **scope**, and the honest position is that the design is more sophisticated than eight days can carry in full.

The execution plan (`sih2026-execution-plan.md`) makes the cuts explicitly. The short version: the grid engine, the reference map, ghost removal, and the offline plan-regret curve are the spine and must land. The conservative pyramid, residual MOS, instance tracking, and the online LOD policy are ranked stretch goals that get dropped in that order.

Cut from the top of that list, never from the spine.
