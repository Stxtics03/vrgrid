# Team Assignments — Named

*Overrides Part 2 of the execution plan. The day-by-day gates in that document are unchanged; substitute names for D1/D2/D3 using the mapping below.*

| | Person | Role |
|---|---|---|
| **Build** | **Aakash** | Grid engine + evaluation harness — *the contribution and its proof* |
| | **JP** | Perception front-end + dashboard — *data in, pixels out* |
| | **Shrestha** | GPU, kernels, scaling, memory — *the horizontal layer* |
| **Research** | **Srinivas** | R1 — Representation & prior art |
| | **Hriday** | R2 — Dynamics & segmentation |
| | **Pratyushi** | R3 — Traversability, evaluation & the novelty claim |

**Pairing:** Srinivas ↔ Aakash · Hriday ↔ JP · Pratyushi ↔ Aakash (evaluation side).

Shrestha has no dedicated researcher — his work is implementation and measurement, not literature. Pratyushi doubles up because the plan-regret metric and the reference map both live with Aakash.

---

## Shrestha — GPU, kernels, scaling, memory

**The shape of this role is different from the other two.** Aakash and JP own vertical slices; Shrestha owns a horizontal layer that both of them sit on top of. That has one consequence worth stating up front: **his early kernels are a dependency for everyone**, so the Day 0–1 items are not negotiable and cannot slip. After Day 2 the dependency mostly reverses — he becomes the one optimising what others have built.

### Owns

**Kernels and correctness-under-parallelism**
- `scatter()` — the one custom kernel in the project. Points → cells, parallel, with **fixed-point int32 atomics** so results are bit-identical run to run. Float atomics are non-associative and produce bugs that move when you look at them; this must be right from the first commit, never retrofitted.
- Toroidal shift kernel — O(perimeter) boundary clear, not O(area). Ring 3 clears 1,000 cells per shift, not 250,000.
- Visibility cleanup kernel (§10.4) — O(1) per cell range comparison, fully parallel, with the never-clear-a-current-return guard.
- Separate-arrays-per-field layout (SoA, not AoS) for coalesced access.

**Memory — he owns the headline number**
- Preallocate everything at startup: grid arrays per ring, refinement pool (512 × 16 × 12 B), transient layer as a fixed grid, tracked-object list capped at N. No allocation inside the frame loop, ever.
- Enforce and *prove* the compile-time bound. Flaw E2 said the bound was false; Shrestha is the person who makes it true and can demonstrate it holds under a worst-case dense-crowd scene.
- **The allocating dense-3D baseline stub.** The problem statement says *demonstrating* memory reduction, not calculating it. A stub that genuinely allocates 2.56 GB, with both counters ticking live on the dashboard, is worth ten times a number in a table. This is his, and it's a demo-critical item.

**Performance**
- Per-stage latency instrumentation, p50 **and p99**. A system averaging 40 FPS with 200 ms spikes is unsafe, and the spikes almost always come from allocation or the map-shift path — both his.
- Name the GPU, benchmark against the 10 Hz sensor rate, compute headroom (FPS ÷ 10 — 40 FPS is 4× headroom, not 3×).
- Decouple rendering from processing so JP's dashboard can't throttle the pipeline.

**The important non-plumbing piece: the conservative pyramid (§7.2–7.3)**

This is his stretch item and it's deliberately his rather than Aakash's. It's a parallel reduction — build max/min/AND pyramids bottom-up over the ring arrays — which is exactly the shape of work he's already doing, and it comes with a theorem (no false negatives, provable in two lines) and an exhaustive unit test. If it lands, he owns a named contribution rather than only the infrastructure underneath everyone else's.

### Does not own

Split/merge mathematics (Aakash), segmentation (JP), metric definitions (Aakash/Pratyushi). He implements fast; he doesn't define semantics.

### Gates

| Day | Exit criterion |
|---|---|
| 0 | SoA layout committed against the frozen cell struct. Timing harness stubbed. |
| 1 | `scatter()` running on a real scan with integer atomics. **Same sequence twice → byte-identical map hash.** |
| 1 | Toroidal shift: +1 then −1 restores every cell bit-exactly. |
| 2 | Everything preallocated. Zero allocations inside the loop — verify with a profiler, not by reading the code. |
| 3 | Visibility cleanup kernel live. Per-stage p50/p99 on the dashboard. |
| 4 | Allocating 3D baseline running, both counters ticking on screen. |
| 5 | Conservative pyramid + exhaustive no-false-negative test *(stretch)*. |
| 6 | Memory bound demonstrated under worst case. Latency table final, GPU named, headroom computed. |

---

## Aakash — Grid engine + evaluation harness

**This is the contribution and the proof that it works.** Heaviest correctness load on the team. Everything here is either a claim on a slide or the test that backs it.

### Owns

**The grid engine**
- Lattice and indexing (§2) — `i_L = ⌊i_fine/k_L⌋`, integer-exact, with the 10⁶-point partition test. This is the "no alignment errors" claim the problem statement explicitly asks about, and it's provable rather than tuned.
- Ring assignment and per-frame migration. Anisotropic foveation with hysteresis (§6.2–6.3) as a stretch.
- **`split()` / `merge()` with the variance mathematics** (§4–5) — law of total variance on merge, inflation on split, the `derived` bit, and both theorems as passing unit tests. This is the mathematically distinctive core; nobody else touches it.
- Refinement pool with priority eviction and automatic release when the ring schedule overtakes a block (fixes flaw E1).
- Kalman height fusion, three-state log-odds, Boyer–Moore class fusion.
- Traversability bitfield (§7.1) — six conditions, gradient by central differences.
- The `query()` / `is_traversable()` API and the config validator (both checks: integer ratio *and* the sampling-divergence warning).

**The evaluation harness**
- Reference map construction — aggregate a held-out sequence with GT poses, strip `moving-*`, rasterise at 5 cm. **Start Day 1; it's the long pole.**
- Per-ring height RMSE, per-ring mIoU, the coarsening ratio ρ = IL/spread (§9.3).
- **The plan-regret study (§8.1)** — grid A\* on the traversability map, both paths scored on the reference map, schedule sweep, the memory-vs-regret curve. The headline figure.

**Why these two together:** he needs the reference map to validate his own split/merge behaviour, and plan regret needs the traversability layer he built. Splitting them across two people would mean handing off exactly at the point where the maths has to be right.

### Gates

| Day | Exit criterion |
|---|---|
| 0 | Partition test passes on 10⁶ random points. |
| 1 | Reference map for the tuning sequence exists on disk. |
| 2 | **Round-trip exact. Variance strictly up on a synthetic slope, unchanged on flat ground.** Per-ring RMSE printed. |
| 3 | Refinement pool with release condition. Semantic gate → transient layer. |
| 4 | Traversability bitfield. **Memory-vs-regret curve produced, ≥4 schedules.** |
| 6 | ρ computed per ring. Every number reproducible from a script. |

---

## JP — Perception front-end + dashboard

**Owns both ends of the pipeline** — raw points in at one end, the thing judges actually look at at the other. Lower correctness risk than Aakash, higher demo risk.

### Owns

**Front-end**
- Loader, cached preprocessing (start the 3-sequence download in hour one — it's the only critical-path item cleverness can't speed up).
- **Coordinate transforms and the static-wall test.** Frame confusion is the single most common silent bug in this class of project: the map looks plausible and slowly rotates. Map a known wall across 100 frames, assert it doesn't move. Day 0.
- Range-image projection at 64×512 with the inverse index (reversibility is mandatory — you must map a pixel back to its exact source point).
- Semantic labels: 19-class, read straight from the raw `.label` files. FRNet was the plan but the available standalone port does not reproduce the trained network (~15% accuracy) — it is flagged non-functional; a real mmdet3d install is a possible later swap.
- Motion labels read from the raw `.label` files (`moving-*` IDs 250–259). Both semantic and motion are ground truth — disclose it; it isolates the mapping contribution from segmentation error.
- Patchwork++ for ground segmentation — use as-is, do not reimplement.
- Reflectivity normalisation (§10.3): `ρ̂ = I·r²/cos θ_inc`. Lane markings and wet road, one byte, no extra sensor.
- *Stretch:* residual-image MOS; instance clustering by connected components **on the range image**, not DBSCAN on raw points.

**Dashboard**
- Rerun, built Day 0 against a **mock** grid so every level afterwards plugs into something already visible. The failure mode to avoid: a beautiful visualisation of nothing, mistaken for progress. The mock must be replaced level by level, not admired.
- **Visible cell boundaries.** The entire thesis is the variable grid; if a judge can't see cells growing with distance, the contribution is hidden.
- Ghost toggle, schedule dropdown, plan overlay. Memory counters fed by Shrestha, metrics by Aakash.
- Unknown rendered distinctly from free, blind cone shown, colourblind-safe palette.

### Gates

| Day | Exit criterion |
|---|---|
| 0 | Static-wall test passes. Dashboard renders a mock map. Download running. |
| 1 | Range image + inverse index. Semantic labels from `.label` files (FRNet dropped). |
| 2 | Motion labels wired. Patchwork++ integrated. Reflectivity live. |
| 3 | **Ghost toggle: off → trails behind moving cars. On → gone.** Cell boundaries visible. |
| 4 | Full sequence runs without crashing. Instance clustering *(stretch)*. |
| 6 | All three sequences end to end. Dashboard final. |

---

## Research — Srinivas, Hriday, Pratyushi

Modules unchanged from `sih2026-research-modules.md`; swap owners freely if someone's already read into a topic.

| | Owner | Paired with | Day-3 hard deliverable |
|---|---|---|---|
| R1 Representation & prior art | **Srinivas** | Aakash | The cite-or-get-caught list: MLS (2006), Droeschel (2014), Losasso (2004) |
| R2 Dynamics & segmentation | **Hriday** | JP | ⚑ **Day 0, 4 hours:** confirm `moving-*` IDs 250–259 in raw `.label` files |
| R3 Traversability & evaluation | **Pratyushi** | Aakash | ⚑ **Day 3:** verdict on plan-regret novelty vs Psomiadis 2024 |

**Two assignments gate the whole plan.** Hriday's Day-0 label check decides whether the no-retraining plan survives — if those IDs aren't in the raw files, JP falls back to residual MOS the same day and the schedule shifts. Pratyushi's Day-3 verdict decides whether plan regret is the headline claim or gets reframed as validation. Both go to the whole team the hour they land, not in a memo.

Research stops Day 5. All three move to writing.

---

## Load check

| | Spine items | Stretch items | Risk if they slip |
|---|---|---|---|
| Aakash | 3 of 5 | 2 | **Fatal** — the contribution and its proof |
| JP | 2 of 5 | 2 | **Demo dies** — no visible output |
| Shrestha | supports all 5 | 1 | **Everything slows**, nothing stops |

Aakash is the bottleneck. If he's behind on Day 2, Shrestha moves to help on split/merge before touching anything else — Day 2 is the mathematically load-bearing day and it cannot slip. Watch that gate specifically.
