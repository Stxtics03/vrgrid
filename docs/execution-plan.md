# Execution Plan — 6 People, 8 Days, Deadline 5 September Morning

*Companion to master plan v4, sih-math.md, and the research modules.*

**Today is Friday 28 August. You have 8 working days: 28, 29, 30, 31 August and 1, 2, 3, 4 September, with the morning of 5 September for submission only.**

---

## Part 1 — The scope decision, made explicitly

The design in master v4 is more sophisticated than eight days can carry in full. The most dangerous thing this team can do is start everything and finish nothing. So the cuts are made here, in advance, in writing.

### The spine — must land, non-negotiable

If these five things work on 4 September you have a strong submission, even with every stretch goal dropped.

1. **Grid engine** — lattice, rings, toroidal shift, split/merge with variance, hard memory bound
2. **Reference map + per-ring metrics** — without this you are tuning blind and have no defence
3. **Ghost removal with a live toggle** — the five most persuasive seconds of the demo
4. **Dashboard with live memory counters and a real allocating 3D baseline**
5. **The offline plan-regret curve** — the one figure that makes this research rather than engineering

### Stretch goals, ranked — drop from the top of this list

6. Conservative pyramid with the no-false-negative test *(strong, cheap, do it if Day 5 is clean)*
7. Residual-image MOS *(the learned motion path)*
8. Instance clustering + tracked objects with velocity arrows
9. Online corridor-based LOD policy
10. KISS-ICP parallel run and the GT-vs-odometry gap
11. DynamicMap Benchmark comparison
12. Anisotropic speed-scaled foveation

**Cut from the top of the stretch list, never from the spine.** If on Day 5 the spine is not done, cut items 6–12 entirely and spend Days 6–7 polishing the spine. A perfect spine beats a broken everything.

### Three decisions that save the deadline

**A. Do not retrain anything — and, as it turned out, do not run inference for labels either.** The plan was pretrained FRNet 19-class for semantics; the available standalone port does not reproduce the trained network (~15% accuracy — see the research log and master v4 §3.6), so **both** the 19-class semantic label and the `moving-*` motion flag are now read straight from SemanticKITTI's raw `.label` files (`src/perception/semantics.py`). This removes several days of GPU time from the critical path and **isolates the mapping contribution from segmentation error**, which is what a careful evaluator wants anyway. Disclose it plainly. A real mmdet3d FRNet install remains a possible later swap.

**B. Do not download all of SemanticKITTI.** You need three sequences, not twenty-two. Sequence **00** (development), **07** (mapping-hyperparameter tuning, held out from reporting), **08** (validation, reported). ~40 GB instead of ~200 GB. Start the download in the first hour of Day 0 — it is the only thing on the critical path that money and cleverness cannot speed up.

**C. Freeze interfaces before parallel work begins.** The cell struct and the five function signatures — `scatter()`, `fuse()`, `split()`, `merge()`, `query()` — are frozen on Day 0 and everyone builds against stubs. If three people build against unfrozen interfaces for four days, integration on Day 5 will fail and there is no recovery time.

---

## Part 2 — Team structure

Six people. Three research, three development, **paired one-to-one**. The pairing is the point: research that isn't attached to a developer produces a bibliography nobody reads.

> **Names are assigned in `sih2026-team-assignments.md`, which overrides this section.** Throughout the day-by-day below, read **D1 = Aakash** (grid engine + evaluation harness), **D2 = JP** (perception front-end + dashboard), **D3 = Shrestha** (GPU, kernels, scaling, memory). Research: **R1 = Srinivas**, **R2 = Hriday**, **R3 = Pratyushi**.

| Pair | Research | Development | Owns |
|---|---|---|---|
| **α** | R1 Srinivas — Representation & prior art | D1 Aakash — Grid engine + evaluation | The contribution and its proof |
| **β** | R2 Hriday — Dynamics & segmentation | D2 JP — Perception + dashboard | Data in, pixels out |
| **γ** | R3 Pratyushi — Traversability & evaluation | (paired to Aakash, evaluation side) | The novelty claim |
| — | — | D3 Shrestha — GPU, scaling, memory | The horizontal layer everyone sits on |

**How a pair operates.** The researcher's job is to keep their developer from building the wrong thing or rebuilding an existing thing. They sit together. The researcher delivers a decision memo every 48 hours (see research modules doc) and is available for same-day questions. The researcher does **not** write code and the developer does **not** read papers — that separation is what makes six people faster than three, rather than slower.

**Escalation rule:** if a researcher finds prior art that invalidates a claim, it goes to the whole team the same hour, not in the next memo. Three such landmines are already known: MLS maps (2006) vs our two-layer cell, Droeschel (2014) vs our ring grid, and Psomiadis (2024) vs our plan-regret claim. R1 and R3 must resolve those by Day 3.

### Individual ownership

**D1 — Grid engine.** The lattice and indexing (§2), ring assignment and migration, toroidal shift, `scatter()`, Kalman fusion, `split()`/`merge()` with the variance mathematics and the `derived` bit (§4–5), the refinement pool, the memory bound. Then the conservative pyramid if time. **This is the contribution; D1 should be your strongest systems programmer and should touch nothing else.**

**D2 — Perception front-end.** Loader and cached preprocessing, coordinate transforms and the static-wall test, range-image projection with inverse index, GT semantic + motion labels from the raw `.label` files (FRNet dropped — see decision A), Patchwork++ ground segmentation, reflectivity normalisation. Then residual MOS and instance clustering if time.

**D3 — Evaluation and dashboard.** Reference-map construction, the metrics harness, the Rerun dashboard, the allocating 3D baseline stub, the plan-regret study. **D3 owns the two things judges actually see** — the demo and the numbers — which is why it is a full-time role and not an afterthought.

**R1 / R2 / R3** as specified in `sih2026-research-modules.md`.

### Standing meetings — two per day, both short

- **09:00, 10 minutes, standing.** What's blocked. Not what you did.
- **20:00, 20 minutes.** Gate review against the day's exit criterion below. If the gate fails, the recovery decision is made *that evening*, not the next morning.

---

## Part 3 — Day by day

Each day has an **exit gate**. A gate is binary and demonstrable. "Mostly working" fails the gate.

---

### Day 0 — Friday 28 August · Decisions and scaffolding

**Everyone, first two hours, together.** No code until this is done.

- Start the SemanticKITTI download (sequences 00, 07, 08). **First thing, before the meeting.**
- Freeze the cell struct: 12 bytes exactly as master v4 §3.3. Write it as a header file, commit it, and treat it as immutable for eight days.
- Freeze the five function signatures and the `CellQuery` output struct (§3.7).
- Decide and write down: vertical extent −2 to +6 m; ring schedule 5/10/20/40 default with 5/10/50 ablation; benchmark GPU named; 1 cm height quantisation.
- Repo scaffolded, CI running a stub test, everyone can push.

**Then split:**

| | Task |
|---|---|
| D1 | Lattice + indexing + the partition unit test (§2.4). Nothing else. |
| D2 | Loader, transforms, the static-wall test. Begin preprocessing to cached format. |
| D3 | Rerun dashboard skeleton against a **mock** grid — random data, correct shape. Memory counters visible and ticking. |
| R1 | Triebel 2006 and Droeschel 2014. **Memo by tonight.** |
| R2 | ⚑ **Confirm `moving-*` IDs exist in the raw `.label` files.** One line, sent within four hours — decision C depends on it. |
| R3 | Psomiadis 2024 and Larsson 2021. Begin the novelty verdict. |

> **GATE 0:** Cell struct committed and immutable. Partition test passes on 10⁶ random points. Dashboard renders a mock map. Download running. R2's label verdict delivered.

---

### Day 1 — Saturday 29 August · Core mechanics

| | Task |
|---|---|
| D1 | Ring assignment, toroidal shift with the O(perimeter) clear, `scatter()` with **fixed-point integer atomics**. Shift round-trip test. |
| D2 | Range-image projection + inverse index at 64×512. FRNet inference running on one frame end-to-end. |
| D3 | Reference-map builder: aggregate sequence 07 with GT poses, strip `moving-*`, rasterise at 5 cm. **This is the long pole in evaluation — start it today.** |
| R1 | Losasso 2004, Fankhauser 2014. Confirm the measurement-variance model matches §3.2. |
| R2 | FRNet setup memo: exact config, checkpoint, class map, gotchas. |
| R3 | **Novelty verdict on plan regret.** Metric pseudocode to D3. |

> **GATE 1:** A real scan scatters into the real grid and renders on the dashboard. Ugly is fine. Shift round-trip is bit-exact. Reference map for one sequence exists on disk.

---

### Day 2 — Sunday 30 August · Fusion and the mathematics

| | Task |
|---|---|
| D1 | Kalman height update with range-dependent variance (§3). Three-state log-odds. Boyer–Moore class fusion. `split()`/`merge()` with the law of total variance and the `derived` bit. **All four unit tests from §4.4 and §5.5 must pass.** |
| D2 | Motion labels wired in. Patchwork++ integrated for ground. Reflectivity normalisation (§10.3). |
| D3 | Per-ring height RMSE against the reference map. First real numbers. Allocating dense-3D baseline stub. |
| R1 | PCT, Adaptive Patched Grid Mapping. Positioning paragraph. |
| R2 | DynamicMap metric definitions → D3. |
| R3 | Comparison-table skeleton. Traversability decomposition from SALON/EVORA. |

> **GATE 2:** Round-trip test passes exactly. Variance strictly increases on split over a synthetic slope, and is unchanged on flat ground. Per-ring RMSE printed for the default schedule. **This is the mathematically load-bearing day — do not let it slip.**

---

### Day 3 — Monday 31 August · Ghosts

| | Task |
|---|---|
| D1 | Semantic gate → transient layer (shared geometry, preallocated). Visibility cleanup (§10.4) **with the never-clear-a-current-return guard.** Decay. Refinement pool with priority eviction and the automatic release when the schedule overtakes a block. |
| D2 | Full pipeline runs a whole sequence without crashing. Frame-time instrumentation per stage, p50 and p99. |
| D3 | **Ghost toggle live on the dashboard.** DR / SP / F metrics. Cell-boundary rendering — the judge must *see* the cells growing. |
| R1 | Tevs 2008. Verify nobody has already imported max-mipmaps to traversability. |
| R2 | Baseline number table for the comparison slide. |
| R3 | Related-work paragraphs begin. |

> **GATE 3:** ⚑ **Toggle ghost removal off — you see trails behind moving cars. Toggle it on — they vanish.** If that works, you have a demo. Everything after this is improving a thing that already exists, which is a far better position than building toward one.

---

### Day 4 — Tuesday 1 September · The headline result

| | Task |
|---|---|
| D1 | Traversability bitfield (§7.1) — six conditions, gradient by central differences. Schedule switching live via config, validator with both checks. |
| D2 | Instance clustering by connected components **on the range image**, not DBSCAN on raw points. Per-cluster class vote and velocity. |
| D3 | ⚑ **The plan-regret study.** Grid A\* on the traversability map; `R(S)` per §8.1, both paths scored on the reference map. Sweep the schedules. **Produce the memory-vs-regret curve.** |
| R1 | Related-work section assembled. |
| R2 | Segmentation and dynamic-removal paragraphs. |
| R3 | Traversability and evaluation paragraphs. Verify the plot's framing. |

> **GATE 4:** The memory-vs-regret curve exists as a figure, with at least four schedules on it. This is the slide that distinguishes you from every other team attempting this problem statement.

---

### Day 5 — Wednesday 2 September · Stretch, and the honest cut

**Morning: full-team scope review, 30 minutes.** Look at the spine. If any of the five spine items is incomplete, **cancel all stretch work now** and spend Days 5–7 on the spine. Make the call in the morning, not at midnight.

If the spine is clean, take stretch goals in ranked order:

| | Task |
|---|---|
| D1 | Conservative pyramid (§7.2) + the exhaustive no-false-negative test. Then anisotropic foveation with hysteresis. |
| D2 | Residual-image MOS. Object tracking that survives brief occlusion. |
| D3 | Sector-split metrics (front/side/rear). Online corridor LOD if D1 and D3 are both clear. KISS-ICP comparison. |
| R1–R3 | **Research stops. All three move to writing** — report, slide content, figure captions. |

> **GATE 5:** Every stretch item either works and is integrated, or is reverted cleanly. **No half-merged branches after tonight.** A half-integrated feature on Day 6 costs more than it is worth.

---

### Day 6 — Thursday 3 September · Freeze and harden

**Code freeze at 18:00.** After that, bug fixes only — no new features, no exceptions, no "it's only a small change."

| | Task |
|---|---|
| D1 | Bug fixes. Confirm the memory bound holds under a worst-case dense-crowd scene. Final numbers recomputed against the actual cell size. |
| D2 | Run all three sequences end to end. Latency table, p50 and p99, with the GPU named and headroom computed (FPS ÷ 10, not guessed). |
| D3 | Every metric table filled with real numbers. Dashboard polished, colourblind-safe palette, unknown visually distinct from free, blind cone shown. |
| R1–R3 | Report draft complete. Every citation verified against a real PDF. |

> **GATE 6:** Every number that will appear on a slide has been produced by a script that can be re-run. **No numbers typed from memory.** Nothing on a slide that isn't reproducible.

---

### Day 7 — Friday 4 September · Demo and deck

No code today. If something is broken, ship without it.

- **Record the demo video.** Multiple takes. The three moments, in order: cell boundaries growing with distance → ghost toggle → schedule dropdown with the memory counter jumping. Under three minutes.
- **Build the deck.** Suggested order: the pitch (v4 §0) → ring-sweep filling figure (the derived argument) → the grid → the split/merge theorems → ghost toggle → **the plan-regret curve** → memory table with all three baselines → limitations, stated plainly.
- **Rehearse the hard questions.** Assign one person per question:
  - *"Isn't this just a clipmap / just Droeschel 2014?"* → R1
  - *"Planners want uniform grids, so you give the savings back."* → the three-part answer in v4 Part 4
  - *"Why not full 3D / wavemap?"* → R1
  - *"Your Ring 0 shows no improvement."* → accuracy-per-megabyte framing (E3)
  - *"Can you detect a pothole at 50 m?"* → no, and here is the equation that says why (§1.4)
  - *"Did you tune on your test set?"* → held-out sequence 07, thresholds frozen
- **Submission package assembled and checked** by someone who did not build it.

> **GATE 7:** Video recorded, deck complete, package assembled and independently checked, by 20:00.

---

### 5 September, morning — Submit

Buffer only. Submit early. Do not touch the code.

---

## Part 4 — Risk register

| Risk | Likelihood | Trigger | Response |
|---|---|---|---|
| Download not finished by Day 1 | Medium | Started late | Work on sequence 00 alone; 07/08 arrive later. Never blocks D1. |
| `moving-*` not in raw labels | Low | R2 Day 0 | Fall back to residual MOS immediately; the map contribution is unaffected. |
| FRNet won't run / checkpoint mismatch | ~~Medium~~ **realised, Day 1** | D2 Day 1 | **Done:** standalone port is non-functional (~15% acc); shipped with GT 19-class semantic labels from the raw `.label` files. Segmentation is not the contribution. Real mmdet3d install is a possible later swap. |
| Plan-regret claim already published | Low–Medium | R3 Day 3 | Reframe as *validation* of the approach in a new domain; keep the curve, it is still the best figure in the deck. |
| Split/merge maths doesn't pass tests | Medium | D1 Day 2 | **Spine item — stop everything and fix.** All three devs on it if needed. |
| Integration fails on Day 5 | Medium | Interfaces drifted | Prevented by Gate 0. If it happens anyway, revert to last green commit and ship that. |
| Dashboard slower than the pipeline | High if ignored | D3 Day 1 | Decouple rendering from processing on Day 0. Render at 30 Hz from a snapshot. |
| Non-deterministic results | High if ignored | D1 Day 1 | Fixed-point integer atomics from the first commit, never retrofitted. |
| Scope creep | **Very high** | Continuous | Gate 5's morning review is the mechanism. Somebody must be willing to say no. |

---

## Part 5 — What "done" means

On the morning of 5 September, you should be able to make these six statements, each backed by a number you can reproduce:

1. *"Our map uses 8.9 MB where a uniform 5 cm 2.5D grid uses 192 MB and a dense 3D voxel grid uses 2.56 GB — 21× and 286× respectively, with both counters allocating live on screen."*
2. *"Near-field accuracy is identical to the uniform baseline; per-ring height RMSE against our reference map is ▢ cm at 0–10 m rising to ▢ cm at 50–100 m."*
3. *"Coarsening is justified: ρ = IL/spread is ▢ across all rings, meaning we lose approximately what the terrain's own sub-cell variability costs and no more."*
4. *"Coarsening is free where it matters: plan regret is zero below ▢ MB — the planner produces an identical path on our map and on the 5 cm reference."*
5. *"Ghosts are removed at ▢% while preserving ▢% of static structure, and you can watch the difference with a toggle."*
6. *"Split and merge are an exact inverse pair, variance strictly increases when we assert detail we did not measure, and here are the tests."*

Statements 3, 4 and 6 are the ones no other team will be able to make.

**And be equally ready to say what you cannot do:** no potholes beyond ~8 m, no pedestrian motion beyond ~25 m geometrically, 11% of Ring 0 permanently blind, no overpasses, ground-truth poses assumed with the odometry gap reported. Stating limits precisely reads as mastery. Being caught by one reads as an oversight.
