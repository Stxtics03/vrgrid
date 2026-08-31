# Gate 3 review — Monday 31 August, 20:00

**Shrestha — GPU, kernels, scaling, memory.** Prepared before the meeting. Every number below comes from a script in `scripts/`, per Gate 6; the command is named next to each one so anyone can re-run it on their own clone.

---

## The 20-minute agenda, in the order I want to spend the time

| # | Item | Who decides | Time |
|---|---|---|---|
| 1 | §10.4 has been written twice — one copy is a stub with Aakash's name on it | Aakash + me | 3 min |
| 2 | The ghost toggle demonstrates a label filter, not the engine | JP + room | 5 min |
| 3 | Point→slot binning is the largest stage in the frame and nobody owns it | Aakash + room | 5 min |
| 4 | 19 classes do not fit in 4 bits — the first real frame raises | Room | 3 min |
| 5 | Visibility scratch: pick a cap, or accept it stays undeclared | Room | 4 min |
| 6 | Two ratifications: eq (32)'s delta, and §7.2's memory figure | Room | 2 min |

Items 1–4 are blockers. Item 5 moves a number on a slide. Item 6 is for the record.

**Suite: 390 passed, 20 skipped. `ruff check .` clean. `main` is green.**

---

## Gate 3 — met on paper, and I do not think it is met in substance

> **GATE 3:** ⚑ Toggle ghost removal off — you see trails behind moving cars. Toggle it on — they vanish. If that works, you have a demo.

### First, the thing I would fix before anything else in this document

**§10.4 exists twice.** `src/gpu/visibility.py: visibility_cleanup()` is implemented, tested and benchmarked — eq (32), the never-clear-a-current-return guard, a preallocated scratch, p50 9.0 / p99 10.6 ms at 200,000 candidates. `src/grid/fusion.py: visibility_cleanup()` is:

```python
def visibility_cleanup(soa, range_image, thresholds) -> None:
    """O(1) per cell by range-image comparison, no ray casting. Math §10.4.
    Hard guard: never clear a cell that has a return in the current scan."""
    raise NotImplementedError("Aakash — Day 3")
```

Same name, same section, same guard in the docstring, different signature, and `tests/test_fusion.py` skips its test with `pytest.skip("visibility_cleanup — Aakash, Day 3")`. The execution plan's Day-3 row does assign §10.4 to D1; I built it in `src/gpu` because the kernel is a range-image comparison and that is my directory. Neither of us was wrong, and that is the point — **this is the failure mode the ownership table exists to prevent, and it still happened.**

So: Aakash, before you spend Day 3 on it — it is done. What is genuinely yours and still open is the half I deliberately did not write: log-odds and the three-state decision (§10.1). Mine returns a `see_through` mask and `apply_miss()` applies it; the decision about what a miss *means* is fusion's. If we agree, the `fusion.py` stub should be deleted rather than left to be filled in, because a `NotImplementedError` with a name on it reads as work outstanding.

**Two minutes of the room's time, and it saves a day of Aakash's.**

### Second, the toggle itself

It does that. `dashboard/pipeline_view.py` splits the moving returns into a separate `world/ghosts` entity, and toggling that entity's eye icon in Rerun makes them appear and disappear. The five seconds work.

**But what is being toggled is the raw `moving-*` label on the point cloud, not the map.** `get_display_points(frame, ghost_removal)` filters points before they are drawn. `visibility_cleanup()` — the §10.4 kernel that actually removes ghost *cells* from the persistent map, with the never-clear-a-current-return guard — is not on that path. It is not called anywhere outside its own tests.

Why this matters more than it looks: the trails a judge sees behind a moving car are cells that were fused from that car and then never cleared. Filtering the input points means those cells are never created in the first frame, which looks identical on screen and demonstrates nothing about the mapping engine. The claim the demo is supposed to support is *"our rolling local map removes dynamic objects online"*; what it currently supports is *"we can hide points whose ground-truth label says they moved."* With GT labels, that is close to circular.

The gap is one wiring change, not new work — `visibility_cleanup` returns a `see_through` mask and `apply_miss()` applies it to `log_odds`. The frame loop has to call them. I am not going to wire it inside `dashboard/` because that is JP's directory and the run loop is the integration point, not the viewer.

**What I want from the room:** agreement that the toggle drives the map path before it is demonstrated to anyone outside the team, and a decision on who wires it. I am volunteering for the `src/run/` side of it tomorrow if nobody objects.

---

## 2. Point→slot binning: the largest stage in the frame, and it has no owner

`scripts/timing_table.py` — new today, this is the Day-6 latency table.

```
stage       owner         p50 ms   p99 ms   max ms   x10Hz  MB/frame
bin         ⚑ nobody       14.57    17.45    17.65    5.7x      6.96
scatter     Shrestha        5.80     8.21     8.54   12.2x      0.55
fuse        Aakash          6.66     9.10    10.33   11.0x        ~0
cleanup     Shrestha        9.00    10.58    10.60    9.5x      0.07
pyramid     Shrestha        2.69     3.98     4.04   25.1x      0.05
shift       Shrestha        2.85     3.48     3.94   28.8x      0.29
MEASURED                   41.87    49.49    49.69    2.0x      7.92
```

Broken down, because the split matters for who fixes what:

| sub-step | p50 ms | in |
|---|---|---|
| `ring_of` — ring membership | 4.39 | `src/grid/lattice.py` |
| `i_ring` — lattice index, ×4 rings | 5.79 | `src/grid/lattice.py` |
| `flat_slot` — index → storage slot, ×4 rings | 4.60 | `src/gpu/shift.py` |
| mask + gather per ring | 3.85 | the composition itself |

Turning a sweep into flat slots — `ring_of` for membership, `i_ring` per ring, `flat_slot` into the toroidal window — is a stage the frame loop must run every frame. **No module exports it.** It is composed by hand in three places already: in `scripts/timing_table.py` (all four rings), in `scripts/baseline_demo.py` (ring 0 only), and in `src/grid/transient.py` (a third spelling). Three hand-rolled copies of the one step between perception and the grid is an integration defect waiting for the day the three disagree, and they will disagree silently — a binning bug produces a plausible map, not a crash.

Two things make it worse than a tidiness complaint:

- **It is the largest single stage,** larger than `fuse` and larger than my `cleanup`.
- **It allocates 6.96 MB per frame**, against the "no allocation in the frame loop" invariant, and more than twice what the scatter scratch cost per frame before I preallocated it — the defect I already found once, with a profiler, behind a docstring that said the opposite.

**On that 6.96 MB: I had this wrong when I first drafted this section, and the correction points at a different file than I did.** I assumed it was the masks and fancy-index copies in my own composition. Measured, it is not: `ring_of` allocates 6.96 MB per call on its own, 4.80 MB of it inside `d_aniso`, and the composition around it contributes essentially nothing. I rewrote the binning as a single fused pass with every intermediate preallocated and the per-frame figure did not move by a byte. **So the allocation is `src/grid/lattice.py`'s to fix, and no amount of restructuring on my side touches it.** Roughly seven full-length float64 temporaries per call, at 0.96 MB each for a 120,000-point sweep.

**What I have already done, in my half.** `flat_slot` is now `flat_slot_into` — division-free and zero-allocation, and pinned bit-identical to the old method across ring sides, offsets and shifted windows by `test_flat_slot_into_matches_flat_slot`. In view means `x0 <= ix < x0 + W`, so `ix - x0` is already in `[0, W)` and `(x0 + c) mod W` is `(x0 mod W) + c` minus `W` once if it overflowed: one masked subtract instead of an integer division per point per axis. That is 28% off the `flat_slot` sub-step, and it is what took `bin` from 16.20 to 14.57 ms p50 and the frame subtotal from 59.15 to 49.49 ms p99 — the table above is the after. The odd-side (33) case is in the parametrisation deliberately — the identity has nothing to do with powers of two, and a version that assumed one would pass on 400 and 500.

**And it is easy to get wrong in a way that survives casual testing.** I got it wrong twice today. Ring membership is a question about distance from the *sensor*, so `ring_of` takes the vehicle-frame point; the lattice index is *global*, so `i_ring` takes the world-frame one. Feed world coordinates to `ring_of` and every point reads as OUTSIDE once the vehicle has driven past ring 3's half-width. Hold the sweep at the vehicle origin while the ring windows advance and by frame 13 everything bins to −1 — after which `scatter` and `fuse` post sub-millisecond p50s for doing nothing at all, and the latency table reads *better*. Both failure modes look correct for the first few seconds. Both are written up in `bin_points`' docstring.

**My recommendation:** one vectorised `bin_points(xv, yv, xw, yw, schedule, …) -> idx` in `src/grid/`, next to `ring_of`, with a preallocated output like every other frame-path buffer, and the three call sites deleted. It is lattice semantics, so it is Aakash's directory by the ownership table — but I will write it as a same-day cross-directory PR if he would rather spend Day 4 on the regret curve. **That is the decision I want: whose, and today or tomorrow.**

---

## 3. 19 classes do not fit in 4 bits — the first real frame raises, it does not degrade

`fusion.boyer_moore_update()` rejects any class id above 15. §10.2 specifies a 4-bit candidate plus a 4-bit counter, and Aakash correctly made it raise rather than wrap — a silent `% 16` would relabel class 16 as class 0, which is `unlabeled`.

SemanticKITTI is 19 classes, ids 0–18. So the moment JP's GT labels reach `fuse()`, it raises. Not degrades — raises. I hit this within a minute of wiring the real `fuse` into the timing table, and had to draw `class_id` over 0..15 to make my own script run.

Three files currently assume three different class ranges: `gpu/kernels.py` packs its sort key assuming ids < 32, `grid/fusion.py` enforces < 16, and the project is 19.

Aakash has already pinned it in `test_nineteen_classes_do_not_fit` and proposed the fix in the docstring: **a 5-bit candidate and a 3-bit counter**, which holds all 19 and caps the counter at 7. Boyer–Moore's majority guarantee does not depend on where the counter saturates, so the change is safe. It stays one byte, so the frozen 12-byte cell struct does not move.

**This needs 30 seconds of the room's time and a yes.** It is flagged as a room decision because it changes the semantics of a frozen struct's field, and it is the kind of thing that should not be decided quietly inside one directory. But it is a blocker on the first end-to-end frame, and we are on Day 3.

---

## 4. Visibility scratch — pick a cap, or decide to leave it undeclared

`visibility_scratch_bytes()` computes it; `allocate()` does not include it. Folding it in means fixing a cap on candidate cells per frame, which moves the number on the dashboard counter. That is the same class of decision as the transient-layer line, so it goes to the room rather than into my directory quietly.

The candidate set is the currently-**occupied** cells — a free cell has nothing to clear. 68 B per candidate cell against a float64 range image, 64 B against float32.

| Cap | Scratch (float32 image) | Preallocated total | vs uniform 2.5D | vs dense 3D |
|---|---|---|---|---|
| undeclared (today) | — | **29.06 MB** | 6.6× | 88× |
| 100,000 | 6.40 MB | 35.46 MB | 5.4× | 72× |
| 150,000 | 9.60 MB | 38.66 MB | 5.0× | 66× |
| 200,000 | 12.80 MB | 41.86 MB | 4.6× | 61× |

*(Add 3.11 MB to every total if the stretch pyramid ships enabled.)*

**The report's ratios are unaffected** — 21.5× uniform and 286× dense are cell-count ratios over our 8.94 MB of *map*, and scratch is working memory. But the dashboard shows the total, and the two must not meet on a slide without someone being ready to explain which is which.

**I cannot pick the cap honestly yet.** It depends on how many cells are occupied at once on sequence 07, and that needs the data. What I would like agreed tonight is the *shape*: a `visibility.max_candidate_cells` key in `configs/thresholds.yaml`, exactly like the `scatter.max_points_per_frame: 150000` knob that is already there, declared in the budget printout, with the value set from a measurement once the download lands. That way the bound is honest and movable, and nobody discovers on Day 6 that the counter and the slide disagree.

**Interim recommendation: 150,000**, on the reasoning that a sweep touches ~105,000 distinct cells per frame and the occupied set should not be several times that in a rolling 100 m local map. That is an argument, not a measurement, and I will replace it with one.

---

## 5. Two ratifications — no work attached, just say yes

Neither changes a theorem or a proof. Both are already implemented and tested; I want them on the record rather than discovered in the report.

### (a) eq (32)'s `delta` — the math doc and the config disagreed, and I followed the math

§10.4 specifies `delta = 3σ(r)` precisely so the clearing band widens with range. `configs/thresholds.yaml: visibility.range_tolerance_m` says a flat `0.30`. Those are very different rules: 0.30 m is **26.9σ at 5 m** and **1.7σ at 100 m** — so the flat value barely clears in the near field, where ghosts matter most, and clears real structure at range, which is exactly what the guard exists to prevent.

Implemented as `3σ(r)` with the config value as a **floor** for pose error. No threshold was changed; one was given a new job. `clear_tolerance_m(range_m, sensor, floor_m=0.30)`.

### (b) §7.2's pyramid memory is low by about half

§7.2 quotes 1.24 MB, from `745,000 × 5 / 3`. Two compounding errors: a node stores the **reductions**, not the source fields, so ground contributes both `H_max` and `H_min` — 8 B per node by §7.2's own list, 9 with `OR_mask`, not 5. And `N` is the ring **windows**, 910,000 allocated slots, not 745,000 logical cells.

Corrected: **2.73 MB of nodes**, plus 0.38 MB of shared reduction scratch, 3.11 MB total. The `N/3` claim itself is exactly right — measured 3.00. `scripts/bench_pyramid.py` prints all of it.

---

## 6. Latency, for the record: the back end alone is 59 ms of the 100 ms budget

From the table in item 2. 42 ms p50, 49 ms p99, **2.0× headroom at p99 — before a single line of perception runs.**

Whatever `load`, `transform`, `range_image`, `semantics` and `motion` cost has to fit in the remaining ~51 ms. I am not raising this as an alarm: `bin` alone is 14.6 ms of it and is only half-optimised (item 2), and the whole thing is the numpy CPU reference path. But **the 10 Hz claim is currently bounded, not demonstrated**, and I would rather say that at Day 3 than have it emerge at Day 6. JP: a per-stage p50/p99 from your front end, whenever you have one, and I will put the real total on the table.

Stable to within about 10% across three runs of 200 frames. `Intel i7-14650HX`, numpy 2.5.2, single-threaded CPU path. No GPU kernel exists yet; that is a Day-6 item and it is where the headroom comes back.

---

## What I've built

`src/gpu/`, `scripts/`. Nothing outside my directories except this document.

**Preallocation, math §3.3 / §7.2** — `allocate()` returns every frame-path buffer, committed at startup. 29.06 MB claimed, 28.82 MB resident per the OS. Residency is `mincore(2)` on the array, not a process-RSS delta: glibc raises its mmap threshold once it has seen a large block freed, so a later allocation of about that size reuses pages the process already holds, and a correctly committed 64 MB baseline reported a 42 MB delta after the allocator tests had run and the full 64 MB when it ran alone.

**Scatter, two paths, bit-identical** — `scatter_sorted` (scratch sized by points) and `scatter_atomic` (dense accumulator, the literal §3.5 reading). `tests/test_kernels.py` asserts they agree field-for-field, and `bench_scatter.py` prints the hash proving it. Re-measured tonight on this machine at 120,000 returns into 745,000 cells: **sorted p50 6.65 / p99 9.81 ms, atomic p50 20.56 / p99 30.51 ms** — 3.1× apart on the median. Note these are higher than the 5.9 / 6.1 recorded in `src/gpu/CLAUDE.md`, which was measured on a quieter machine; the *ratio* between the two paths is what the design decision rests on and it has not moved. I am re-running every latency figure I own before any of them reaches a slide.

**Visibility cleanup, §10.4** — eq (32) plus the never-clear-a-current-return guard, producing a mask. p50 9.0 / p99 10.6 ms at 200,000 candidates. Log-odds and the three-state decision stay in fusion, which is Aakash's.

**Conservative pyramid, §7.2–7.3 (stretch)** — off by default. Rebuild p50 2.9 / p99 4.4 ms over 910,000 slots. The theorem test is mutation-checked: six deliberate breakages, all six fail the suite.

**Toroidal shift, §2.4** — O(perimeter). 25,500 cells cleared at 15 m/s across four rings, 3.0 ms p50.

**Six numbers scripts**, now seven with `timing_table.py`. Gate 6 holds for everything I own.

Two things worth saying about correctness specifically:

- **`mean_height_cm()` rounded every negative mean 1 cm low**, and the ground plane is almost entirely negative. 540 of 600 negative probes were wrong; all 203 positive probes were fine, which is why it read as correct. A systematic 1 cm sag over the whole ground plane against a §3.2 noise floor of 0.8 cm at 5 m, and per-ring RMSE is the only place it would ever have shown.
- **`EMPTY_CELL` is one shared definition** used by both `allocate()` and the strip `shift()` clears. Only `ceiling_height` differs from zero, and 0 cm reads as solid ground at the datum — so a zero clear marks the whole world untraversable and never recovers, because `fuse()` only ever lowers a ceiling. Fixing it in `allocate()` alone gives a map that is correct exactly until the vehicle moves.

## Not done, and why

| Item | Status |
|---|---|
| Visibility scratch in `allocate()` | Blocked on a cap decision — item 4 |
| Real-data latency numbers | Blocked — no SemanticKITTI on disk |
| GPU kernels (CuPy path) | Day 6. Everything today is the numpy CPU reference path |
| `bin_points` in `src/grid/` | Blocked on an ownership decision — item 2 |
| `ring_of`'s 6.96 MB/frame | `src/grid/lattice.py`, not mine to fix — item 2 |

## Tomorrow

1. Wire `visibility_cleanup` + `apply_miss` into `src/run/` so the ghost toggle drives the map, if the room agrees in item 2
2. `bin_points`, if it lands with me
3. `ablation_table.py` — schedule comparison, thresholds frozen first

**One risk to say out loud:** every latency and fill number I have is against a synthetic sweep. The shapes are right — gamma-tailed range distribution, 47% of returns in ring 1, 6.6% in ring 3 — but the numbers are not reportable and I have been careful to label them that way in every script. The day the data lands, all of them get re-run before anything goes on a slide, and I would rather the room expects that than is surprised by a column of changed figures on Day 5.
