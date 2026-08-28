# Gate 0 review — Friday 28 August, 20:00

**Aakash — grid engine + evaluation harness.** Prepared before the meeting; numbers are reproducible from `pytest -q` on my clone.

---

## The 20-minute agenda, in the order I want to spend the time

| # | Item | Who decides | Time |
|---|---|---|---|
| 1 | Is the SemanticKITTI download running? | JP | 2 min |
| 2 | Memory bound: the headline is 745,000 cells, the buffers need 910,000 | Shrestha + room | 8 min |
| 3 | Allocator is my hard dependency for `scatter()` | Shrestha | 2 min |
| 4 | Two corrections already applied to `docs/sih-math.md` — ratify | Room | 5 min |
| 5 | Anisotropy has an unresolved conflict with fixed buffers (stretch #12) | Note only | 3 min |

Items 1–3 are blockers. Items 4–5 are for the record; nothing waits on them.

---

## Gate 0 — met

> **GATE 0:** Cell struct committed and immutable. Partition test passes on 10⁶ random points. Dashboard renders a mock map. Download running. R2's label verdict delivered.

My half of it:

- **Partition test passes on 10⁶ random points**, both frozen schedules. CI-blocking, seeded, `pytest -m partition` → 5 passed.
- Suite: **33 passed, 11 skipped**, `ruff` clean. The 11 skips are Day-2 split/merge (4), and fusion/determinism (7) which are blocked — see item 3.

I can't speak to the dashboard, the download, or Hriday's label verdict. Those are items 1 and the standing questions for JP and Hriday.

## Gate 1 — one of two met, and the missed one is the long pole

> **Day 1, Aakash:** Reference map for the tuning sequence exists on disk.
> **Day 1, D1 (execution plan):** Ring assignment, toroidal shift with the O(perimeter) clear, `scatter()` with fixed-point integer atomics. Shift round-trip test.

- Ring assignment — **done**
- Toroidal shift + round-trip test — **done**
- `scatter()` — **blocked**, item 3
- Reference map — **blocked**, item 1. This is the one flagged in the assignments doc as *"the long pole; start Day 1."* It is not started because there is no data.

I'd rather say that plainly tonight than let it look on track for another day.

---

## 1. Download — the only question that matters tonight

`data/` in my clone holds its README and nothing else. That is expected if JP is downloading on his own machine, so **this is a question, not an accusation**: is it running, and what is the ETA?

It gates:

- my reference map, and therefore every metric, and therefore the Day-4 memory-vs-regret curve
- JP's entire front-end from Day 1 onward

The execution plan calls it *"the one item on the critical path that neither cleverness nor effort can accelerate."* If it hasn't started, that is tonight's recovery decision, not tomorrow's.

**What I'll do either way:** build `reference_map.build()`/`load()` against the documented sequence layout and test it on a synthetic miniature sequence I generate, so it runs the hour the data lands instead of starting then.

## 2. Memory — the headline number and the buffers disagree by 22%

**This is the item I most want a decision on, because Shrestha's Day-2 gate is "everything preallocated" and he should not build against the wrong number.**

`schedule.total_cells = 745,000` counts square **annuli** — math §6.1 eq. (19), `N_L = 4(R_L² − R_{L−1}²)/c_L²`. That is where 8.94 MB and 21.5× come from.

But §2.4 addresses each ring as a full `N_L × N_L` square with toroidal wraparound, and it has to. The annulus hole is centred on the vehicle, so as we drive it **travels through the buffer**: cells cross between hole and annulus on every shift. There is nowhere to put them unless the square is stored.

| Schedule | Annulus (claimed) | Square buffers (required) | Delta |
|---|---|---|---|
| 5/10/20/40 | 745,000 → **8.94 MB**, 21.5× uniform, 286× dense-3D | 910,000 → **10.92 MB**, 17.6× uniform, 234× dense-3D | +165,000 (+22%) |
| 5/10/50 | 520,000 → 6.24 MB, 30.8× | 570,000 → 6.84 MB, 28.1× | +50,000 (+10%) |

Ring extents for the default schedule are 400, 500, 500, 500 — so rings 1–3 each allocate 250,000 cells against annulus counts of 210,000 / 187,500 / 187,500.

**Three ways out, my recommendation last:**

1. **Store the squares, restate the number.** 10.92 MB, 17.6× uniform, 234× dense-3D. Costs 2 MB and one slide edit. Still a strong claim.
2. **Store each annulus as four independently-shifted rectangular strips.** Keeps 8.94 MB and 21.5×. The shift stays O(perimeter). The corners and the strip-to-strip handoff are fiddly and are exactly the kind of indexing bug that shows up as a plausible-looking map on Day 5.
3. **Allocate squares, report the annulus figure.** Not an option — we would be quoting a number we don't allocate, and the demo has live memory counters on screen.

**My recommendation: option 1.** We are four days from a demo with a live allocating baseline next to our counter, and the counter must match the slide. 17.6× against a uniform grid and 234× against dense 3D is not a weaker story, and "we found this in our own arithmetic and corrected it" survives a hostile question far better than a number that disagrees with the running system. If Shrestha wants option 2, it is his call and his directory — but it should be a deliberate choice made tonight, not discovered on Day 5.

Pinned in `test_allocated_cells_exceed_the_annulus_count` so both numbers stay visible whichever way we go. I have changed no config and no claim.

## 3. Allocator — my dependency, and it's already documented as one

`gpu/allocators.py: allocate()` is still `NotImplementedError("Shrestha — Day 0, hour 3")`. Its own docstring says: *"This blocks Aakash: `scatter()` needs somewhere to write."*

Consequences right now: `scatter()`, `fuse()`, and the **CI-blocking determinism test** cannot be written. That's 7 of my 11 skipped tests.

**Not asking anyone to rush.** I've written `lattice.alloc_ring_buffers()` as a stand-in that wraps the frozen `alloc_soa()` and adds a `ring_origin` array — enough to build and test the toroidal shift today. When the real `allocate()` lands it should return the same shape, and I'll drop mine. Worth two minutes to agree that seam so we don't integrate two different layouts on Day 5.

## 4. Two corrections applied to `docs/sih-math.md` — ratifying, not proposing

Both are in the shared source of truth, so the room should know. Neither changes a theorem, a formula, or a proof.

### §2.4(b) — the specified unit test asserts something false

The section asks us to assert `i_L == ⌊x/(k_L·c₀)⌋` computed directly, bit-exact, for all rings. That is false for `k = 10`, at **every positive boundary** — 4000 of 4000 out to 200 m.

```
fl(0.05) = 3602879701896397/2⁵⁶, slightly greater than 1/20
  → ten fine cells span 0.5000000000000000277…
  → but fl(10 × 0.05) rounds to exactly 0.5

x = 0.5, k = 10:   ⌊i_fine/k⌋ = ⌊9/10⌋   = 0     ← derived, and correct
                   ⌊x/(k·c₀)⌋ = ⌊0.5/0.5⌋ = 1     ← naive, off by one
```

The sharper half: for `k ∈ {1,2,4,8}` the naive lattice isn't merely close, it is **the same lattice** — scaling a double by 2ᵐ is exact. Our default schedule is all powers of two. **So a naive implementation passes everything we run against 5/10/20/40 and is off by one cell on 5/10/50 — the schedule the memory claim is made on.**

And the defect has measure zero: at ±4 ulps around 4000 boundaries, 72,009 probes, the only disagreements are the boundary doubles themselves. No amount of random sampling finds it, which is why §2.4(b) as written passed on a broken implementation. Rewritten to compare against exact rational arithmetic, and (d) added requiring both schedules.

*Srinivas — this is prior-art adjacent, it's the kind of detail Droeschel's ring-buffer paper would have hit. Worth a line in the research log.*

### §2.4(c) — the round-trip test as worded is unachievable

*"Shifting the map by +1 then −1 cell restores every cell value identically."* A shift clears the newly exposed strip — it destroys information by design. +d then −d exposes the **same** slots twice, so the map is identical everywhere **provided that strip started empty**. Both readings are now tested separately.

## 5. Anisotropy vs fixed buffers — for the record, no decision needed

Stretch item #12, bottom of the drop list, so this costs us nothing today. But whoever picks it up should know it before they start.

At 15 m/s the forward stretch `a_f = 2` maps a return 58 m ahead to `d = 29`, filing it in ring 2 — whose buffer stops at 50 m. The index wraps toroidally and lands on a cell on the **far side of the map**. Same failure for the rear floor read literally: a point at (−10, −70) satisfies "x < 0 and |x| < 50" and clamping it into ring 2 aliases it onto the cell at +30 m.

Rule I've enforced: **a point may only be assigned to a ring that physically contains it.** Eq. (20) may push a point outward — coarser rings are larger, so it still fits — never inward past its geometric ring.

**The consequence is the interesting part: under fixed square buffers, eq. (20) reduces to its lateral half.** The squeeze works. The forward stretch is clamped away entirely. Buying it back means allocating each ring for its maximum stretch, ~1.5× the cells, on top of item 2.

§6.2's ⚑ note says anisotropy "changes which ring a cell belongs to; it does not change the lattice." True, and beside the point — it changes the required **extent** of every ring buffer. That sentence should be widened when someone next touches the section.

---

## What I've built

`src/grid/`, `tests/test_lattice.py`, `docs/sih-math.md`. Nothing outside my directories. Nothing committed — local only, as agreed.

**Lattice, math §2** — `i_fine`, `i_ring`. One lattice at 5 cm, ring indices by integer division. Scalar and vectorised through the same operator, so `query()`'s path and `scatter()`'s path cannot drift. `i_ring` rejects non-integer or non-positive `k` at the index, not only at config load.

**Ring assignment, math §6** — `ring_of` (Chebyshev with the scaled-L∞ stretch, rear floor, containment rule, `OUTSIDE` for out-of-map), `migrate_ring` (hysteresis, eq. 21), `d_aniso`, `stretch_factors`.

**Toroidal buffers, math §2.4** — `toroidal_shift` (offset-based, nothing copied), `ring_extent`, `ring_slice`, `buffer_cells`, `alloc_ring_buffers`. Returns the cleared count so O(perimeter) is measured rather than asserted.

**Schedule** — `load()` now parses the `anisotropy:` block, which it was silently dropping on the floor.

Measured, not claimed:

- 40 cm ego-motion step clears **6,700 of 910,000 cells (0.74%)**, matching the analytic perimeter sum exactly
- ring 0 shifts 8 cells, ring 1 four, ring 2 two, ring 3 one — whole cells at every ring, per the §2.4 constraint
- round trip restores every cell bit-for-bit

### Tests

CI-blocking partition suite: exactly one cell per ring per point on 10⁶ points × both schedules; no gap at ring boundaries; floor-not-truncation at the origin; the k=10 disagreement counterexample. Ring assignment: isotropic boundaries, out-of-map, membership-only invariance, rear floor, the containment guard on 20,000 random points × 2 speeds, hysteresis thrash bound. Toroidal: bit-exact round trip, strip-only loss, O(perimeter), whole-cell shifts, and the memory arithmetic.

**Two things worth saying about the tests specifically**, because both are the kind of thing that quietly makes a suite worthless:

- My first partition test was **weak and I caught it with a negative control**. Counting how many of `{i−1, i, i+1}` contain a point gives 1 *even when the returned index is off by one* — the cells are disjoint by construction. A truncating implementation passed clean. Existence is now anchored at the returned index; that version catches truncation on 500,077 of 10⁶ points. I've recorded the trap in §2.4(a) so nobody rebuilds it.
- Every claim in the §2.3 correction is asserted in a test, so the document and the code can't drift apart later.

### Not done, and why

| Item | Status |
|---|---|
| Reference map (`src/eval/`) | Blocked — no data. Item 1. |
| `scatter()`, `fuse()`, determinism test | Blocked — no allocator. Item 3. |
| Split/merge (§4–5) | Day 2, on schedule |

## Tomorrow

1. Reference map builder against a synthetic sequence, so it runs the hour data lands
2. Split/merge with the variance mathematics — Day 2 is the mathematically load-bearing day and the assignments doc says explicitly it cannot slip
3. `scatter()`/`fuse()` the moment the allocator exists

**One risk to say out loud:** the load check has me as the bottleneck, and two of my four current work items are blocked on other people. Days 0–1 have absorbed that because the lattice work was independent. Day 2 is split/merge, which is also independent — so the squeeze doesn't land tomorrow. It lands Day 3–4, when metrics and plan regret both need the reference map, which needs the data. If the download slips past tomorrow, the Day-4 memory-vs-regret gate is the one that fails, and that is the headline figure.
