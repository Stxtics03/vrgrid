# Four decisions, 2 September — Shrestha

Drafted for review, not taken unilaterally. Each one states the evidence, the
options, what was actually done, and how to reverse it. Three of the four are
one-line changes either way; the fourth is a metric-semantics question that
belongs to whoever owns the §8.2 figure.

Everything below was measured on real SemanticKITTI, sequences 07 and 08, on
2 September. Sequence 00 finished downloading during the work and its numbers
are noted where they exist.

---

## Decision 1 — three additions to a file marked `frozen: true`

`configs/thresholds.yaml` opens with *"FROZEN before any schedule comparison. If
these move between runs you are comparing tuning effort, not schedules."* Three
blocks were added to it today. **No existing value was changed.**

| key | value | reads it |
|---|---|---|
| `traversability.baseline_m` | `0.50` | §7.1 bits 1 and 2, eq. (22a) |
| `features:` (whole block) | — | §7.4 only, a new module |
| `visibility.max_candidate_cells` | `150000` → `null` | §10.4 scratch sizing |

**Why this is not a freeze violation, and where that reasoning could be wrong.**

`features:` is safe without argument: nothing in §7.1, §8 or §9 reads it, and
deleting the block changes no reported number.

`visibility.max_candidate_cells` sizes *working* memory, not map memory.
`allocators.py` is explicit that the report's cell-count ratios are computed
over map memory. The declared total moves only when `with_visibility=True`,
which is off by default.

`traversability.baseline_m` **is** a predicate change, and it is the one that
needs a real answer. It alters which cells set bits 1 and 2. Measured on the
synthetic scene it changes no R(S) — verified by an A/B with the key present
and set to `null`, byte-identical results — because M\_S is already blocked to
`plan.cell_m` before §7.1 runs there. It changes the map's own traversability
layer, which the dashboard, ghost removal and the per-ring table read. **The
per-ring table has not been regenerated since.** That is the concrete risk.

**Recommendation:** accept all three, then regenerate the per-ring and ablation
tables on 07/08 before anything goes on a slide. If the freeze is read strictly,
`baseline_m` is the only one that needs a waiver.

---

## Decision 2 — the visibility candidate cap

`visibility.max_candidate_cells` bounds how many occupied cells §10.4 may test
per frame. It has been a placeholder since Gate 3 because nobody could size it
without data. Measured now, whole sequences, real `iter_pipeline → MapEngine.step`:

| sequence | frames | median | p99 | **max** |
|---|---|---|---|---|
| 07 | 1,101 | 187,921 | 305,350 | **314,442** |
| 08 | 4,071 | 225,629 | 435,308 | **455,714** |

The retired 150,000 dropped **52.3% of 07's peak and 67.1% of 08's**.

**Truncation is silent in the dangerous direction.** `engine._cleanup` does
`occupied[:max_candidates]` deterministically, so the determinism test passes.
Dropped cells keep their occupancy, are never tested against the range image,
and cannot appear in `cleared` — because `cleared` only counts what was
offered. A truncating run prints a healthy ghost count while the map keeps its
ghosts. This is now counted (`StepCounters.truncated`) and printed by
`vrgrid.run`, which is worth having whatever number is chosen.

**Why not simply fit a number.** The peak scales with sequence length: frames
×3.70 from 07 to 08, peak ×1.45. Sequences 00 (4,541 frames), 02 (4,661) and
19 (4,981) are all longer than 08. Any number fitted to the two sequences that
happened to download first is a bet that nobody runs the others.

| option | value | scratch @ 5/10/20/40 | truncation |
|---|---|---|---|
| measured | 600,000 | 38.40 MB | possible on a longer sequence |
| **structural** | `null` → 910,000 slots | **58.24 MB** | **impossible by construction** |

**Taken:** `null`. The occupied set cannot exceed the grid — a cell must exist
to be occupied — so this turns the bound back into the compile-time guarantee
the memory claim is supposed to be. An explicit integer is still honoured for a
constrained target, and truncation is now visible if one is used.

**Reverse it** by writing `600000` in place of `null`. Sequence 00's
measurement is running and will either support or undermine the scaling
argument; if 00's peak lands near 08's rather than above it, the case for the
cheaper option strengthens.

---

## Decision 3 — R(S) = 0.207 on the frozen schedules, 0.000 on every uniform

This looked like a residual asymmetry defect. It is not.

Scoring both paths on M\* by hand: π\* and π\_S are both 42 cells, π\* runs down
column 16 of the planning window and π\_S down column 17 — out and back. Two
diagonal steps instead of two straight ones is

```
2 · (√2 − 1) · plan.cell_m  =  2 · 0.4142 · 0.25  =  0.2071
```

which is the reported figure to three decimals. **0.207 is the smallest
non-zero regret this lattice can express.**

The cause is one cell:

```
column j=16   43 cells clear, 1 with bit 5 (n < n_min)   M_S cost mean 1.0909
column j=17   44 cells clear                             M_S cost mean 1.0000
M*            cost mean 1.0000 on both columns
```

The uniform baselines pool more observations per cell, nothing falls under
`n_min`, and they go straight. So the gap between 0.207 and 0.000 is fill rate
at fine resolution steering the *path* — `restrict()` masks but does not
neutralise bit 5, and planning happens on M\_S before scoring.

**Reportable consequence:** on this scene the frozen schedules' regret is one
lattice quantum and the uniforms' is exactly zero. That supports *"the
coarsening did not change the decision"* with the honest caveat that the scene
cannot resolve anything below 0.207. **It is not a knee and must not be drawn
as one.**

Separately fixed while tracing this: §7.1 bit 4 was evaluated on one side of
eq. (23) only. `ReferenceMap` carries `class_id`, but `costmap_from_reference`
built from `block_stats` — heights alone — so M\* charged 0 class penalties
while the frozen schedules charged 18, and a schedule paid pure regret for
routing around ground it had correctly labelled non-drivable. Now symmetric,
via a new `ReferenceMap.block_class()`. It does not move the synthetic number
(the whole window is drivable ground) and it will matter for a lateral query
and on real data.

---

## Decision 4 — the `PLAN_LANE_CELLS` query design

Parked on Pratyushi, who has no commits since 30 August. JP asked that if it
fell to us it be *"a deliberate, documented decision — not us guessing at her
intended framing."* This is that document. **It is reversible and it is not a
claim about what she intended.**

The current query is longitudinal: `PLAN_LANE_CELLS = 6` puts it at
`PLAN_N//2 − 6`, column 16, y = −1.50 m, running the length of the window.
Today's evidence says it cannot carry the §8.2 figure:

- M\* over the window has **one** passable cost value — 1,924 cells at 1.00×.
  A graded curve needs a graded cost field and this window has none.
- The only decision available is around a wall, and the lane misses the one
  hazard: the pothole sits at (18, 0), on the centreline, not at y = −1.50 m.
- The finest difference expressible is the 0.207 lattice jog of Decision 3.

**A lateral query — road to verge, across the kerb — is the natural
alternative**, and it only became a fair test today: until bit 4 was put on
both sides of eq. (23) a lateral query would have been measured almost entirely
through an asymmetry, since crossing the kerb is exactly where the class
penalty lives. The survey's §4 note says as much.

**Recommendation, not taken:** add the lateral query alongside the longitudinal
one rather than replacing it, and report both. Replacing it would silently
change what every previous R(S) number meant. Adding it costs one start/goal
pair in `eval_synthetic.plan_query` and leaves the existing series comparable.

**Not done here** because it changes what the headline figure measures, three
days from submission, in someone else's absence. It needs one person to say yes.

---

## What is still outstanding after these

- Regenerate the per-ring, memory, ablation and ghost-removal tables on 07/08.
  Everything above changed code that feeds them.
- Sequence 00's cap measurement, to test the scaling argument in Decision 2.
- The remaining 19 sequences are downloading; 07, 08 and 00 are complete and
  verified. `python scripts/data_status.py` is the check.
