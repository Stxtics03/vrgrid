# Repositioning the plan-regret query — Shrestha's half of the agreement

**Day 5, Tuesday 2 September.** Answering the open item at the foot of
`scripts/regret_plot.py`: *"Posing the query so it has to decide about the kerb
or the pothole is still the open item, still yours to agree with me before
either of us moves it."*

Everything below comes from `scripts/plan_query_survey.py`, which is new today
and is built so that it **cannot** be used to tune: it never computes R(S), not
once, not for a candidate query. It reports what the costmaps *contain*. A
query chosen from that is chosen from the terrain; a query chosen from an R(S)
table is chosen from the answer.

```
python scripts/plan_query_survey.py --frames 12 --map
```

---

## Short version

**Do not reposition the query yet — it is not what is wrong.** Two of the three
hazards you named are not hazards at planning resolution, the third cannot
produce a knee, and there are two defects underneath the figure that no start
and goal can fix:

1. **The fill-rate confound is not closed.** `common_support()` restricts on
   `CostMap.unknown` (never-observed). The penalty is charged on bit 5
   (`n < n_min`), which is a different and far larger set. After restriction
   our schedules still pay `w_unknown` on **91.9%** of the surviving window
   against uniform 20 cm's **4.1%**.
2. **The two sides of eq. (23) apply §7.1 at different lattices**, and §7.1's
   thresholds are not scale-invariant. Our maps wall off the kerb; M\* does
   not. That accounts for 154 of the 156 walls in our schedule's window.

Both are metric semantics, which is yours and Pratyushi's. I have the
measurement and have not touched `eval_synthetic.py` or `plan_regret.py`.

---

## 1. `common_support()` does not restrict what the note says it restricts

This is the one to fix first, because it is the confound you documented as
closed on 1 September.

`costmap_from_gridmap` sets `unknown[i,j] = not seen`, where `seen` is true if
**any one** of 25 sub-samples was observed. Bit 5 — confidence, `n < n_min` —
is OR-ed across those same 25 samples, so **one** thin sub-cell sets it for the
whole planning cell. `_cost_from_bits` then charges the same `w_unknown` for
either. `common_support()` masks on `unknown` only, so it removes the first set
and leaves the second entirely.

`--frames 12`, after `restrict()` to the 99.1% common support:

| schedule | below `n_min` | still inside the mask | of the mask |
|---|---|---|---|
| **5/10/20/40** | 91.9% | **1,762** | **91.9%** |
| **5/10/50** | 91.9% | **1,762** | **91.9%** |
| uniform 10 cm | 17.6% | 341 | 17.8% |
| **uniform 20 cm** | 4.1% | **80** | **4.2%** |
| uniform 40 cm | 9.1% | 176 | 9.2% |
| uniform 80 cm | 18.2% | 352 | 18.4% |

(99.1% of the window carries bit 5 on our schedules; 156 of those cells are
already walls, so 1,762 are cells where it shows up as a *weight*.)

Our map costs `w_base + w_unknown` = **5.0 across 92% of the restricted
window**; uniform 20 cm costs `w_base` = 1.0 across 95% of it. Every R(S) in
the current table is dominated by that ratio. It is exactly the confound in
your own words — *"a finer schedule holds fewer returns per cell, so more of
its window is below `n_min`; it pays `w_unknown` and the planner routes around
a map that is merely SPARSE"* — and `common_support()` is not the fix for it.

**Why it looked closed.** The diagnostic reads the wrong array.
`PlanResult.unknown_fraction`, and `eval_synthetic`'s "window low-confidence"
column, are both `np.mean(costmap.unknown)`. That is the never-observed
fraction: **0.9%**. The below-`n_min` fraction is **91.9%**. The column headed
*"cells low-confidence in the window"* in `plan_regret.py`'s note, reading 1%
against uniform's 0%, is off by two orders of magnitude in the direction of
"looks fine" — and it is the number the note uses to conclude that restricting
the window changes almost nothing.

## 2. The kerb is not a hazard, and cannot be made into one by moving the query

§7.1 is evaluated on M\* at the **planning** cell size, because
`costmap_from_reference` blocks the 5 cm reference down to `plan.cell_m` before
setting a single bit. At 25 cm the kerb is:

| bit | value | threshold | fires |
|---|---|---|---|
| step | 0.120 m | `s_max_m` 0.15 | no |
| slope | 0.240 | `tan(20°)` 0.364 | no |

The kerb is 12 cm by construction and `s_max_m` is 15 cm, so its **step never
fires at any cell size**; its slope fires only below 20 cm cells. On M\* it is
flat ground. A start and goal either side of it are not asking the planner to
decide anything — and if it *were* impassable on M\*, the query would be worse
than useless, because the kerb is an unbroken line across the whole window, so
there would be no path on M\* at all and R(S) would be undefined rather than
large.

The ramp is the same from the other end: 6% is 0.06 against 0.364, and at
`--frames 16` and below it is not inside the window at all. **Of the three
features, only the pothole is a hazard.**

## 3. The pothole is a hazard, and it still cannot produce a knee

At `--frames 12` the window is `x [11, 22] × y [−5.5, 5.5]` and M\* over it is:

```
1936 cells:  impassable 12  unknown 0  below n_min 0  roughness 0  class 0  plain w_base 1924
passable cost spread:  1.00 x 1924
```

**One cost value.** No roughness bit fires, no class bit fires, nothing is
unknown. There is no graded cost field anywhere in this scene at planning
resolution, so the only decision available to any planner is *go around the
12-cell hole*, in an otherwise empty 11 × 11 m field where the detour costs a
few tenths of a cost unit.

That is a binary outcome, not a curve. A schedule that resolves the pothole
detours cheaply; one that smooths it away plans through it and is
`blocked_on_reference`. **No placement of start and goal turns that into a
knee**, because a knee needs coarsening to change the *cost* of the best route,
and here it can only change whether one 60 cm hole exists.

## 4. The two sides of eq. (23) are not on the same lattice

`--frames 12`, wall sets compared cell by cell against M\*:

| schedule | impassable | agreement with M\* on the wall set |
|---|---|---|
| 5/10/20/40 | 156 | both 2, **M\* only 10, M_S only 154** |
| 5/10/50 | 156 | both 2, **M\* only 10, M_S only 154** |
| uniform 10 cm | 178 | both 2, M\* only 10, M_S only 176 |
| **uniform 20 cm** | **12** | **both 12, M\* only 0, M_S only 0** |
| uniform 40 cm | 0 | both 0, M\* only 12 |
| uniform 80 cm | 0 | both 0, M\* only 12 |

Our two frozen schedules **invent 154 walls M\* does not have and miss 10 of
the 12 real ones.** I localised the 154: two vertical lines at y = −3.25/−3.00
and y = +2.75/+3.00 m, 32 to 37 cells long each. That is the kerb, on both
sides, walling the road into a corridor — in the map under test, on ground M\*
calls flat.

The mechanism is one line of arithmetic. A step of height `h` reads as a
gradient of `h / 2c` at cell size `c`, and §7.1 compares that against a single
`tan(θ_max)` whatever `c` is:

| cell size | \|grad\| at the kerb | wall? | who evaluates §7.1 there |
|---|---|---|---|
| 0.05 m | 1.200 | **WALL** | rings 0 of both frozen schedules |
| 0.10 m | 0.600 | **WALL** | uniform 10 cm |
| 0.20 m | 0.300 | – | uniform 20 cm |
| **0.25 m** | **0.240** | **–** | **M\*** — `costmap_from_reference` blocks to `plan.cell_m` |
| 0.40 m | 0.150 | – | uniform 40 cm |
| 0.80 m | 0.075 | – | uniform 80 cm |

**The crossing sits between 10 cm and 20 cm, and M\* is on the far side of it
from every ring we care about.** Three consequences, none about coarsening:

- A schedule is charged regret for **resolving** the kerb. Same shape as the
  fill-rate confound, different mechanism, and unaffected by either fix to it.
- **The only schedule that reproduces M\* exactly is uniform 20 cm**, and it
  does so because its lattice is nearest M\*'s evaluation lattice, not because
  it is a good map. That is the same 10 cm / 20 cm boundary the anomalous
  R(S) = 1.536 spike sits on.
- It moves with the frame count, which is the dependence you flagged. At
  `--frames 24` the window covers x = 35–46 m, M\* walls **88** cells — both
  kerb lines, full length — and the sign of the disagreement flips: the coarse
  uniforms now miss 88 real walls and the fine ones both invent 181 and miss 83.

And the figure's own default is the worst case:

> **At `--frames 16` — `regret_plot.py`'s default, the one that goes on the
> slide — M\* contains ZERO impassable cells.** The window is x = 19–30 m and
> the pothole is at x = 18, just outside it. Our schedules still wall 160
> cells. The headline figure is planning across an empty field on one side of
> eq. (23) and down a walled corridor on the other.

## 5. Two smaller things the survey turned up

**Bit 4 (class) is on one side of eq. (23) only.**
`costmap_from_reference` states that clearance is absent from M\* and why.
Class is absent too and that is not stated: `ReferenceMap` carries `class_id`,
but the reference costmap is built from `block_stats`, which is heights. So
M_S can pay `w_class` on ground M\* charges nothing for. It costs nothing on
this scene — both sides report 0 — but it goes live the moment a real sequence
has a verge, and it is exactly what a lateral query would have been measured
through. Either implement it from `class_id` or state it beside the clearance
sentence.

**`PLAN_LANE_CELLS`' stated justification has expired.** The comment says the
centreline "drops out of the common support and the centreline corridor is
severed". At `--frames 12` the centreline is **100.0% supported** and the whole
window is 99.1%. Whatever was true when that was written was fixed by the
beam-intersection correction. Worth updating whichever way we go, so the next
person does not inherit a reason that no longer holds.

## 6. What I would like agreed

The measurement is mine and it is done. The next step is metric semantics,
which is yours and Pratyushi's — I have touched neither `eval_synthetic.py`
nor `plan_regret.py`, and the survey only reads.

**First, and I think uncontroversial: restrict on the confidence bit, not on
`unknown`.** Either `common_support()` also masks bit 5, or `w_unknown` stops
being charged for it and bit 5 is reported separately. And
`unknown_fraction` should report the set that is actually being charged, or
report both — as it stands the diagnostic that exists to make this confound
visible is reading the array that hides it. Until this changes I do not think
any R(S) in the table is interpretable.

**Second, the lattice. Pick one:**

**(a) Evaluate §7.1 at the planning lattice on both sides.** Block M_S down to
`plan.cell_m` through `query()`'s existing sampling before setting bits, so
eq. (23) subtracts like from like. Defensible on its own terms — a planner
stepping in 25 cm cannot act on 5 cm detail anyway — but it concedes some of
the fine map's advantage by construction, and we should say that out loud
rather than be asked.

**(b) Make the geometric bits scale-aware.** Test step and slope over a fixed
physical baseline rather than over one cell, so a 12 cm kerb reads the same at
5 cm and at 80 cm. The better metric, and a change to §7.1 itself, which
touches more than this figure.

**(c) Accept that this scene cannot carry the §8.2 money plot and say so.**
Run it on sequence 08 and keep the synthetic sweep as a correctness harness.
This is my honest reading of §3 above: even with both defects fixed, one 60 cm
hole in an empty field will not draw a knee.

**My recommendation: the confidence-bit fix and (a) now, (c) as the real
answer.** The first two are an afternoon between them and make the number
interpretable; (c) is what the figure needs to be evidence, and it is the one I
cannot do anything about, because —

> ⚑ **`data/` is empty on Day 5.** No sequence 00, 07 or 08 on disk. The
> execution plan's decision B put this in the first hour of Day 0 as *"the one
> item on the critical path that neither cleverness nor effort can
> accelerate"*, and `data/README.md` assigns it to JP. Every reportable number
> in the project is downstream of it, this figure included — and so is the
> `visibility.max_candidate_cells` cap I could not pick honestly at Gate 3.
> Raising it here because it decides which of the three options is even
> available, not to relitigate the assignment.

If (a): it is your file and your semantics, so I will review rather than write.
If you would rather I take it as a same-day cross-directory PR, say so and I
will.
