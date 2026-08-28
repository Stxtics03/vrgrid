# sih-math.md
## Mathematical Foundations — Adaptive Variable-Resolution 2.5D LiDAR Mapping

*Companion to master plan v4. Every claim in the plan that has a number behind it is derived here.*

**How to use this document.** Each section states what is being computed, why that mathematics and not something simpler, the derivation, and the unit test that proves the implementation matches. If a section has a ⚑, it contains a result that is defensible as a contribution rather than a standard technique.

**Notation.**

| Symbol | Meaning |
|---|---|
| `r` | range from sensor (m) |
| `h_s` | sensor height above ground (m); 1.73 for KITTI |
| `Δθ` | azimuthal angular step (rad); 0.2° = 3.49×10⁻³ |
| `Δφ` | vertical beam spacing (rad); 26.9°/63 = 7.45×10⁻³ |
| `c_L` | cell size of ring `L` (m) |
| `R_L` | outer half-width of ring `L` (m) |
| `μ, σ²` | cell height estimate and its variance |
| `z` | a height measurement |
| `n` | observation count |
| `∇z` | local ground gradient (dimensionless) |

---

## 1. Sensor sampling geometry — why the rings are what they are ⚑

This section is the physical justification for the entire representation. It is also, in its second half, the most original analysis in the project.

### 1.1 Azimuthal spacing — the easy axis

A LiDAR fires at fixed *angles*. Two consecutive returns in the same laser ring, at range `r`, are separated along the arc by

```
s_az(r) = r · Δθ                                                    (1)
```

Linear in range. With Δθ = 0.2°:

| r (m) | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| `s_az` (cm) | 3.5 | 8.7 | 17.5 | 34.9 |

Compare the schedule 5 / 10 / 20 / 40 cm. **Cell size tracks the sensor's own sample spacing to within a factor of 1.15 at every ring boundary.** This is a Nyquist-style argument: making cells finer than the sample spacing cannot add information, it only adds empty cells.

### 1.2 Radial ground spacing — the axis everybody forgets ⚑

The gap between where *consecutive laser rings* strike flat ground is a different beast. A beam at depression angle `φ` from a sensor at height `h_s` intersects the ground at

```
r = h_s / tan φ  ≈  h_s / φ     for small φ                         (2)
```

Differentiate:

```
dr/dφ = −h_s/φ²  = −h_s/(h_s/r)²  = −r²/h_s
```

so the radial spacing between adjacent beams is

```
s_rad(r) = (r² / h_s) · Δφ                                          (3)
```

**Quadratic in range.** This is the key result.

| r (m) | 10 | 25 | 50 | 80 | 100 |
|---|---|---|---|---|---|
| `s_rad` (m) | **0.43** | **2.69** | **10.77** | **27.6** | **43.1** |

At 50 m, consecutive laser rings land **10.8 metres apart** on the road.

### 1.3 The consequence: single-frame cell fill rate

A ground cell of size `c` at range `r` receives a return only if a laser ring passes through it *and* an azimuthal sample lands in it:

```
P_fill(r, c) ≈ min(1, c/s_rad(r)) · min(1, c/s_az(r))               (4)
```

| Ring | r | c | c/s_rad | c/s_az | **P_fill** |
|---|---|---|---|---|---|
| 0 | 10 m | 5 cm | 0.116 | 1.0 | **11.6%** |
| 1 | 25 m | 10 cm | 0.037 | 1.0 | **3.7%** |
| 2 | 50 m | 20 cm | 0.019 | 1.0 | **1.9%** |
| 3 | 100 m | 40 cm | 0.009 | 1.0 | **0.9%** |
| *uniform baseline* | 50 m | 5 cm | 0.005 | 0.29 | **0.13%** |

**Three conclusions, all reportable:**

1. **The uniform 5 cm baseline is 99.87% empty at 50 m.** Uniform high resolution at range is not high resolution — it is an empty array with a confident axis label. This is the strongest possible form of your central argument, and it is a derived number, not rhetoric.

2. **Coarsening improves fill rate by 15× at Ring 2.** Your cells are not merely smaller in count, they are individually *better supported by evidence*.

3. ⚑ **Rings 2–3 are filled by ego-motion, not by the sensor.** Since `P_fill < 2%` per frame, the far field is populated only as the vehicle drives forward and the ring pattern sweeps across the ground. Call this **ring-sweep filling**. It implies: temporal accumulation is the sole fill mechanism in the far field; single-frame far-field metrics are meaningless; and far-ring accuracy must be reported as a function of frames-since-first-observation, not as a scalar.

### 1.4 Derived limits (each of these is a scope statement in the report)

**Blind cone.** The lowest beam at depression `φ_min` (−24.8° for HDL-64E) strikes the ground at

```
r_blind = h_s / tan|φ_min| = 1.73 / tan(24.8°) = 3.74 m             (5)
```

Blind disc area `π r_blind² = 43.9 m²`; Ring 0 covers a 20×20 m square = 400 m². **11.0% of Ring 0 is unobservable in any single frame.** Mark unknown, never free. Report the persistent-unknown fraction separately, since ego-motion fills most of it.

**Negative obstacles.** A pothole of width `W` is sampled only if `W > s_rad(r)`. Inverting (3):

```
r_max(W) = √(W · h_s / Δφ)                                          (6)
```

| W | 30 cm | 50 cm | 1.0 m |
|---|---|---|---|
| `r_max` | **8.3 m** | **10.8 m** | **15.2 m** |

**Slow-motion detectability.** An object at speed `v` moves `d = v·Δt` between frames (Δt = 0.1 s at 10 Hz). It is *geometrically* detectable only if

```
v · Δt  >  max( c_L(r),  s_az(r),  3σ_r )                           (7)
```

A pedestrian at 1.4 m/s moves 14 cm/frame. That exceeds Ring 0 (5 cm) and Ring 1 (10 cm) but **not Ring 2 (20 cm)**. Solving `c_L(r) = v·Δt` puts the crossover at **≈ 25 m**. Beyond 25 m, pedestrian motion detection is a *semantic prior*, not a measurement. A car at 15 m/s moves 1.5 m/frame and is detectable in all rings.

**Unit test.** Assert `s_rad(50) / s_rad(25) ≈ 4` (quadratic scaling) and that `P_fill` computed empirically from a real scan matches (4) to within 20%.

---

## 2. The lattice — a proof of alignment, not a tolerance ⚑

The problem statement explicitly warns about "alignment errors or data loss during the projection." Most teams will handle this with an epsilon. You can *prove* it away.

### 2.1 Construction

Define one global integer lattice at the base resolution `c₀ = 5 cm`:

```
i_fine(x) = ⌊x / c₀⌋                                                (8)
```

Every coarser ring index is derived from it by integer division, never recomputed in floating point:

```
i_L(x) = ⌊ i_fine(x) / k_L ⌋ ,     k_L = c_L / c₀ ∈ ℤ⁺              (9)
```

For the default schedule `k = (1, 2, 4, 8)`; for the ablation `k = (1, 2, 10)`.

### 2.2 Theorem (Exact nesting)

> For any real `x`, any `c₀ > 0` and any integer `k ≥ 1`:
> ```
> ⌊ ⌊x/c₀⌋ / k ⌋  =  ⌊ x / (k c₀) ⌋
> ```

**Proof.** Let `m = ⌊⌊x/c₀⌋/k⌋`. Then `mk ≤ ⌊x/c₀⌋ < (m+1)k`. Since `mk` and `(m+1)k` are integers and `⌊x/c₀⌋` is the greatest integer `≤ x/c₀`, the left inequality gives `mk ≤ x/c₀`, and the right gives `x/c₀ < (m+1)k` (if `x/c₀ ≥ (m+1)k` then `⌊x/c₀⌋ ≥ (m+1)k`, contradiction). Hence `m ≤ x/(kc₀) < m+1`, i.e. `m = ⌊x/(kc₀)⌋`. ∎

*(This is the nested-floor identity, Graham, Knuth & Patashnik,* Concrete Mathematics*, eq. 3.11.)*

### 2.3 Corollary (Partition)

The ring-`L` lattice is **exactly** the direct lattice of cell size `k_L c₀`. Therefore the ring cells form a partition of the plane: every point lands in exactly one cell, never zero, never two. **There is no tolerance to tune, no epsilon, and no boundary case.**

Contrast with the naive implementation, computing `⌊x/c_L⌋` independently per ring in IEEE-754 from a decimal literal. The ring lattice is then built on `fl(c_L)`, a different real number from the `k_L·fl(c₀)` that `k_L` fine cells actually span, so the two lattices drift apart and near a boundary you get points that fall in both cells or in neither.

> **Correction, 28 Aug — Aakash.** *This paragraph previously named `⌊x/0.20⌋` and `⌊x/0.40⌋` as the counterexample. Those are exactly the two values where the naive code is **accidentally correct**, so as written the section was arguing its case from the one family of examples that does not support it. The claim is right; the numbers were wrong. Corrected here and in §2.4(b). Nothing about (8), (9), the theorem or its proof changes.*

**Where the drift actually bites — and why the default schedule hides it.** `fl(c₀)` = 3602879701896397/2⁵⁶, slightly greater than 1/20. Multiplying a double by 2^m is exact, so for `k = 2^m` the quantity `k·fl(c₀)` is representable and `fl(c_L)` **is** that same double: the naive lattice coincides with the derived one exactly. The default schedule's ratios are 1, 2, 4, 8 — all powers of two — so a naive implementation passes every test you run against `5/10/20/40` and is genuinely bit-identical there.

It fails on the ablation. For `k = 10`, `k·fl(c₀)` = 0.5000000000000000277… is **not** representable and rounds to exactly 0.5, so a ring cell of the naive lattice is a hair narrower than the ten fine cells it is supposed to contain. The double 0.5 lies in fine cell 9 — ring cell 0 — while the naive lattice calls it ring cell 1:

```
x = 0.5, k = 10:   ⌊i_fine(x)/k⌋ = ⌊9/10⌋   = 0     ← derived, and correct
                   ⌊x / (k·c₀)⌋  = ⌊0.5/0.5⌋ = 1     ← naive, off by one
```

This is not one unlucky value. It is **every** positive boundary of the naive lattice: 4000 of 4000 out to 200 m, and the same for `k = 5` and `k = 20`. The failure is one-sided — the naive cell is narrower than the fine cells it should contain, so on the negative side the flooring absorbs the shortfall and the two agree.

Note *where* it is not: at ±4 ulps around each of those 4000 boundaries, 72,009 probes in total, the only disagreements are at the boundary doubles themselves — 4000 of them, zero in the neighbourhood. The defect has measure zero. **That is what makes it dangerous, not what makes it safe.** No amount of uniform random sampling will find it, which is why the test specified in §2.4(b) has to compare against exact arithmetic rather than against a second float computation, and why this went unnoticed long enough to reach a frozen document. A LiDAR return at exactly 0.5 m, 1.0 m or 1.5 m is not exotic. Recorded as `test_direct_float_lattice_disagrees_at_ring_boundaries`.

**Why powers of two are convenient but not required.** For `k = 2^m`, equation (9) is a bit-shift `i_fine >> m`. For any other integer (e.g. `k=10` in the ablation schedule) it is an integer divide. Both are exact. The validator must therefore check **integer ratio**, not power-of-two.

Note that the power-of-two case is doubly special: it is also the case where the float shortcut is safe. That is precisely why equation (9) is not optional. Written the naive way, this project would ship a lattice that is provably correct on the schedule it was developed against and silently off by one cell on the schedule it is compared to — and the ablation is where the memory claim is made.

### 2.4 Map shifting — O(perimeter), not O(area)

The map follows the vehicle by toroidal (wrap-around) indexing. Ring `L` of extent `N_L × N_L` cells is addressed as

```
addr(i, j) = ( (i + o_L^x) mod N_L ,  (j + o_L^y) mod N_L )         (10)
```

Shifting by one cell increments the offset and clears only the newly exposed strip: **`2N_L` cells cleared, not `N_L²`.** For Ring 3 (`N = 500`), that is 1,000 cells instead of 250,000 — the difference between a sub-millisecond shift and a 40 ms stall.

**Constraint:** the map origin must move in whole **coarsest**-cell steps (40 cm), otherwise every ring boundary shifts by a fraction and you must resample — which is precisely the "data loss during projection" the brief warns about. Expected side effect: the nominal 25 m ring boundary wobbles by up to 40 cm. That is correct behaviour, not a bug.

**Unit test.** Generate 10⁶ random points, seeded — a CI-blocking gate must fail reproducibly or not at all. Assert:

**(a)** each point maps to exactly one cell per ring. Anchor existence at the index actually returned: `i·k ≤ i_fine < (i+1)·k`, then assert that neither neighbour also contains it. Counting how many of `{i−1, i, i+1}` contain the point is **not** sufficient — the cells are disjoint by construction, so that count is 1 even when `i` is off by one, and a truncating implementation passes.

**(b)** `i_L` computed by (9) equals the true index of the size-`k_L c₀` lattice, evaluated in **exact rational arithmetic** — `⌊Fraction(x) / (Fraction(c₀)·k_L)⌋` — not as `⌊x/(k_L c₀)⌋` in floating point. The theorem in §2.2 is a statement about reals, and `k_L c₀` is itself a rounded double; evaluating the right-hand side in floats measures that rounding, not the theorem, and for `k = 10` it is false at every boundary (§2.3). Exact arithmetic is slow, so run it on a 2·10⁴ subsample and keep the full 10⁶ for (a), which is pure integer work.

**(c)** shifting the map by +1 then −1 cell restores every cell value identically.

**(d)** run (a) and (b) against **both** frozen schedules. `5/10/20/40` is all powers of two and cannot catch a lattice bug that only appears at non-power-of-two ratios; `5/10/50` is what exercises `k = 10`.

*(b) and (d) revised 28 Aug — see the correction in §2.3.*

---

## 3. Per-cell height estimation — Kalman with a range-dependent measurement model

### 3.1 Why a Kalman filter and not a running mean

A running mean weights every measurement equally. But a point returned at 80 m at grazing incidence is dramatically less informative about elevation than one at 5 m head-on, and averaging them equally throws away that structure. The Kalman filter is the minimum-variance linear estimator when measurement variances differ — which is exactly our situation. This is the standard elevation-mapping formulation (Fankhauser et al.).

### 3.2 The measurement variance model

A point's height is `z = r sin φ`. Propagating range noise `σ_r` and angular noise `σ_φ` through first-order error propagation:

```
∂z/∂r = sin φ ,      ∂z/∂φ = r cos φ

σ²_z = sin²φ · σ²_r  +  r² cos²φ · σ²_φ                             (11)
```

For a beam striking the ground, `φ` is small, so `sin φ ≈ h_s/r` and `cos φ ≈ 1`:

```
σ²_z ≈ (h_s/r)² σ²_r  +  r² σ²_φ                                    (12)
```

The **second term dominates at range and grows as `r²`.** With `σ_φ = 0.1°`: σ_z = 8.7 cm at 50 m, 17.5 cm at 100 m. The first term is the near-field floor.

**Incidence-angle inflation.** On a surface whose normal makes angle `θ_inc` with the beam, a lateral positioning error maps into height error amplified by `1/cos θ_inc`. Grazing hits on the road at range are the worst case:

```
σ²_z(r, φ, θ_inc) = [ sin²φ σ²_r + r² cos²φ σ²_φ ] / cos²θ_inc      (13)
```

Clamp `cos θ_inc ≥ 0.1` to avoid a singularity at pure grazing.

### 3.3 The scalar update

```
K   = σ²_prior / (σ²_prior + σ²_z)                                  (14)
μ  ← μ + K (z − μ)
σ² ← (1 − K) σ²_prior
```

Add process noise each frame to model pose drift and terrain change: `σ² ← σ² + q Δt`, with `q` small (≈ 10⁻⁴ m²/s). Without it, `σ²` collapses toward zero and the cell stops responding to new evidence — the classic filter-lock failure.

### 3.4 Fixed-point accumulation and determinism

IEEE-754 addition is **not associative**: `(a+b)+c ≠ a+(b+c)` in general. GPU atomic float adds complete in nondeterministic order, so two identical runs produce different maps. That breaks debugging (you cannot bisect a bug whose location moves) and quietly invalidates any lossless claim.

Since heights are quantised to 1 cm anyway, accumulate in **int32 fixed-point**. Integer addition is exactly associative, so results are bit-identical run to run.

**Quantisation error budget.** Uniform quantisation with step `q` has variance `q²/12`. For `q = 1 cm`: `σ_quant = 2.9 mm`. Compare to (12): σ_z ≈ 8 mm at 5 m, 87 mm at 50 m. Quantisation is **≤ 1/3 of sensor noise at the closest range and negligible beyond**, and a 12 cm kerb resolves into 12 levels. int16 at 1 cm spans ±327 m. **1 cm is justified, not a default.**

**Unit test.** Run the same sequence twice; assert byte-identical map hashes. Assert `σ²` decreases monotonically under repeated consistent measurements and increases under process noise alone.

---

## 4. Merge — the law of total variance ⚑

**This is a correction to plan v2, which called merging "standard, uncontroversial."**

### 4.1 Why inverse-variance fusion is the wrong tool

Inverse-variance fusion, `1/σ²_fused = Σ 1/σ²_i`, is the maximum-likelihood combination of **repeated measurements of one quantity**. Four child cells are not four measurements of one height — they are measurements of **four different places**. Merging them is *marginalisation over a footprint*.

The failure is not academic. Applied naively to four children straddling a kerb — say heights 0.00, 0.00, 0.12, 0.12 m, each with σ = 2 cm — inverse-variance fusion returns σ = 1 cm. **The merged cell claims to be twice as certain as any child, while sitting on top of a 12 cm step it has just erased.** Your map becomes most confident exactly where it is least justified.

### 4.2 The correct rule

Model the parent as the distribution of surface height over its footprint — a mixture of the children. By the law of total variance:

```
μ_p  = Σ w_i μ_i                                                    (15)

σ²_p = Σ w_i σ²_i        +   Σ w_i (μ_i − μ_p)²                     (16)
       └─ within-cell ─┘      └─── between-cell ───┘
         E[Var(z|child)]         Var(E[z|child])
```

with weights `w_i = n_i / Σ n_j` (observation counts), or uniform if counts are equal.

The second term is what naive fusion discards. On the kerb example, (16) gives `σ²_p = 0.0004 + 0.0036`, so **σ_p = 6.3 cm** — six times the naive answer, and correctly reflecting that the cell now spans a step.

### 4.3 Corollary

```
σ²_p ≥ Σ w_i σ²_i  ≥  min_i σ²_i
```

**Merging never decreases variance below the average child variance, and increases it strictly whenever the children disagree.** Combined with §5 this gives the symmetry for the slide: *variance rises on split because we assert detail we never measured; variance rises on merge because we hide detail we did measure.* Both directions are honest.

**Unit test.** Four children with identical means → `σ²_p = Σw σ²_i` exactly (between-term vanishes). Four children on a synthetic step of height `Δ` → `σ²_p ≥ Δ²/4`.

---

## 5. Split — variance inflation and the round-trip theorem ⚑

This is the mathematically distinctive core of the project.

### 5.1 The problem

Splitting a parent of size `c_p` into four children of size `c_c = c_p/2`, we know `(μ_p, σ²_p)` and nothing else. The children must be assigned values. The mean is forced: with no information distinguishing them, `μ_i = μ_p` for all `i` (any other assignment invents structure).

The variance is the interesting question. **Setting `σ²_i = σ²_p` is wrong** — not because it under-reports the magnitude of uncertainty, but because it misrepresents its *structure*. Four children carrying `μ_p` are perfectly correlated, yet every downstream consumer will treat them as four independent finer estimates. The map asserts resolution it does not possess.

### 5.2 The inflation term

Model the ground locally as a plane with gradient `∇z` estimated by finite differences over the parent's neighbours (§7.1). A child centre sits at offset `d = c_p/4` from the parent centre, so the true child mean differs from `μ_p` by approximately `∇z · d`. Adding a roughness term `α` for sub-cell terrain variability:

```
σ²_child = σ²_parent  +  κ ‖∇z‖² (c_p² − c_c²)  +  α                (17)
                         └──────── slope term ────────┘
```

with `κ = 1/16` from the offset geometry (`d² = c_p²/16`) and `α` calibrated against the reference map (§9).

> **Note, 28 Aug — Aakash. `κ = 1/16` does not follow from the geometry it cites, and the correct constant is `1/12`.** *Not applied: κ is frozen in `configs/thresholds.yaml` and changing it is a room decision. Both values are pinned in `test_kappa_from_geometry_is_one_twelfth_at_every_ratio` so the choice stays visible. No theorem changes either way.*
>
> *The offset geometry is right — a child centre of a 2×2 split sits `c_p/4` off the parent centre on each axis, so `d² = c_p²/16`. But (17) multiplies κ by `(c_p² − c_c²)`, not by `c_p²`. At `c_c = c_p/2` that factor is `(3/4)c_p²`, so `κ = 1/16` delivers `3c_p²/64` where the stated geometry asks for `4c_p²/64` — a uniform 25% under-inflation. Setting `κ = 1/12` reproduces the geometry exactly.*

**Generalisation to `m × m`, which the ablation needs.** §5.1 and §5.2 are written for `c_c = c_p/2`, i.e. four children. `5/10/50` refines **5×** between rings 1 and 2, so a split there produces **25** children. (17) already handles this and the merge rule of §4.2 is stated for an arbitrary number of children, so nothing needs rewriting — but two things are worth stating rather than leaving to be rediscovered:

- The mean-square child-centre offset per axis for an `m × m` split is `c_p²(m² − 1)/(12m²)`, which is exactly `(c_p² − c_c²)/12`. So (17)'s `(c_p² − c_c²)` form is the **m-independent** one, and the geometric κ above is `1/12` at every ratio — not a per-schedule constant. The `m = 2` case, `d² = c_p²/16`, is the special case, not the general rule.
- Consequently the 25% shortfall from `κ = 1/16` is the same 25% at `m = 2` and `m = 5`. One constant to decide, once.

`split()` reads `m` from the schedule rather than assuming four. `test_split_follows_the_schedule_not_the_number_four` asserts 4 children across `10 → 5` and 25 across `50 → 10`.

### 5.3 Theorem 1 (Variance monotonicity)

> For `c_c < c_p` and `‖∇z‖ > 0`, `σ²_child > σ²_parent` strictly.

Immediate from (17), since `c_p² − c_c² > 0`. **Limiting behaviour is correct:** on a perfectly flat road `∇z = 0` and splitting costs nothing — which is right, because splitting a flat surface genuinely loses no information.

> **Note, 28 Aug — Aakash. The flat-ground limit above, and §5.4 unit test (c), are true only for `α = 0`.** *(17) adds `α` unconditionally, so any `α > 0` charges for splitting flat ground and both statements become false. Theorem 1 itself survives — `α > 0` only strengthens a strict inequality — which is exactly why this is easy to walk past.*
>
> *`α` is `0.0` in `configs/thresholds.yaml` today, which is honest rather than convenient: §5.2 calibrates it against the reference map and the reference map is blocked on the download. **Whoever calibrates `α` must restate this paragraph and rewrite unit test (c) in the same commit.** `test_alpha_would_break_the_flat_ground_remark` fails the moment `α` moves, so the commit cannot be a quiet one.*

### 5.4 Theorem 2 (Round-trip idempotence) — and why it needs one bit

Naively, split-then-merge does **not** return the original. Substituting `μ_i = μ_p` into (16): the between-term vanishes, so `σ²_merged = σ²_child = σ²_p + Δ > σ²_p`. Variance has been inflated by a no-op.

This is a real bug, not a formality. A cell oscillating across a ring boundary while the vehicle changes speed would inflate its variance *every frame*, and the map would drift toward uncertainty without any physical cause. (This is also why §6.3's hysteresis matters.)

**Fix:** one `derived` bit in the cell's flags byte, set on split, cleared by any new measurement. Merge rule:

```
if all four children derived AND no observations since split:
        restore (μ_p, σ²_p) exactly            ← inverse of split
else:
        apply law of total variance (16)       ← genuine marginalisation
```

> **Theorem 2.** With the `derived` flag, `merge(split(c)) = c` exactly, in both mean and variance, when no measurement intervenes.

**Proof.** Split sets `μ_i = μ_p` (mean preserved by construction) and marks all children derived. With no intervening measurement, the merge branch restores `σ²_p` by definition. ∎

> **Implementation note, 28 Aug — Aakash. "Restores `σ²_p`" has to mean *reads it back*, not *recomputes it*, and that is a constraint on the map layout.** *The proof is a statement about reals and is not in question; this is about what makes it hold in float64.*
>
> *Deflating — `σ²_p = σ²_child − Δ` — is exact in real arithmetic and is not exact in IEEE-754. It is worst precisely where the map is best: a confident cell (`σ²_p ≈ 10⁻⁶ m²`) split on a slope (`Δ ≈ 10⁻² m²`) loses most of its significant digits in the subtraction and does not come back bit-identical. "Bit-identical" in unit test (a) is the right requirement — a round trip accurate to 10⁻¹² per cycle is still unbounded drift over a sequence at 10 Hz, which is the drift §5.4 exists to eliminate.*
>
> *So the restore branch returns the parent value rather than computing anything, which is available because **split does not destroy the parent**: it writes children into the finer ring / refinement pool while the ring-`L` cell stays resident in its own buffer. ⚑ **If a future SoA split reuses the parent's slot, Theorem 2 stops being exact.** Recorded here because it is invisible at the call site and cheap to break.*

Cost: one bit. Return: split and merge form an exact inverse pair, provable and testable.

**Unit test.** (a) Random cell → split → merge → assert bit-identical mean and variance. (b) Split on a synthetic slope → assert `σ²_child > σ²_parent` for every child. (c) Split on flat ground → assert `σ²_child = σ²_parent`. (d) Split → inject one measurement into one child → merge → assert the result now follows (16), not the restore path.

---

## 6. Ring assignment, anisotropy and hysteresis

### 6.1 Isotropic base

Rings are square annuli, so ring membership uses the Chebyshev (L∞) norm:

```
d(x, y) = max(|x|, |y|)
L(x, y) = min { L : d(x,y) < R_L }                                  (18)
```

Cell count for square annulus `L` follows directly:

```
N_L = 4 (R_L² − R_{L−1}²) / c_L²                                    (19)
```

| L | `R_{L−1}→R_L` | `c_L` | `N_L` |
|---|---|---|---|
| 0 | 0 → 10 | 0.05 | 160,000 |
| 1 | 10 → 25 | 0.10 | 210,000 |
| 2 | 25 → 50 | 0.20 | 187,500 |
| 3 | 50 → 100 | 0.40 | 187,500 |
| | | **Σ** | **745,000** |

Ablation (5/10/50): `160,000 + 210,000 + 4(100²−25²)/0.25 = 520,000`.

### 6.2 Anisotropic foveation

Circular foveation wastes resolution behind the vehicle. In the vehicle frame (`x` forward, `y` left), replace (18) with a scaled L∞ norm:

```
d_aniso = max( x⁺/a_f(v),  x⁻/a_r,  |y|/a_s(v) )                    (20)

a_f(v) = clamp(1 + κ_f v/v_ref,  1,  2)      forward stretch
a_s(v) = 1 / (1 + κ_s v/v_ref)                lateral squeeze
a_r    = 1                                    rear never stretched
```

subject to a hard rear floor: `c_L ≤ 0.20 m` whenever `x < 0 ∧ |x| < 50`.

⚑ **Alignment is preserved, and this is not obvious.** Equation (20) changes which *ring* a cell belongs to; it does not change the *lattice*. Every cell at every ring remains a `k_L`-fold aggregate of the same base 5 cm lattice, so §2's partition theorem is untouched. State this explicitly — it looks like it should break alignment, and a reviewer may assume it does.

### 6.3 Hysteresis — mandatory, not optional

A cell sitting exactly on a ring boundary while `v` fluctuates will split and merge every frame. Consequences: refinement-pool thrash, and (by §5.4) unbounded variance inflation if the `derived` flag is ever cleared mid-cycle. Use asymmetric thresholds:

```
split  when  d_aniso < R_L
merge  when  d_aniso > R_L (1 + ε)          ε ≈ 0.1                 (21)
```

**Unit test.** Drive a synthetic trajectory with sinusoidal speed across a ring boundary; assert the number of split/merge events per cell is bounded and that variance does not grow monotonically over 1,000 frames.

---

## 7. Traversability and the conservative pyramid ⚑

### 7.1 The predicate

Traversability is a **bitfield**, not a scalar — six independent conditions that fail for different reasons and that a planner should be able to distinguish:

```
bit 0  clearance   ceiling − ground  <  h_vehicle
bit 1  slope       ‖∇z‖              >  tan(θ_max)
bit 2  step        max|z_c − z_nbr|  >  s_max
bit 3  roughness   σ²                >  σ²_max
bit 4  class       class ∉ drivable_set
bit 5  confidence  n                 <  n_min          (fail safe)
```

Gradient by central differences over the four neighbours, scaled by the cell size of the ring:

```
∂z/∂x ≈ (z_{i+1,j} − z_{i−1,j}) / (2 c_L)                           (22)
```

⚑ **Geometry decides, semantics filters.** A road with a 40 cm pothole has class `road` and is not drivable; a packed grass verge has class `vegetation` and often is. Class is one bit among six, not the decision. This is what the problem statement's Requirement 1 actually asks for, and it is also *evidence for* the grid: slope and step are finite differences over neighbours, which are trivial on a grid and effectively impossible on a raw point cloud without first building one.

### 7.2 The conservative pyramid

Build a 4-ary pyramid over each ring storing, per block `B`:

```
H_max(B) = max ground        H_min(B) = min ground
C_min(B) = min ceiling       n_min(B) = min observation count
AND_mask(B) = ⋀ traversability bitfields
```

**Not means.** Averaging heights hides hazards: a coarse cell straddling a kerb reports the mean and looks flat. Max/min preserves the worst case.

### 7.3 Theorem 3 (No false negatives)

> Define
> ```
> SAFE(B) ⟺ H_max(B) − H_min(B) < s_max
>          ∧ C_min(B) − H_max(B) > h_vehicle
>          ∧ n_min(B) ≥ n_min_threshold
> ```
> If `SAFE(B)` then **every** cell in `B` is traversable on conditions 0, 2 and 5.

**Proof.** For any cell `c ∈ B`, its clearance is `C(c) − H(c) ≥ C_min(B) − H_max(B) > h_vehicle`, satisfying bit 0. For any pair `c, c' ∈ B`, `|H(c) − H(c')| ≤ H_max(B) − H_min(B) < s_max`, satisfying bit 2. And `n(c) ≥ n_min(B) ≥ threshold`, satisfying bit 5. ∎

Symmetrically, if `AND_mask(B)` has bit `k` set, **every** cell fails condition `k`, so the block is certainly blocked. Otherwise the block is MIXED and the query descends one level.

**Result:** coarse queries are cheap *and* safe. You pay fine resolution only where the coarse answer is genuinely ambiguous, and a false "traversable" is impossible by construction.

**Cost.** A 4-ary pyramid over `N` cells adds `N(1/4 + 1/16 + …) = N/3`. Built over ground, ceiling and the traversability byte (5 bytes of the 12): `745,000 × 5 / 3 ≈ 1.24 MB`.

**Unit test.** Exhaustive: for 10⁴ random blocks, if `SAFE(B)` then assert every constituent cell is individually traversable. Any counterexample is a proof failure, not a tuning issue.

---

## 8. Plan sensitivity — coarsening measured in units of decision ⚑

The headline contribution. Every adaptive-mapping paper measures reconstruction error; reconstruction error is a *proxy* for what matters. This measures the thing itself.

### 8.1 The offline metric

Let `M*` be the reference map (5 cm, no LOD, offline-aggregated). Let a planner `P` produce path `π_S = P(M_S)` on the map under schedule `S`, and `π* = P(M*)`. Define the path cost functional `J_M(π) = Σ_{c ∈ π} w_M(c) · Δℓ`, with `w` derived from the traversability bitfield.

```
Plan regret:   R(S) = J_{M*}(π_S) − J_{M*}(π*)   ≥ 0                (23)
```

⚑ **The critical detail: both paths are scored on `M*`.** Scoring `π_S` on `M_S` measures self-consistency, not quality — a badly coarsened map will happily report that its own bad plan is cheap. Non-negativity of (23) follows because `π*` minimises `J_{M*}` by construction.

Report alongside a purely geometric measure, the discrete Fréchet distance `d_F(π_S, π*)`, which catches the case where a detour costs the same but goes somewhere quite different.

### 8.2 The money plot

Sweep `S` over schedules (5/10/20/40, 5/10/50, uniform 5, uniform 10, uniform 20, …). Plot memory on x, `R(S)` on y. The curve has a knee. The result reads:

> *"Below 8.9 MB the plan is unchanged — regret is exactly zero — and above the knee it degrades measurably. Our schedule sits at the knee."*

That single figure is worth more than every memory bar chart in the deck, because it answers the only question a sceptic actually has.

### 8.3 The online policy — one extra O(N) pass

Refining a region is only worth compute if refinement could change the decision. Run two Dijkstra passes on the coarse traversability grid:

```
f(c) = cost-to-come from start     (forward Dijkstra)
g(c) = cost-to-go to goal          (backward Dijkstra)
T(c) = f(c) + g(c)                 best path cost *through* c        (24)
```

`T(c) − J(π*)` is the *detour penalty* for routing through `c`. Refine only where

```
T(c) − J(π*) < τ                                                     (25)
```

— the corridor of near-optimal alternatives. **Cells outside that band cannot change the plan no matter how finely resolved, so refining them is provably wasted compute.** τ sets the budget and maps directly onto the refinement pool size.

This is the online form of the offline metric, and it is cheap: two Dijkstras over ~10⁵ coarse cells, well under a millisecond.

**Unit test.** Construct a synthetic map with a narrow gap. Assert `R(S) = 0` for schedules fine enough to resolve the gap and `R(S) > 0` for schedules coarser than the gap width. Assert cells flagged by (25) form a connected corridor containing `π*`.

---

## 9. Reference map and information-loss metrics

### 9.1 Construction

Aggregate all scans of a held-out sequence into one static cloud using GT poses and GT labels, remove `moving-*` points, rasterise at 5 cm with no LOD and no time limit. That is `M*`. It is schedule-independent, which is what makes cross-schedule comparison valid.

### 9.2 Per-ring height RMSE

For each coarse cell `c`, let `F(c)` be the set of reference fine cells it subsumes, and `h*_f` their reference heights.

```
RMSE_L = sqrt( (1/|C_L|) Σ_{c ∈ C_L} ( μ_c − h̄*(c) )² ),
         h̄*(c) = mean_{f ∈ F(c)} h*_f                               (26)
```

### 9.3 ⚑ The coarsening-justification ratio

The number that expresses the thesis. For coarse cell `c`, define information loss against the individual fine cells (not their mean):

```
IL(c)² = (1/|F|) Σ_{f ∈ F(c)} ( μ_c − h*_f )²
       = ( μ_c − h̄*(c) )²   +   Var_{f}( h*_f )
         └──── bias² ────┘       └── spread² ──┘                    (27)
```

by the standard bias–variance decomposition. `spread` is the **intrinsic** sub-cell terrain variability — it is what any single-value cell must pay, irrespective of algorithm. So define

```
ρ(c) = IL(c) / spread(c)                                            (28)
```

- `ρ ≈ 1` — coarsening cost only the intrinsic sub-cell variability. **Optimal; the saving was free.**
- `ρ ≫ 1` — your estimate is biased beyond the terrain's own roughness. The schedule is too aggressive, or fusion is wrong.

Report `ρ` per ring. It is the entire argument compressed to one dimensionless number, and it separates *what the representation costs* from *what the algorithm costs* — which nobody in the adaptive-mapping literature reports.

### 9.4 Dynamic removal — both directions

```
DR = removed_dynamic / total_dynamic          (removal rate)
SP = preserved_static / total_static          (preservation rate)
F  = 2·DR·SP / (DR + SP)                                            (29)
```

Report all three. **DR alone is gameable** — delete the whole map and score 100%. The harmonic mean prevents that.

---

## 10. Occupancy, class fusion, reflectivity, visibility

### 10.1 Three-state occupancy

Log-odds with an explicit unknown state:

```
l ← clamp( l + log(p_meas/(1−p_meas)),  l_min,  l_max )             (30)

state = UNKNOWN   if n < n_min
        OCCUPIED  if l > l_occ
        FREE      otherwise
```

**Unknown is decided by observation count, not by log-odds.** "I looked and it's empty" and "I couldn't see" are different facts; a log-odds value near zero conflates them. Clamping prevents saturation — an unclamped cell that has seen 500 free observations needs 500 occupied ones to change its mind, which is why unclamped maps fail to register newly-appeared obstacles.

### 10.2 Class fusion in one byte — Boyer–Moore majority ⚑

A Dirichlet count vector over K classes needs K bytes; the cell budget allows one. Boyer–Moore streaming majority solves this in constant memory:

```
on observing class y:
    if counter == 0:        candidate ← y ;  counter ← 1
    elif y == candidate:    counter ← min(counter + 1, 15)
    else:                   counter ← counter − 1
```

Packed as 4-bit candidate + 4-bit counter = 1 byte. **Guaranteed to return the true majority class whenever one exists** (>50% of observations), and the counter doubles as a confidence readout. Never average softmax vectors across frames — the mean of two confident, contradictory distributions is a confident-looking lie.

### 10.3 Reflectivity normalisation

Raw intensity confounds surface reflectance with geometry. The LiDAR equation gives `I ∝ ρ cos θ_inc / r²`, so recover the intrinsic reflectance:

```
ρ̂ = I · r² / max(cos θ_inc, 0.1)      then normalise to [0,255]     (31)
```

Lane paint has `ρ ≈ 0.5`, dry asphalt `≈ 0.1`; **wet asphalt reflects specularly and returns almost nothing**, so `ρ̂ ≈ 0` on a cell classified `road` is a wet-surface indicator. One byte, no extra sensor, directly serves the drivable/non-drivable requirement.

### 10.4 Visibility cleanup — O(1) per cell, no ray casting

For map cell `c` at position `p`, project into the current range image:

```
(u, v) = proj(p) ,     r_expected = ‖p − p_sensor‖
clear c   iff   R_current(u,v) > r_expected + δ                     (32)
```

If the current beam returned from *further away* than the cell, the beam passed through it, so the cell is empty. This is a range comparison, not a 3D traversal — **O(1) per cell, fully parallel.**

**Guard, mandatory:** never clear a cell that has a return in the current scan. Without it, thin structures — fences, poles, sign posts — get eaten within a few frames. `δ` absorbs range noise; set `δ = 3σ_r(r)` from (12), so the guard band widens with distance automatically rather than being a hand-tuned constant.

### 10.5 Residual images for motion

Transform scan `t−k` into frame `t` and compare range images:

```
D_k(u,v) = | R_t(u,v) − R_{t←t−k}(u,v) |                            (33)
```

Static geometry cancels; moving objects leave a bright residual. Feed `{D_1, D_2, D_4}` as extra input channels. Note the detectability floor is (7): residuals cannot beat the geometric limit, so beyond ~25 m a pedestrian produces no residual and only the semantic prior remains.

---

## 11. Memory arithmetic — every number, with its assumptions

Ratios are pure cell-count ratios and therefore **invariant to bytes-per-cell**; absolute sizes are not. State both.

```
N_ours    = 745,000 cells                                 (19)
N_uniform = (200/0.05)² = 16,000,000 cells
N_dense3D = (200/0.05)² × (8/0.05) = 2.56 × 10⁹ voxels
```

| Representation | Assumption | Size | Ratio vs ours |
|---|---|---|---|
| Dense 3D voxel | 1 B/voxel, 8 m vertical extent | 2.56 GB | **286×** |
| Sparse/hashed 3D | surface-only, 8 B/voxel incl. hash overhead | ~130–240 MB | ~15–27× |
| Uniform 5 cm 2.5D | same 12 B cell | 192 MB | **21.5×** |
| **Ours, 4-ring** | 12 B cell | **8.94 MB** | — |
| Ours, 3-ring ablation | 12 B cell | 6.24 MB | 30.8× vs uniform |
| Conservative pyramid | 5 of 12 B, ×1/3 | +1.24 MB | overhead |
| Refinement pool | 512 × 16 × 12 B | +0.10 MB | fixed |
| **Total bound** | | **≈ 10.3 MB** | compile-time |

**Report the dense-3D, sparse-3D and uniform-2.5D numbers together, in that order.** The problem statement asks for the 3D comparison, so lead with it; volunteering the sparse figure before someone else raises it reads as good faith; the uniform-2.5D figure is the one that actually isolates *your* contribution from the 3D→2.5D reduction.

---

## 12. Summary — which mathematics is load-bearing

| § | Result | Status | Test |
|---|---|---|---|
| 1.2 | `s_rad = r²Δφ/h` — quadratic radial sampling | ⚑ original analysis | empirical fill rate |
| 1.3 | Ring-sweep filling; uniform 5 cm is 99.87% empty at 50 m | ⚑ the core argument, quantified | fill-rate plot |
| 1.4 | Blind cone 3.74 m; potholes ≤ 8.3 m; motion ≤ 25 m | derived scope limits | geometry check |
| 2.2 | Nested-floor theorem ⇒ exact partition | standard, correctly applied | 10⁶-point partition test |
| 3.2 | `σ²_z ∝ r²σ²_φ / cos²θ_inc` | standard (Fankhauser) | monotonicity |
| 4.2 | Merge by law of total variance | ⚑ corrects a real error | kerb-step test |
| 5.2 | Variance inflation on split | ⚑ contribution | slope test |
| 5.4 | Round-trip idempotence via `derived` bit | ⚑ contribution | exact round-trip |
| 7.3 | Conservative pyramid, no false negatives | ⚑ imported from graphics | exhaustive block test |
| 8.1 | Plan regret `R(S)` | ⚑ headline contribution | synthetic-gap test |
| 8.3 | Corridor rule `T(c) − J(π*) < τ` | ⚑ contribution | corridor connectivity |
| 9.3 | Coarsening ratio `ρ = IL/spread` | ⚑ contribution | bias/spread decomposition |
| 10.2 | Boyer–Moore class fusion in 1 byte | ⚑ neat, nobody else will | majority guarantee |

Seven ⚑ results. Three of them (5.4, 8.1, 9.3) are the ones to put on slides.
