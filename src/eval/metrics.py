"""Information-loss metrics. Math §9. [Aakash]

Report per-ring, never as a single scalar — the whole claim is that error is
allowed to grow with range, so an aggregate number hides the result.

Far-ring accuracy must be reported as a function of frames-since-first-
observation, not as a scalar. P_fill < 2% beyond 25 m means the far field is
filled by ego-motion sweeping the ring pattern across the ground, not by any
single frame ("ring-sweep filling", math §1.3). Single-frame far-field numbers
are meaningless.

--- how a coarse cell finds its reference cells ---------------------------

Every metric here needs `F(c)`, the set of 5 cm reference cells a ring-L cell
subsumes. Because both live on the same base lattice and `i_L = i_fine // k_L`
(§2.1), that set is exactly

    [i_L·k_L, (i_L+1)·k_L) x [j_L·k_L, (j_L+1)·k_L)

— a rectangle of integers, with no resampling, no interpolation and no
tolerance. This is the partition theorem of §2.2 being spent rather than
merely proved, and it is worth saying in the report: the alignment guarantee
is not only a correctness property, it is what makes the evaluation exact.

--- what is deliberately NOT scored --------------------------------------

A ring cell with no observed reference cell underneath it is dropped, not
counted as agreement. The far rings extend well past where a short sequence
ever drove, and scoring "we predicted nothing and the truth is nothing" as a
hit would make every metric improve with range, which is the exact opposite
of the effect being measured.

⚑ **And a ring is scored only where it is still the ring that ANSWERS.** Ring
L's buffer is a square of half-width `R_L`, so it physically covers the hole
that the finer rings serve; `ring_of` hands a place to the FINEST ring that
contains it, so ring L only ever receives returns from the annulus
`[R_{L-1}, R_L)`. The vehicle then drives, that annulus sweeps outward across
the world, and every cell it leaves behind keeps its last far-range value for
as long as it stays in the window. Nothing clears it -- a toroidal shift only
clears the edge coming into view (§2.4) -- and nothing reads it either, because
`query()` routes that place to a finer ring now.

Scoring those cells compares a value frozen at 60 m against a reference that
went on accumulating the close-range returns the cell never received, and the
population is not small. On the synthetic sequence, stale share of each ring's
scored cells:

    driven      ring 1    ring 2
     22 m         19%       21%
     46 m         20%       26%

Ring 3 shows 13 cells and not a fraction, and that is this scene rather than
the effect. The synthetic terrain is flat to x = 30 m and then ramps at 6%,
and a rising surface closes the forward horizon: forward returns past 50 m
appear at frame 0 and never again. Ring 3's annulus is otherwise lateral and
rear, where a straight-line drive leaves nothing behind. On a real sequence
the far band is fed continuously for kilometres and ring 3 is the ring that
carries most of this.

⚑ Nothing here is a cell moving between rings. The buffers are static and
world-anchored and a cell never changes ring -- what moves is the vehicle, and
with it which ring is RESPONSIBLE for a given place. "Migration" below always
means that responsibility passing inward, never storage being relocated.

⚑ **A uniform baseline has one ring, so it structurally cannot carry this --
and that is the direct test of how much it costs.** `known-limitations.md` §6
runs exactly that comparison on seq 07 with the datum fixed: one-ring uniform
20 cm against the four-ring schedule's ring 2, **-0.26 cm and -0.41 cm**.
Migration costs nothing measurable in height bias. An earlier version of this
note claimed the confound distorted §8.2's money plot across schedules; that
claim is **withdrawn**, because it rested on synthetic runs made before §6's
datum fix and §6's real-data experiment is the better evidence.

What the defect does change is WHICH CELLS ARE SCORED, so it lands on the
population metrics rather than on height: ring 3's fill rate read 0.23 against
0.83 over the cells it actually answers for -- the §1.3 ring-sweep claim
understated three and a half times, because the unwritten interior counted as
unfilled. That is a different metric from the one §6 measured and is not
covered by it.

So `_ring_cells` returns the cells the ring SERVES, not the cells it stores,
and every metric in this file inherits that. The predicate is `ring_of` on the
cell centre -- the same function `query()` routes with, so the scored set
cannot drift from the set the map actually answers with. See
`test_a_ring_is_scored_only_where_it_still_answers`.

⚑ **What that does NOT fix, and why it is left.** A cell the ring still serves
is scored against every reference return in its footprint, including returns
fired from outside the ring's band -- the ground behind the vehicle was driven
over at 2 m before it fell back to 40 m, and M* kept all of it while ring 2's
cell only ever integrated the 25-50 m returns. Fixing that properly means a
range-stratified M*: (n, sum, sum^2) per band per 5 cm cell, roughly 4x the
reference's memory and 4x its summed-area tables, on an array that is already
205 MB for a 12-frame synthetic scene. Measured against a reference rebuilt
from band-restricted returns instead, on the 12-frame sequence:

    ring    band      RMSE vs M*   RMSE vs M*|band
     1     10-25 m       0.37            0.32
     2     25-50 m       0.32            0.30
     3    50-100 m       0.33            0.32

At most 0.05 cm, which is under the 0.29 cm quantisation floor §9.3 already
puts on any of these numbers -- so it is second order to the migration
confound above (0.40 -> 0.37 on the same ring, and 0.39 -> 0.72 on its IoU)
and the memory is not worth spending on it. Stated rather than hidden, and
the measurement is `scripts/`-free on purpose: it is a one-off, and it wants
re-running on real data, where the rear band has a kerb in it and this
sequence has smooth analytic terrain.

⚑ **THE SIGN OF THE CORRECTION IS NOT SETTLED, and no figure for it is
recorded here on purpose.** Every before/after number above is the synthetic
sequence, where dropping the stale cells LOWERS RMSE -- the stale value was
written at grazing incidence on the 6% ramp and is worse than the live
annulus.

An external measurement against seq 07/08 was reported on 3 Sep and revised
the same day: first as a consistent 3-12% UNDERSTATEMENT, then withdrawn in
favour of "no consistent bias, -40% to +21% depending on ring, sequence and
frame count". Its traced example was revised too -- 350.7 cm retracted as a
separate M* defect, restated as 169.5 cm. Neither figure is reproducible here:
there is no data root on this machine (`VRGRID_DATA_ROOT` unset, `data/` holds
only its README) and no M* artefact for either sequence.

So the honest state is that the direction on real data is **unknown**, not
that it is any particular percentage. A withdrawn number quoted as if it stood
is worse than no number, which is why the 3-12% that was briefly written into
this file, §9.2 and `known-limitations.md` has been taken out rather than
swapped for its replacement.

⚑ And it touches the headline. `known-limitations.md` §2b leads with
**rho = 1.45 median** at ring 1 over eleven sequences; this fix changes the
scored population, and rho moves by up to 0.06 per ring on the synthetic
sequence. **§2b's table should be regenerated with this fix before rho is
quoted to two decimals.** The finding survives in shape -- 0.06 does not move
rho out of its band -- but the second decimal is not currently earned.
"""

import numpy as np
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.gpu.kernels import Z_MAX_CM, Z_MIN_CM
from vrgrid.grid.fusion import occupancy_state
from vrgrid.grid.lattice import ring_of
from vrgrid.grid.query import window_cells


def _cell_centres_m(gm, ring: int, ix, iy):
    """Vehicle-frame centres of a ring's cells, in metres.

    The vectorised twin of `gate._cell_centre`, which is O(1) per cell because
    it is called from the frame loop; this one is handed a whole window at
    once and is not. Same arithmetic, and
    `test_cell_centres_agree_with_the_frame_path` pins that they agree --
    two spellings of a cell's position is how a metric ends up scoring the
    cell next door.
    """
    cell_m = gm.schedule.rings[ring].cell_m
    return ((ix + 0.5) * cell_m - gm.vehicle_xy_m[0],
            (iy + 0.5) * cell_m - gm.vehicle_xy_m[1])


def _ring_cells(gm, ring: int):
    """(slots, i_lo, j_lo) for the cells ring `ring` still SERVES, and the
    base-lattice corner of the reference block each one subsumes.

    Not every slot in the window. The window is a square of half-width `R_L`
    and therefore contains the hole the finer rings serve; the cells in it hold
    whatever they were last written with before the vehicle migrated them
    inward, and nobody reads them. See the note in the module docstring for
    what scoring them costs.

    `ring_of` on the cell centre is the predicate, because it is exactly what
    `slot_of` routes a query with -- pinned in
    `test_the_scored_set_is_the_set_query_routes_to`. The centre is the
    convention for a cell straddling a ring boundary; §2.4 already says that
    boundary wobbles by up to one coarsest cell as the window shifts, so no
    finer rule would mean anything.
    """
    buf = gm.buffers[ring]
    ix, iy = window_cells(buf)
    slots = np.arange(buf.slots, dtype=np.int64) + buf.offset
    k = gm.schedule.k(ring)

    serves = ring_of(*_cell_centres_m(gm, ring, ix, iy),
                     gm.schedule, gm.speed_ms) == ring
    return slots[serves], ix[serves] * k, iy[serves] * k


def _compared(gm, reference, ring: int, require_observed=True):
    """The cells of `ring` that can honestly be scored, with their reference
    statistics. Returns (slots, n_ref, ref_mean_cm, ref_var_cm2, mine_cm).

    Three conditions, and they are three different questions: `_ring_cells`
    asks whether the ring still answers for the place, `n_ref > 0` whether the
    reference knows anything about it, and `require_observed` whether this map
    ever wrote it. A cell has to pass all three to be an honest comparison.
    """
    slots, i_lo, j_lo = _ring_cells(gm, ring)
    k = gm.schedule.k(ring)
    n_ref, ref_mean, ref_var = reference.block_stats(i_lo, j_lo, k)

    keep = n_ref > 0
    if require_observed:
        # ⚑ `obs_count > 0` is NOT enough: it counts every return, and a cell
        #   whose returns were all NON-ground has no measured ground height at
        #   all. `fuse` leaves such a cell's `ground_height` at its initial 0,
        #   and 0 cm is not a neutral height -- it is THE DATUM. Scoring those
        #   cells makes the metric depend on where the datum happens to sit:
        #   on seq 07, moving the datum from -1.64 m to -2.00 m took ring 1's
        #   RMSE from 3.19 cm to 6.15 cm without changing a single measurement.
        #   A metric whose answer moves with an arbitrary offset is measuring
        #   the offset.
        #
        #   `height_variance > 0` is the predicate. The variance codec maps
        #   code 0 to MAXIMUM variance precisely so that "never fused" is
        #   distinguishable from "fused and confident" (fusion.initialise), so
        #   a non-zero code means ground evidence actually reached this cell.
        keep &= gm.soa["obs_count"][slots] > 0
        keep &= gm.soa["height_variance"][slots] > 0
        # ⚑ And not saturated at the band edge. The map is a LOCAL 8 m band
        #   that slides with the vehicle (gpu.shift.track_datum); M* is global.
        #   On a sequence with more relief than the band, ground the vehicle
        #   has left behind falls out of it and clamps -- 58.6% of seq 08's
        #   observed cells after only 40 frames of its 45.7 m climb. A clamped
        #   cell holds the band edge, not a measurement, and scoring it against
        #   a global reference measures the band rather than the map: it put
        #   seq 08's ring 1 RMSE at 213 cm.
        #
        #   Excluded, not silently: `saturated_fraction_per_ring` reports how
        #   much of each ring this removes, and a ring that loses most of
        #   itself is telling you the band is too narrow for the sequence, not
        #   that the map is accurate over what survives.
        g = gm.soa["ground_height"][slots]
        keep &= (g > Z_MIN_CM) & (g < Z_MAX_CM)
    # ⚑ `+ z_datum_m`. Stored heights are relative to the run's datum; M* is
    #   world-absolute. Compare them without this and the whole difference is
    #   the vehicle's starting elevation -- 162 cm on seq 07 -- dressed up as
    #   map error.
    return (slots[keep], n_ref[keep], ref_mean[keep], ref_var[keep],
            gm.soa["ground_height"][slots[keep]].astype(np.float64)
            + getattr(gm, "z_datum_m", 0.0) * 100.0)


def height_rmse_per_ring(gm, reference):
    """RMSE_L against the mean reference height in each footprint. Eq. (26).

    In centimetres, which is the unit the claim is made in ("the kerb is
    12 cm") and the unit the map stores. Returns {ring: rmse_cm}, with nan for
    a ring that has no comparable cell -- nan rather than 0.0, because a ring
    nobody drove through has no error and reporting 0 would read as perfect.
    """
    out = {}
    for ring in range(len(gm.schedule.rings)):
        _, _, ref_mean, _, mine = _compared(gm, reference, ring)
        out[ring] = (float(np.sqrt(np.mean((mine - ref_mean) ** 2)))
                     if mine.size else float("nan"))
    return out


def coarsening_ratio_per_ring(gm, reference):
    """⚑ rho = IL / spread, per ring. Math §9.3, eqs. (27)-(28).

    The number that expresses the thesis, and the one nobody else in the
    adaptive-mapping literature reports. By the bias-variance decomposition,

        IL(c)^2 = (mu_c - mean_ref)^2  +  Var_ref
                  \\___ bias^2 ___/       \\_ spread^2 _/

    `spread` is the INTRINSIC sub-cell terrain variability -- what any
    single-value cell must pay whatever the algorithm is. So rho ~ 1 means the
    coarsening cost only what the terrain itself costs and the memory saving
    was free; rho >> 1 means the estimate is biased beyond the terrain's own
    roughness, and the schedule is too aggressive or the fusion is wrong.

    Returns {ring: {il_cm, bias_cm, spread_cm, rho, n}}.

    Cells whose reference footprint holds a single observation are excluded
    from rho: their spread is 0 by construction, not by flatness, and dividing
    by it manufactures an infinite ratio out of thin evidence. They still
    count toward RMSE, where they are perfectly legitimate.

    ⚑ rho is a RATIO OF AGGREGATES, `rms(IL) / rms(spread)`, not the mean of
      the per-cell ratios. §9.3 defines IL and spread per cell and asks for
      rho per ring, and the two readings are not close: a mean of ratios is
      dominated by cells whose reference footprint is nearly flat, where
      spread -> 0 and any error at all gives an enormous ratio. Measured on
      the synthetic scene, writing each cell the EXACT mean of its footprint
      -- bias identically zero, so rho must be 1 by construction -- the mean
      of ratios reports 4.8 and the ratio of aggregates reports 1.0.

    ⚑ And rho has a floor the schedule cannot beat. Heights are stored to
      1 cm, so IL can never fall below the quantisation noise of q/sqrt(12) =
      0.29 cm however good the estimate is. Where the terrain's own spread is
      finer than that -- smooth asphalt -- rho is bounded below by roughly
      0.29/spread and is measuring the STORAGE, not the coarsening. Read rho
      on rough ground; on glass-smooth ground read RMSE instead.
    """
    out = {}
    for ring in range(len(gm.schedule.rings)):
        _, n_ref, ref_mean, ref_var, mine = _compared(gm, reference, ring)
        usable = n_ref > 1
        bias2 = (mine - ref_mean) ** 2
        il2 = bias2 + ref_var

        if not np.any(usable):
            out[ring] = {"il_cm": float("nan"), "bias_cm": float("nan"),
                         "spread_cm": float("nan"), "rho": float("nan"), "n": 0}
            continue

        il_cm = float(np.sqrt(np.mean(il2[usable])))
        spread_cm = float(np.sqrt(np.mean(ref_var[usable])))
        # ⚑ `bias_cm` is RMS(bias), not mean(bias), and the two answer different
        #   questions. RMS cannot tell "systematically 20 cm high" from
        #   "randomly +/-20 cm", and on real data those are different defects
        #   with different causes. Measured on seq 07, 40 frames: ring 1 reads
        #   bias_cm 20.72 but its MEAN bias is +3.98 cm -- almost all
        #   dispersion. Ring 2 reads 36.02 with a mean of +23.46 and 75.9% of
        #   cells above the reference, which is a real systematic offset that
        #   grows with range. Both numbers are reported so neither can be
        #   mistaken for the other.
        out[ring] = {
            "il_cm": il_cm,
            "bias_cm": float(np.sqrt(np.mean(bias2[usable]))),
            "mean_bias_cm": float(np.mean((mine - ref_mean)[usable])),
            "above_frac": float(np.mean((mine - ref_mean)[usable] > 0)),
            "spread_cm": spread_cm,
            "rho": il_cm / spread_cm if spread_cm > 1e-9 else float("nan"),
            "n": int(usable.sum()),
        }
    return out


def occupancy_iou_per_ring(gm, reference, thresholds=None):
    """Intersection over union of OCCUPIED cells, per ring.

    Truth is "the reference has at least one static return in this footprint".
    IoU rather than accuracy because the map is mostly empty: predicting FREE
    everywhere scores 97% accuracy and 0% IoU, and only one of those two
    numbers notices.
    """
    th = thresholds if thresholds is not None else gm.thresholds
    out = {}
    for ring in range(len(gm.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(gm, ring)
        k = gm.schedule.k(ring)
        n_ref, _, _ = reference.block_stats(i_lo, j_lo, k)

        truth = n_ref > 0
        mine = occupancy_state(gm.soa, th, slots) == OCC_OCCUPIED

        union = np.count_nonzero(truth | mine)
        out[ring] = (float(np.count_nonzero(truth & mine) / union)
                     if union else float("nan"))
    return out


def fill_rate_per_ring(gm, reference=None):
    """Fraction of cells with at least one observation, per ring.

    Measured over the cells the reference says exist, when one is given --
    otherwise over everything the ring serves, which still includes cells no
    return has ever reached and so reports a fill rate diluted by memory that
    was never meant to hold anything.

    ⚑ The hole this docstring used to warn about is gone at the source:
      `_ring_cells` no longer returns the cells the finer rings serve. It had
      been dropping them here by `n_ref > 0`, which is not the same test and
      does not do it -- the reference has plenty of returns under the hole,
      which is the whole problem. Ring 3 counted cells frozen at 60 m as
      "filled" and reported a far-field fill rate the ring had not earned.

    Expect this to be low and to RISE with frame count, not with a single
    frame's geometry: past ~25 m the far field is filled by ego-motion sweeping
    the ring pattern across the ground, at P_fill < 2% per frame (§1.3).
    """
    out = {}
    for ring in range(len(gm.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(gm, ring)
        seen = gm.soa["obs_count"][slots] > 0
        if reference is None:
            out[ring] = float(np.mean(seen))
            continue
        k = gm.schedule.k(ring)
        n_ref, _, _ = reference.block_stats(i_lo, j_lo, k)
        scope = n_ref > 0
        out[ring] = float(np.mean(seen[scope])) if np.any(scope) else float("nan")
    return out


def footprint_coverage_per_ring(gm, reference):
    """Median fraction of each scored cell's k x k footprint that M* observed.

    ⚑ Read rho against this, and do not report rho without it. §9.3 defines
      `spread` as the variability of the reference heights across `F(c)`, and
      `block_stats` estimates it from the cells of `F(c)` M* actually observed
      -- `n` is a count of observed 5 cm cells, capped at k^2 by construction.
      On the 12-frame synthetic sequence the median is 0.25 at ring 1, 0.06 at
      ring 2 and 0.02 at ring 3: ring 3's sub-cell terrain variability is being
      estimated from roughly ONE reference cell in sixty-four.

      A spread estimated from two points is biased low, and rho divides by it,
      so rho on the coarse rings is biased HIGH -- the conservative direction
      for a number we want near 1, which is why this is a disclosure rather
      than a correction. `coarsening_ratio_per_ring` already drops `n_ref <= 1`
      for the same reason; that guard is simply far too weak at k = 8, where it
      admits a spread computed from two cells of sixty-four.

      Returns {ring: median_coverage}, nan for a ring with nothing scored.
    """
    out = {}
    for ring in range(len(gm.schedule.rings)):
        _, n_ref, _, _, _ = _compared(gm, reference, ring)
        k = gm.schedule.k(ring)
        out[ring] = (float(np.median(n_ref / (k * k))) if n_ref.size
                     else float("nan"))
    return out


def dynamic_removal(removed_dynamic, total_dynamic,
                    preserved_static, total_static):
    """DR, SP and their harmonic mean. Math §9.4 eq. (29).

    All three, always. DR alone is gameable -- delete the whole map and score
    100% removal -- which is why the section insists on both directions and
    the F-score that punishes trading one for the other.
    """
    dr = removed_dynamic / total_dynamic if total_dynamic else float("nan")
    sp = preserved_static / total_static if total_static else float("nan")
    f = 2 * dr * sp / (dr + sp) if (dr + sp) else 0.0
    return {"DR": float(dr), "SP": float(sp), "F": float(f)}


def memory_bytes(allocation) -> int:
    """Total preallocated bytes.

    Deferred to `gpu.allocators.bytes_allocated()` rather than recomputed. Two
    functions that both "know" the memory figure is how the report and the
    running system end up disagreeing, and the memory number is a headline
    claim with a live counter next to it in the demo.
    """
    from vrgrid.gpu.allocators import bytes_allocated

    return bytes_allocated(allocation)


def saturated_fraction_per_ring(gm, reference):
    """Share of each ring's OBSERVED cells sitting at the vertical band edge.

    The companion to `_compared`'s exclusion of them. A high value does not
    mean the map is wrong -- it means the 8 m band cannot hold this sequence's
    relief, so the map has forgotten ground the vehicle drove past, and the
    per-ring accuracy figures describe only what is still inside the band.
    Report it beside the RMSE or the RMSE is quoted over an unstated subset.
    """
    out = {}
    for ring in range(len(gm.schedule.rings)):
        slots, _, _ = _ring_cells(gm, ring)
        seen = gm.soa["obs_count"][slots] > 0
        if not seen.any():
            out[ring] = float("nan")
            continue
        g = gm.soa["ground_height"][slots][seen]
        out[ring] = float(np.mean((g <= Z_MIN_CM) | (g >= Z_MAX_CM)))
    return out
