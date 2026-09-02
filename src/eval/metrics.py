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
"""

import numpy as np
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.gpu.kernels import Z_MAX_CM, Z_MIN_CM
from vrgrid.grid.fusion import occupancy_state
from vrgrid.grid.query import window_cells


def _ring_cells(gm, ring: int):
    """(slots, i_lo, j_lo) for one ring: every slot in its window, and the
    base-lattice corner of the reference block each one subsumes."""
    buf = gm.buffers[ring]
    ix, iy = window_cells(buf)
    slots = np.arange(buf.slots, dtype=np.int64) + buf.offset
    k = gm.schedule.k(ring)
    return slots, ix * k, iy * k


def _compared(gm, reference, ring: int, require_observed=True):
    """The cells of `ring` that can honestly be scored, with their reference
    statistics. Returns (slots, n_ref, ref_mean_cm, ref_var_cm2, mine_cm)."""
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
    otherwise over the ring's whole window, which includes the toroidal
    padding and the hole covered by the finer ring, and would report a fill
    rate diluted by memory that was never meant to hold anything.

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
