"""Scatter and the fixed-point accumulation path. [Shrestha]

`scatter()` is the one custom kernel in the project: points in, per-cell
aggregates out, in parallel, with a result that is **bit-identical run to
run**. IEEE-754 addition is not associative, so GPU atomic float adds -- which
complete in nondeterministic order -- produce a different map every run. You
cannot bisect a bug whose location moves. Heights are quantised to 1 cm
anyway, so everything here accumulates in integers, which are exactly
associative. See math §3.4.

Two implementations, identical outputs, tested against each other:

  `scatter_sorted`  (default) sorts point indices and reduces each segment.
                    Scratch is sized by POINTS (15.0 MB at 150,000/frame),
                    independent of grid size, and nothing needs clearing
                    between frames. Allocates nothing per frame.
  `scatter_atomic`  the literal reading of master v4 §3.5 level 4: unbuffered
                    integer atomic adds into a dense per-cell accumulator.
                    Scratch is sized by CELLS (~11.9 MB at 745,000) and must be
                    cleared every frame.

Both are deterministic -- that is the point of the integers, and it is why the
choice between them is a cost decision rather than a correctness one. The
default is `sorted` because a dense accumulator costs more than the entire
grid it accumulates into. `tests/test_kernels.py` asserts the two agree
field-for-field on random input.

This file does NOT own the Kalman update or the class-fusion rule -- those are
semantics (math §3.3, §10.2, Aakash). It owns making them fast and repeatable.

One piece of semantics does live here, because it is a property of the
reduction rather than of the filter: **the height sums are taken over GROUND
returns only.** §3 estimates the elevation of the ground, so a canopy return is
not weak evidence about it. `w_sum == 0` therefore means "no ground evidence
this frame" and `mean_height_cm()` is meaningless for those cells --
`has_ground_evidence()` is the predicate, and `fuse()` leaves such a cell's
height and variance untouched. Every other column stays over all returns.
"""

import numpy as np
from vrgrid.cell import CELL_BYTES

# Inverse-variance weights are quantised to integers so the accumulation stays
# exactly associative. 1024 gives ~3 decades of dynamic range between a 5 m
# return and a 100 m one, with the smallest weight still >= 1.
WEIGHT_SCALE = 1024
WEIGHT_MAX = 1 << 20

# Vertical extent -2 to +6 m, in the 1 cm units heights are stored in.
Z_MIN_CM, Z_MAX_CM = -200, 600

# Packed class key: point_id * CLASS_RADIX + class_id. Taking the integer
# minimum of that key yields the lowest-numbered point in the cell AND its
# class in one reduction -- deterministic, and one array instead of two.
CLASS_RADIX = 32

# Packed sort key for the sorted path, one int64 per point:
#
#     key = (cell * POINT_RADIX + point_id) * POINT_RADIX + position
#
# Sorting that single array IS the (cell, point_id) lexicographic order -- the
# same order `np.lexsort` produced, in a buffer we own. Because every position
# in a frame is distinct the keys are unique, so an in-place introsort is
# exactly as deterministic as a stable sort and needs no merge workspace:
# measured 2.6 KB, against ~1.2 MB for the argsort it replaces.
#
# The bottom field must be POSITION and not point_id, even though they are
# equal in the default case. `point_id` orders the points; `position` is where
# the payload columns actually live. The determinism test hands in a shuffled
# scan carrying its original ids, and conflating the two gathers heights from
# the wrong points while still producing a perfectly plausible map.
POINT_RADIX = 1 << 18                    # 262,144 -- the cap on points per frame
CELL_MAX = 1 << 27                       # what is left of int64 above the two fields
KEY_DROPPED = np.iinfo(np.int64).max     # holes and off-map returns sort to the end

CEILING_NONE = np.iinfo(np.int16).max  # sentinel: nothing overhead seen
# Sentinel for the packed class key: the max of whatever dtype holds it, so it
# always loses the integer minimum against a real key. It cannot share
# CEILING_NONE -- 32767 is beaten by any point past index 1023, which silently
# made every far cell report class 32767 % 32 = 31.

# Sensor defaults, HDL-64E on KITTI. Overridden from configs/thresholds.yaml.
SIGMA_R_M = 0.02
SIGMA_PHI_RAD = np.radians(0.1)
SENSOR_HEIGHT_M = 1.73


def measurement_variance_cm2(range_m, sigma_r_m=SIGMA_R_M, sigma_phi_rad=SIGMA_PHI_RAD,
                             h_s_m=SENSOR_HEIGHT_M, cos_incidence=1.0):
    """Height variance of a single return, in cm^2. Math §3.2, eq (12)-(13).

        sigma_z^2 ~= (h_s/r)^2 sigma_r^2  +  r^2 sigma_phi^2   , all over cos^2(theta_inc)

    The second term dominates at range and grows as r^2: 8.7 cm at 50 m,
    17.5 cm at 100 m. The first is the near-field floor. A return at 80 m at
    grazing incidence is dramatically less informative than one at 5 m
    head-on, and this is the function that says so numerically.
    """
    r = np.maximum(np.asarray(range_m, dtype=np.float64), 1e-3)
    cos_inc = np.maximum(np.asarray(cos_incidence, dtype=np.float64), 0.1)  # no singularity
    var_m2 = ((h_s_m / r) ** 2 * sigma_r_m ** 2 + r ** 2 * sigma_phi_rad ** 2) / cos_inc ** 2
    return var_m2 * 1e4  # m^2 -> cm^2


def quantise_weight(variance_cm2) -> np.ndarray:
    """Inverse variance, quantised to int32. Never zero: a far return is weak
    evidence, not absent evidence, and a zero weight would make a cell observed
    only at range have no height at all."""
    w = WEIGHT_SCALE / np.maximum(np.asarray(variance_cm2, dtype=np.float64), 1e-9)
    return np.clip(np.rint(w), 1, WEIGHT_MAX).astype(np.int32)


def quantise_height(z_m, datum_m: float = 0.0) -> np.ndarray:
    """Metres -> int16 centimetres, clamped to the vertical extent.

    `datum_m` is the world-frame elevation the band is measured FROM. It
    defaults to 0.0 -- the world datum, and exactly what every caller meant
    while the band was world-absolute, bit for bit.

    ⚑ **The band is 8 m wide; it is not 8 m above sea level.** Clamping to a
    world-absolute [-2, +6] m works only while the vehicle stays near z = 0,
    which is true of seq 00 and 07 and false of seq 08's hill. Above the
    ceiling every nearby cell's height saturates at +6 m while the sensor sits
    tens of metres higher, so `visibility_cleanup` -- which is handed these
    heights and documents them as VEHICLE-frame -- computes a viewing angle
    far above the sensor's FOV limit, finds nothing in view, and clears no
    ghosts at all. Measured on seq 08: inert above ~11.7 m world-z, 57% of its
    frames.

    Subtracting ego-z at the cleanup's end does not repair this, and that is
    the non-obvious part: by then the clamp has already destroyed the value it
    would subtract from. Driving the real projection with ground-level cells,
    seq 07 (world-z -5.8 m, which saturates the FLOOR of the band) goes from 42
    of 46 cells in view to 0 -- a saturated -2.0 m minus an ego of -5.8 m reads
    as +3.8 m and leaves through the TOP of the image. The frame and the clamp
    have to move together, which is what this argument is for: pass the
    vehicle's elevation and the band follows it, keeping the 8 m of vertical
    extent the schedule's `vertical_extent_m` declares and the memory claim
    is computed from.
    """
    z_cm = np.asarray(z_m, dtype=np.float64) * 100.0
    if datum_m:
        z_cm -= datum_m * 100.0
    np.rint(z_cm, out=z_cm)
    return np.clip(z_cm, Z_MIN_CM, Z_MAX_CM).astype(np.int16)


class CellAggregate:
    """One frame's evidence, per touched cell. Input to fuse() (math §3.3).

    Deliberately not a Kalman state: scatter produces sufficient statistics and
    nothing else, so the filter stays in one place.
    """

    __slots__ = ("ceiling_cm", "cells", "class_id", "n", "refl_sum", "w_sum", "wz_sum")

    def __init__(self, cells, wz_sum, w_sum, n, ceiling_cm, refl_sum, class_id):
        self.cells = cells            # int64, sorted, unique
        self.wz_sum = wz_sum          # int64, sum of w_q * z_cm over GROUND returns
        self.w_sum = w_sum            # int64, sum of w_q over GROUND returns; 0 = none
        self.n = n                    # int32, returns in the cell this frame, all of them
        self.ceiling_cm = ceiling_cm  # int16, lowest non-ground return
        self.refl_sum = refl_sum      # int32, over all returns
        self.class_id = class_id      # uint8, class of the lowest-indexed return

    def mean_height_cm(self) -> np.ndarray:
        """Integer-weighted mean of the GROUND returns, rounded half-away-from-
        zero in integers so the result does not depend on float rounding mode.

        **Meaningless where `has_ground_evidence()` is False**, and it returns 0
        there rather than raising, because the frame path wants a branch-free
        column. 0 is not a neutral value -- the road sits near -173 cm in the
        vehicle frame -- so a caller that writes this into the map without
        consulting `has_ground_evidence()` puts every wall and car body 1.7 m
        above the road with a plausible-looking variance behind it.
        """
        w = np.maximum(self.w_sum, 1)
        # Rounded on the MAGNITUDE, with the sign put back afterwards. The
        # obvious spelling -- `(2*wz + sign(wz)*w) // (2*w)` -- is correct for
        # positive sums and one short for negative ones: `//` floors, so after
        # the away-from-zero nudge a negative quotient takes one extra step
        # down. Every negative mean came out 1 cm low, exact values included,
        # and only the exact half-centimetres agreed.
        #
        # That is not a rounding curiosity here. Vehicle frame is z up with the
        # sensor at 1.73 m, so the road sits near -173 cm and essentially every
        # ground cell in the map has a negative mean: the bias was a systematic
        # 1 cm sag over the whole ground plane, against a §3.2 noise floor of
        # 0.8 cm at 5 m. It reads as a real height, and per-ring RMSE is the
        # only place it would ever have surfaced.
        s = np.sign(self.wz_sum)
        return (s * ((2 * np.abs(self.wz_sum) + w) // (2 * w))).astype(np.int32)

    def has_ground_evidence(self) -> np.ndarray:
        """Which cells saw at least one ground return this frame.

        `quantise_weight()` clips every point's weight to >= 1, so a zero sum
        cannot mean "ground returns that happened to weigh nothing". It means
        the cell held nothing but canopy, wall or vehicle, and §3 has no height
        measurement to offer for it.
        """
        return self.w_sum > 0

    def __len__(self) -> int:
        return len(self.cells)

    def as_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


def _validate(idx, z_cm, w_q, refl, class_id, is_ground):
    """Length and range checks only. Resolving a default `point_id` is left to
    the caller: the sorted path takes it from its preallocated `iota` buffer,
    and building an `np.arange` here would allocate inside the frame loop."""
    n = len(idx)
    for name, arr in (("z_cm", z_cm), ("w_q", w_q), ("refl", refl),
                      ("class_id", class_id), ("is_ground", is_ground)):
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} entries, idx has {n}")
    if n and int(np.asarray(class_id).max()) >= CLASS_RADIX:
        raise ValueError(f"class ids must be < {CLASS_RADIX} to pack into the class key")
    return n


def scatter_sorted(idx, z_cm, w_q, refl, class_id, is_ground, point_id=None,
                   scratch=None) -> CellAggregate:
    """Default path. Sort by cell, reduce each segment. **Allocates nothing.**

    `idx` is the flat cell index from `allocators.annulus_index()`; entries < 0
    (the ring's hole, or a point outside the map) are dropped here rather than
    silently landing in cell 0.

    Pass the `scratch` from `allocate()` and this runs entirely in preallocated
    buffers: every step below writes through an `out=` parameter, the sort is
    in place, and the returned aggregate is a set of VIEWS into that scratch.
    That last part is the contract worth reading twice --

        **The returned aggregate is valid only until the next `scatter_sorted`
        call on the same scratch.** `fuse()` consumes it inside the frame; if
        you need it to outlive the frame, copy it.

    -- and it is the price of the no-allocation-in-the-loop invariant. Omitting
    `scratch` allocates a private one per call, which is fine for tests and
    wrong for the frame loop.

    Nothing is cleared between frames: every buffer is written before it is
    read, so the sorted path pays no per-frame clear at all, unlike the dense
    accumulator in `scatter_atomic`.
    """
    n = _validate(idx, z_cm, w_q, refl, class_id, is_ground)
    if scratch is None:
        scratch = new_sorted_scratch(max(n, 1))
    cap = len(scratch["key"])
    if n > cap:
        raise ValueError(
            f"{n:,} points exceeds the scratch capacity of {cap:,}. Raise "
            f"scatter.max_points_per_frame in configs/thresholds.yaml -- do not "
            f"grow the buffer here, that is an allocation in the frame loop")
    if n == 0:
        return _empty_aggregate()

    idx = np.asarray(idx)
    # The gathers below need each payload column to arrive in exactly the width
    # its scratch buffer declares -- `np.take` refuses a wider `out`. These are
    # free views when the caller already speaks the contract, which
    # `fusion.scatter()` does; a caller that does not pays one copy per column
    # per frame, which is why the boundary coercion lives there and not here.
    z_cm = np.asarray(z_cm, dtype=np.int16)
    w_q = np.asarray(w_q, dtype=np.int32)
    refl = np.asarray(refl, dtype=np.int32)
    is_ground = np.asarray(is_ground, dtype=np.bool_)
    class_id = np.asarray(class_id, dtype=np.uint8)
    pid = scratch["iota"][:n] if point_id is None else np.asarray(point_id)
    if int(pid.max()) >= POINT_RADIX or int(pid.min()) < 0:
        raise ValueError(f"point ids must lie in [0, {POINT_RADIX}) to pack into the sort key")
    if int(idx.max()) >= CELL_MAX:
        raise ValueError(f"cell index {int(idx.max()):,} exceeds {CELL_MAX:,}, the largest "
                         "the packed sort key can carry")

    # One packed key per point, dropped returns pushed past every real key so
    # they land in one contiguous tail we can simply ignore.
    key = scratch["key"][:n]
    drop = scratch["drop"][:n]
    np.multiply(idx, POINT_RADIX, out=key, casting="unsafe")
    np.add(key, pid, out=key, casting="unsafe")
    np.multiply(key, POINT_RADIX, out=key)
    np.add(key, scratch["iota"][:n], out=key, casting="unsafe")  # position: the gather index
    np.less(idx, 0, out=drop)
    np.copyto(key, KEY_DROPPED, where=drop)
    m = n - int(np.count_nonzero(drop))
    if m == 0:
        return _empty_aggregate()

    key.sort(kind="quicksort")  # in place; keys are unique, so order is total
    key = key[:m]

    cell, order = scratch["cell"][:m], scratch["order"][:m]
    np.floor_divide(key, POINT_RADIX * POINT_RADIX, out=cell, casting="unsafe")
    np.remainder(key, POINT_RADIX, out=order, casting="unsafe")

    # Segment starts: the first point of each cell, in one pass over the sorted
    # cell column. This replaces `np.unique`, which sorted the column a second
    # time to rediscover an order we already have.
    bnd = scratch["bnd"][:m]
    bnd[0] = True
    np.not_equal(cell[1:], cell[:-1], out=bnd[1:])
    k = int(np.count_nonzero(bnd))
    seg = scratch["seg"][:k]
    # `np.compress(bnd, iota, out=seg)` is the natural spelling and is a
    # numpy >= 2.5 spelling only: before that, take/compress wrap `out` in the
    # SOURCE dtype, so a wider `out` is rejected outright -- int32 `iota` into
    # an intp `seg` raises "cannot cast int64 to int32 under rule 'safe'". This
    # builds the same single index temporary compress builds internally (8 bytes
    # per touched cell, the residual the Day-2 gate accounts for) and writes it
    # straight into the scratch, on every numpy we support.
    np.copyto(seg, np.flatnonzero(bnd))

    # Cell id per segment, taken from `key` rather than from `cell`: both carry
    # the same number, but `key` is int64 like `cells` is, and take demands the
    # two match. `key` is dead after this -- `cell` and `order` are already out
    # of it -- so reusing it costs nothing.
    cells = scratch["cells"][:k]
    np.take(key, seg, out=cells, mode="clip")
    np.floor_divide(cells, POINT_RADIX * POINT_RADIX, out=cells)

    # Gather the payloads into sorted order. `np.take` does NOT widen into the
    # output dtype -- before numpy 2.5 it wraps `out` in the SOURCE dtype, so a
    # wider `out` is refused outright -- which is why every gather below lands
    # in a buffer of exactly the column's own width and the widening is done
    # afterwards, where it can be written down. `reduceat` DOES accumulate at
    # the output's width, so the int32 weight column sums into int64 with no
    # overflow and no intermediate copy; `np.multiply` does not, hence the
    # explicit `dtype` on the one product that needs it.
    # mode="clip" throughout: `order` and `seg` are constructed in range, and
    # the default "raise" pays for a bounds check by copying the index array.
    z = scratch["z_cm"][:m]
    w = scratch["w_q"][:m]
    wz = scratch["wz"][:m]
    r = scratch["refl"][:m]
    g = scratch["is_ground"][:m]
    np.take(z_cm, order, out=z, mode="clip")
    np.take(w_q, order, out=w, mode="clip")
    np.take(refl, order, out=r, mode="clip")
    np.take(is_ground, order, out=g, mode="clip")

    # §3 estimates the height of the GROUND, so a non-ground return is not a
    # weak measurement of it -- it is a measurement of something else. Zeroing
    # the weight drops it from both height sums and leaves every other column
    # (count, reflectivity, class, ceiling) over all returns, which is what
    # those want. Without this a cell holding a road return at 0 cm and a
    # canopy return at 200 cm reports a ground height of 100 cm, and the map
    # it produces looks entirely plausible.
    #
    # Multiplying by the bool column rather than masking with `~g` on purpose:
    # `~g` is a temporary, and this path's whole contract is that it allocates
    # nothing. int32 * bool writes back into the int32 buffer in place.
    np.multiply(w, g, out=w)

    # w peaks at 2^20 and z at 600 cm: the product needs 64 bits and the int32
    # inputs would silently wrap without `dtype`, which is the overflow that
    # would look like a plausible map. Buffered per chunk, so no temporary.
    np.multiply(w, z, out=wz, dtype=np.int64)

    wz_sum, w_sum = scratch["wz_sum"][:k], scratch["w_sum"][:k]
    refl_sum, ceiling = scratch["refl_sum"][:k], scratch["ceiling_cm"][:k]
    np.add.reduceat(wz, seg, out=wz_sum)
    np.add.reduceat(w, seg, out=w_sum)
    np.add.reduceat(r, seg, out=refl_sum)

    # `z` is finished as a height column once wz exists, so the ceiling source
    # is written over it rather than into a second buffer of its own.
    np.copyto(z, CEILING_NONE, where=g)
    np.minimum.reduceat(z, seg, out=ceiling)

    # Returns per cell are the gaps between segment starts -- no per-point
    # array of ones to reduce over.
    count = scratch["n"][:k]
    np.subtract(seg[1:], seg[:-1], out=count[:-1], casting="unsafe")
    count[-1] = m - int(seg[-1])

    # Class of the lowest-numbered point in each cell: the key sort already put
    # that point first in its segment, so this is a gather, not a reduction.
    class_src = scratch["class_src"][:k]
    out_class = scratch["class_id"][:k]
    np.take(order, seg, out=class_src, mode="clip")
    np.take(class_id, class_src, out=out_class, mode="clip")

    return CellAggregate(cells, wz_sum, w_sum, count, ceiling, refl_sum, out_class)


def scatter_atomic(idx, z_cm, w_q, refl, class_id, is_ground, n_cells,
                   scratch=None, point_id=None) -> CellAggregate:
    """The literal integer-atomics path: unbuffered adds into a dense per-cell
    accumulator, exactly as a CUDA kernel would issue atomicAdd.

    `np.add.at` is the CPU stand-in for an atomic add -- unbuffered, so
    repeated indices accumulate rather than overwrite. The order in which they
    land is irrelevant because the accumulators are integers.

    Costs a dense accumulator (16 B/cell) that must be cleared every frame.
    Kept because it is what master v4 §3.5 specifies and because it is the
    honest reference the sorted path is checked against.
    """
    n = _validate(idx, z_cm, w_q, refl, class_id, is_ground)
    if point_id is None:
        point_id = np.arange(n, dtype=np.int64)

    keep = np.asarray(idx) >= 0
    idx = np.asarray(idx)[keep].astype(np.int64)
    if idx.size == 0:
        return _empty_aggregate()
    z_cm = np.asarray(z_cm)[keep].astype(np.int64)
    w_q = np.asarray(w_q)[keep].astype(np.int64)
    refl = np.asarray(refl)[keep].astype(np.int64)
    class_id = np.asarray(class_id)[keep].astype(np.int64)
    is_ground = np.asarray(is_ground)[keep]
    point_id = np.asarray(point_id)[keep].astype(np.int64)

    acc = scratch if scratch is not None else new_dense_scratch(n_cells)
    clear_dense_scratch(acc)

    # Ground mask, as in the sorted path -- the two must stay bit-identical, so
    # this is the same rule spelled the way an allocating reference path may
    # spell it. See the note beside `np.multiply(w, g, out=w)` above.
    w_ground = w_q * is_ground

    np.add.at(acc["wz_sum"], idx, w_ground * z_cm)
    np.add.at(acc["w_sum"], idx, w_ground.astype(np.int32))
    np.add.at(acc["n"], idx, 1)
    np.add.at(acc["refl_sum"], idx, refl.astype(np.int32))
    np.minimum.at(acc["ceiling_cm"], idx,
                  np.where(is_ground, CEILING_NONE, z_cm).astype(np.int16))
    np.minimum.at(acc["class_key"], idx,
                  (point_id * CLASS_RADIX + class_id).astype(np.int32))

    cells = np.flatnonzero(acc["n"])
    return CellAggregate(
        cells.astype(np.int64), acc["wz_sum"][cells].astype(np.int64),
        acc["w_sum"][cells].astype(np.int64), acc["n"][cells].astype(np.int32),
        acc["ceiling_cm"][cells].astype(np.int16), acc["refl_sum"][cells].astype(np.int32),
        (acc["class_key"][cells] % CLASS_RADIX).astype(np.uint8),
    )


def _empty_aggregate() -> CellAggregate:
    return CellAggregate(
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
        np.zeros(0, np.int32), np.zeros(0, np.int16), np.zeros(0, np.int32),
        np.zeros(0, np.uint8),
    )


# Per-point columns. `iota` is a constant 0..max_points-1 written once at
# startup: it is the default `point_id` and the index source `np.compress`
# selects segment starts out of, and building either with `np.arange` per frame
# would be an allocation in the loop.
# `cell` and `iota` are int32 because a cell index fits in CELL_MAX and a
# position in POINT_RADIX, and at 150,000 points that narrowing is worth 1.2 MB
# each off the largest line in the budget. `order` is deliberately NOT narrowed:
# it indexes every gather below, and np.take copies a non-intp index array to
# widen it -- measured 1.6 MB a frame at 200,000 elements, which is the
# allocation this whole path exists to avoid. Trading 0.6 MB of declared
# scratch for 1.6 MB of per-frame garbage is a bad trade twice over.
# The payload columns carry the width of the INPUT column they gather, not the
# width the reduction wants: `np.take` will not widen into `out`. `w_q` is
# therefore int32 like `quantise_weight` returns, and `wz` is the int64 buffer
# that the product is widened into -- w*z peaks at 2^20 * 600, which fits int32
# with only 3x margin, and a silent overflow there would look like a plausible
# map.
SORTED_SCRATCH_POINT_FIELDS = (
    ("key", np.int64), ("cell", np.int32), ("order", np.intp), ("iota", np.int32),
    ("drop", np.bool_), ("bnd", np.bool_), ("z_cm", np.int16), ("w_q", np.int32),
    ("wz", np.int64), ("refl", np.int32), ("is_ground", np.bool_),
)

# Per-touched-cell columns -- the aggregate itself, returned as views. Sized for
# the worst case of one cell per point, which is what makes the bound a bound.
SORTED_SCRATCH_CELL_FIELDS = (
    ("seg", np.intp), ("cells", np.int64), ("wz_sum", np.int64), ("w_sum", np.int64),
    ("n", np.int32), ("ceiling_cm", np.int16), ("refl_sum", np.int32),
    ("class_id", np.uint8), ("class_src", np.intp),
)


def new_sorted_scratch(max_points: int, n_cells: int | None = None) -> dict:
    """Scratch for `scatter_sorted`. Sized by the sensor's point count, not by
    the grid, so it does not grow when the map does.

    A frame can touch at most one cell per point, so the aggregate columns are
    capped at `min(max_points, n_cells)` -- still independent of grid size for
    any grid larger than a scan, which every schedule we ship is.
    """
    if max_points > POINT_RADIX:
        raise ValueError(f"max_points {max_points:,} exceeds POINT_RADIX {POINT_RADIX:,}; "
                         "the packed sort key cannot address that many points")
    max_cells = max_points if n_cells is None else min(max_points, n_cells)
    s = {name: np.zeros(max_points, dtype=dt) for name, dt in SORTED_SCRATCH_POINT_FIELDS}
    s.update({name: np.zeros(max_cells, dtype=dt) for name, dt in SORTED_SCRATCH_CELL_FIELDS})
    np.copyto(s["iota"], np.arange(max_points, dtype=np.int64))
    return s


# Widths are chosen against real bounds, not defensively: wz_sum needs int64
# (w_q up to 2^20 times z_cm up to 600, summed over a cell), but a ceiling is
# an int16 height and a class key is at most max_points * CLASS_RADIX.
DENSE_SCRATCH_FIELDS = (
    ("wz_sum", np.int64), ("w_sum", np.int32), ("n", np.int32),
    ("ceiling_cm", np.int16), ("refl_sum", np.int32), ("class_key", np.int32),
)


def new_dense_scratch(n_cells: int) -> dict:
    """Dense accumulator for `scatter_atomic`. Allocate once at startup."""
    return {name: np.zeros(n_cells, dtype=dt) for name, dt in DENSE_SCRATCH_FIELDS}


def clear_dense_scratch(acc: dict) -> None:
    """Every frame, for the atomic path. This clear is a real per-frame cost
    over the whole grid, which the sorted path does not pay at all."""
    for name, arr in acc.items():
        if name == "ceiling_cm":
            arr[:] = CEILING_NONE
        elif name == "class_key":
            arr[:] = np.iinfo(arr.dtype).max
        else:
            arr[:] = 0


def scatter_scratch_bytes(mode: str, n_cells: int, max_points: int) -> int:
    """What each path costs, for the memory bound. Numbers, not adjectives."""
    if mode == "sorted":
        per_point = sum(np.dtype(dt).itemsize for _, dt in SORTED_SCRATCH_POINT_FIELDS)
        per_cell = sum(np.dtype(dt).itemsize for _, dt in SORTED_SCRATCH_CELL_FIELDS)
        return max_points * per_point + min(max_points, n_cells) * per_cell
    if mode == "atomic":
        return n_cells * sum(np.dtype(dt).itemsize for _, dt in DENSE_SCRATCH_FIELDS)
    raise ValueError(f"unknown scatter mode {mode!r}")


def map_hash(arrays) -> str:
    """Blake2b over the SoA arrays in a fixed field order.

    The Day-1 gate: run the same sequence twice, compare this string. Field
    order is sorted so two callers cannot disagree about it, and the field name
    goes into the digest so swapping two same-typed arrays does not collide.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def grid_bytes(n_cells: int) -> int:
    return n_cells * CELL_BYTES
