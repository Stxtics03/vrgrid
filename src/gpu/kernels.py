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
                    Scratch is sized by POINTS (~1.8 MB), independent of grid
                    size, and nothing needs clearing between frames.
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


def quantise_height(z_m) -> np.ndarray:
    """Metres -> int16 centimetres, clamped to the vertical extent."""
    return np.clip(np.rint(np.asarray(z_m, dtype=np.float64) * 100.0),
                   Z_MIN_CM, Z_MAX_CM).astype(np.int16)


class CellAggregate:
    """One frame's evidence, per touched cell. Input to fuse() (math §3.3).

    Deliberately not a Kalman state: scatter produces sufficient statistics and
    nothing else, so the filter stays in one place.
    """

    __slots__ = ("ceiling_cm", "cells", "class_id", "n", "refl_sum", "w_sum", "wz_sum")

    def __init__(self, cells, wz_sum, w_sum, n, ceiling_cm, refl_sum, class_id):
        self.cells = cells            # int64, sorted, unique
        self.wz_sum = wz_sum          # int64, sum of w_q * z_cm
        self.w_sum = w_sum            # int64, sum of w_q
        self.n = n                    # int32, returns in the cell this frame
        self.ceiling_cm = ceiling_cm  # int16, lowest non-ground return
        self.refl_sum = refl_sum      # int32
        self.class_id = class_id      # uint8, class of the lowest-indexed return

    def mean_height_cm(self) -> np.ndarray:
        """Integer-weighted mean, rounded half-away-from-zero in integers so the
        result does not depend on float rounding mode."""
        w = np.maximum(self.w_sum, 1)
        return ((2 * self.wz_sum + np.sign(self.wz_sum) * w) // (2 * w)).astype(np.int32)

    def __len__(self) -> int:
        return len(self.cells)

    def as_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


def _validate(idx, z_cm, w_q, refl, class_id, is_ground, point_id):
    n = len(idx)
    for name, arr in (("z_cm", z_cm), ("w_q", w_q), ("refl", refl),
                      ("class_id", class_id), ("is_ground", is_ground)):
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} entries, idx has {n}")
    if point_id is None:
        point_id = np.arange(n, dtype=np.int64)
    if np.any(class_id >= CLASS_RADIX):
        raise ValueError(f"class ids must be < {CLASS_RADIX} to pack into the class key")
    return point_id


def scatter_sorted(idx, z_cm, w_q, refl, class_id, is_ground, point_id=None) -> CellAggregate:
    """Default path. Sort by cell, reduce each segment.

    `idx` is the flat cell index from `allocators.annulus_index()`; entries < 0
    (the ring's hole, or a point outside the map) are dropped here rather than
    silently landing in cell 0.

    Scratch is sized by point count, not cell count, and nothing is cleared
    between frames -- both of which matter more than they look at 10 Hz.
    """
    point_id = _validate(idx, z_cm, w_q, refl, class_id, is_ground, point_id)

    keep = np.asarray(idx) >= 0
    idx = np.asarray(idx)[keep]
    if idx.size == 0:
        return _empty_aggregate()
    z_cm = np.asarray(z_cm)[keep].astype(np.int64)
    w_q = np.asarray(w_q)[keep].astype(np.int64)
    refl = np.asarray(refl)[keep].astype(np.int64)
    class_id = np.asarray(class_id)[keep].astype(np.int64)
    is_ground = np.asarray(is_ground)[keep]
    point_id = np.asarray(point_id)[keep].astype(np.int64)

    # Sorting by (cell, point_id) makes the class representative well defined
    # without a second reduction. Stable sort so the result cannot depend on
    # the sort implementation.
    order = np.lexsort((point_id, idx))
    idx = idx[order]
    cells, start = np.unique(idx, return_index=True)

    wz = np.add.reduceat((w_q[order] * z_cm[order]), start)
    ws = np.add.reduceat(w_q[order], start)
    n = np.add.reduceat(np.ones(idx.size, dtype=np.int32), start)
    refl_sum = np.add.reduceat(refl[order], start)

    ceil_src = np.where(is_ground[order], CEILING_NONE, z_cm[order])
    ceiling = np.minimum.reduceat(ceil_src, start)

    class_first = class_id[order][start]  # lowest point_id in each cell

    return CellAggregate(cells.astype(np.int64), wz.astype(np.int64), ws.astype(np.int64),
                         n.astype(np.int32), ceiling.astype(np.int16),
                         refl_sum.astype(np.int32), class_first.astype(np.uint8))


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
    point_id = _validate(idx, z_cm, w_q, refl, class_id, is_ground, point_id)

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

    np.add.at(acc["wz_sum"], idx, w_q * z_cm)
    np.add.at(acc["w_sum"], idx, w_q.astype(np.int32))
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


SORTED_SCRATCH_FIELDS = (
    ("order", np.int32), ("idx", np.int32), ("z_cm", np.int16), ("w_q", np.int32),
    ("refl", np.uint8), ("class_id", np.uint8), ("is_ground", np.bool_),
    ("point_id", np.int32), ("wz", np.int64),
)


def new_sorted_scratch(max_points: int) -> dict:
    """Per-point scratch for `scatter_sorted`. Sized by the sensor's point
    count, not by the grid, so it does not grow when the map does."""
    return {name: np.zeros(max_points, dtype=dt) for name, dt in SORTED_SCRATCH_FIELDS}


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
        return max_points * sum(np.dtype(dt).itemsize for _, dt in SORTED_SCRATCH_FIELDS)
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
