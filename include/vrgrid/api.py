"""The five frozen signatures + CellQuery — FROZEN Day 0.

Master v4 §3.7. Plain Python core, no ROS dependency: the ROS 2 adapter is
optional and lives under `adapters/`. Every implementation lives elsewhere;
this file is signatures only so that three people can build against real
interfaces from hour 3 instead of imagined ones.

Changing anything here requires all three devs to agree, in the same room.

⚑ The functions below raise, and that is the design -- this file declares the
  contract, it does not serve it. But each message carried a Day-1/Day-2 OWNER
  where it should have carried a DESTINATION, so as the work landed they went
  stale in the one direction that matters: every one of them now reads as
  outstanding work on a contract that has been met for days. `grid/fusion.py`
  has the same lesson written at the bottom of it, about a §10.4 stub that a
  second reader found before the real implementation and concluded from it
  that the ghost gate was unbuilt.

  Each message now names where the contract is met. `export_gridmap` is the
  one that is genuinely still open, and it is the only one that still reads
  that way.

  The frozen surface -- names, signatures, `CellQuery`, `QueryLOD`, `AABB` --
  is untouched. Nothing a caller may rely on changes.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from vrgrid.cell import OCC_UNKNOWN


class QueryLOD(IntEnum):
    """Result of a conservative block query (math §7.2)."""

    SAFE = 0     # every cell in the block is traversable -- provable, see Theorem 3
    BLOCKED = 1  # every cell fails the same condition
    MIXED = 2    # descend one level


@dataclass
class AABB:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class CellQuery:
    """Resolution-agnostic answer. The caller never learns which ring served it."""

    ground_height: float      # metres
    ceiling_height: float     # metres
    semantic_class: int
    traversability: int       # bitfield, 0 = traversable (cell.TRAV_*)
    confidence: int
    occupancy: int = OCC_UNKNOWN
    dynamic: bool = False


# --- the five -----------------------------------------------------------------


def scatter(points: np.ndarray, labels: np.ndarray, pose: np.ndarray) -> None:
    """Scatter one scan into the variable-resolution grid.

    Fixed-point int32 accumulation in 1 cm units -- never float atomics, which
    are non-associative and destroy run-to-run determinism. See math §3.4.
    """
    raise NotImplementedError("implemented: vrgrid.grid.fusion.scatter")


def fuse() -> None:
    """Fold this frame's accumulators into the persistent map.

    Kalman update with a range-dependent measurement model. See math §3.
    """
    raise NotImplementedError("implemented: vrgrid.grid.fusion.fuse")


def split(cell_index: int) -> None:
    """One cell -> four children on the next finer ring.

    Children inherit mu_p with a strictly larger variance and the `derived`
    bit set. The bit is what makes merge(split(c)) == c exact. See math §5.
    """
    raise NotImplementedError("implemented: vrgrid.grid.splitmerge.split")


def merge(child_indices) -> None:
    """Four children -> one parent, by the law of total variance.

    sigma^2_p = sum(w_i sigma_i^2) + sum(w_i (mu_i - mu_p)^2). Children measure
    different *places*, not one quantity, so inverse-variance fusion is wrong
    here -- it makes merged cells most confident exactly where they straddle a
    kerb. See math §4.
    """
    raise NotImplementedError("implemented: vrgrid.grid.splitmerge.merge")


def query(x: float, y: float) -> CellQuery:
    """Point query in vehicle frame (x forward, y left, z up).

    Returns the union of the persistent and transient layers: OCCUPIED if
    either says so, with dynamic=True when the transient layer supplied it.
    One merge rule, defined here, so no consumer inherits the problem.
    """
    raise NotImplementedError("implemented: vrgrid.grid.query.query")


# --- the rest of the output interface ----------------------------------------


def is_traversable(x: float, y: float) -> bool:
    raise NotImplementedError("implemented: vrgrid.grid.query.is_traversable")


def query_conservative(region: AABB) -> QueryLOD:
    """Conservative pyramid query. Never reports SAFE for a block containing an
    untraversable cell -- proved in math §7.2, Theorem 3."""
    raise NotImplementedError(
        "implemented: vrgrid.gpu.pyramid.classify, over a pyramid built by "
        "pyramid.build(). §7.3's predicate on its own is pyramid.theorem3_safe, "
        "which covers bits 0, 2 and 5; classify() uses OR_mask for all six")


def dynamic_objects() -> list:
    """Tracked objects with velocity. Persists ~1 s with constant-velocity
    prediction, so a pedestrian briefly hidden by a parked car does not vanish."""
    raise NotImplementedError("implemented: vrgrid.grid.transient.TrackList")


def export_gridmap():
    """Lossy near-field export for interop. grid_map::GridMap is uniform
    resolution and cannot hold this map -- convenience only, never the
    primary output. Label it as lossy everywhere it appears."""
    raise NotImplementedError("adapters/ -- optional")
