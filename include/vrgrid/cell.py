"""The cell — FROZEN Day 0. Whole-team change only.

Master v4 §3.3. Structure-of-arrays: one numpy array per field, never an
array-of-structs, so GPU access stays coalesced. `CELL_DTYPE` exists to pin
the layout and the 12-byte budget; allocate with `alloc_soa()`.

Changing this file means recomputing every memory figure in the report
(8.94 MB total, 21.5x vs uniform 5 cm 2.5D, 286x vs dense 5 cm 3D).
"""

import numpy as np

# --- the 12 bytes ------------------------------------------------------------
# Master v4 §3.3 lists these ten fields plus a "reserved (1 B, alignment)" row,
# which sums to 13, not the 12 the memory arithmetic depends on. The ten fields
# below are exactly 12 B and naturally aligned (int16s first, at offsets 0 and
# 2), so no padding byte is needed. Confirm at Day-0 sign-off before anyone
# builds on it.
CELL_FIELDS = [
    ("ground_height", np.int16),      # 2 B, 1 cm units, +-327 m
    ("ceiling_height", np.int16),     # 2 B, 1 cm units, lowest thing overhead
    ("height_variance", np.uint8),    # 1 B, log-quantised
    ("log_odds", np.int8),            # 1 B, occupancy
    ("semantic_class", np.uint8),     # 1 B, Boyer-Moore: 5-bit candidate | 3-bit counter
    ("reflectivity", np.uint8),       # 1 B, range-normalised
    ("obs_count", np.uint8),          # 1 B, saturating
    ("frames_since_seen", np.uint8),  # 1 B, saturating
    ("traversability", np.uint8),     # 1 B, bitfield -- see TRAV_* below
    ("flags", np.uint8),              # 1 B, bitfield -- see FLAG_* below
]

CELL_DTYPE = np.dtype(CELL_FIELDS)
CELL_BYTES = 12

# --- traversability bitfield (math §7.1) -- 0 means traversable --------------
TRAV_CLEARANCE = 1 << 0   # ceiling - ground  <  h_vehicle
TRAV_SLOPE = 1 << 1       # ||grad z||        >  tan(theta_max)
TRAV_STEP = 1 << 2        # max|z_c - z_nbr|  >  s_max
TRAV_ROUGHNESS = 1 << 3   # sigma^2           >  sigma^2_max
TRAV_CLASS = 1 << 4       # class not in drivable_set
TRAV_CONFIDENCE = 1 << 5  # n < n_min  (fail safe: unobserved is not traversable)

# --- flags bitfield ----------------------------------------------------------
FLAG_DERIVED = 1 << 0  # value came from split(), not measurement. Math §5:
                       # this bit is what makes merge(split(c)) == c exact.
FLAG_REFINED = 1 << 1  # semantics forced a finer resolution than range alone
FLAG_BLIND = 1 << 2    # inside the 3.74 m blind cone -- unknown, never free
FLAG_DYNAMIC = 1 << 3  # supplied by the transient layer this frame

# --- occupancy: three states, unknown is not "log-odds near zero" ------------
OCC_UNKNOWN = 0  # decided by obs_count, per master v4 §3.6
OCC_FREE = 1
OCC_OCCUPIED = 2


def alloc_soa(n_cells: int) -> dict:
    """Preallocate one array per field. Called once at startup, never in the
    frame loop -- the compile-time memory bound is a headline claim."""
    return {name: np.zeros(n_cells, dtype=dt) for name, dt in CELL_FIELDS}


def soa_bytes(n_cells: int) -> int:
    """Total bytes for `n_cells`. Used by the memory-bound test and by
    scripts/memory_table.py, so the report and the code cannot disagree."""
    return n_cells * CELL_BYTES
