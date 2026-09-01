"""The semantic gate — what forces refinement below what range gives. [Aakash]

Master v4 §3.4. Range decides the ring; semantics can override it downward,
but only into the preallocated pool, and only where it would change something.

This is the piece that makes the refinement pool a mechanism rather than a data
structure. Without it `pool.acquire()` is called from nowhere, the 98 KB sits
untouched, and "semantics can force local refinement" is a sentence in a
document rather than a thing the map does.

--- what fires the gate ---------------------------------------------------

Three reasons, in priority order, all configurable in `configs/thresholds.yaml`:

1. **Something is standing here.** A cell the transient layer wrote this frame.
   A pedestrian at 30 m sits in a 20 cm cell that also contains the road she is
   standing on, and the planner needs the two separated.
2. **The class is one that matters at a finer scale than its range gives.**
   A pole, a traffic sign, a person: thin structures whose whole geometry is
   smaller than the cell they land in past ~25 m.

   ⚑ Two of those three cannot currently fire it. The cell's class nibble is
     4 bits, `pole` is learning id 18 and `traffic-sign` is 19, so no cell can
     ever report them -- the criterion is dead code that reads as working.
     `unstorable_refine_classes()` names them rather than letting a `% 16`
     turn `pole` into `other-vehicle`. This is the §10.2 width conflict doing
     its most expensive damage yet, and it is a whole-team call on a frozen
     struct.
3. **The cell is ambiguous where it matters.** High variance on a GEOMETRIC
   hazard -- a step or a slope -- meaning the map knows there is an edge here
   and is unsure exactly where it runs. That boundary is what the planner will
   steer along, so half a cell of doubt about it is worth a block.

   ⚑ Geometric, not `TRAV_ROUGHNESS`, and the distinction is not cosmetic:
     bit 3 IS the predicate `sigma^2 > sigma^2_max`, so "roughness bit AND high
     variance" is the same condition written twice. Written that way the gate
     fired on 20,954 of 143,587 observed cells -- 8x the whole pool per frame,
     with the priority ordering left doing all the selection the gate was
     supposed to do. A conjunction of a bit with its own definition looks like
     a careful two-part test and is one part.

⚑ What deliberately does NOT fire it: being interesting. A refinement that
  cannot change a decision is wasted pool, and the pool is 512 blocks. §8.3
  makes the sharp version of this argument -- cells outside the near-optimal
  corridor cannot change the plan however finely resolved -- and
  `corridor_mask` below is where that plugs in when the planner runs online.

--- refinement is a split, not an allocation ------------------------------

A block's children are produced by `splitmerge.split()`, which means they
inherit mu_p, carry FLAG_DERIVED and an inflated variance, and merge back
exactly (Theorem 2). Filling a block with zeros instead would give the planner
four confident readings of 0 m where the parent knew nothing, and the map
would look sharper precisely where it had just admitted it was guessing.
"""

import numpy as np
from vrgrid.cell import FLAG_DYNAMIC, TRAV_SLOPE, TRAV_STEP
from vrgrid.grid.fusion import unpack_class
from vrgrid.grid.lattice import migrate_ring
from vrgrid.grid.pool import FREE, priority
from vrgrid.grid.quantise import dequantise_variance_cm2, quantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.splitmerge import CellValue, split

# Thin or safety-critical structures whose geometry is finer than the ring they
# land in past the near field. SemanticKITTI learning ids, provisional with the
# rest of that table (see traversability.class_ids()).
DEFAULT_REFINE_CLASSES = ("person", "bicyclist", "motorcyclist", "pole",
                          "traffic-sign", "bicycle", "motorcycle")


def _config(thresholds):
    th = thresholds if thresholds is not None else load_thresholds()
    return th.get("refinement_pool", {}), th


def refine_class_ids(thresholds=None) -> np.ndarray:
    """The classes that fire the gate, by id -- those that CAN fire it.

    See `unstorable_refine_classes()`. Classes whose id will not fit the
    cell's class field are dropped here, because a cell can never report them
    and matching against them is dead code that reads as working.

    Since the byte was re-split 5/3 (1 Sep) the field holds ids to 31 and the
    whole label set fits, so nothing is dropped and this returns every
    configured name. The filter stays because it is what makes that fact
    checkable rather than assumed.
    """
    from vrgrid.grid.fusion import CLASS_MAX
    from vrgrid.grid.traversability import class_ids

    _, _th = _config(thresholds)
    names = _names(thresholds)
    ids = class_ids()
    return np.array(sorted(ids[n] for n in names
                           if ids[n] <= CLASS_MAX), dtype=np.int32)


def _names(thresholds=None):
    from vrgrid.grid.traversability import class_ids

    cfg, _ = _config(thresholds)
    names = cfg.get("refine_classes", DEFAULT_REFINE_CLASSES)
    unknown = [n for n in names if n not in class_ids()]
    if unknown:
        raise ValueError(f"refine_classes names no class in the label map: {unknown}")
    return names


def unstorable_refine_classes(thresholds=None) -> list:
    """Classes the gate is configured to refine on and the cell cannot hold.

    **Empty since 1 Sep**, and it emptied itself: the §10.2 byte was re-split
    5/3, the class field went from 16 ids to 32, and every configured name now
    fits. Nothing here changed.

    It is kept, and still called, because of what it used to hold. With a
    4-bit nibble `pole` (18) and `traffic-sign` (19) did not fit -- and those
    are **the two classes semantic refinement exists for**, thin structures
    whose entire geometry is smaller than the cell they land in past 25 m. The
    gate matched against ids no cell could ever report, fired on nothing, and
    nothing failed. A dead criterion that reads as working is the failure mode
    this function exists to make visible, and it costs one list comprehension
    per config load to keep it visible for the next class someone adds.
    """
    from vrgrid.grid.fusion import CLASS_MAX
    from vrgrid.grid.traversability import class_ids

    ids = class_ids()
    return [n for n in _names(thresholds) if ids[n] > CLASS_MAX]


def ring_of_slot(gm, slot: int) -> int:
    """Which ring a flat slot belongs to. The allocation lays the rings out
    end to end, so this is a search over at most four offsets."""
    for level in range(len(gm.allocation.rings) - 1, -1, -1):
        if slot >= gm.allocation.rings[level].offset:
            return level
    return 0


def candidates(gm, slots, thresholds=None):
    """Which of `slots` the gate fires on, and why. Returns a boolean mask.

    Takes the slots touched this frame rather than the whole map: the gate is
    a per-frame decision about new evidence, and walking 910,000 cells to find
    the few hundred that changed is the kind of O(area) work the toroidal shift
    exists to avoid.
    """
    _, th = _config(thresholds)
    slots = np.asarray(slots, dtype=np.int64)
    if slots.size == 0:
        return np.zeros(0, dtype=bool)

    cls = unpack_class(gm.soa["semantic_class"][slots])[0].astype(np.int32)
    fires = np.isin(cls, refine_class_ids(th))

    if gm.transient is not None:
        inside = slots < gm.transient["flags"].size
        dynamic = np.zeros(slots.shape, dtype=bool)
        dynamic[inside] = (gm.transient["flags"][slots[inside]] & FLAG_DYNAMIC) != 0
        fires |= dynamic

    # Ambiguous where it matters: uncertain AND on a GEOMETRIC hazard. Either
    # alone is not enough -- uncertain open road is just unobserved road, and a
    # confidently-known kerb does not need a finer look. TRAV_ROUGHNESS is
    # excluded because it is the same predicate as the variance test; see the
    # note in the module docstring.
    sigma2_m2 = dequantise_variance_cm2(gm.soa["height_variance"][slots]) * 1e-4
    edge = (gm.soa["traversability"][slots] & (TRAV_STEP | TRAV_SLOPE)) != 0
    fires |= edge & (sigma2_m2 > th["traversability"]["sigma2_max_m2"])

    return fires


def apply(gm, slots, vehicle_speed_ms: float = 0.0, thresholds=None,
          corridor_mask=None, grad_z=None) -> dict:
    """Run the gate over this frame's touched cells. Master v4 §3.4.

    Order matters and it is the E1 fix: **release first, then acquire.** A
    block whose cell has migrated inward is buying resolution the schedule now
    provides free, and holding it while new requests are refused is exactly
    the degradation flaw E1 describes -- the pool going useless precisely as
    you approach the things you cared about.

    ⚑ **That fix was inert until 2 Sep, and the symptom was in every run.**
      `release_overtaken` was handed `ring_of_slot`, which answers which ring
      a flat SLOT is stored in -- a property of the allocation layout, fixed
      at startup, and not a function of where the vehicle is. So `now` was
      always exactly `ring`, the release test `now <= ring - levels` could
      never hold with `levels >= 1`, and nothing was ever released. Twelve
      frames of the synthetic sequence ended `released 0 ... pool 512/512
      blocks` with 15,791 requests refused: the pool filled once, early, and
      then refused every later request -- E1 exactly, with the fix for it
      sitting in the file.

      What it needs is where the cell IS, which is `_cell_centre` (already
      vehicle-relative) put through `migrate_ring`. That also puts §6.3's
      hysteresis on the frame path for the first time: `migrate_ring` had no
      caller outside its own unit test, so a cell on a ring boundary while
      the speed fluctuated would have thrashed the pool -- the exact failure
      §6.3 calls mandatory to prevent.

      Measured on the same 14 frames, before -> after:

          fired    116,684 -> 395        refused  15,791 -> 0
          acquired   2,954 -> 395        released      0 -> 324
          pool     512/512 -> 62/512

      `fired` collapses because a refused cell never gets FLAG_REFINED and so
      re-fires every frame: 116,684 was the same few hundred cells asking
      again and again against a full pool. Memory is unchanged -- the pool is
      512 x 16 x 12 B whether or not it is full -- so no headline number
      moves. What changes is that the refinement pool now does its job.

    `corridor_mask(ring, slot) -> bool` is §8.3's band, when a planner is
    running: cells outside the near-optimal corridor cannot change the plan
    however finely resolved, so refining them is provably wasted. Optional,
    because the offline metric does not need it and the online policy is a
    Day-5 item.

    Returns counts, so the dashboard and the report can both say what the pool
    is doing rather than inferring it.
    """
    _, th = _config(thresholds)
    pool = gm.pool
    if pool is None:
        return {"released": 0, "fired": 0, "acquired": 0, "refused": 0, "unfit": 0}

    released = pool.release_overtaken(
        lambda ring, slot: migrate_ring(*_cell_centre(gm, ring, slot),
                                        gm.schedule, ring, vehicle_speed_ms))

    slots = np.asarray(slots, dtype=np.int64)
    fires = candidates(gm, slots, th)
    fired = slots[fires]

    acquired = refused = unfit = 0
    for slot in fired:
        slot = int(slot)
        ring = ring_of_slot(gm, slot)
        if ring < 1:
            continue                       # already at the base lattice
        if corridor_mask is not None and not corridor_mask(ring, slot):
            continue
        if pool.levels_available(gm.schedule, ring) < 1:
            unfit += 1                     # the ablation's 5x boundary
            continue

        x_m, y_m = _cell_centre(gm, ring, slot)
        score = priority(float(np.hypot(x_m, y_m)),
                         is_dynamic=_is_dynamic(gm, slot),
                         ttc_s=_time_to_collision(x_m, vehicle_speed_ms))

        block = pool.acquire(gm.schedule, ring, slot, levels=1, score=score)
        if block == FREE:
            refused += 1                   # designed degradation, not an error
            continue
        _fill(gm, pool, block, ring, slot, grad_z)
        gm.soa["flags"][slot] |= 2         # FLAG_REFINED
        acquired += 1

    return {"released": released, "fired": int(fires.sum()),
            "acquired": acquired, "refused": refused, "unfit": unfit}


def _is_dynamic(gm, slot: int) -> bool:
    if gm.transient is None or slot >= gm.transient["flags"].size:
        return False
    return bool(gm.transient["flags"][slot] & FLAG_DYNAMIC)


def _time_to_collision(x_m: float, speed_ms: float) -> float:
    """Crude and forward-only: how long until the vehicle reaches this range.

    Lateral offset is ignored on purpose -- a cell 2 m to the side at 30 m is
    still on the corridor the vehicle is about to occupy, and pretending
    otherwise would refuse to refine the kerb it is about to drive along.
    """
    if speed_ms <= 1e-6 or x_m <= 0.0:
        return np.inf
    return x_m / speed_ms


def _cell_centre(gm, ring: int, slot: int):
    """Vehicle-frame centre of one cell, in metres. O(1).

    ⚑ Deliberately NOT `window_cells()`, which is the readable way to do this
      and builds the ring's entire side x side coordinate grid -- 250,000
      cells for ring 3 -- to read one of them. Called once per gated cell
      inside the frame loop, that turned an 8-frame run into minutes: O(area)
      work per cell, in the one place the whole toroidal design exists to
      keep O(perimeter). The inverse of `RingBuffer.slot()` is two modulos.
    """
    buf = gm.buffers[ring]
    local = int(slot) - buf.offset
    side = buf.side
    sx, sy = local % side, local // side
    ix = buf.x0 + (sx - buf.x0) % side
    iy = buf.y0 + (sy - buf.y0) % side
    cell_m = gm.schedule.rings[ring].cell_m
    return ((ix + 0.5) * cell_m - gm.vehicle_xy_m[0],
            (iy + 0.5) * cell_m - gm.vehicle_xy_m[1])


def _fill(gm, pool, block, ring: int, slot: int, grad_z=None) -> None:
    """Populate a freshly acquired block by SPLITTING the parent. Math §5.

    Not zeros: a zeroed block reads as four confident measurements of 0 m
    where the parent knew nothing, so the map would look sharpest exactly
    where it had just admitted to guessing. `split()` gives the children the
    parent's mean, an inflated variance and FLAG_DERIVED -- which is also what
    makes the block mergeable back to the parent exactly if the gate stops
    firing (Theorem 2).
    """
    parent = CellValue(
        mu_m=float(gm.soa["ground_height"][slot]) / 100.0,
        sigma2_m2=float(dequantise_variance_cm2(gm.soa["height_variance"][slot])) * 1e-4,
        n=int(gm.soa["obs_count"][slot]),
        flags=int(gm.soa["flags"][slot]),
    )
    slope = 0.0 if grad_z is None else float(grad_z)
    children = split(parent, gm.schedule, ring, slope)

    cells = pool.block_cells(block)
    lo = cells.start
    for i, child in enumerate(children[: pool.cells_per_block]):
        gm.pool.cells["ground_height"][lo + i] = round(child.mu_m * 100.0)
        gm.pool.cells["height_variance"][lo + i] = quantise_variance_cm2(
            child.sigma2_m2 * 1e4)
        gm.pool.cells["obs_count"][lo + i] = min(child.n, 255)
        gm.pool.cells["semantic_class"][lo + i] = gm.soa["semantic_class"][slot]
        gm.pool.cells["ceiling_height"][lo + i] = gm.soa["ceiling_height"][slot]
        gm.pool.cells["log_odds"][lo + i] = gm.soa["log_odds"][slot]
        gm.pool.cells["flags"][lo + i] = child.flags
