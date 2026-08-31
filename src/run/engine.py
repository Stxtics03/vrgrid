"""The map back end, as one frame loop. [Shrestha]

`__main__.py` runs perception and hands each frame here. This file owns the
ORDER of the map stages and nothing else -- every computation below lives in
its owner's module, and if you find yourself writing arithmetic here, it
belongs somewhere else:

    bin      ring_of / i_ring  (grid, Aakash) + flat_slot_into  (gpu, mine)
    scatter  gpu.kernels.scatter_sorted
    fuse     grid.fusion.fuse
    cleanup  gpu.visibility.visibility_cleanup + apply_miss
    shift    gpu.shift.shift, tracking the vehicle

**Why this exists: the ghost toggle has to drive the map, not the point cloud.**
`dashboard/pipeline_view.py` splits the moving returns into a `world/ghosts`
entity and toggling it hides them. That is a filter on the input, and it
demonstrates nothing about the engine -- the trails a viewer sees in a 2.5D map
are *cells* that were fused from a moving car and never cleared. Removing them
is §10.4's job, and until this file existed §10.4 was never called outside its
own tests. `ghost_removal=False` here leaves the trails in the map, which is
what the "off" half of the Gate 3 demo is supposed to show.

Everything is preallocated in `__init__`. The frame loop allocates nothing.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.gpu.allocators import allocate
from vrgrid.gpu.kernels import (
    CEILING_NONE,
    measurement_variance_cm2,
    quantise_height,
    quantise_weight,
    scatter_sorted,
)
from vrgrid.gpu.shift import RingBuffer, flat_slot_into, new_slot_scratch, shift
from vrgrid.gpu.visibility import Sensor, apply_miss, visibility_cleanup

# fusion packs the semantic class into 5 bits since 1 Sep (math §10.2, Gate 3
# item 3), so ids above 31 raise rather than wrap -- a silent % 32 would relabel
# class 32 as 0, which is `unlabeled`. The SemanticKITTI learning set stops at
# 19, so every real frame now fits and nothing on this path clips. Imported
# rather than restated: this file held its own `CLASS_MAX = 15` and would have
# gone on clipping perfectly storable ids after the split landed.
from vrgrid.grid.fusion import CLASS_MAX, fuse, occupancy_state
from vrgrid.grid.lattice import i_ring, ring_of
from vrgrid.grid.schedule import load_thresholds


@dataclass
class StepCounters:
    """What the frame did, so "the cleanup works" is checkable rather than
    asserted. These are what the dashboard's ghost counter should read."""

    index: int
    points: int
    binned: int              # points that landed in a live slot
    cells_touched: int
    occupied: int            # occupied cells offered to the cleanup
    tested: int
    cleared: int
    protected: int           # would have cleared; had a return this scan
    out_of_view: int

    @property
    def protected_fraction(self) -> float:
        would = self.protected + self.cleared
        return self.protected / would if would else 0.0


def class_ids_fit(semantic) -> bool:
    """Whether `fuse()` will accept these ids, without asking it to find out."""
    s = np.asarray(semantic)
    return bool(s.size == 0 or s.max() <= CLASS_MAX)


class MapEngine:
    """Allocate once, then fold frames in. See the module docstring."""

    def __init__(self, schedule, thresholds=None, max_points: int = 150_000,
                 max_candidates: int | None = None, ghost_removal: bool = True,
                 sensor: Sensor | None = None, clip_class_ids: bool = False,
                 timer=None):
        self.sched = schedule
        self.thresholds = thresholds if thresholds is not None else load_thresholds()
        if max_candidates is not None:
            # An explicit cap overrides the config, but it has to reach
            # `allocate()` too or the declared bound would describe a different
            # scratch than the one the loop uses.
            self.thresholds = dict(self.thresholds)
            self.thresholds["visibility"] = dict(self.thresholds.get("visibility", {}))
            self.thresholds["visibility"]["max_candidate_cells"] = max_candidates
        self.ghost_removal = ghost_removal
        self.clip_class_ids = clip_class_ids
        # Optional gpu.timing.Timer. Stage names are timing.STAGES', so one
        # Timer shared with iter_pipeline covers the whole frame rather than
        # the back end alone -- see scripts/timing_table.py --seq.
        self.timer = timer
        self.sensor = sensor or Sensor.from_config(self.thresholds)

        # `with_visibility=True`: the cleanup's scratch is part of THIS loop's
        # footprint, so it belongs in this allocation's budget rather than
        # being conjured per call. `allocate()` leaves it off by default
        # because switching it on moves the headline total, and that is the
        # room's call -- but a frame loop that actually runs §10.4 is not the
        # place to leave 9.60 MB undeclared.
        self.handle = allocate(schedule, thresholds=self.thresholds,
                               with_visibility=True)

        self.max_points = max_points
        self.max_candidates = self.thresholds["visibility"]["max_candidate_cells"]

        # One toroidal window per ring, centred on the vehicle's start. `x0`
        # defaulting to the lattice origin would put half of every ring out of
        # view, since a sweep is centred on the sensor and not on the origin.
        self.buffers = [RingBuffer(side=r.side, offset=r.offset,
                                   x0=-(r.side // 2), y0=-(r.side // 2))
                        for r in self.handle.rings]
        self._k = [round(r.cell_m / schedule.base_cell_m) for r in self.handle.rings]
        self._origin = None          # world xy of the vehicle at frame 0

        self.slot_scratch = new_slot_scratch(max_points)
        self.slot_scratch["out"] = np.zeros(max_points, np.int64)
        self.idx = np.zeros(max_points, np.int64)

        # From the allocation, not a fresh one: the range image is JP's and it
        # is float32, and `allocate()` sizes the gather buffer to match --
        # np.take does not widen into `out`, so the dtypes have to agree.
        self.vis_scratch = self.handle.visibility
        self.range2d = np.zeros((0, 0), np.float32)
        cap = self.max_candidates
        self._cand = {n: np.zeros(cap, np.float64) for n in "xyz"}
        self._cand_slots = np.zeros(cap, np.int64)
        self._has_return = np.zeros(cap, np.bool_)

    # -- binning ------------------------------------------------------------

    def bin(self, xs, ys, xw, yw):
        """Points -> flat slots.

        **Two frames, and mixing them is the whole difficulty.** Ring
        MEMBERSHIP is a question about distance from the sensor, so `ring_of`
        takes the sensor/vehicle-frame point (§6.1). The lattice INDEX is
        global -- the map does not move when the vehicle does -- so `i_ring`
        takes the world-frame one (§2.1). Feed world coordinates to `ring_of`
        and every point reads as OUTSIDE once the vehicle has driven past the
        last ring's half-width; feed vehicle coordinates to `i_ring` and the
        map slides along under the vehicle. Both look right for a few seconds.
        """
        n = len(xs)
        level = ring_of(xs, ys, self.sched)
        idx = self.idx[:n]
        idx[:] = -1
        for layout, buf, k in zip(self.handle.rings, self.buffers, self._k):
            sel = level == layout.ring
            m = int(np.count_nonzero(sel))
            if not m:
                continue
            idx[sel] = flat_slot_into(
                buf,
                i_ring(xw[sel], self.sched.base_cell_m, k),
                i_ring(yw[sel], self.sched.base_cell_m, k),
                self.slot_scratch["out"][:m], self.slot_scratch,
            )
        return idx

    # -- the inverse, for the cleanup ---------------------------------------

    def _centres(self, slots, ego_xy, out_x, out_y, out_z):
        """Occupied slots -> cell centres, minus `ego_xy`.

        Pass the vehicle's world xy for vehicle-frame centres, which is what
        the cleanup needs; pass zeros for world-frame ones, which is what a
        map view needs.

        The cleanup projects cell centres into JP's range image, so it needs
        them where the sensor is, not where the lattice origin is. Slot to
        lattice cell is the inverse of `flat_slot`: within a ring of side W the
        slot is `(iy mod W) * W + (ix mod W)`, and the window pins which
        multiple of W is meant -- `ix = x0 + ((col - x0) mod W)`.
        """
        n = len(slots)
        for layout, buf in zip(self.handle.rings, self.buffers):
            hi = layout.offset + layout.slots
            sel = (slots >= layout.offset) & (slots < hi)
            if not sel.any():
                continue
            local = slots[sel] - layout.offset
            W = buf.side
            row, col = local // W, local % W
            ix = buf.x0 + np.mod(col - buf.x0, W)
            iy = buf.y0 + np.mod(row - buf.y0, W)
            out_x[:n][sel] = (ix + 0.5) * layout.cell_m - ego_xy[0]
            out_y[:n][sel] = (iy + 0.5) * layout.cell_m - ego_xy[1]

        # **Not `ground_height`.** A cell whose returns are all non-ground has
        # `w_sum == 0`, so `ground_height` is 0 -- and 0 cm is not a neutral
        # height, it is the datum. Projecting a parked car at 0 aims the ray
        # 1.73 m below the sensor, lands it on a different image row, and
        # compares the cell against a beam that never went near it. Measured
        # on the Gate 3 scene before this was fixed: every one of 379 car
        # cells read `ground_height == 0` while `ceiling_height` carried the
        # real 34 cm, and not one ghost was cleared.
        #
        # `ceiling_height` is the LOWEST thing overhead (§7.1's clearance bit,
        # fused with a min), which is exactly the surface that stops the beam.
        # Where nothing overhead was ever seen the sentinel stands and the
        # ground height is the only evidence there is.
        ceiling = self.handle.grid["ceiling_height"][slots]
        ground = self.handle.grid["ground_height"][slots]
        out_z[:n] = np.where(ceiling != CEILING_NONE, ceiling, ground) / 100.0
        return out_x[:n], out_y[:n], out_z[:n]

    # -- the frame ----------------------------------------------------------

    def step(self, frame) -> StepCounters:
        """Fold one `PerceptionFrame` into the map. See the module docstring
        for the stage order; everything here is bookkeeping around it."""
        stage = (self.timer.stage if self.timer is not None
                 else (lambda _name: nullcontext()))
        pts = frame.points_sensor
        n = min(len(pts), self.max_points)
        xs, ys, zs = pts[:n, 0], pts[:n, 1], pts[:n, 2]
        world = frame.points_world[:n]
        ego = np.asarray(frame.vehicle_xyz_world, float)[:2]
        if self._origin is None:
            self._origin = ego.copy()

        with stage("shift"):
            self._track_vehicle(ego)
        with stage("bin"):
            idx = self.bin(xs, ys, world[:, 0], world[:, 1])

        semantic = np.asarray(frame.semantic)[:n]
        cls = np.where(semantic < 0, 0, semantic).astype(np.uint8)
        if not class_ids_fit(cls):
            if not self.clip_class_ids:
                raise ValueError(
                    f"semantic class {int(cls.max())} exceeds the {CLASS_MAX} that "
                    "fusion's 5-bit candidate holds (math §10.2). The learning set "
                    "stops at 19, so an id above 31 means RAW SemanticKITTI ids "
                    "(10, 11, 40, 252, ...) are reaching the map where learning "
                    "ids are expected -- see `perception.semantics.semantic_labels`. "
                    "Clipping would be the wrong repair for that: pass "
                    "clip_class_ids=True (--clip-class-ids) only to get a frame "
                    "through, and know that it corrupts the class layer.")
            np.clip(cls, 0, CLASS_MAX, out=cls)

        rng_m = np.sqrt(xs * xs + ys * ys + (zs) * (zs))
        with stage("scatter"):
            aggregate = scatter_sorted(
                idx,
                quantise_height(world[:, 2]),
                quantise_weight(measurement_variance_cm2(np.maximum(rng_m, 1e-3))),
                np.asarray(frame.reflectivity8)[:n].astype(np.uint8),
                cls,
                np.asarray(frame.ground)[:n].astype(bool),
                scratch=self.handle.scratch,
            )
        touched = np.asarray(aggregate.cells).copy()
        with stage("fuse"):
            fuse(self.handle.grid, aggregate, self.thresholds)

        counters = StepCounters(
            index=frame.index, points=len(pts), binned=int((idx >= 0).sum()),
            cells_touched=len(touched), occupied=0, tested=0, cleared=0,
            protected=0, out_of_view=0)
        if self.ghost_removal:
            with stage("cleanup"):
                self._cleanup(frame, touched, ego, counters)
        return counters

    def _track_vehicle(self, ego_xy):
        """Slide each ring's window to keep the vehicle centred, in whole cells
        of that ring -- the §2.4 constraint. Sub-cell remainders are carried,
        not dropped, or the map drifts behind the vehicle at a few cm a frame."""
        for layout, buf in zip(self.handle.rings, self.buffers):
            want_x = int(np.floor(ego_xy[0] / layout.cell_m)) - buf.side // 2
            want_y = int(np.floor(ego_xy[1] / layout.cell_m)) - buf.side // 2
            dx, dy = want_x - buf.x0, want_y - buf.y0
            if dx or dy:
                shift(buf, dx, dy, self.handle.grid)

    def _cleanup(self, frame, touched, ego_xy, counters):
        """§10.4 against this frame's range image, then fold the misses into
        occupancy. This is the half the Gate 3 toggle is supposed to switch."""
        state = occupancy_state(self.handle.grid, self.thresholds)
        occupied = np.flatnonzero(state == OCC_OCCUPIED)
        counters.occupied = len(occupied)
        if not len(occupied):
            return
        if len(occupied) > self.max_candidates:
            # Deterministic, not random: a cap that changes which cells it drops
            # from run to run would break the determinism test.
            occupied = occupied[:self.max_candidates]

        m = len(occupied)
        self._cand_slots[:m] = occupied
        cx, cy, cz = self._centres(occupied, ego_xy, self._cand["x"],
                                           self._cand["y"], self._cand["z"])

        image = np.asarray(frame.range_image)
        if self.range2d.shape != image.shape[:2]:
            self.range2d = np.zeros(image.shape[:2], np.float32)
        np.copyto(self.range2d, image[:, :, 0])

        # The guard: a cell with a return in THIS scan is never cleared.
        guard = self._has_return[:m]
        np.copyto(guard, np.isin(occupied, touched))

        result = visibility_cleanup(
            cx, cy, cz, self.range2d, has_return_now=guard,
            sensor=self.sensor,
            floor_m=self.thresholds["visibility"]["range_tolerance_m"],
            protect_current_returns=True, scratch=self.vis_scratch)

        occ = self.thresholds["occupancy"]
        apply_miss(self.handle.grid["log_odds"], self._cand_slots[:m],
                   result.see_through, occ["log_odds_miss"],
                   tuple(occ["log_odds_clamp"]))

        counters.tested = result.tested
        counters.cleared = result.cleared
        counters.protected = result.protected
        counters.out_of_view = result.out_of_view

    # -- readout ------------------------------------------------------------

    def occupied_slots(self) -> np.ndarray:
        """Flat slots the map currently calls OCCUPIED. What a 2.5D map view
        should draw, and what the ghost trails live in."""
        return np.flatnonzero(
            occupancy_state(self.handle.grid, self.thresholds) == OCC_OCCUPIED)

    def occupied_cells(self):
        """`(slots, x, y, z)` for every OCCUPIED cell, in the WORLD frame.

        The readout a 2.5D map view needs, and the one thing the Gate 3 demo
        is missing: the dashboard draws returns, and the ghost toggle moves
        cells. `z` is the cell's visibility height -- its ceiling where one was
        ever seen, its ground height otherwise -- which is the same height the
        cleanup tests against, so what is drawn and what is reasoned about
        cannot disagree.

        This allocates, deliberately: it is a readout, called by a viewer or a
        figure at its own rate, and sizing it to `max_candidate_cells` would
        make the drawing silently truncate at the cleanup's cap. Nothing on the
        frame path calls it.
        """
        slots = self.occupied_slots()
        n = len(slots)
        x, y, z = (np.zeros(n), np.zeros(n), np.zeros(n))
        if n:
            # ego (0, 0) leaves the centres in the world frame -- the lattice
            # is global, and it is the subtraction that makes them vehicle.
            self._centres(slots, np.zeros(2), x, y, z)
        return slots, x, y, z
