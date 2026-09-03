"""The transient layer and the tracked-object list. Master v4 §3.6-3.7. [Aakash]

Dynamic returns do not belong in the persistent map. A car parked in frame 12
and gone by frame 20 leaves a wall in an elevation map that averages it in, and
no amount of visibility cleanup gets it out cleanly. So moving points are
routed here instead, and `query()` returns the union of the two layers with one
merge rule defined in one place (`grid/query.py`).

⚑ This is not a nicety. `python scripts/eval_synthetic.py --frames 14
  --keep-moving` bypasses it, and the car 12 m ahead pulls ring 1's height
  RMSE from **0.41 cm to 12.72 cm** -- thirty times the entire error budget of
  that ring, from one object. `eval_synthetic.py` used to strip moving points
  by hand to get an interpretable number, which is exactly the kind of
  hand-holding that stops being possible the moment the real sequence lands.

  (Re-measured 2026-09-01 after the beam-intersection fix in `eval/synthetic`;
  it read 0.48 -> 11.71 cm on the old sampler, which is the same conclusion.)

--- where the motion labels come from, and why that is a feature ----------

Master v4 §3.6: `moving-*` is read straight from the raw `.label` files, ids
250-259. Nothing is retrained. Disclose it plainly -- *"motion labels are
ground truth; the mapping contribution is evaluated independently of
segmentation quality"* -- because it isolates the contribution from
segmentation error, which is what a careful evaluator wants.

⚑ `separate()` therefore takes RAW label ids, not learning ids. The 19-class
  collapse happens in `learning_map` and it is where the motion information is
  destroyed: every `moving-*` id maps onto its static counterpart, so a scan
  that has already been through the learning map cannot be separated at all.
  Order matters, and getting it wrong produces a map that looks right and
  quietly contains every car that ever drove past.

--- what persists and what does not --------------------------------------

Master v4 §3.7, and the sentence is easy to read past: **"Do not wipe the
transient layer's memory, only its grid."**

The GRID is frame-fresh -- cleared every frame, because a dynamic obstacle's
position is only true for the frame it was seen in. The TRACKED OBJECT LIST is
not: it persists ~1 s with constant-velocity prediction, so a pedestrian
briefly hidden by a parked car does not vanish. Clearing both is the failure
mode that matters most, and it is one line away from correct.
"""

import numpy as np
from vrgrid.cell import FLAG_DYNAMIC
from vrgrid.gpu.allocators import TRACK_DTYPE, TRANSIENT_FIELDS
from vrgrid.grid.schedule import load_thresholds

# Raw SemanticKITTI `moving-*` ids. The one definition; `eval.reference_map`
# imports it rather than keeping a second copy, because a reference map that
# strips a different set of ids than the pipeline does is not a reference.
MOVING_ID_LO, MOVING_ID_HI = 250, 259

# A track is dropped after this many frames unseen. ~1 s at 10 Hz, per §3.7.
TRACK_TTL_FRAMES = 10


def is_moving(label_id) -> np.ndarray:
    """Raw label ids 250-259. Master v4 §3.6."""
    lid = np.asarray(label_id, dtype=np.int64) & 0xFFFF
    return (lid >= MOVING_ID_LO) & (lid <= MOVING_ID_HI)


def separate(label_id):
    """(static_mask, moving_mask) from RAW label ids. See the note above about
    ordering: this must run before any learning-map collapse."""
    moving = is_moving(label_id)
    return ~moving, moving


def clear_grid(transient) -> int:
    """Wipe the transient GRID, not the tracked list. §3.7.

    Called at the top of every frame. The grid holds where dynamic things were
    last frame, which is not where they are now; the tracks hold what they are
    and how fast, which survives.
    """
    for name, _ in TRANSIENT_FIELDS:
        transient[name][:] = 0
    return transient["flags"].size


def ingest(gm, points_m, class_id, moving_mask, points_world_m=None) -> int:
    """Write this frame's dynamic returns into the transient grid.

    Same two frames as `fusion.scatter()` and for the same reasons: the RING
    comes from the vehicle frame, the CELL from the world frame. The transient
    layer shares the grid's geometry exactly (§3.7), so it shares the slot
    index too -- which is what lets `query()` merge the layers without a
    second addressing scheme to keep in sync.

    Height is the MAXIMUM of the dynamic returns in a cell, not the mean. A
    pedestrian is 1.7 m of person over 0 m of road and the mean is a metre of
    neither; the planner needs the top of the obstacle, and a transient cell
    exists precisely to say "something is standing here".

    ⚑ THREE frames, not two, and the third is vertical. Height is taken from
      the WORLD z, re-based to `gm.z_datum_m`, which is what the persistent
      layer stores (`engine.step` -> `quantise_height(world[:, 2], z_datum)`;
      `harness.run_sequence` -> `world[:, 2] - gm.z_datum_m`). This used to
      quantise the VEHICLE-frame z instead, so the two layers were on
      different vertical origins and `query()` returned both through the same
      `CellQuery.ground_height` field with nothing to say which was which.
      The gap is `frac(ego_z)` -- up to a metre, and it moved every time the
      band stepped. Measured: ego_z 3.4, datum 3.0, a return at world z 4.60
      came back as 1.20 m where every static cell around it was on 1.60 m.

      With no `points_world_m` the world points ARE the vehicle points and
      `z_datum_m` is 0.0, so the stationary case every unit test uses is
      unchanged, bit for bit.

    Returns the number of points written.
    """
    from vrgrid.gpu.kernels import quantise_height
    from vrgrid.grid.lattice import bin_points

    moving_mask = np.asarray(moving_mask, dtype=bool)
    if not np.any(moving_mask):
        return 0

    pts = np.asarray(points_m, dtype=np.float64)[moving_mask]
    world = pts if points_world_m is None else np.asarray(
        points_world_m, dtype=np.float64)[moving_mask]

    # The third spelling of the binning stage, now the same one as the other
    # two. `bin_points` already returns -1 for OUTSIDE, so the separate
    # `np.where(rings == OUTSIDE, ...)` mask this used to need is gone with it.
    scratch, out = gm.bin_scratch(pts.shape[0])
    slots = bin_points(pts[:, 0], pts[:, 1], world[:, 0], world[:, 1],
                       gm.schedule, gm.buffers, out, scratch,
                       gm.speed_ms).copy()

    keep = (slots >= 0) & (slots < gm.transient["flags"].size)
    if not np.any(keep):
        return 0
    slots = slots[keep]
    z_cm = quantise_height(world[keep, 2], getattr(gm, "z_datum_m", 0.0))

    # Maximum per cell, deterministically: sort by (slot, height) and take the
    # last of each run. np.maximum.at would also work and is slower; either
    # way it must not be an unordered scatter, which would make the map depend
    # on point order (§3.4's argument, applied to a different array).
    order = np.lexsort((z_cm, slots))
    s, z = slots[order], z_cm[order]
    last = np.r_[s[1:] != s[:-1], True]
    top_slots, top_z = s[last], z[last]

    gm.transient["ground_height"][top_slots] = np.maximum(
        gm.transient["ground_height"][top_slots], top_z)
    gm.transient["flags"][top_slots] |= FLAG_DYNAMIC
    return int(keep.sum())


class TrackList:
    """Tracked objects with velocity, capped and preallocated. §3.4, §3.7.

    Persists ~1 s with constant-velocity prediction so that a pedestrian
    briefly hidden by a parked car does not vanish -- the failure mode master
    v4 singles out as the one that matters most.

    Fixed capacity, evicted by the same priority as the refinement pool: the
    memory bound is a compile-time claim and a list that grows with traffic
    makes it false.
    """

    def __init__(self, capacity: int = 256, arrays=None):
        self.capacity = int(capacity)
        self.tracks = (arrays if arrays is not None
                       else np.zeros(self.capacity, dtype=TRACK_DTYPE))
        self.active = np.zeros(self.capacity, dtype=bool)
        self._next_id = 1

    @property
    def count(self) -> int:
        return int(self.active.sum())

    def predict(self, dt_s: float) -> None:
        """Constant velocity, one frame. Ages every track by one frame."""
        idx = np.flatnonzero(self.active)
        if idx.size == 0:
            return
        self.tracks["x_m"][idx] += self.tracks["vx_ms"][idx] * dt_s
        self.tracks["y_m"][idx] += self.tracks["vy_ms"][idx] * dt_s
        aged = np.minimum(self.tracks["frames_since_seen"][idx].astype(np.int32) + 1, 255)
        self.tracks["frames_since_seen"][idx] = aged.astype(np.uint8)
        self.active[idx[aged > TRACK_TTL_FRAMES]] = False

    def update(self, detections, dt_s: float = 0.1, gate_m: float = 2.5) -> int:
        """Associate detections to tracks by nearest centre, then spawn.

        `detections` is (N, 3) of dynamic-cluster centroids with a class in the
        4th column if given; nearest-neighbour association inside `gate_m`.
        Deliberately simple: this exists so the transient layer has memory,
        not to be a tracker. Real association is JP's `dynamic_objects()`.

        Returns the number of tracks after the update.
        """
        det = np.asarray(detections, dtype=np.float64)
        if det.ndim != 2 or det.shape[0] == 0:
            return self.count

        for row in det:
            x, y = float(row[0]), float(row[1])
            idx = np.flatnonzero(self.active)
            matched = -1
            if idx.size:
                d = np.hypot(self.tracks["x_m"][idx] - x, self.tracks["y_m"][idx] - y)
                nearest = int(np.argmin(d))
                if d[nearest] <= gate_m:
                    matched = int(idx[nearest])

            if matched >= 0:
                # Velocity from the observed displacement, not from a filter:
                # one frame of lag is cheaper than a filter nobody tuned.
                self.tracks["vx_ms"][matched] = (x - self.tracks["x_m"][matched]) / dt_s
                self.tracks["vy_ms"][matched] = (y - self.tracks["y_m"][matched]) / dt_s
                self.tracks["x_m"][matched] = x
                self.tracks["y_m"][matched] = y
                self.tracks["frames_since_seen"][matched] = 0
                continue

            free = np.flatnonzero(~self.active)
            if free.size == 0:
                # Full: drop the stalest, which is the closest thing to the
                # pool's priority rule that a track list has.
                stale = np.flatnonzero(self.active)
                slot = int(stale[np.argmax(self.tracks["frames_since_seen"][stale])])
            else:
                slot = int(free[0])
            self.tracks[slot] = (x, y, 0.0, 0.0,
                                 int(row[3]) if row.size > 3 else 0, 0,
                                 self._next_id % 65536)
            self.active[slot] = True
            self._next_id += 1
        return self.count

    def alive(self):
        """Active tracks, as a view. What `api.dynamic_objects()` returns."""
        return self.tracks[self.active]


def cluster(points_m, class_id=None, cell_m: float = 1.0):
    """Dynamic returns -> object centroids, by grid cell.

    A stand-in for real clustering: bin by a 1 m grid and take each bin's
    centroid. Enough to give the track list something to associate, and
    honest about what it is -- segmentation and instance clustering are JP's,
    and this is here so the transient layer's memory can be built and tested
    before they land.
    """
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 4))
    key = np.floor(pts[:, :2] / cell_m).astype(np.int64)
    flat = key[:, 0] * 100_000 + key[:, 1]
    order = np.argsort(flat, kind="stable")
    flat, pts = flat[order], pts[order]
    cls = (np.zeros(len(pts), np.int64) if class_id is None
           else np.asarray(class_id, dtype=np.int64)[order])

    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    out = []
    for a, b in zip(starts, np.r_[starts[1:], len(flat)]):
        out.append([pts[a:b, 0].mean(), pts[a:b, 1].mean(),
                    pts[a:b, 2].max(), cls[a]])
    return np.array(out)


def step(gm, points_m, label_id, points_world_m=None, dt_s=None, tracks=None):
    """One frame of the transient layer: clear, ingest, track. §3.7.

    Returns (n_dynamic_points, n_tracks). The caller keeps the `TrackList`
    across frames -- that is the memory §3.7 says not to wipe.
    """
    th = gm.thresholds if gm.thresholds else load_thresholds()
    dt = th.get("fusion", {}).get("frame_dt_s", 0.1) if dt_s is None else dt_s

    clear_grid(gm.transient)
    _, moving = separate(label_id)
    written = ingest(gm, points_m, None, moving, points_world_m)

    if tracks is not None:
        tracks.predict(dt)
        pts = np.asarray(points_m, dtype=np.float64)[np.asarray(moving, dtype=bool)]
        tracks.update(cluster(pts), dt_s=dt)
        return written, tracks.count
    return written, 0
