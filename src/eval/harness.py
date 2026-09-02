"""The evaluation harness. Master v4 §3.8. [Aakash]

The harness is the product — judges see the numbers and the demo, nothing
else. So this is the one place that owns the whole loop:

    sequence -> reference map (M*)          math §9.1, schedule-independent
             -> map under test, per schedule
             -> per-ring metrics             §9.2, §9.3
             -> one table                    Gate 6: every number from a script

Two things it is built around, both of which are constraints rather than
preferences:

**The reference is built once and reused across schedules.** M* is on the base
lattice and knows nothing about rings, which is exactly what makes comparing
5/10/20/40 against 5/10/50 a comparison of schedules rather than of two
separately-tuned pipelines (flaw E6, and `src/eval/CLAUDE.md`).

**Nothing here tunes anything.** Thresholds come from the frozen config. A
harness that can adjust a threshold to improve its own number is not a
harness, and the moment one exists somebody will use it the night before the
deadline.
"""

from dataclasses import dataclass, field

import numpy as np
from vrgrid.eval import metrics
from vrgrid.eval.reference_map import ReferenceMap
from vrgrid.gpu.allocators import allocate, bytes_allocated
from vrgrid.gpu.kernels import CEILING_NONE
from vrgrid.gpu.shift import RingBuffer, shift, track_datum
from vrgrid.grid import gate, traversability
from vrgrid.grid.fusion import fuse, initialise, scatter
from vrgrid.grid.pool import RefinementPool
from vrgrid.grid.query import GridMap
from vrgrid.grid.schedule import load, load_thresholds
from vrgrid.grid.transient import TrackList, separate
from vrgrid.grid.transient import step as transient_step


def uniform_schedule(cell_m: float, half_width_m: float = 100.0,
                     base_cell_m: float = 0.05, hysteresis_eps: float = 0.1):
    """A single-ring schedule: the uniform-grid baselines of §8.2's sweep.

    The money plot needs points that are not our own schedule, and a uniform
    grid is the honest comparison -- it is what everyone else builds and what
    the 21.5x claim is measured against. Built through the same `Schedule` the
    frozen configs load into, so it goes through `validate()` and through
    exactly the same lattice, fusion and metric code. A baseline evaluated by
    a second code path is not a baseline.

    ⚑ The two FROZEN schedules share rings 0 and 1 (5 cm to 10 m, 10 cm to
      25 m) and differ only beyond 25 m. So a planning problem inside 25 m
      cannot tell them apart and will report identical regret for both -- not
      a bug, and not evidence that the ablation is free. Either plan into the
      far field or read the curve against these uniform points, which differ
      from us everywhere.
    """
    from vrgrid.grid.schedule import Anisotropy, Ring, Schedule, validate

    side = 2 * half_width_m / cell_m
    if abs(side - round(side)) > 1e-9 or round(side) % 2:
        raise ValueError(
            f"{half_width_m} m at {cell_m} m is {side} cells; the allocator "
            "needs an even whole number so the ring has a centre"
        )
    cells = round(side) ** 2
    name = f"uniform_{cell_m * 100:.0f}cm"
    s = Schedule(
        name=name, base_cell_m=base_cell_m,
        rings=[Ring(0, half_width_m, cell_m, cells, 0.0)],
        total_cells=cells, vertical_extent_m=(-2.0, 6.0),
        hysteresis_eps=hysteresis_eps, anisotropy=Anisotropy(),
    )
    validate(s)
    return s


def build_gridmap(schedule, thresholds=None, with_pool: bool = True) -> GridMap:
    """A ready-to-use map: allocation, ring windows, pool, initialised fields.

    The windows are centred on the vehicle, so ring L's absolute lattice runs
    from -N_L/2 to +N_L/2 and the vehicle sits at index 0. Centring is a
    convention rather than a requirement -- the toroidal addressing does not
    care -- but it is the convention the whole project reads x forward, y left
    in, so it is set here once instead of in every caller.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    alloc = allocate(schedule, th)

    buffers = [RingBuffer(side=r.side, offset=r.offset,
                          x0=-r.side // 2, y0=-r.side // 2)
               for r in alloc.rings]

    initialise(alloc.grid)   # ceiling sentinel; see fusion.initialise
    pool_cfg = th.get("refinement_pool", {})
    pool = RefinementPool(pool_cfg.get("blocks", 512),
                          pool_cfg.get("cells_per_block", 16),
                          arrays=alloc.pool) if with_pool else None

    gm = GridMap(soa=alloc.grid, schedule=schedule, buffers=buffers,
                 thresholds=th, transient=alloc.transient, pool=pool,
                 scatter_mode=alloc.scatter_mode)
    gm.allocation = alloc
    return gm


def recenter(gm: GridMap, x_m: float, y_m: float) -> int:
    """Slide every ring window so it is centred on the vehicle. Math §2.4.

    Returns the number of slots cleared, which is the O(perimeter) claim as a
    measurement rather than an assertion.

    The origin moves in whole COARSEST cells and every finer ring moves a whole
    multiple of that -- §2.4's constraint, and the reason it exists: a fractional
    step would shift each ring boundary by a fraction of a cell and force a
    resample, which is precisely the "data loss during projection" the problem
    statement warns about. Expected side effect: the nominal ring boundary
    wobbles by up to 40 cm. That is correct behaviour, not a bug.
    """
    coarsest = gm.schedule.rings[-1]
    k_coarsest = gm.schedule.k(len(gm.schedule.rings) - 1)

    # Where the vehicle sits, in coarsest cells, and where the window wants
    # its low corner to be so the vehicle stays in the middle.
    want_x = int(np.floor(x_m / coarsest.cell_m)) - gm.buffers[-1].side // 2
    step = want_x - gm.buffers[-1].x0
    want_y = int(np.floor(y_m / coarsest.cell_m)) - gm.buffers[-1].side // 2
    step_y = want_y - gm.buffers[-1].y0

    # query() converts vehicle frame to world with this, so it has to move
    # even when the window does not: the vehicle drifts within a coarsest cell
    # between shifts, and 40 cm of unrecorded drift is a whole ring-0 cell.
    gm.vehicle_xy_m = (x_m, y_m)
    if step == 0 and step_y == 0:
        return 0

    cleared = 0
    for level, buf in enumerate(gm.buffers):
        scale = k_coarsest // gm.schedule.k(level)   # integer, by validate()
        slots = shift(buf, step * scale, step_y * scale, gm.soa,
                      fill={"ceiling_height": CEILING_NONE})
        cleared += int(slots.size)
    return cleared


@dataclass
class RunStats:
    """What the frame loop did, so the dashboard and §9.4 can both read it."""

    frames: int = 0
    static_points: int = 0
    dynamic_points: int = 0
    dynamic_to_transient: int = 0
    tracks: int = 0
    gate_fired: int = 0
    gate_acquired: int = 0
    gate_refused: int = 0
    gate_released: int = 0

    @property
    def removal(self) -> dict:
        """§9.4's DR / SP / F, from the counters the loop already keeps.

        DR is the fraction of dynamic returns kept OUT of the persistent map;
        SP the fraction of static returns kept IN. Both directions, always:
        DR alone is gameable -- delete the whole map and score 100%.
        """
        from vrgrid.eval.metrics import dynamic_removal

        return dynamic_removal(self.dynamic_points, self.dynamic_points,
                               self.static_points, self.static_points)


class FrameConventionError(AssertionError):
    """The world-frame points are not in the convention the lattice needs."""


def assert_world_is_z_up(world, ground, tolerance_m: float = 2.0) -> None:
    """Ground returns must sit near z = 0 in the world frame. Math §2.1.

    ⚑ This exists because frame confusion is the most common silent bug in
      this project (CLAUDE.md) and this is the exact seam it enters through.

    `run_sequence` takes a 4x4 **vehicle -> world** transform and hands the
    result to `i_ring`, which decides cell identity. A raw KITTI `poses.txt`
    row is NOT that transform. It is Camera-0 -> World_cam, in the camera
    convention (x right, y down, z forward), and it needs `Tr` from calib.txt
    on the right and the axis permutation on the left before it means anything
    to this map:

        T_vehicle_to_world = R_CAM0_TO_VEH @ pose_4x4 @ Tr

    which is exactly what `perception.transforms.vehicle_to_world()` returns.
    Pass the raw row instead and every return is rotated into a different
    world: the elevation map fills, the metrics compute, nothing raises, and
    the map slowly rotates. There is no downstream assertion that would catch
    it, because a wrong cell is a perfectly valid cell.

    The check is on the DATA rather than on the matrix, deliberately. At frame
    0 a camera-convention pose and a vehicle-convention pose are both close to
    the identity and cannot be told apart by inspecting the rotation; the
    ground plane tells them apart immediately, because in the camera
    convention the ground is a plane of constant y, not constant z.

    Checked once per run, on the first frame, over ground-flagged returns only.
    """
    g = np.asarray(ground, dtype=bool)
    if not g.any():
        return
    z = np.asarray(world, dtype=np.float64)[g, 2]
    median = float(np.median(z))
    if abs(median) <= tolerance_m:
        return
    raise FrameConventionError(
        f"ground returns have a median world z of {median:.2f} m, not ~0 -- "
        f"the world points are not z-up.\n"
        f"  Most likely: a raw `poses.txt` row was passed as the pose. That is "
        f"Camera-0 -> World_cam (x right, y down, z forward), not "
        f"vehicle -> world.\n"
        f"  Fix: pass `perception.transforms.vehicle_to_world(pose, sequence)`, "
        f"which composes the axis permutation and `Tr` from calib.txt.\n"
        f"  If this really is z-up terrain, raise `tolerance_m`."
    )


# How far the vehicle must have travelled before the frame guard's second
# look. See `FrameGuard`.
GUARD_BASELINE_M = 10.0


class FrameGuard:
    """Checks the world-frame convention on the first frame, and again once
    the vehicle has actually gone somewhere.

    ⚑ **Checking frame 0 alone is nearly worthless, and that is not obvious.**
      A KITTI `poses.txt` starts at the identity by construction, so on frame 0
      the wrong composition and the right one produce almost the same points:
      feeding raw sensor points straight through an identity pose leaves them
      in the sensor frame, which is x-forward, y-left, z-up with the ground at
      -1.73 m -- inside this check's 2 m tolerance. The conventions only
      separate once the pose carries real rotation or translation, which is to
      say once the vehicle has driven.

      Found by a test that was written to prove the guard fires and instead
      proved it does not: `test_build_refuses_a_camera_convention_pose`.

    So: frame 0, because a gross error should not survive one frame, and then
    the first frame at least `GUARD_BASELINE_M` from the start, which is where
    a convention error has become unmistakable. Two looks, then it stops
    costing anything.
    """

    def __init__(self, baseline_m: float = GUARD_BASELINE_M,
                 tolerance_m: float = 2.0):
        self.baseline_m = baseline_m
        self.tolerance_m = tolerance_m
        self._origin = None
        self._moved = False

    @property
    def done(self) -> bool:
        return self._moved

    def check(self, world, ground, translation) -> None:
        """`translation` is the vehicle -> world transform's t, or None to
        check unconditionally (a caller with no pose to hand)."""
        if self._moved:
            return
        first = self._origin is None
        if translation is None:
            assert_world_is_z_up(world, ground, self.tolerance_m)
            self._moved = True
            return
        t = np.asarray(translation, dtype=np.float64)[:3]
        if first:
            self._origin = t
        far = float(np.linalg.norm(t - self._origin)) >= self.baseline_m
        if first or far:
            assert_world_is_z_up(world, ground, self.tolerance_m)
        self._moved = far


def learning_ids(raw_labels):
    """RAW SemanticKITTI ids -> 0-19 learning ids, or pass through if already
    mapped. Math §10.2.

    The synthetic sequences already write learning ids, and the real loader
    writes raw ones (10 = car, 40 = road, 252 = moving-car). Both reach this
    harness, so it detects rather than assumes: anything already inside the
    class field is left alone, and anything above it is mapped.

    ⚑ Order matters and is the reason this is not done earlier. The learning
      map collapses every `moving-*` id onto its static counterpart, so a scan
      already through it cannot be separated into static and dynamic at all.
      `separate()` must see the raw ids; this runs after it, on what survived.
    """
    import numpy as _np
    from vrgrid.grid.fusion import CLASS_MAX, CLASS_UNLABELLED

    ids = _np.asarray(raw_labels, dtype=_np.int64) & 0xFFFF
    if ids.size and int(ids.max()) <= CLASS_MAX:
        return ids.astype(_np.uint8)

    from vrgrid.perception.semantics import semantic_labels

    mapped = _np.asarray(semantic_labels(ids), dtype=_np.int32)
    # ⚑ `semantic_labels` reports -1 for `unlabeled` (raw 0) and for any id
    #   outside the SemanticKITTI scheme. Straight through `astype(uint8)` that
    #   is 255, which does not fit the 5-bit class field, and `scatter_sorted`
    #   rejects it -- "class ids must be < 32 to pack into the class key" on
    #   the first real frame. The synthetic sequences write learning ids and
    #   never contain an unlabelled point, so this could not surface until the
    #   loader was pointed at sequence 08.
    #
    #   Mapped, not dropped: an unlabelled return still has geometry, and a
    #   wall nobody labelled is still a wall. CLASS_UNLABELLED is in no
    #   drivable set, so the cell fails safe on §7.1 bit 4 -- which is the
    #   honest verdict for a class that is not known.
    return _np.where(mapped < 0, CLASS_UNLABELLED, mapped).astype(_np.uint8)


def real_scans(sequence: str, max_frames=None, start_frame: int = 0,
               use_patchworkpp: bool = True):
    """SemanticKITTI as `run_sequence` wants it: (points in VEHICLE frame, RAW
    label ids, is_ground, vehicle -> world T).

    The seam the synthetic writer stands in for. `eval_synthetic`'s docstring
    has said since Day 0 to "swap `read_sequence` for `perception.loader` and
    sequence 07 when the download lands" -- this is that swap, in one place
    rather than copied into each script that needs it.

    Three conversions the loader does NOT do, each of which produced a wrong
    answer rather than an error when guessed at:

    ⚑ `loader.scans` yields (N, 4) -- x, y, z, INTENSITY -- and `run_sequence`
      wants (N, 3). Intensity is consumed by the reflectivity path, separately.

    ⚑ The points are in the SENSOR frame; the harness wants the VEHICLE frame.
      They differ by the 1.73 m HDL-64E mounting height and nothing else
      (docs/frames.md), so this is `transforms.sensor_to_vehicle()` and not a
      no-op. Skip it and every height in the map is 1.73 m too high, which
      looks like a map -- just a wrong one.

    ⚑ `pose` is a raw KITTI row, Camera-0 -> World_cam, and must NOT be handed
      over as a vehicle -> world transform. `transforms.vehicle_to_world` is
      the composition that applies `Tr` and the axis permutation, and per
      `run_sequence`'s own docstring it is the only thing allowed to build it.

    Ground comes from Patchwork++ where the extension is installed and from the
    semantic labels otherwise -- the same fallback `run/__main__.py` uses.

    For `reference_map.build_from_scans`, which wants a 3-tuple without the
    ground mask, drop it: `((p, l, T) for p, l, _, T in real_scans(...))`.
    """
    from vrgrid.perception import ground, loader, semantics, transforms

    t_s_v = transforms.sensor_to_vehicle()
    for pts, labels, pose in loader.scans(sequence, max_frames=max_frames,
                                          start_frame=start_frame):
        vehicle_pts = transforms.transform_points(pts[:, :3], t_s_v)
        if use_patchworkpp and ground._HAVE_PATCHWORKPP:
            gmask = ground.segment_ground(pts)
        else:
            gmask = ground.ground_from_semantics(
                semantics.semantic_labels(labels))
        yield (vehicle_pts, labels, gmask,
               transforms.vehicle_to_world(pose, sequence=sequence))


def final_vehicle_xy(sequence: str, max_frames=None) -> tuple:
    """(x, y) of the vehicle's last pose, in world metres.

    The planning window is placed relative to it. On the synthetic sequence
    that was `(frames - 1) * 2.0` because the car drives straight down y = 0;
    on a real sequence it turns, and `costmaps_for`'s own note says a window
    hardcoded about the origin measures ground the map never saw and comes out
    as a confident zero.
    """
    from vrgrid.perception import loader, transforms

    poses = loader.poses(sequence)
    if max_frames is not None:
        poses = poses[:max_frames]
    T = transforms.vehicle_to_world(poses[-1], sequence=sequence)
    return float(T[0, 3]), float(T[1, 3])


def run_sequence(gm: GridMap, scans, recentre: bool = True,
                 tracks: TrackList | None = None) -> RunStats:
    """Drive the map through a sequence. Returns what it did.

    `scans` yields (points in VEHICLE frame, RAW label ids, is_ground, T).

    ⚑ `T` is a **vehicle -> world** transform (3x4 or 4x4), z-up, NOT a raw
      KITTI `poses.txt` row. A `poses.txt` row is Camera-0 -> World_cam and
      needs `Tr` and the axis permutation applied first --
      `perception.transforms.vehicle_to_world(pose, sequence)` is that
      composition and is the only thing that should be building it. Two
      implementations of one convention is how a map ends up slowly rotating,
      so there is one, it is JP's, and this consumes it.

      `FrameGuard` checks the first frame and one frame after 10 m of travel,
      and raises rather than letting the wrong convention through, because the
      failure downstream is a full, plausible-looking map in the wrong cells.
      The second look is the one that matters: frame 0 of a KITTI sequence is
      the identity pose, where both conventions agree.

    ⚑ RAW label ids, not learning ids. `moving-*` (250-259) is what separates
      dynamic from static, and the 19-class learning map collapses every
      `moving-*` onto its static counterpart -- so a scan already through the
      learning map cannot be separated at all, and every car that ever drove
      past ends up welded into the elevation map. See grid/transient.py.

    ⚑ Both frames are needed and they do different jobs -- see the note on
      `fusion.scatter()`. The ring a point lands in is decided in the vehicle
      frame, because foveation follows the vehicle; the CELL it lands in is
      decided in the world frame, because cell identity is world-anchored and
      that is the entire reason the toroidal shift exists. Scatter every frame
      at the vehicle origin instead and the map still builds, still looks
      plausible, and smears the whole sequence onto one patch of ground.

    The pose is applied here rather than in `scatter()` so there is one place
    that knows the frame convention. When `perception.transforms` lands this
    should call `transform_points()` instead of composing the matrix itself --
    two implementations of one convention is how a map ends up slowly rotating.
    """
    stats = RunStats()
    datum_set = False
    speed = 0.0
    last_xy = None
    guard = FrameGuard()

    for pts, labels, ground, pose in scans:
        # ⚑ MOVING, and re-basing as it moves. A FIXED datum is enough for
        #   seq 07 -- ground at -1.67..-1.59 m -- and not for a sequence with
        #   relief: 08 climbs -1.65 -> +5.63 m in 40 frames and +45.7 m over
        #   the sequence against an 8 m band, and a fixed datum still clipped
        #   16.91% of its ground returns. `track_datum` slides the band in
        #   whole 1 m steps and re-bases every stored height to match, so all
        #   cells stay relative to the SAME current datum and every difference
        #   the map computes -- slope, step, curb height, pothole depth -- is
        #   unaffected. `metrics` adds the final datum back to compare against
        #   the world-absolute M*.
        gm.z_datum_m = track_datum(gm.soa, None if not datum_set else gm.z_datum_m,
                                   float(np.asarray(pose)[2, 3]))
        datum_set = True
        pose = np.asarray(pose, dtype=np.float64)
        pts = np.asarray(pts, dtype=np.float64)
        world = pts @ pose[:3, :3].T + pose[:3, 3]
        if not guard.done:
            guard.check(world, ground, pose[:3, 3])
        xy = (float(pose[0, 3]), float(pose[1, 3]))
        if last_xy is not None:
            dt = gm.thresholds.get("fusion", {}).get("frame_dt_s", 0.1)
            speed = float(np.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) / dt)
        last_xy = xy
        if recentre:
            recenter(gm, *xy)

        # Dynamic returns never reach the persistent map. Before this existed,
        # one car 12 m ahead moved ring 1's RMSE from 0.48 cm to 11.71 cm.
        static, moving = separate(labels)
        stats.static_points += int(static.sum())
        stats.dynamic_points += int(moving.sum())

        written, n_tracks = transient_step(gm, pts, labels, world, tracks=tracks)
        stats.dynamic_to_transient += written
        stats.tracks = n_tracks

        # Learning ids, not `% 16`. The modulo was a stand-in for the 4-bit
        # class field and it was not a harmless one: it mapped `terrain` (17)
        # onto `car` (1), so drivable ground arrived in the map as a blocked
        # class and the §7.1 predicate marked it untraversable. `pole` and
        # `traffic-sign` became `bicycle` and `motorcycle` the same way. The
        # field is 5 bits since 1 Sep and holds the whole set.
        # ⚑ HEIGHT COMES FROM THE WORLD FRAME, and `scatter` takes it from
        #   `pts` -- `fusion.scatter` reads `x, y, z = pts[...]` for the height
        #   while taking only `wx, wy` from `points_world_m`. So the map stored
        #   VEHICLE-frame z against WORLD-anchored cells: the same patch of road
        #   got a different stored height depending on where the vehicle was
        #   when it saw it, and M* -- which stores world z -- disagreed by the
        #   vehicle's elevation. On sequence 07 that is a flat +162.50 cm of
        #   bias with a spread of 1.08 cm, i.e. the entire error.
        #
        #   Invisible on the synthetic scene, where the car drives at z = 0 and
        #   the two frames coincide, which is why it survived to real data.
        #
        #   Fixed here rather than in `scatter` because `MapEngine` has its own
        #   datum machinery (`_z_datum`, added back on readout) and must not be
        #   disturbed; this is the eval harness, which already composes world
        #   coordinates per frame, so one more per-frame array is in keeping.
        #   The deeper question -- whether `scatter` should ever take height
        #   from a different frame than cell identity -- is worth asking once.
        #   Heights go in RELATIVE TO A FIXED DATUM for the run. The band is
        #   8 m wide and world-absolute at datum 0 (kernels.quantise_height),
        #   which holds for seq 07 -- ground at -1.67..-1.59 m -- and fails for
        #   seq 08, whose ground climbs -1.65 -> +5.63 m in 40 frames and
        #   +45.7 m over the sequence. `MapEngine` tracks a moving datum and
        #   adds it back on readout; this harness had none, so world-absolute
        #   heights press against the ceiling on any sequence with relief.
        #
        #   ONE datum for the whole run, not a moving one, and deliberately:
        #   a constant offset cancels in every DIFFERENCE the map computes --
        #   slope, step, curb height, pothole depth -- so §7.1 and §7.4 are
        #   untouched by it. Only the comparison against M*, which is
        #   world-absolute, needs it added back, and `metrics` does that from
        #   `gm.z_datum_m`. A moving datum would not cancel and would put a
        #   spurious step between any two cells last seen at different times.
        #
        #   `height_m` rather than a doctored `pts`: an earlier version of this
        #   fix overwrote `pts[:, 2]` with the world z, which also corrupted the
        #   RANGE that `scatter` computes from the same array for the
        #   measurement-variance weighting. Height and geometry come from
        #   different frames here and each is now named.
        agg = scatter(gm, pts[static], learning_ids(np.asarray(labels)[static]),
                      np.asarray(ground, dtype=bool)[static],
                      points_world_m=world[static],
                      height_m=world[static][:, 2] - gm.z_datum_m)
        fuse(gm.soa, agg, gm.thresholds)

        # Traversability before the gate: the gate consults the hazard bits,
        # so computing it after would gate on the PREVIOUS frame's geometry.
        _update_traversability(gm)
        fired = gate.apply(gm, agg.cells, vehicle_speed_ms=speed,
                           thresholds=gm.thresholds)
        stats.gate_fired += fired["fired"]
        stats.gate_acquired += fired["acquired"]
        stats.gate_refused += fired["refused"]
        stats.gate_released += fired["released"]
        stats.frames += 1

    _update_traversability(gm)
    return stats


def _update_traversability(gm) -> None:
    rings = [(slice(r.offset, r.offset + r.side * r.side), r.side)
             for r in gm.allocation.rings]
    traversability.update(gm.soa, gm.schedule, rings, gm.thresholds)


@dataclass
class Result:
    """One schedule's scorecard. Deliberately plain data: the thing that
    formats a table must not be the thing that computes it, or a formatting
    change becomes a number change."""

    schedule_name: str
    frames: int
    bytes_allocated: int
    logical_cells: int
    rmse_cm: dict
    coarsening: dict
    iou: dict
    fill: dict
    coverage: dict = field(default_factory=dict)

    def rows(self):
        for ring in sorted(self.rmse_cm):
            c = self.coarsening[ring]
            yield {"ring": ring, "rmse_cm": self.rmse_cm[ring], "rho": c["rho"],
               "il_cm": c["il_cm"], "bias_cm": c["bias_cm"],
               "mean_bias_cm": c.get("mean_bias_cm"),
               "above_frac": c.get("above_frac"),
               "spread_cm": c["spread_cm"], "n": c["n"],
               "iou": self.iou[ring], "fill": self.fill[ring],
               "cov": self.coverage.get(ring, float("nan"))}


def evaluate(gm: GridMap, reference: ReferenceMap, frames: int = 0) -> Result:
    """Every §9 metric for one map, per ring."""
    return Result(
        schedule_name=gm.schedule.name,
        frames=frames,
        bytes_allocated=bytes_allocated(gm.allocation),
        logical_cells=gm.allocation.logical_cells,
        rmse_cm=metrics.height_rmse_per_ring(gm, reference),
        coarsening=metrics.coarsening_ratio_per_ring(gm, reference),
        iou=metrics.occupancy_iou_per_ring(gm, reference),
        fill=metrics.fill_rate_per_ring(gm, reference),
        coverage=metrics.footprint_coverage_per_ring(gm, reference),
    )


def format_result(result: Result, schedule) -> str:
    """The per-ring table. One row per ring, because a single aggregate number
    hides the entire claim: error is SUPPOSED to grow with range."""
    head = (f"{result.schedule_name}  --  {result.frames} frames, "
            f"{result.logical_cells:,} logical cells, "
            f"{result.bytes_allocated / 1e6:.2f} MB allocated")
    cols = (f"{'ring':>4} {'cell':>6} {'reach':>7} {'cells':>8} {'RMSE':>8} "
            f"{'|bias|':>7} {'mean_b':>7} {'spread':>7} {'IL':>7} {'rho':>6} "
            f"{'cov':>6} "
            f"{'IoU':>6} {'fill':>6}")
    lines = [head, "", cols, "-" * len(cols)]

    def fmt(v, w, p=2):
        # None as well as nan: a ring with no comparable cell returns a dict
        # without the newer keys, and "--" is the honest rendering of both.
        if v is None or np.isnan(v):
            return f"{'--':>{w}}"
        return f"{v:>{w}.{p}f}"

    for r in result.rows():
        ring = schedule.rings[r["ring"]]
        lines.append(
            f"{r['ring']:>4} {ring.cell_m * 100:>5.0f}c {ring.half_width_m:>6.0f}m "
            f"{r['n']:>8,} {fmt(r['rmse_cm'], 8)} {fmt(r['bias_cm'], 7)} "
            f"{fmt(r.get('mean_bias_cm'), 7)} "
            f"{fmt(r['spread_cm'], 7)} {fmt(r['il_cm'], 7)} {fmt(r['rho'], 6)} "
            f"{fmt(r['cov'], 6)} {fmt(r['iou'], 6)} {fmt(r['fill'], 6)}"
        )
    lines += [
        "",
        "RMSE, |bias|, mean_b, spread, IL in cm against M*. rho = IL/spread (§9.3):",
        "  |bias| is RMS and mean_b is the SIGNED mean. RMS alone cannot tell",
        "  'systematically high' from 'randomly scattered', and on real data",
        "  those are different defects: seq 07 ring 1 reads |bias| 20.7 with a",
        "  mean of +4.0 (dispersion), ring 2 reads 36.0 with a mean of +23.5",
        "  (a real offset that grows with range).",
        "  rho ~ 1  coarsening cost only the terrain's own sub-cell variability",
        "  rho >> 1 the estimate is biased beyond that -- schedule too aggressive",
        "cells = cells the ring still SERVES (not every cell in its window: the",
        "        window is square, only its annulus is ever written, and the",
        "        interior holds stale far-range values a finer ring now answers",
        "        for), with an observed reference footprint AND >1 reference",
        "        return. Everything else is dropped, not scored as agreement.",
        "cov   = median fraction of each cell's k x k reference footprint that M*",
        "        observed. Read rho against it: at cov 0.02 the spread rho divides",
        "        by is estimated from ~1 sub-cell in 64. See metrics.py.",
    ]
    return "\n".join(lines)


def compare(schedules, scans_factory, reference: ReferenceMap,
            thresholds=None) -> list:
    """Run several schedules against one reference. The ablation, in one call.

    `scans_factory()` must return a FRESH iterator each time -- the same scans
    in the same order for every schedule, or the comparison measures which
    schedule got the better half of the sequence.
    """
    results = []
    for name in schedules:
        s = load(name) if isinstance(name, str) else name
        gm = build_gridmap(s, thresholds)
        frames = run_sequence(gm, scans_factory()).frames
        results.append((s, gm, evaluate(gm, reference, frames)))
    return results


def _nanmean(values) -> float:
    """np.nanmean over an all-nan list warns and returns nan; a ring nobody
    drove through is the ordinary case here, not an anomaly worth a warning."""
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def memory_vs_regret_row(result: Result, regret=None) -> dict:
    """One point of the Day-4 headline curve. Math §8.2.

    x is memory, y is R(S). The curve has a knee and the schedule should sit
    at it: "below 8.9 MB the plan is unchanged -- regret is exactly zero --
    and above the knee it degrades measurably."

    `regret` is a `plan_regret.Regret`. Its `unknown_fraction` travels with it
    on purpose: zero regret along a mostly-unknown path says the sequence was
    too short to fill the map, not that the coarsening was free, and the two
    numbers are only meaningful side by side.
    """
    rings = sorted(result.rmse_cm)
    finite = [result.rmse_cm[r] for r in rings if not np.isnan(result.rmse_cm[r])]
    return {
        "schedule": result.schedule_name,
        "megabytes": result.bytes_allocated / 1e6,
        "logical_cells": result.logical_cells,
        "worst_ring_rmse_cm": max(finite) if finite else float("nan"),
        "mean_rho": _nanmean([result.coarsening[r]["rho"] for r in rings]),
        "regret": None if regret is None else regret.regret,
        "frechet_m": None if regret is None else regret.frechet_m,
        "unknown_fraction": None if regret is None else regret.unknown_fraction,
        "blocked_on_reference": None if regret is None else regret.blocked_on_reference,
    }
