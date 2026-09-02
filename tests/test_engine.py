"""The map back end as one frame loop, and the Gate 3 claim as an assertion.
[Shrestha]

`test_a_departed_car_leaves_no_ghost` is the requirement — the five seconds of
demo the whole project is judged on. `test_without_cleanup_the_ghost_remains`
is the negative control that proves the first one is testing the engine and
not the scene: the same frames, the same assertions, `ghost_removal=False`,
and the ghost must survive. Without the pair, a test that passes because the
car's cells were never occupied in the first place looks exactly like a test
that passes because the cleanup works.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from vrgrid.cell import OCC_OCCUPIED
from vrgrid.grid.fusion import CLASS_MAX, occupancy_state, unpack_class
from vrgrid.grid.schedule import load
from vrgrid.run.engine import Z_DATUM_STEP_M, MapEngine, class_ids_fit

SENSOR_H = 1.73          # docs/frames.md: the sensor sits this far above the vehicle
WALL_X = 30.0            # a static wall the beams return from
CAR_X, CAR_Y = 15.0, 0.0  # a car parked between the sensor and the wall


GROUND_R = 12.0          # the ground disc stops short of the car, on purpose


def _scene(rng):
    """The static half of the world, generated ONCE and reused every frame.

    Re-randomising the returns per frame would touch different cells each time
    and the occupied set would grow without bound — which is a property of the
    generator, not of the engine, and it would drown the signal this test is
    looking for.

    **The ground stops at 12 m and the car sits at 15 m.** The first version of
    this scene ran the ground disc out to 45 m, under the car, and only 178 of
    379 car cells ever cleared. That was correct behaviour being measured
    wrongly: a cell holding both car returns and ground returns is still
    occupied after the car leaves, because the ground is really there. Keeping
    the two apart is what makes "the ghost is gone" a statement about ghosts.

    The wall spans -4 to +2 m so it subtends every elevation the car does. A
    wall that stopped at the car's lowest beam would leave those beams
    returning from nothing, and NO_RETURN must never clear anything — the
    cells would survive for a correct reason and the test would read as a
    cleanup failure.
    """
    n_g, n_w = 6000, 9000
    r = rng.uniform(4.0, GROUND_R, n_g)
    a = rng.uniform(-np.pi, np.pi, n_g)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n_g, -SENSOR_H)])
    wall = np.column_stack([np.full(n_w, WALL_X),
                            rng.uniform(-8.0, 8.0, n_w),
                            rng.uniform(-4.0, 2.0, n_w)])
    return ground, wall


def _car(rng, n=1500):
    """A box between the sensor and the wall. It occludes the wall while it is
    there, and when it leaves the beams reach the wall — which is exactly the
    evidence eq (32) needs to clear the cells it left behind."""
    return np.column_stack([rng.uniform(CAR_X - 1.0, CAR_X + 1.0, n),
                            rng.uniform(CAR_Y - 1.0, CAR_Y + 1.0, n),
                            rng.uniform(-1.5, -0.2, n)])


def _frame(index, points, ground_mask, elevation_m=0.0):
    """A stand-in for `run.__main__.PerceptionFrame`, built the way the real
    one is: the range image comes from JP's projector over the same points, so
    the image and the cloud cannot disagree.

    `elevation_m` puts the whole rig on a hill. The SENSOR-frame points do not
    move -- a car 15 m ahead is 15 m ahead whatever the altitude -- so this
    lifts only `points_world` and `vehicle_xyz_world`, which is exactly the
    difference a climbing sequence presents and the one the map has to absorb.
    """
    ri_mod = pytest.importorskip("vrgrid.perception.range_image")
    p4 = np.column_stack([points, np.full(len(points), 0.5)])
    image, inverse = ri_mod.project(p4)
    return SimpleNamespace(
        index=index,
        points_sensor=p4,
        points_world=points + np.array([0.0, 0.0, SENSOR_H + elevation_m]),
        pose=np.eye(4)[:3],
        vehicle_xyz_world=np.array([0.0, 0.0, elevation_m]),
        semantic=np.zeros(len(points), np.int8),
        moving=np.zeros(len(points), bool),
        ground=ground_mask,
        reflectivity8=np.full(len(points), 100, np.uint8),
        range_image=image,
        inverse_index=inverse,
    )


def _sequence(rng, present_for: int, total: int, elevation_m=0.0):
    """The car is there for `present_for` frames, then gone. Same static
    returns throughout."""
    ground, wall = _scene(rng)
    car = _car(rng)
    for i in range(total):
        parts = [ground, wall] + ([car] if i < present_for else [])
        points = np.vstack(parts)
        mask = np.zeros(len(points), bool)
        mask[:len(ground)] = True
        yield _frame(i, points, mask, elevation_m), len(ground) + len(wall)


def _run(ghost_removal, present_for=3, total=12, seed=0, elevation_m=0.0):
    rng = np.random.default_rng(seed)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, ghost_removal=ghost_removal)
    car_slots, counters = None, []
    for frame, n_static in _sequence(rng, present_for, total, elevation_m):
        counters.append(engine.step(frame))
        if car_slots is None:
            # The cells the car itself occupies, taken from the engine's own
            # binning so the test cannot disagree with it about geometry.
            pts = frame.points_sensor
            idx = engine.bin(pts[n_static:, 0], pts[n_static:, 1],
                             pts[n_static:, 0], pts[n_static:, 1] )
            car_slots = np.unique(idx[idx >= 0])
    return engine, car_slots, counters


def _occupied(engine, slots):
    state = occupancy_state(engine.handle.grid, engine.thresholds)
    return int(np.count_nonzero(state[slots] == OCC_OCCUPIED))


def test_a_departed_car_leaves_no_ghost():
    """GATE 3, as an assertion rather than a screenshot.

    The car is present for three frames and gone for nine. Its cells are
    occupied while it is there; once it leaves, every beam that used to stop
    on it returns from the wall 15 m further out, so eq (32) sees through
    those cells and `apply_miss` walks their log-odds down.
    """
    engine, car_slots, counters = _run(ghost_removal=True)

    assert len(car_slots) > 20, "the car has to occupy a real number of cells"
    assert any(c.cleared > 0 for c in counters), "the cleanup never fired at all"

    left = _occupied(engine, car_slots)
    assert left <= 0.25 * len(car_slots), (
        f"{left} of {len(car_slots)} car cells are still occupied nine frames "
        "after the car left — that is a ghost trail")


def test_without_cleanup_the_ghost_remains():
    """The negative control. Same frames, same assertions, cleanup off.

    If this passed too, the first test would be measuring the scene rather
    than the engine — the car's cells might simply never have been occupied.
    """
    engine, car_slots, counters = _run(ghost_removal=False)

    assert all(c.cleared == 0 for c in counters), "cleanup ran with it disabled"
    left = _occupied(engine, car_slots)
    assert left >= 0.75 * len(car_slots), (
        f"only {left} of {len(car_slots)} car cells survived with the cleanup "
        "off — something other than §10.4 is removing them")


def test_the_guard_protects_the_wall():
    """A cell with a return in the current scan is never cleared. Without this
    the cleanup eats fences, poles and sign posts within a few frames — and
    the wall here is the fence."""
    engine, _, counters = _run(ghost_removal=True, present_for=3, total=12)

    wall_slots = engine.bin(np.array([WALL_X]), np.array([0.0]),
                            np.array([WALL_X]), np.array([0.0]))
    assert wall_slots[0] >= 0
    state = occupancy_state(engine.handle.grid, engine.thresholds)
    assert state[wall_slots[0]] == OCC_OCCUPIED, "the cleanup ate the wall"
    assert sum(c.protected for c in counters) > 0, (
        "the guard never fired, so this test would pass without it")


def test_the_frame_loop_allocates_almost_nothing():
    """`No allocation in the frame loop` is the invariant the memory claim
    rests on, and this asserts it of the WHOLE step.

    It used to subtract two allocations in `src/grid` that were not this
    file's to fix -- `occupancy_state` at 8.19 MB a call and `ring_of` at
    ~7 MB per 120,000-point sweep -- and its own docstring named the weakness
    that created: "once grid is fixed, keep passing while a megabyte crept
    back in here". Both are fixed (Gate 3, item 2), so the subtraction is gone
    and this is a flat cap on the real thing.

    The cap is deliberately close to the measurement. A frame that allocated
    one more full-grid int64 temporary would add 7.28 MB and a per-sweep
    float64 one would add ~1 MB; there is no room for either. What remains is
    small per-call bookkeeping -- numpy's fixed casting buffer, the
    `np.flatnonzero` of the occupied set, a few Python objects.
    """
    import tracemalloc

    def peak(fn, reps=4):
        fn()
        tracemalloc.start()
        try:
            worst = 0
            for _ in range(reps):
                tracemalloc.reset_peak()
                before = tracemalloc.get_traced_memory()[0]
                fn()
                worst = max(worst, tracemalloc.get_traced_memory()[1] - before)
        finally:
            tracemalloc.stop()
        return worst

    rng = np.random.default_rng(0)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000)
    frames = [f for f, _ in _sequence(rng, 3, 6)]
    for f in frames[:2]:
        engine.step(f)                       # warm up

    step = peak(lambda: engine.step(frames[3]))
    assert step < 2_000_000, (
        f"engine.step() allocates {step:,} B a frame -- something on the "
        "frame path is not preallocated"
    )


def test_the_two_grid_allocations_stay_fixed():
    """⚑ Regression guard for Gate 3, item 2, asserted at the source.

    `occupancy_state` allocated 8.19 MB per call in full-grid int64
    temporaries -- `np.where` picking between two Python ints chooses int64,
    and one int64 array over 910,000 slots is 7.28 MB on its own, for a uint8
    answer. `ring_of` allocated 6.96 MB per 120,000-point sweep in seven
    float64 temporaries. Together they were 15.15 MB a frame, and the frame
    loop called both every frame.

    Neither is visible in a latency table, which is why this is a test and not
    a note in a doc.
    """
    import tracemalloc

    from vrgrid.grid.fusion import new_occupancy_scratch, occupancy_state
    from vrgrid.grid.lattice import new_bin_scratch, ring_of_into

    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000)
    grid = engine.handle.grid
    n_slots = grid["log_odds"].size
    occ_out = np.zeros(n_slots, np.uint8)
    occ_scratch = new_occupancy_scratch(n_slots)

    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, 40_000)
    y = rng.uniform(-100, 100, 40_000)
    bin_scratch = new_bin_scratch(len(x), engine.sched)
    level = bin_scratch["level"][:len(x)]

    def peak(fn):
        for _ in range(3):
            fn()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        fn()
        out = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        return out

    occ = peak(lambda: occupancy_state(grid, engine.thresholds,
                                       out=occ_out, scratch=occ_scratch))
    ring = peak(lambda: ring_of_into(x, y, engine.sched, 0.0, level, bin_scratch))

    assert occ < 16_000, f"occupancy_state allocates {occ:,} B over {n_slots:,} slots"
    assert ring < 16_000, f"ring_of_into allocates {ring:,} B over {len(x):,} points"


def test_the_whole_label_set_now_fuses_and_raw_ids_still_raise():
    """Was `test_nineteen_class_labels_are_refused_with_a_useful_message`.

    The candidate is 5 bits since 1 Sep, so a 19-class frame is no longer an
    error to be accommodated -- it is the normal case, and the engine must
    fuse it without clipping. What still has to be refused is a RAW label id
    reaching the map: raw ids run to 259 and are a different numbering, so
    seeing one means the learning map was never applied. Clipping that would
    turn `moving-car` into a static class and weld every passing car into the
    elevation layer.
    """
    assert class_ids_fit(np.array([0, 15, 17, 18, 19]))
    assert class_ids_fit(np.array([CLASS_MAX]))
    assert not class_ids_fit(np.array([0, 18, CLASS_MAX + 1]))
    assert not class_ids_fit(np.array([252]))          # raw moving-car

    rng = np.random.default_rng(0)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000, max_candidates=80_000)
    frame, _ = next(iter(_sequence(rng, 1, 1)))

    # pole (18) -- one of the two classes the refinement gate exists for, and
    # one a 4-bit candidate could never hold
    frame.semantic = np.full(len(frame.points_sensor), 18, np.int8)
    engine.step(frame)                                  # no raise, no clip

    slots = engine.occupied_slots()
    assert slots.size, "the frame fused nothing, so this asserts nothing"
    cand, _ = unpack_class(engine.handle.grid["semantic_class"][slots])
    assert set(np.unique(cand).tolist()) == {18}, (
        "the class came back as something other than pole -- the byte is being "
        "packed or unpacked at the wrong width somewhere"
    )

    frame.semantic = np.full(len(frame.points_sensor), 100, np.int8)
    with pytest.raises(ValueError, match="5-bit candidate"):
        engine.step(frame)



# --- elevation: the vertical band has to follow the vehicle ------------------


@pytest.mark.parametrize("elevation_m", [0.0, -5.8, 6.0, 12.0, 39.0])
def test_the_ghost_clears_at_any_vehicle_elevation(elevation_m):
    """Gate 3 again, with the whole rig on a hill.

    Heights used to be clamped to a WORLD-ABSOLUTE [-2, +6] m band, so a
    vehicle above the ceiling saw every nearby cell's height saturate while the
    sensor sat tens of metres higher. `visibility_cleanup` documents its inputs
    as vehicle-frame, computed a viewing angle far outside the sensor's FOV,
    found nothing in view and cleared nothing: on SemanticKITTI seq 08 that was
    2,304 of 4,071 frames with ghost removal silently inert.

    The elevations here are the ones that mattered. -5.8 m is seq 07's floor,
    which saturates the BOTTOM of the band and is the case a naive "subtract
    ego-z in the cleanup" fix breaks -- a stored -2.0 m minus an ego of -5.8 m
    reads as +3.8 m and leaves through the top of the image, taking the flat
    sequence the demo runs on from working to inert. 6.0 m is the old ceiling,
    12.0 m the measured point where clearing stopped entirely, and 39.0 m the
    top of seq 08's climb.
    """
    engine, car_slots, counters = _run(ghost_removal=True, elevation_m=elevation_m)

    assert len(car_slots) > 20, "the car has to occupy a real number of cells"
    assert any(c.cleared > 0 for c in counters), (
        f"the cleanup never cleared a single cell at {elevation_m} m -- it is "
        "inert at this elevation, not merely worse")

    left = _occupied(engine, car_slots)
    assert left <= 0.25 * len(car_slots), (
        f"{left} of {len(car_slots)} car cells are still occupied nine frames "
        f"after the car left, with the vehicle at {elevation_m} m")


def test_the_band_follows_the_vehicle_rather_than_the_world_datum():
    """The mechanism behind the test above, asserted directly.

    Storage stays 8 m tall -- widening it would have fixed the symptom and cost
    the report its memory claim, since `dashboard/_config.py` counts the dense
    baseline's voxels over exactly this extent. So the datum moves instead, and
    a cell at the vehicle's own feet must read as being at its feet, not at the
    band's edge, however high the vehicle is.
    """
    engine, _, _ = _run(ghost_removal=True, elevation_m=39.0)

    assert engine.z_datum == pytest.approx(39.0, abs=Z_DATUM_STEP_M)
    slots, _, _, z = engine.occupied_cells()
    assert len(slots), "nothing occupied, so this asserts nothing"
    # `occupied_cells` reads out in the WORLD frame, so the ground the vehicle
    # is standing on comes back near its world elevation -- not pinned to the
    # +6 m the old absolute clamp would have saturated it to.
    assert np.median(z) > 30.0, (
        f"median occupied height {np.median(z):.1f} m with the vehicle at 39 m "
        "-- heights are still being clamped against the world datum")


def test_the_visibility_cap_reports_what_it_dropped():
    """A cap that silently drops cells is the dangerous half of §10.4.

    Truncated cells keep their occupancy and are never tested against the range
    image, so a ghost among them is permanent -- and `cleared` cannot reveal it,
    because `cleared` only counts what was offered. On sequence 07 the
    provisional cap of 150,000 dropped 164,442 cells at peak, 52.3% of the
    occupied set, and nothing said so.
    """
    from vrgrid.run.engine import StepCounters

    c = StepCounters(index=0, points=0, binned=0, cells_touched=0,
                     occupied=200_000, tested=150_000, cleared=0, protected=0,
                     out_of_view=0, truncated=50_000)
    assert c.truncated_fraction == 0.25

    held = StepCounters(index=0, points=0, binned=0, cells_touched=0,
                        occupied=120_000, tested=120_000, cleared=7, protected=1,
                        out_of_view=0)
    assert held.truncated == 0, "the default must be 'nothing was dropped'"
    assert held.truncated_fraction == 0.0

    empty = StepCounters(index=0, points=0, binned=0, cells_touched=0,
                         occupied=0, tested=0, cleared=0, protected=0,
                         out_of_view=0)
    assert empty.truncated_fraction == 0.0, "no divide by zero on an empty frame"
