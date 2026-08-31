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
from vrgrid.grid.fusion import occupancy_state
from vrgrid.grid.lattice import ring_of
from vrgrid.grid.schedule import load
from vrgrid.run.engine import MapEngine, class_ids_fit

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


def _frame(index, points, ground_mask):
    """A stand-in for `run.__main__.PerceptionFrame`, built the way the real
    one is: the range image comes from JP's projector over the same points, so
    the image and the cloud cannot disagree."""
    ri_mod = pytest.importorskip("vrgrid.perception.range_image")
    p4 = np.column_stack([points, np.full(len(points), 0.5)])
    image, inverse = ri_mod.project(p4)
    return SimpleNamespace(
        index=index,
        points_sensor=p4,
        points_world=points + np.array([0.0, 0.0, SENSOR_H]),
        pose=np.eye(4)[:3],
        vehicle_xyz_world=np.zeros(3),
        semantic=np.zeros(len(points), np.int8),
        moving=np.zeros(len(points), bool),
        ground=ground_mask,
        reflectivity8=np.full(len(points), 100, np.uint8),
        range_image=image,
        inverse_index=inverse,
    )


def _sequence(rng, present_for: int, total: int):
    """The car is there for `present_for` frames, then gone. Same static
    returns throughout."""
    ground, wall = _scene(rng)
    car = _car(rng)
    for i in range(total):
        parts = [ground, wall] + ([car] if i < present_for else [])
        points = np.vstack(parts)
        mask = np.zeros(len(points), bool)
        mask[:len(ground)] = True
        yield _frame(i, points, mask), len(ground) + len(wall)


def _run(ghost_removal, present_for=3, total=12, seed=0):
    rng = np.random.default_rng(seed)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, ghost_removal=ghost_removal)
    car_slots, counters = None, []
    for frame, n_static in _sequence(rng, present_for, total):
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
    rests on, and this asserts it of THIS FILE rather than of the whole stack.

    Two functions in `src/grid` allocate per call, and neither is mine to fix:

        occupancy_state   8.19 MB   full-grid temporaries over 910,000 slots
        ring_of           0.87 MB   at this scene's 15,000 points; ~7 MB at 120,000

    So the test measures a step, subtracts what those two cost on the same
    frame, and requires the remainder to be small. Written as a flat cap on
    the step it would either fail for someone else's reason or, once grid is
    fixed, keep passing while a megabyte crept back in here.
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

    frame = frames[3]
    pts = frame.points_sensor
    step = peak(lambda: engine.step(frame))
    grid_side = (peak(lambda: occupancy_state(engine.handle.grid, engine.thresholds))
                 + peak(lambda: ring_of(pts[:, 0], pts[:, 1], engine.sched)))

    assert step - grid_side < 1_000_000, (
        f"the engine itself allocates {step - grid_side:,} B a frame on top of "
        f"grid's {grid_side:,} B — something here is not preallocated")


def test_nineteen_class_labels_are_refused_with_a_useful_message():
    """The engine must not quietly clip. SemanticKITTI is 19-class and
    fusion's candidate is 4 bits, so this is every real frame — the caller
    has to be told, not accommodated."""
    assert class_ids_fit(np.array([0, 15]))
    assert not class_ids_fit(np.array([0, 16, 18]))

    rng = np.random.default_rng(0)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000, max_candidates=80_000)
    frame, _ = next(iter(_sequence(rng, 1, 1)))
    frame.semantic = np.full(len(frame.points_sensor), 18, np.int8)

    with pytest.raises(ValueError, match="4-bit candidate"):
        engine.step(frame)

    engine.clip_class_ids = True
    engine.step(frame)          # opt in explicitly, and it runs
