"""The §7.4 / §7.5 dashboard layers. [JP]

`world/map/curbs`, `world/map/potholes` and `world/map/confidence` are pure
readouts over fields `MapEngine.step` already fills -- no new computation, only
drawing. What these tests pin is the wiring that is easy to get silently wrong:
the slot->flat offset (a `Curbs.slot` is an index WITHIN a ring window, so
merging rings without the offset aliases ring 0's slot 5 onto ring 3's), the
off-by-default contract, and that the confidence ramp survives colourblindness.

⚑ `rerun` is the optional `[dash]` extra and CI does not install it, so the
  whole module skips there. The import therefore cannot sit at the top --
  `vrgrid.dash.pipeline_view` imports rerun unconditionally. It goes inside a
  fixture rather than behind a module-level `importorskip` + `# noqa: E402`,
  because whether E402 is enabled differs between our ruff and CI's: with it
  on, the bare import is an error; with it off, the `noqa` silencing it is an
  unused directive (RUF100). Importing inside the fixture is an error under
  neither, and it is the pattern `test_run.py` already uses.
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None,
    reason="rerun not installed -- the optional [dash] extra",
)


@pytest.fixture(scope="module")
def dash():
    """The dashboard symbols under test, imported once rerun is known present."""
    from vrgrid.dash.pipeline_view import (
        FEATURE_INTERVAL,
        PipelineView,
        _confidence_ramp,
    )
    from vrgrid.grid import schedule as schedule_mod
    from vrgrid.run.engine import MapEngine

    return SimpleNamespace(
        FEATURE_INTERVAL=FEATURE_INTERVAL, PipelineView=PipelineView,
        _confidence_ramp=_confidence_ramp, schedule_mod=schedule_mod,
        MapEngine=MapEngine,
    )


@pytest.fixture(scope="module")
def view_and_engine(dash):
    sched = dash.schedule_mod.load("5/10/20/40")
    engine = dash.MapEngine(sched)
    view = dash.PipelineView(sched, spawn=False, save_path=None, engine=engine,
                             features=True)
    return view, engine


# --- the confidence ramp ---------------------------------------------------


def test_confidence_ramp_shape_and_range(dash):
    c = dash._confidence_ramp(np.linspace(0.0, 1.0, 32))
    assert c.shape == (32, 3) and c.dtype == np.uint8


def test_confidence_ramp_is_ordered_in_luminance_not_hue(dash):
    """A red-to-green ramp is the one a deuteranope cannot read at all. This
    ramp has to carry its ordering in lightness so it survives every CVD type
    -- assert that directly rather than trusting the hex values."""
    lum = (dash._confidence_ramp(np.linspace(0, 1, 24))
           * np.array([0.2126, 0.7152, 0.0722])).sum(axis=1)
    assert np.all(np.diff(lum) > 0), "confidence ramp is not monotone in luminance"
    assert lum[-1] - lum[0] > 100, "ramp spans too little lightness to read"


def test_confidence_ramp_clips_out_of_range_inputs(dash):
    lo = dash._confidence_ramp(np.array([-5.0, 0.0]))
    hi = dash._confidence_ramp(np.array([1.0, 5.0]))
    assert np.array_equal(lo[0], lo[1]) and np.array_equal(hi[0], hi[1])


# --- the slot -> flat-slot lift, which is the aliasing bug -----------------


def test_ring_slices_cover_the_allocation_exactly_once(view_and_engine):
    view, engine = view_and_engine
    rings = view._ring_slices()
    assert len(rings) == len(engine.sched.rings)
    covered = 0
    prev_stop = 0
    for sl, side in rings:
        assert sl.start == prev_stop, "ring slices must be contiguous"
        assert sl.stop - sl.start == side * side
        covered += sl.stop - sl.start
        prev_stop = sl.stop
    assert covered == engine.handle.grid["obs_count"].size


def test_flat_adds_the_ring_offset_so_rings_cannot_alias(view_and_engine):
    view, engine = view_and_engine
    slot = np.array([0, 5, 17], dtype=np.int64)
    for level, layout in enumerate(engine.handle.rings):
        got = view._flat(level, slot)
        assert np.array_equal(got, slot + layout.offset)
    # the aliasing this prevents: ring 0 slot 5 and ring 3 slot 5 are different
    assert view._flat(0, slot)[1] != view._flat(3, slot)[1]


# --- the off-by-default contract -------------------------------------------


def test_features_are_off_by_default(dash):
    sched = dash.schedule_mod.load("5/10/20/40")
    v = dash.PipelineView(sched, spawn=False, save_path=None,
                          engine=dash.MapEngine(sched))
    assert v.features is False


def test_features_need_an_engine(dash):
    """`features=True` with no map back end is a no-op, not a crash: the
    detectors read the engine's SoA and there is not one."""
    sched = dash.schedule_mod.load("5/10/20/40")
    v = dash.PipelineView(sched, spawn=False, save_path=None, engine=None,
                          features=True)
    assert v.features is False
    v.log_features()   # must not raise


def test_log_features_is_a_noop_when_disabled(dash):
    sched = dash.schedule_mod.load("5/10/20/40")
    v = dash.PipelineView(sched, spawn=False, save_path=None,
                          engine=dash.MapEngine(sched), features=False)
    v.log_features()   # must not raise, and must not run the 1.1 s detector


def test_the_recompute_interval_is_not_every_frame(dash):
    """The whole reason these layers are opt-in: `features.detect` is a
    full-window neighbourhood pass measured at 1,137 ms, eleven times the
    100 ms frame budget. If this ever becomes 1, the viewer stops being
    usable and every --save recording silently triples in cost."""
    assert dash.FEATURE_INTERVAL > 1


# --- the real path, on an empty map and on a stepped one -------------------


def test_log_features_draws_on_an_empty_map_without_raising(view_and_engine):
    """Every layer has to handle "nothing detected" -- an empty map is the
    first frame of every run, and `rr.Clear` is the correct answer there."""
    view, _ = view_and_engine
    view.log_features()


def test_log_features_runs_on_a_stepped_map(view_and_engine):
    """One synthetic frame through the real engine, then the real draw path."""
    from vrgrid.perception.range_image import project
    from vrgrid.perception.transforms import SENSOR_HEIGHT_M

    view, engine = view_and_engine
    rng = np.random.default_rng(0)
    n = 4000
    pts = np.column_stack([rng.uniform(-12, 12, n), rng.uniform(-12, 12, n),
                           np.full(n, -SENSOR_HEIGHT_M)])
    p4 = np.column_stack([pts, np.full(n, 0.5)])
    image, inverse = project(p4)
    frame = SimpleNamespace(
        index=0, points_sensor=p4,
        points_world=pts + np.array([0.0, 0.0, SENSOR_HEIGHT_M]),
        pose=np.eye(4)[:3], vehicle_xyz_world=np.zeros(3),
        semantic=np.full(n, 8, np.int8), moving=np.zeros(n, bool),
        ground=np.ones(n, bool), reflectivity8=np.full(n, 100, np.uint8),
        range_image=image, inverse_index=inverse,
    )
    engine.step(frame)
    view.log_features()
