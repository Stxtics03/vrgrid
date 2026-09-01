"""Dashboard controls: ghost toggle, blind cone, schedule selector. [JP]

`get_display_points()` is the ghost-removal swap point -- it filters is_moving()
points from the rendered cloud, and its body is the only thing that changes when
the grid's transient layer is the source. The blind-cone radius and the schedule
selector are read from `configs/` through `vrgrid.dash._config` (no rerun), so
those tests run in CI without the `[dash]` extra.
"""

import numpy as np
import pytest

pytest.importorskip("rerun")

from vrgrid.dash._config import (
    DENSE_VOXEL_BYTES,
    available_schedules,
    blind_cone_radius_m,
    dense_3d_baseline,
    grid_memory_stats,
    memory_overlay_markdown,
    schedule_legend_markdown,
)
from vrgrid.dash.pipeline_view import COLOR_BY, get_display_points
from vrgrid.grid.schedule import CONFIG_DIR, load_thresholds
from vrgrid.grid.schedule import load as load_schedule
from vrgrid.perception.loader import _velodyne_path, verify_sequence_exists

_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", 10).exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")


class _Frame:
    """Minimal PerceptionFrame stand-in with a known motion mask."""

    def __init__(self, n=200, n_moving=17, seed=0):
        rng = np.random.default_rng(seed)
        self.points_sensor = rng.random((n, 4)).astype(np.float32)
        self.points_world = (rng.random((n, 3)) * 40 - 20).astype(np.float32)
        self.semantic = rng.integers(-1, 19, n)
        self.ground = rng.random(n) > 0.5
        self.reflectivity8 = rng.integers(0, 256, n).astype(np.uint8)
        self.moving = np.zeros(n, dtype=bool)
        self.moving[rng.choice(n, n_moving, replace=False)] = True


# --------------------------------------------------------------------------
# synthetic -- exact filtering
# --------------------------------------------------------------------------


def test_ghost_removal_on_drops_exactly_the_moving_points():
    f = _Frame(n=200, n_moving=17)
    xyz, colors = get_display_points(f, ghost_removal=True)
    assert len(xyz) == 200 - 17
    assert len(colors) == len(xyz)
    # every kept point is a static one, and all static points are kept
    kept = {tuple(p) for p in xyz}
    assert kept == {tuple(p) for p in f.points_world[~f.moving]}
    assert not (kept & {tuple(p) for p in f.points_world[f.moving]})


def test_ghost_removal_off_shows_everything():
    f = _Frame(n=200, n_moving=17)
    xyz, colors = get_display_points(f, ghost_removal=False)
    assert len(xyz) == 200 and len(colors) == 200
    assert np.array_equal(np.sort(xyz, axis=0), np.sort(f.points_world, axis=0))


def test_static_points_identical_with_toggle_either_way():
    f = _Frame(n=300, n_moving=25)
    on, _ = get_display_points(f, ghost_removal=True)
    off, _ = get_display_points(f, ghost_removal=False)
    static = f.points_world[~f.moving]
    on_set = {tuple(p) for p in on}
    off_set = {tuple(p) for p in off}
    assert {tuple(p) for p in static} <= on_set
    assert {tuple(p) for p in static} <= off_set
    assert on_set < off_set  # ON is a strict subset of OFF


@pytest.mark.parametrize("color_by", COLOR_BY)
def test_colours_stay_aligned_with_points_for_every_layer(color_by):
    f = _Frame(n=150, n_moving=12)
    for gr in (True, False):
        xyz, colors = get_display_points(f, ghost_removal=gr, color_by=color_by)
        assert colors.shape == (len(xyz), 3) and colors.dtype == np.uint8


def test_frame_with_no_moving_points_is_a_noop():
    f = _Frame(n=100, n_moving=0)
    on, _ = get_display_points(f, ghost_removal=True)
    off, _ = get_display_points(f, ghost_removal=False)
    assert len(on) == len(off) == 100


# --------------------------------------------------------------------------
# real scan -- frame 10 has a moving motorcyclist + pedestrian
# --------------------------------------------------------------------------


@needs_data
def test_frame_10_ghost_toggle_removes_the_moving_objects():
    from vrgrid.run.__main__ import iter_pipeline

    frame = list(iter_pipeline("00", max_frames=11))[10]
    n_moving = int(frame.moving.sum())
    total = len(frame.points_sensor)
    assert 40 < n_moving < 120, f"frame 10 moving count {n_moving} (expected ~66)"

    on_xyz, _ = get_display_points(frame, ghost_removal=True)
    off_xyz, _ = get_display_points(frame, ghost_removal=False)

    assert len(off_xyz) == total
    assert len(on_xyz) == total - n_moving
    # the removed set is exactly the moving points
    removed = {tuple(p) for p in off_xyz} - {tuple(p) for p in on_xyz}
    assert removed == {tuple(p) for p in frame.points_world[frame.moving].astype(np.float32)}
    # the moving objects are near the vehicle, not scattered across the map
    ghosts = frame.points_world[frame.moving]
    assert np.linalg.norm(ghosts - frame.vehicle_xyz_world, axis=1).max() < 60


# --------------------------------------------------------------------------
# blind cone -- radius read from config, the corrected 3.74 m value
# --------------------------------------------------------------------------


def test_blind_cone_radius_is_374_and_comes_from_config():
    from_config = load_thresholds()["sensor"]["blind_cone_m"]
    assert from_config == pytest.approx(3.74)
    assert blind_cone_radius_m() == pytest.approx(from_config)
    # the corrected value -- master v4 flagged the earlier 1-2 m assumption
    assert blind_cone_radius_m() > 3.0


# --------------------------------------------------------------------------
# schedule selector -- reads configs/schedule_*.yaml, no hardcoded ring sizes
# --------------------------------------------------------------------------


def test_available_schedules_are_discovered_from_config_dir():
    got = available_schedules()
    on_disk = sorted(p.stem.removeprefix("schedule_") for p in CONFIG_DIR.glob("schedule_*.yaml"))
    assert got == on_disk
    assert "5_10_20_40" in got and "5_10_50" in got


def test_schedule_legend_matches_the_config_ring_boundaries():
    md = schedule_legend_markdown("5_10_20_40")
    assert "**(active)**" in md
    for name in available_schedules():
        s = load_schedule(name)
        assert f"`{name}`" in md
        for r in s.rings:
            # half-width / cell-cm pair, straight from the yaml, appears verbatim
            assert f"{r.half_width_m:g}/{r.cell_m * 100:g}" in md
        assert f"{s.total_cells:,}" in md


def test_pipeline_view_logs_rings_from_the_passed_schedule(tmp_path):
    from vrgrid.dash.pipeline_view import PipelineView

    # both schedules build without error and use their own ring count
    for name, n_rings in [("5/10/20/40", 4), ("5/10/50", 3)]:
        s = load_schedule(name)
        assert len(s.rings) == n_rings
        PipelineView(s, spawn=False, save_path=str(tmp_path / f"{n_rings}.rrd"))


# --------------------------------------------------------------------------
# Gate 3 -- the occupied-cell surface, drawn from MapEngine.occupied_cells()
# --------------------------------------------------------------------------


def _wall_frame(index: int):
    """A minimal PerceptionFrame: a ground disc + a static wall at x = 25 m,
    built the way test_engine.py builds its scenes (range image from the same
    points, so cloud and image agree)."""
    from types import SimpleNamespace

    ri = pytest.importorskip("vrgrid.perception.range_image")
    rng = np.random.default_rng(index)
    ground = np.column_stack([
        (r := rng.uniform(3.0, 12.0, 5000)) * np.cos(a := rng.uniform(-np.pi, np.pi, 5000)),
        r * np.sin(a), np.full(5000, -1.73)])
    wall = np.column_stack([np.full(7000, 25.0), rng.uniform(-8, 8, 7000),
                            rng.uniform(-3.0, 2.0, 7000)])
    pts = np.vstack([ground, wall])
    p4 = np.column_stack([pts, np.full(len(pts), 0.4)])
    image, inverse = ri.project(p4)
    gmask = np.zeros(len(pts), bool)
    gmask[:5000] = True
    return SimpleNamespace(
        index=index, points_sensor=p4,
        points_world=pts + np.array([0.0, 0.0, 1.73]),
        pose=np.eye(4)[:3], vehicle_xyz_world=np.zeros(3),
        semantic=np.zeros(len(pts), np.int8), moving=np.zeros(len(pts), bool),
        ground=gmask, reflectivity8=np.full(len(pts), 90, np.uint8),
        range_image=image, inverse_index=inverse)


def test_pipeline_view_draws_the_engine_occupied_surface(tmp_path):
    from vrgrid.dash.pipeline_view import PipelineView, _height_ramp
    from vrgrid.run.engine import MapEngine

    sched = load_schedule("5/10/20/40")
    engine = MapEngine(sched, ghost_removal=True)
    view = PipelineView(sched, spawn=False, save_path=str(tmp_path / "map.rrd"),
                        engine=engine)

    for i in range(3):
        f = _wall_frame(i)
        engine.step(f)
        view.log_frame(f)          # must not raise -- draws world/map/occupied

    slots, _x, _y, z = engine.occupied_cells()
    assert len(slots) > 100, "the wall + ground should occupy cells"

    # per-slot cell size comes from the ring: near cells are 5 cm, the wall at
    # 25 m falls in ring 1 (10 cm) or ring 2 (20 cm) -- strictly more than one
    # distinct size, which is the foveation the surface is meant to show
    cell_m = view._cell_m_per_slot(slots)
    assert np.isclose(cell_m.min(), 0.05)          # near cells are the base 5 cm
    assert cell_m.max() > cell_m.min()             # farther rings are coarser
    assert np.unique(np.round(cell_m, 3)).size >= 2

    # colour tracks height and is a valid uint8 triple per cell
    c = _height_ramp(z)
    assert c.shape == (len(z), 3) and c.dtype == np.uint8


def test_pipeline_view_separates_occupied_free_unknown(tmp_path):
    """math §10.1 / CLAUDE.md: unknown is not free. The view must keep the
    three occupancy states on distinct entities, driven only by the engine's
    occ_state (refreshed by the occupied_cells() call log_frame already makes)."""
    from vrgrid.cell import OCC_FREE, OCC_OCCUPIED, OCC_UNKNOWN
    from vrgrid.dash.pipeline_view import PipelineView
    from vrgrid.run.engine import MapEngine

    sched = load_schedule("5/10/20/40")
    engine = MapEngine(sched, ghost_removal=True)
    view = PipelineView(sched, spawn=False, save_path=str(tmp_path / "occ.rrd"),
                        engine=engine)
    for i in range(4):
        f = _wall_frame(i)
        engine.step(f)
        view.log_frame(f)   # logs world/map/{occupied,free,unknown}, must not raise

    st = engine.occ_state
    assert set(np.unique(st)) <= {OCC_UNKNOWN, OCC_FREE, OCC_OCCUPIED}
    assert (st == OCC_OCCUPIED).sum() > 100
    # centres for the free set resolve through the engine's own inverse, no NaN
    free = np.flatnonzero(st == OCC_FREE)
    fx, fy, fz = view._centres_world(free)
    assert np.isfinite(np.concatenate([fx, fy, fz])).all()
    # never-observed slots are UNKNOWN and are NOT handed to the renderer
    obs = engine.handle.grid["obs_count"]
    assert ((st == OCC_UNKNOWN) & (obs == 0)).sum() > 0
    drawn_unknown = np.flatnonzero((st == OCC_UNKNOWN) & (obs > 0))
    assert len(drawn_unknown) < (st == OCC_UNKNOWN).sum()   # the bulk is left undrawn


def test_pipeline_view_without_engine_skips_the_surface(tmp_path):
    from vrgrid.dash.pipeline_view import PipelineView

    view = PipelineView(load_schedule("5/10/20/40"), spawn=False,
                        save_path=str(tmp_path / "n.rrd"), engine=None)
    view.log_frame(_wall_frame(0))  # no engine -> no occupied surface, no error


# --------------------------------------------------------------------------
# live memory overlay -- occupied cells * CELL_BYTES vs the dense-3D baseline
# --------------------------------------------------------------------------


def test_dense_3d_baseline_is_derived_not_a_magic_number():
    from vrgrid.cell import CELL_BYTES

    sched = load_schedule("5/10/20/40")
    d = dense_3d_baseline(sched)

    # exactly the documented formula, recomputed from the schedule
    footprint = 2.0 * sched.rings[-1].half_width_m
    lo, hi = sched.vertical_extent_m
    vertical = hi - lo
    res = sched.base_cell_m
    assert d["footprint_m"] == footprint == 200.0
    assert d["vertical_m"] == vertical == 8.0
    assert d["res_m"] == res == 0.05
    assert d["voxels"] == (footprint / res) ** 2 * (vertical / res) == 2.56e9
    assert d["bytes"] == d["voxels"] * DENSE_VOXEL_BYTES == 2.56e9   # 1 B/voxel

    # matches the report's 286x headline against the 8.94 MB logical map
    assert d["bytes"] / (sched.total_cells * CELL_BYTES) == pytest.approx(286.4, abs=0.5)


def test_grid_memory_stats_is_exactly_occupied_count_times_cell_bytes():
    from vrgrid.cell import CELL_BYTES

    sched = load_schedule("5/10/20/40")
    for n in (0, 1, 42, 187_808, 745_000):
        s = grid_memory_stats(n, sched)
        assert s["live_bytes"] == n * CELL_BYTES        # exact, no rounding
        assert s["cell_bytes"] == CELL_BYTES
        assert s["dense_bytes"] == dense_3d_baseline(sched)["bytes"]
        if n:
            assert s["ratio"] == s["dense_bytes"] / (n * CELL_BYTES)


@needs_data
def test_memory_overlay_tracks_the_real_occupied_count(tmp_path):
    """The overlay number must equal len(occupied_cells()) * CELL_BYTES for the
    frame it was logged on -- no drift between what is drawn and what is counted."""
    from vrgrid.cell import CELL_BYTES
    from vrgrid.dash.pipeline_view import PipelineView
    from vrgrid.run.__main__ import iter_pipeline
    from vrgrid.run.engine import MapEngine

    sched = load_schedule("5/10/20/40")
    engine = MapEngine(sched, ghost_removal=True)
    view = PipelineView(sched, spawn=False, save_path=str(tmp_path / "mem.rrd"),
                        engine=engine)

    seen = []
    for f in iter_pipeline("00", 20):
        engine.step(f)
        view.log_frame(f)
        n = len(engine.occupied_slots())
        assert view._last_occupied_n == n
        s = grid_memory_stats(n, sched)
        assert s["live_bytes"] == n * CELL_BYTES
        assert str(f"{n:,}") in memory_overlay_markdown(n, sched)
        seen.append((n, s["live_bytes"], s["ratio"]))

    ns = [x[0] for x in seen]
    assert ns[-1] > ns[0] > 0                     # the map fills as frames arrive
    assert all(0 < x[2] < 1e7 for x in seen)      # ratio stays a sane finite number
