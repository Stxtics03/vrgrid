"""Dashboard controls: ghost toggle, blind cone, schedule selector. [JP]

`get_display_points()` is the ghost-removal swap point -- it filters is_moving()
points from the rendered cloud, and its body is the only thing that changes when
the grid's transient layer is the source. The blind-cone radius and the schedule
selector are read from `configs/` through `vrgrid.dash._config` (no rerun).
"""

import numpy as np
import pytest

pytest.importorskip("rerun")

from vrgrid.dash._config import (
    available_schedules,
    blind_cone_radius_m,
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
