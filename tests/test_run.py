"""End-to-end pipeline wiring. [JP]

`iter_pipeline` chains loader -> transforms -> range_image -> semantics ->
ground -> reflectivity into one PerceptionFrame per scan. The grid `scatter()`
step is not on this branch yet (Aakash's), so it is not exercised here.
"""

import numpy as np
import pytest

from vrgrid.perception.loader import _velodyne_path, verify_sequence_exists
from vrgrid.perception.transforms import sensor_to_world
from vrgrid.run.__main__ import PerceptionFrame, iter_pipeline

_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", 0).exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")


@needs_data
def test_iter_pipeline_yields_aligned_perception_frames():
    frames = list(iter_pipeline("00", max_frames=3))
    assert len(frames) == 3

    for f in frames:
        assert isinstance(f, PerceptionFrame)
        n = len(f.points_sensor)
        assert f.points_world.shape == (n, 3)
        assert f.semantic.shape == (n,) and f.semantic.min() >= -1 and f.semantic.max() <= 18
        assert f.moving.shape == (n,) and f.moving.dtype == bool
        assert f.ground.shape == (n,) and f.ground.dtype == bool
        assert f.reflectivity8.shape == (n,) and f.reflectivity8.dtype == np.uint8
        assert f.range_image.shape == (64, 512, 5)


@needs_data
def test_world_points_are_a_rigid_transform_of_the_scan():
    f = next(iter_pipeline("00", max_frames=1))
    origin_world = sensor_to_world(f.pose, sequence="00")[:3, 3]
    d_sensor = np.linalg.norm(f.points_sensor[:, :3], axis=1)
    d_world = np.linalg.norm(f.points_world - origin_world, axis=1)
    assert np.abs(d_sensor - d_world).max() < 1e-3  # rigid: ranges preserved


@needs_data
def test_ground_split_matches_semantics_roughly():
    f = next(iter_pipeline("00", max_frames=1))
    road = f.semantic == 8
    building = f.semantic == 12
    assert f.ground[road].mean() > 0.9
    assert f.ground[building].mean() < 0.15


def test_pipeline_view_builds_headless_and_colours(tmp_path):
    pytest.importorskip("rerun")
    from vrgrid.dash.pipeline_view import _COLORERS
    from vrgrid.grid import schedule as schedule_mod

    from vrgrid.dash.pipeline_view import PipelineView

    sched = schedule_mod.load("5/10/20/40")
    PipelineView(sched, spawn=False, save_path=str(tmp_path / "t.rrd"), color_by="class")

    class _F:
        points_sensor = np.random.default_rng(0).random((100, 4)).astype(np.float32)
        points_world = np.random.default_rng(1).random((100, 3)).astype(np.float32)
        semantic = np.random.default_rng(2).integers(-1, 19, 100)
        moving = np.zeros(100, dtype=bool)
        ground = np.random.default_rng(3).random(100) > 0.5
        reflectivity8 = np.random.default_rng(4).integers(0, 256, 100).astype(np.uint8)

    for name, fn in _COLORERS.items():
        c = fn(_F())
        assert c.shape == (100, 3) and c.dtype == np.uint8, name
