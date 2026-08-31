"""Range-image projection: reversibility and FOV binning. [JP]

Two guarantees this module has to make:

1. Reversibility -- every filled pixel maps back to its EXACT source point, so
   labels inferred on the image can be scattered back onto the point cloud
   without a nearest-neighbour lookup.
2. No silent point loss on the azimuth axis, and a documented, counted policy
   for points outside the vertical FOV (clamp to the edge ring, never drop --
   see the module docstring).
"""

import numpy as np
import pytest
from vrgrid.perception.loader import (
    _velodyne_path,
    load_velodyne_scan,
    verify_sequence_exists,
)
from vrgrid.perception.range_image import bin_widths, project

SENSOR_CFG = {
    "num_rings": 64,
    "num_azimuth": 512,
    "phi_min_deg": -24.8,
    "phi_max_deg": 2.0,
}

_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", 0).exists()
needs_data = pytest.mark.skipif(
    not _HAS_DATA, reason="KITTI sequence 00 not present -- set VRGRID_DATA_ROOT"
)


def _synthetic_points(bearings, ranges, intensities=None):
    """bearings: list of (azimuth_rad, elevation_rad). Returns (N, 4)."""
    az = np.array([b[0] for b in bearings])
    el = np.array([b[1] for b in bearings])
    rng = np.asarray(ranges, dtype=np.float64)
    x = rng * np.cos(el) * np.cos(az)
    y = rng * np.cos(el) * np.sin(az)
    z = rng * np.sin(el)
    inten = np.ones_like(rng) if intensities is None else np.asarray(intensities)
    return np.column_stack([x, y, z, inten]).astype(np.float32)


# --------------------------------------------------------------------------
# 1. reversibility -- byte-exact
# --------------------------------------------------------------------------


@needs_data
def test_inverse_index_is_byte_exact_reversible():
    points = load_velodyne_scan(_velodyne_path("00", 0))  # (N, 4) float32
    range_image, inv = project(points, SENSOR_CFG)

    filled = inv >= 0
    assert filled.sum() > 10_000, "projection produced almost no pixels"

    src = inv[filled]
    # every filled pixel points at a real, in-range source index
    assert src.min() >= 0 and src.max() < len(points)
    # no source point is claimed by two pixels
    assert len(np.unique(src)) == len(src)

    # the stored geometry IS the source point, bit for bit
    assert np.array_equal(range_image[filled, 1:4], points[src, :3])
    assert np.array_equal(range_image[filled, 4], points[src, 3])

    # the stored range is the norm of that exact source point (float32 tol)
    recomputed = np.linalg.norm(points[src, :3].astype(np.float64), axis=1)
    assert np.allclose(range_image[filled, 0], recomputed, atol=1e-3)

    # empty pixels carry no geometry
    assert np.isnan(range_image[~filled, 0]).all()


@needs_data
def test_closest_point_wins_per_pixel():
    """Two points on the same bearing at different ranges -> the near one is kept."""
    az, el = 0.3, -0.05
    near = _synthetic_points([(az, el)], [5.0])
    far = _synthetic_points([(az, el)], [40.0])
    points = np.vstack([far, near])  # far listed first on purpose

    range_image, inv = project(points, SENSOR_CFG)
    hit = np.argwhere(inv >= 0)
    assert len(hit) == 1
    v, u = hit[0]
    assert inv[v, u] == 1  # the "near" point, index 1
    assert range_image[v, u, 0] == pytest.approx(5.0, abs=1e-3)


# --------------------------------------------------------------------------
# 2. binning correctness
# --------------------------------------------------------------------------


def test_bin_widths_derived_from_fov_and_size():
    d_theta, d_phi = bin_widths(SENSOR_CFG)
    assert d_theta == pytest.approx(2 * np.pi / 512)
    assert d_phi == pytest.approx(np.deg2rad(2.0 - (-24.8)) / 64)


def test_known_bearings_land_in_expected_pixels():
    d_theta, d_phi = bin_widths(SENSOR_CFG)
    phi_max = np.deg2rad(2.0)
    cases = [
        (-np.pi + 0.5 * d_theta, phi_max - 0.5 * d_phi, 0, 0),       # top-left
        (0.0, phi_max - 0.5 * d_phi, 256, 0),                        # top-centre
        (-np.pi + 0.5 * d_theta, np.deg2rad(-24.8) + 0.5 * d_phi, 0, 63),  # bottom-left
    ]
    for az, el, exp_u, exp_v in cases:
        pts = _synthetic_points([(az, el)], [12.0])
        _, inv = project(pts, SENSOR_CFG)
        hit = np.argwhere(inv >= 0)
        assert len(hit) == 1, f"bearing {az:.3f},{el:.3f} produced {len(hit)} pixels"
        v, u = hit[0]
        assert (u, v) == (exp_u, exp_v), f"expected ({exp_u},{exp_v}) got ({u},{v})"


def test_full_azimuth_sweep_drops_nothing():
    """512 points, one per column, all inside the vertical FOV -> 512 pixels,
    zero clamped, zero lost. The old fixed-degree binning kept only ~28%."""
    d_theta, _ = bin_widths(SENSOR_CFG)
    az = -np.pi + (np.arange(512) + 0.5) * d_theta
    el = np.full(512, np.deg2rad(-10.0))
    pts = _synthetic_points(list(zip(az, el)), np.full(512, 20.0))

    _, inv, stats = project(pts, SENSOR_CFG, return_stats=True)
    assert stats["n_input"] == 512
    assert stats["n_clamped_below"] == 0 and stats["n_clamped_above"] == 0
    assert stats["n_pixels_filled"] == 512
    assert sorted(inv[inv >= 0].tolist()) == list(range(512))


def test_azimuth_wraps_at_plus_pi():
    """azimuth = +pi must fold into column 0, not overflow to column W."""
    pts = _synthetic_points([(np.pi, 0.0), (-np.pi, 0.0)], [10.0, 10.0])
    _, inv, _stats = project(pts, SENSOR_CFG, return_stats=True)
    cols = np.argwhere(inv >= 0)[:, 1]
    assert cols.max() < SENSOR_CFG["num_azimuth"]
    assert 0 in cols


# --------------------------------------------------------------------------
# 3. out-of-FOV policy: clamp to the edge ring, count it, never drop
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:range_image.project")
def test_out_of_fov_points_are_clamped_not_dropped():
    _d_theta, d_phi = bin_widths(SENSOR_CFG)
    az = 0.0
    above = np.deg2rad(2.0) + 5 * d_phi   # 5 rows above phi_max
    below = np.deg2rad(-24.8) - 5 * d_phi  # 5 rows below phi_min
    inside = np.deg2rad(-10.0)
    pts = _synthetic_points([(az, above), (az, below), (az + 0.05, inside)],
                            [10.0, 10.0, 10.0])

    _range_image, inv, stats = project(pts, SENSOR_CFG, return_stats=True)

    assert stats["n_clamped_above"] == 1
    assert stats["n_clamped_below"] == 1
    # nothing was dropped -- all three points appear
    assert stats["n_pixels_filled"] == 3
    assert set(inv[inv >= 0].tolist()) == {0, 1, 2}

    # clamped to the correct edge rows: above -> row 0, below -> row H-1
    row_of = {idx: tuple(np.argwhere(inv == idx)[0]) for idx in range(3)}
    assert row_of[0][0] == 0
    assert row_of[1][0] == SENSOR_CFG["num_rings"] - 1


def test_out_of_fov_warning_only_above_threshold(recwarn):
    _d_theta, _d_phi = bin_widths(SENSOR_CFG)
    inside = _synthetic_points(
        [(a, np.deg2rad(-10.0)) for a in np.linspace(-3, 3, 90)], np.full(90, 15.0)
    )
    outside = _synthetic_points(
        [(a, np.deg2rad(20.0)) for a in np.linspace(-3, 3, 10)], np.full(10, 15.0)
    )
    project(np.vstack([inside, outside]), SENSOR_CFG)  # 10% out of FOV -> no warn
    assert not any("vertical FOV" in str(w.message) for w in recwarn.list)

    outside_big = _synthetic_points(
        [(a, np.deg2rad(20.0)) for a in np.linspace(-3, 3, 40)], np.full(40, 15.0)
    )
    with pytest.warns(UserWarning, match="vertical FOV"):
        project(np.vstack([inside, outside_big]), SENSOR_CFG)  # ~31% -> warn


@needs_data
def test_real_scan_out_of_fov_fraction_is_small():
    """Sanity: on a real HDL-64E scan only a few percent land outside the
    nominal FOV, and it's dominated by points above phi_max (tall structure /
    top rings), not below."""
    points = load_velodyne_scan(_velodyne_path("00", 0))
    _, _, stats = project(points, SENSOR_CFG, return_stats=True)
    assert stats["out_of_fov_fraction"] < 0.10
    assert stats["n_clamped_above"] > stats["n_clamped_below"]
    assert stats["fill_fraction"] > 0.5  # was ~0.15 with the broken binning
