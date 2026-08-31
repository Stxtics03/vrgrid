"""Reflectivity normalisation -- math appendix §10.3 eq (31). [JP]

KITTI path (default): rho_hat = I (firmware already range/incidence normalised);
byte = round(I * 255).
Raw-power path: rho_hat = I * r^2 / max(cos(theta_inc), 0.1)  -- eq (31) verbatim.
"""

import numpy as np
import pytest
from vrgrid.grid.lattice import ring_of
from vrgrid.grid.schedule import load as load_schedule
from vrgrid.perception.loader import (
    _label_path,
    _velodyne_path,
    load_labels,
    load_velodyne_scan,
    verify_sequence_exists,
)
from vrgrid.perception.range_image import project
from vrgrid.perception.reflectivity import (
    COS_INC_MIN,
    FLAG_GRAZING,
    FLAG_NO_NORMAL,
    RHO_SATURATION,
    incidence_cos,
    normalise,
    scatter_to_points,
)

H, W = 64, 512

_LANE_FRAME = 4431
# lane-marking-rich frames of seq 00, for the ring-by-ring statistics
_STAT_FRAMES = [4431, 4438, 4417, 4424, 4396, 4403, 0, 7, 126, 147, 1568, 1575]
_SCHED = load_schedule("5/10/20/40")

_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", _LANE_FRAME).exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")

# A road byte should not pile up at either rail on a near-uniform surface. 15%
# is a generous ceiling: it leaves room for genuinely bright returns (paint,
# signs) and genuinely dark ones (wet / specular) as real signal, while still
# catching a systematic scaling error -- the old `* r^2` term pinned 62% of
# ring-1 road at 255, which this threshold rejects by a wide margin.
MAX_SATURATED_FRAC = 0.15


def _plane_range_image(a: float, b: float, c: float, x0: float = 4.0, intensity: float = 0.25):
    """A range image whose central block lies exactly on z = a*x + b*y + c."""
    ri = np.full((H, W, 5), np.nan, dtype=np.float32)
    for v in range(20, 40):
        for u in range(200, 320):
            x = x0 + 0.15 * (u - 200)
            y = -1.0 + 0.15 * (v - 20)
            z = a * x + b * y + c
            r = float(np.sqrt(x * x + y * y + z * z))
            ri[v, u] = (r, x, y, z, intensity)
    return ri


def _analytic_cos_inc(a: float, b: float, c: float, v: int, u: int, x0: float = 4.0) -> float:
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    x = x0 + 0.15 * (u - 200)
    y = -1.0 + 0.15 * (v - 20)
    z = a * x + b * y + c
    beam = np.array([x, y, z])
    beam /= np.linalg.norm(beam)
    return abs(float(n @ beam))


# --------------------------------------------------------------------------
# 1. incidence-angle geometry (finite-difference normal)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a,b,c", [(0.0, 0.0, -2.0), (0.3, 0.1, -2.0), (-0.2, 0.4, -1.5)])
def test_incidence_cos_matches_analytic_plane_normal(a, b, c):
    cos_inc, has_normal = incidence_cos(_plane_range_image(a, b, c))
    v, u = 30, 260
    assert has_normal[v, u]
    assert cos_inc[v, u] == pytest.approx(_analytic_cos_inc(a, b, c, v, u), abs=1e-4)


def test_head_on_surface_has_cos_inc_near_one():
    cos_inc, _ = incidence_cos(_plane_range_image(0.0, 0.0, -2.0, x0=0.05))
    assert cos_inc[30, 205] > 0.9


# --------------------------------------------------------------------------
# 2. the two normalisation paths
# --------------------------------------------------------------------------


def test_kitti_path_is_the_intensity_byte():
    """Default: firmware already range/incidence normalised -> rho_hat = I."""
    ri = _plane_range_image(0.3, 0.1, -2.0, intensity=0.42)
    res = normalise(ri)
    v, u = 30, 260
    assert res.rho_hat[v, u] == pytest.approx(0.42, rel=1e-5)
    assert res.rho8[v, u] == round(0.42 * 255)
    # cos_inc is still computed and available even though it is not applied
    assert np.isfinite(res.cos_inc[v, u])


def test_raw_power_path_is_eq31_verbatim():
    a, b, c = 0.3, 0.1, -2.0
    ri = _plane_range_image(a, b, c, intensity=0.4)
    res = normalise(ri, range_compensated=False, incidence_compensated=False)
    v, u = 30, 260
    r = float(ri[v, u, 0])
    expected = 0.4 * r**2 / max(_analytic_cos_inc(a, b, c, v, u), COS_INC_MIN)
    assert res.rho_hat[v, u] == pytest.approx(expected, rel=1e-4)
    assert res.rho8[v, u] == np.clip(round(expected / RHO_SATURATION * 255.0), 0, 255)


def test_raw_power_grazing_is_clamped_flagged_and_finite():
    ri = _plane_range_image(0.0, 0.0, -2.0, x0=30.0, intensity=0.25)  # cos_inc ~ 0.05
    with np.errstate(all="raise"):
        res = normalise(ri, range_compensated=False, incidence_compensated=False)
    v, u = 30, 260
    assert res.cos_inc[v, u] < COS_INC_MIN
    assert res.flags[v, u] & FLAG_GRAZING
    r = float(ri[v, u, 0])
    assert res.rho_hat[v, u] == pytest.approx(0.25 * r**2 / COS_INC_MIN, rel=1e-4)
    assert np.isfinite(res.rho_hat[np.isfinite(res.rho_hat)]).all()


def test_empty_and_edge_pixels_get_no_normal_and_zero_rho():
    res = normalise(_plane_range_image(0.1, 0.0, -2.0))
    assert res.flags[0, 0] & FLAG_NO_NORMAL
    assert res.rho8[0, 0] == 0 and np.isnan(res.rho_hat[0, 0])
    assert (res.flags[0] & FLAG_NO_NORMAL).all() and (res.flags[-1] & FLAG_NO_NORMAL).all()


def test_output_contract():
    res = normalise(_plane_range_image(0.2, 0.1, -2.0))
    assert res.rho8.shape == (H, W) and res.rho8.dtype == np.uint8
    assert res.flags.shape == (H, W) and res.flags.dtype == np.uint8


# --------------------------------------------------------------------------
# 3. real scan -- per-point, ring-resolved
# --------------------------------------------------------------------------


def _ring_labelled_rho8(frame: int):
    """(rho8, raw_label, ring) for every returned pixel of one frame, KITTI path."""
    pts = load_velodyne_scan(_velodyne_path("00", frame))
    raw = load_labels(_label_path("00", frame)) & 0xFFFF
    ri, inv = project(pts)
    res = normalise(ri)
    m = inv >= 0
    x = ri[:, :, 1][m].astype(np.float64)
    y = ri[:, :, 2][m].astype(np.float64)
    return res.rho8[m].astype(np.float64), raw[inv[m]], ring_of(x, y, _SCHED)


@needs_data
def test_lane_marking_reflectivity_separates_from_road():
    """Per-point, ring 0. The pooled all-pixel median (previously ~1.46x) mixed
    label sets with different range/incidence distributions; at the per-point /
    per-cell scale fusion.py actually sees, lane paint is ~1.25-1.55x brighter
    than plain asphalt, range-stable."""
    rho8, lab, ring = _ring_labelled_rho8(_LANE_FRAME)
    r0 = ring == 0
    road = r0 & (lab == 40)
    lane = r0 & (lab == 60)
    assert road.sum() > 300 and lane.sum() > 100

    road_med, lane_med = np.median(rho8[road]), np.median(rho8[lane])
    road_mean, lane_mean = rho8[road].mean(), rho8[lane].mean()
    print(
        f"\n[frame {_LANE_FRAME}, ring 0]  road n={road.sum()}  lane n={lane.sum()}\n"
        f"  rho8 median: road {road_med:.0f}  lane {lane_med:.0f}  ratio {lane_med / road_med:.2f}\n"
        f"  rho8 mean  : road {road_mean:.0f}  lane {lane_mean:.0f}  ratio {lane_mean / road_mean:.2f}"
    )
    assert 1.15 < lane_med / road_med < 2.0
    assert lane_mean > road_mean * 1.2


@needs_data
def test_no_ring_saturates_the_reflectivity_byte():
    """The `* r^2` bug pinned 62% of ring-1 road at byte 255. With rho_hat = I
    every ring that has real road data stays well inside the byte."""
    by_ring: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    for fr in _STAT_FRAMES:
        rho8, lab, ring = _ring_labelled_rho8(fr)
        for L in range(4):
            by_ring[L].append(rho8[(ring == L) & (lab == 40)])

    print("\n  ring  cell   road px   median   mean   sat@255   zero@0")
    for L in range(4):
        road = np.concatenate(by_ring[L])
        if road.size < 500:
            print(f"  {L}     {_SCHED.rings[L].cell_m * 100:.0f}cm   {road.size:>6}   (insufficient)")
            continue
        sat = float((road >= 255).mean())
        zero = float((road <= 0).mean())
        print(
            f"  {L}     {_SCHED.rings[L].cell_m * 100:.0f}cm   {road.size:>6}   "
            f"{np.median(road):>6.0f}   {road.mean():>4.0f}   {sat * 100:>6.1f}%   {zero * 100:>5.1f}%"
        )
        assert sat < MAX_SATURATED_FRAC, f"ring {L}: {sat:.1%} of road pixels saturated at 255"


@needs_data
def test_scatter_to_points_round_trips():
    pts = load_velodyne_scan(_velodyne_path("00", _LANE_FRAME))
    ri, inv = project(pts)
    res = normalise(ri)
    rho8_pts, flags_pts = scatter_to_points(res, inv)

    assert rho8_pts.shape == (len(pts),) and rho8_pts.dtype == np.uint8
    src = inv[inv >= 0]
    assert np.array_equal(rho8_pts[src], res.rho8[inv >= 0])
    projected = np.zeros(len(pts), dtype=bool)
    projected[src] = True
    assert (flags_pts[~projected] & FLAG_NO_NORMAL).all()
    assert rho8_pts[~projected].max(initial=0) == 0
