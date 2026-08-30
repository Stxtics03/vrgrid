"""Reflectivity normalisation -- math appendix §10.3 eq (31). [JP]

rho_hat = I * r^2 / max(cos(theta_inc), 0.1), then one byte.
"""

import numpy as np
import pytest

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

_LANE_FRAME = 4431  # 1669 lane-marking points in seq 00
_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", _LANE_FRAME).exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="KITTI seq 00 not present -- set VRGRID_DATA_ROOT")


def _plane_range_image(a: float, b: float, c: float, x0: float = 4.0, intensity: float = 0.25):
    """A range image whose central block lies exactly on z = a*x + b*y + c."""
    ri = np.full((H, W, 5), np.nan, dtype=np.float32)
    vs, us = np.arange(20, 40), np.arange(200, 320)
    for v in vs:
        for u in us:
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
# 1. synthetic geometry with a known answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a,b,c", [(0.0, 0.0, -2.0), (0.3, 0.1, -2.0), (-0.2, 0.4, -1.5)])
def test_incidence_cos_matches_analytic_plane_normal(a, b, c):
    ri = _plane_range_image(a, b, c)
    cos_inc, has_normal = incidence_cos(ri)
    v, u = 30, 260
    assert has_normal[v, u]
    assert cos_inc[v, u] == pytest.approx(_analytic_cos_inc(a, b, c, v, u), abs=1e-4)


def test_rho_hat_is_I_r2_over_clamped_cos():
    a, b, c = 0.3, 0.1, -2.0
    ri = _plane_range_image(a, b, c, intensity=0.4)
    res = normalise(ri)
    v, u = 30, 260
    r = float(ri[v, u, 0])
    cos_true = _analytic_cos_inc(a, b, c, v, u)
    expected = 0.4 * r**2 / max(cos_true, COS_INC_MIN)
    assert res.rho_hat[v, u] == pytest.approx(expected, rel=1e-4)
    assert res.rho8[v, u] == np.clip(round(expected / RHO_SATURATION * 255.0), 0, 255)


def test_head_on_surface_has_cos_inc_near_one():
    # a near-horizontal plane viewed steeply: put the block right under the sensor
    ri = _plane_range_image(0.0, 0.0, -2.0, x0=0.05)
    cos_inc, _ = incidence_cos(ri)
    v, u = 30, 205
    assert cos_inc[v, u] > 0.9


# --------------------------------------------------------------------------
# 2. grazing incidence -- clamp, flag, never divide by zero
# --------------------------------------------------------------------------


def test_grazing_incidence_is_clamped_and_flagged_not_infinite():
    # horizontal plane, block 30-48 m out, sensor 2 m up -> cos_inc ~ 0.05
    ri = _plane_range_image(0.0, 0.0, -2.0, x0=30.0, intensity=0.25)
    with np.errstate(all="raise"):  # no div-by-zero / overflow anywhere
        res = normalise(ri)
    v, u = 30, 260
    assert res.cos_inc[v, u] < COS_INC_MIN
    assert res.flags[v, u] & FLAG_GRAZING
    r = float(ri[v, u, 0])
    assert res.rho_hat[v, u] == pytest.approx(0.25 * r**2 / COS_INC_MIN, rel=1e-4)  # clamp, not raw cos
    assert np.isfinite(res.rho_hat[v, u])
    assert np.isfinite(res.rho_hat[np.isfinite(res.rho_hat)]).all()


def test_empty_and_edge_pixels_get_no_normal_flag_and_zero_rho():
    ri = _plane_range_image(0.1, 0.0, -2.0)
    res = normalise(ri)
    # a pixel far from the filled block -> empty -> no normal, rho8 = 0
    assert res.flags[0, 0] & FLAG_NO_NORMAL
    assert res.rho8[0, 0] == 0
    assert np.isnan(res.rho_hat[0, 0])
    # top/bottom rows can never get a ring-direction difference
    assert (res.flags[0] & FLAG_NO_NORMAL).all()
    assert (res.flags[-1] & FLAG_NO_NORMAL).all()


def test_output_contract():
    res = normalise(_plane_range_image(0.2, 0.1, -2.0))
    assert res.rho8.shape == (H, W) and res.rho8.dtype == np.uint8
    assert res.flags.shape == (H, W) and res.flags.dtype == np.uint8
    # no-normal pixels are always zero; grazing-flagged pixels keep a real value
    no_normal = (res.flags & FLAG_NO_NORMAL) > 0
    assert res.rho8[no_normal].max(initial=0) == 0
    assert np.isnan(res.rho_hat[no_normal]).all()


# --------------------------------------------------------------------------
# 3. real scan -- lane paint separates from plain road
# --------------------------------------------------------------------------


@needs_data
def test_lane_marking_reflectivity_separates_from_road():
    pts = load_velodyne_scan(_velodyne_path("00", _LANE_FRAME))
    raw = load_labels(_label_path("00", _LANE_FRAME)) & 0xFFFF
    ri, inv = project(pts)
    res = normalise(ri)

    lbl = np.full(inv.shape, -1)
    m = inv >= 0
    lbl[m] = raw[inv[m]]

    road = (lbl == 40) & res.valid       # raw id 40 = road
    lane = (lbl == 60) & res.valid       # raw id 60 = lane-marking
    assert road.sum() > 500 and lane.sum() > 100

    road_med = float(np.median(res.rho8[road]))
    lane_med = float(np.median(res.rho8[lane]))
    intensity = ri[:, :, 4]
    print(
        f"\n[frame {_LANE_FRAME}] valid px: road {road.sum()}, lane {lane.sum()}\n"
        f"  raw intensity median : road {np.median(intensity[road]):.3f}  "
        f"lane {np.median(intensity[lane]):.3f}\n"
        f"  rho8 median          : road {road_med:.0f}  lane {lane_med:.0f}  "
        f"(ratio {lane_med / max(road_med, 1):.2f})\n"
        f"  rho8 mean            : road {res.rho8[road].mean():.1f}  "
        f"lane {res.rho8[lane].mean():.1f}\n"
        f"  no-normal {np.mean((res.flags & FLAG_NO_NORMAL) > 0) * 100:.0f}%  "
        f"grazing {np.mean((res.flags & FLAG_GRAZING) > 0) * 100:.0f}%"
    )
    # lane paint reads clearly brighter than plain asphalt
    assert lane_med > road_med * 1.2


@needs_data
def test_scatter_to_points_round_trips():
    pts = load_velodyne_scan(_velodyne_path("00", _LANE_FRAME))
    ri, inv = project(pts)
    res = normalise(ri)
    rho8_pts, flags_pts = scatter_to_points(res, inv)

    assert rho8_pts.shape == (len(pts),) and rho8_pts.dtype == np.uint8
    filled = inv >= 0
    src = inv[filled]
    assert np.array_equal(rho8_pts[src], res.rho8[filled])
    # points that never projected carry the no-normal flag and zero reflectivity
    projected = np.zeros(len(pts), dtype=bool)
    projected[src] = True
    assert (flags_pts[~projected] & FLAG_NO_NORMAL).all()
    assert rho8_pts[~projected].max(initial=0) == 0
