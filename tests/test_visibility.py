"""Visibility cleanup, math §10.4. [Shrestha]

The pair that matters is `test_a_pole_survives_a_hundred_frames` and
`test_without_the_guard_a_pole_is_eaten`. The first is the requirement; the
second is the negative control that proves the first is testing something.
"""

import numpy as np
import pytest
from vrgrid.gpu.visibility import (
    NO_RETURN,
    Sensor,
    apply_miss,
    clear_tolerance_m,
    new_visibility_scratch,
    spherical_project,
    visibility_cleanup,
)

SHAPE = (64, 1800)          # HDL-64E: 64 beams, 0.2 deg azimuth
SENSOR = Sensor()
FAR = 60.0                  # the wall everything else returns from


def empty_image(fill=FAR):
    return np.full(SHAPE, fill, np.float64)


def cells_at(x, y, z=0.0):
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    z = np.broadcast_to(np.asarray(z, float), x.shape).copy()
    return x, y, z


# --- equation (32) -----------------------------------------------------------


def test_a_beam_returning_from_beyond_clears_the_cell():
    """The beam came back from 60 m through a cell claiming something at 20 m."""
    x, y, z = cells_at(20.0, 0.0)
    r = visibility_cleanup(x, y, z, empty_image(), sensor=SENSOR)
    assert r.see_through.tolist() == [True]
    assert r.cleared == 1


def test_a_beam_stopping_in_front_does_not_clear():
    """Occlusion. The beam stopped at 10 m, so it says nothing at all about a
    cell at 20 m -- it never got there."""
    x, y, z = cells_at(20.0, 0.0)
    r = visibility_cleanup(x, y, z, empty_image(10.0), sensor=SENSOR)
    assert r.cleared == 0


def test_a_beam_returning_from_the_cell_itself_does_not_clear():
    """The observation agrees with the map. Nothing to clean up."""
    x, y, z = cells_at(20.0, 0.0)
    u, v, rng, _ = spherical_project(x, y, z, SHAPE, SENSOR)
    img = empty_image()
    img[v, u] = rng
    assert visibility_cleanup(x, y, z, img, sensor=SENSOR).cleared == 0


def test_a_pixel_with_no_return_clears_nothing():
    """A beam that returned nothing may have been absorbed, hit glass, or gone
    to the sky. Comparing against inf would clear the whole map."""
    x, y, z = cells_at([20.0, 30.0, 40.0], [0.0, 1.0, -1.0])
    r = visibility_cleanup(x, y, z, empty_image(NO_RETURN), sensor=SENSOR)
    assert r.cleared == 0


def test_a_cell_below_the_field_of_view_is_not_cleared():
    """Nothing was observed there, so nothing is known. The blind cone is
    unknown, never free."""
    x, y, z = cells_at(1.0, 0.0, -20.0)   # steeply below the lowest beam
    r = visibility_cleanup(x, y, z, empty_image(), sensor=SENSOR)
    assert r.out_of_view == 1
    assert r.cleared == 0


# --- the tolerance -----------------------------------------------------------


def test_delta_widens_with_range_as_the_doc_requires():
    """math §10.4: 3*sigma(r), not a hand-tuned constant. A fixed band is too
    permissive near and too tight far, and 'too tight far' means clearing real
    structure at range."""
    near = clear_tolerance_m(10.0, SENSOR, floor_m=0.0)
    far = clear_tolerance_m(100.0, SENSOR, floor_m=0.0)
    assert far > 5 * near
    assert far == pytest.approx(3 * 0.175, rel=0.1)   # 17.5 cm sigma at 100 m


def test_the_config_value_acts_as_a_floor_not_a_ceiling():
    """Pose and registration error do not shrink with range; sensor noise does.
    The floor covers the first, 3*sigma covers the second, and the larger wins."""
    assert clear_tolerance_m(5.0, SENSOR, floor_m=0.30) == pytest.approx(0.30)
    assert clear_tolerance_m(100.0, SENSOR, floor_m=0.30) > 0.30


def test_a_cell_inside_the_tolerance_band_is_not_cleared():
    """Just beyond the cell is not evidence -- that is what delta is for."""
    x, y, z = cells_at(20.0, 0.0)
    u, v, rng, _ = spherical_project(x, y, z, SHAPE, SENSOR)
    delta = float(clear_tolerance_m(rng, SENSOR)[0])
    img = empty_image()
    img[v, u] = rng + 0.5 * delta
    assert visibility_cleanup(x, y, z, img, sensor=SENSOR).cleared == 0
    img[v, u] = rng + 1.5 * delta
    assert visibility_cleanup(x, y, z, img, sensor=SENSOR).cleared == 1


# --- the guard, which is the whole point -------------------------------------


def _thin_pole(x_m=20.0, n=12):
    """A pole one cell wide at `x_m`, as a column of cells. Its cells straddle
    two azimuth columns of the range image, which is what makes it fragile:
    the beam beside the pole reports the wall behind it."""
    y = np.full(n, 0.0)
    z = np.linspace(0.2, 3.0, n)
    x = np.full(n, x_m)
    return x, y, z


def _image_hitting_the_pole(x, y, z):
    """This scan sees the pole: the pixels the pole occupies return its range.
    One column is left reporting the wall, standing in for the beam that
    passed beside it -- which is the geometry that eats thin structures."""
    img = empty_image()
    u, v, r, _ = spherical_project(x, y, z, SHAPE, SENSOR)
    img[v, u] = r
    img[v[::2], u[::2]] = FAR          # every other beam missed and saw the wall
    return img


def test_a_pole_survives_a_hundred_frames():
    """The mandatory guard, math §10.4: never clear a cell with a return in the
    current scan. Fences, poles and sign posts are the things this saves, and
    they are exactly the things a robot must not drive into."""
    x, y, z = _thin_pole()
    has_return = np.ones(len(x), bool)          # the scan hit the pole
    survivors = len(x)
    for _ in range(100):
        r = visibility_cleanup(x, y, z, _image_hitting_the_pole(x, y, z),
                               has_return_now=has_return, sensor=SENSOR)
        survivors -= r.cleared
    assert survivors == len(x), "the guard did not protect a pole the scan hit"
    assert r.protected > 0, (
        "no cell was ever protected, so this test would pass with the guard "
        "removed -- see the negative control below")


def test_without_the_guard_a_pole_is_eaten():
    """Negative control. Without it the test above proves nothing: it would
    pass on an implementation that simply never clears anything."""
    x, y, z = _thin_pole()
    r = visibility_cleanup(x, y, z, _image_hitting_the_pole(x, y, z),
                           has_return_now=np.ones(len(x), bool),
                           protect_current_returns=False, sensor=SENSOR)
    assert r.cleared > 0, "the scene is not actually fragile; the test is vacuous"
    assert r.protected == 0


def test_the_guard_only_protects_cells_seen_this_scan():
    """It is a guard, not an off switch. A ghost -- a cell with no return in
    the current scan -- is precisely what cleanup exists to remove."""
    x, y, z = cells_at([20.0, 21.0], [0.0, 2.0])
    guard = np.array([True, False])
    r = visibility_cleanup(x, y, z, empty_image(), has_return_now=guard,
                           sensor=SENSOR)
    assert r.see_through.tolist() == [False, True]
    assert (r.protected, r.cleared) == (1, 1)


# --- frames ------------------------------------------------------------------


def test_range_is_measured_from_the_sensor_not_the_vehicle_origin():
    """Vehicle frame is x forward, y left, z up with the sensor at 1.73 m. Using
    the vehicle origin biases every comparison and still looks plausible --
    docs/frames.md calls this the most common silent bug in the project."""
    _, _, r, _ = spherical_project(np.array([0.0]), np.array([0.0]),
                                   np.array([0.0]), SHAPE, SENSOR)
    assert r[0] == pytest.approx(SENSOR.height_m)

    _, _, r, _ = spherical_project(np.array([10.0]), np.array([0.0]),
                                   np.array([SENSOR.height_m]), SHAPE, SENSOR)
    assert r[0] == pytest.approx(10.0)


def test_projection_covers_the_image_without_going_out_of_bounds():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.uniform(-80, 80, n)
    y = rng.uniform(-80, 80, n)
    z = rng.uniform(-1.5, 5.0, n)
    u, v, _, in_view = spherical_project(x, y, z, SHAPE, SENSOR)
    assert u.min() >= 0 and u.max() < SHAPE[1]
    assert v.min() >= 0 and v.max() < SHAPE[0]
    assert in_view.mean() > 0.5


# --- the frame-loop invariant ------------------------------------------------


def test_cleanup_with_scratch_allocates_little_per_frame():
    """Same rule as scatter: preallocated at startup, nothing grown in the loop."""
    import tracemalloc

    rng = np.random.default_rng(1)
    n = 40_000
    x = rng.uniform(-60, 60, n)
    y = rng.uniform(-60, 60, n)
    z = rng.uniform(-1.0, 3.0, n)
    img = empty_image()
    scratch = new_visibility_scratch(n)

    def frame():
        visibility_cleanup(x, y, z, img, sensor=SENSOR, scratch=scratch)

    frame()
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    for _ in range(3):
        frame()
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    # Measured 0.34 B/cell. The bound is tight on purpose: at 8 B/cell a single
    # full-width index temporary comes back, which is exactly the np.take
    # regression `new_visibility_scratch` documents.
    assert peak < 4 * n, f"{peak:,} B per frame for {n:,} cells"


def test_more_candidates_than_scratch_is_refused():
    scratch = new_visibility_scratch(10)
    x, y, z = cells_at(np.full(20, 20.0), np.zeros(20))
    with pytest.raises(ValueError, match="scratch"):
        visibility_cleanup(x, y, z, empty_image(), scratch=scratch)


# --- handing the mask to occupancy -------------------------------------------


def test_apply_miss_clamps_so_the_map_can_change_its_mind():
    """math §10.1: an unclamped cell with 500 free observations needs 500
    occupied ones to register a new obstacle."""
    log_odds = np.zeros(4, np.int8)
    cells = np.arange(4)
    mask = np.array([True, True, False, False])
    for _ in range(100):
        apply_miss(log_odds, cells, mask, log_odds_miss=-2, clamp=(-64, 63))
    assert log_odds[:2].tolist() == [-64, -64]
    assert log_odds[2:].tolist() == [0, 0]


def test_the_scratch_path_matches_the_readable_one():
    """`_delta_into` is `clear_tolerance_m` written in out= parameters. Two
    versions of one formula is how a fast path quietly stops matching the
    definition it was derived from, so they are pinned to each other here."""
    from vrgrid.gpu.visibility import _delta_into

    r = np.geomspace(0.5, 150.0, 500)
    fast = np.zeros_like(r)
    _delta_into(fast, r, np.zeros_like(r), SENSOR, floor_m=0.30)
    assert np.allclose(fast, clear_tolerance_m(r, SENSOR, floor_m=0.30))
