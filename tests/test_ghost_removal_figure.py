"""The Gate 3 figure's scene and its trail counter. [Shrestha]

Drawing is not tested -- matplotlib is an optional `report` extra and CI does
not install it. What is tested is the part that decides what the picture
CLAIMS: `trail_mask` counts the cells the figure colours red, and if it counted
the wrong strip the two panels would still look different and the caption would
still be wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("vrgrid.perception.range_image")

from ghost_removal_figure import (
    CAR_END_X,
    CAR_HALF,
    CAR_START_X,
    CAR_Y,
    GROUND_R,
    WALL_X,
    build_scene,
    trail_mask,
)


def test_the_trail_strip_holds_nothing_static():
    """The count only means "ghosts" because nothing else lives in that lane.

    The ground disc stops at 11 m and the wall stands at 34 m, so the strip
    between them along y = 0 is empty until the car drives through it. If a
    later change to the scene puts static geometry there, this fails and the
    figure's red count silently stops meaning what its legend says.
    """
    ground, wall = build_scene(np.random.default_rng(0))
    for name, pts in (("ground", ground), ("wall", wall)):
        inside = trail_mask(pts[:, 0], pts[:, 1], CAR_END_X)
        assert not inside.any(), f"{name} intrudes into the trail strip"
    assert GROUND_R < CAR_START_X - CAR_HALF
    assert WALL_X > CAR_END_X + CAR_HALF


def test_the_strip_excludes_the_car_itself():
    """`car now` is drawn as an outline, not counted as a ghost — the cells
    the car currently occupies are not a trail, they are a car."""
    car_x = 20.0
    x = np.array([car_x, car_x - 0.5, car_x - 3.0])
    y = np.full(3, CAR_Y)
    assert not trail_mask(x, y, car_x)[:2].any()
    assert trail_mask(x, y, car_x)[2]


def test_the_strip_is_only_the_lane():
    """A cell three metres off the lane is not in the car's swept path, and
    counting it would fold the static map into the ghost number."""
    x = np.full(3, 20.0)
    y = np.array([CAR_Y, CAR_Y + 3.0, CAR_Y - 3.0])
    got = trail_mask(x, y, CAR_END_X)
    assert got[0] and not got[1] and not got[2]


def test_nothing_before_the_car_started_counts():
    """The car enters at 14 m. Cells nearer than that were never swept, so
    they cannot be its ghosts however occupied they are."""
    x = np.array([CAR_START_X - 5.0, CAR_START_X + 2.0])
    y = np.full(2, CAR_Y)
    got = trail_mask(x, y, CAR_END_X)
    assert not got[0] and got[1]
