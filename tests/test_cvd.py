"""Colour-vision-deficiency guards for the dashboard palettes. [JP]

Simulation + Delta-E per dashboard/cvd.py (Machado 2009 matrices, CIELAB
Delta-E 1976). Delta-E >= 12 == distinguishable; < 6 == effectively identical.

Colours come from dashboard/palettes.py, which imports no rerun -- these tests
check numbers, not the viewer, and must run in CI without the `[dash]` extra.
"""

import numpy as np
import pytest
from vrgrid.dash.cvd import (
    SIM,
    _palettes,
    delta_e,
    ghost_vs_class_min_delta_e,
    min_delta_e,
    simulate,
)

SAFE = 12.0


# --- the simulator itself ---------------------------------------------------


def test_normal_is_identity_and_greyscale_is_cvd_invariant():
    rgb = np.array([[10, 200, 90], [255, 0, 128], [90, 90, 90]], dtype=np.uint8)
    assert np.array_equal(simulate(rgb, "normal"), rgb)
    for v in (0, 64, 128, 192, 255):
        grey = [v, v, v]
        for kind in SIM:
            assert np.allclose(simulate(grey, kind), grey, atol=2), (v, kind)


# --- palettes that must be CVD-safe ---------------------------------------


@pytest.mark.parametrize(
    "name", ["class (groups)", "motion", "ground", "intensity/reflectivity"]
)
def test_palette_is_colourblind_safe(name):
    de, (kind, a, b) = min_delta_e(_palettes()[name])
    assert de >= SAFE, f"{name}: {a} vs {b} under {kind} -> Delta-E {de:.1f}"


def test_group_palette_covers_all_19_classes_exactly_once():
    from vrgrid.dash.palettes import _GROUP_LUT, GROUP_MEMBERS

    members = [c for names in GROUP_MEMBERS.values() for c in names]
    assert len(members) == 20 and len(set(members)) == 20  # 19 classes + "unlabeled"
    assert set(_GROUP_LUT.tolist()) == set(range(7))  # every group is used


def test_ghost_highlight_is_distinct_from_every_class_colour():
    de, closest = ghost_vs_class_min_delta_e()
    assert de >= SAFE, f"ghost highlight collides with {closest} (Delta-E {de:.1f})"


def test_ghost_highlight_is_distinct_from_every_group_colour():
    from vrgrid.dash.palettes import GHOST_RGB, GROUP_NAMES, GROUP_RGB

    for i, name in enumerate(GROUP_NAMES):
        de = min(delta_e(simulate(GHOST_RGB, k), simulate(GROUP_RGB[i], k))
                 for k in ["normal", *SIM])
        assert de >= SAFE, f"ghost vs {name}: Delta-E {de:.1f}"


def test_ghost_is_not_red_or_green():
    from vrgrid.dash.palettes import GHOST_RGB

    r, g, b = GHOST_RGB
    assert not (r > 180 and g < 90 and b < 90), "pure red -> collides with vegetation under deuteranopia"
    assert not (g > 140 and r < 90 and b < 120), "pure green"


# --- class map: documents the known, accepted limitation ------------------


def test_semantickitti_class_map_limitation_is_unchanged():
    """The 19-class map is the SemanticKITTI standard -- kept for recognisability.
    It is NOT colourblind-safe (19 saturated categories exceed any safe
    palette). This pins the worst pair so a change to the LUT is noticed; it is
    a regression sentinel, not a safety assertion."""
    de, (_kind, a, b) = min_delta_e(_palettes()["class (semantickitti)"])
    assert de < 3.0
    assert {a, b} == {"person", "traffic-sign"}  # both near-pure blue, collide even in normal vision
