"""Ground segmentation. [JP]

Patchwork++ (pypatchworkpp) on raw sensor-frame points, with a semantic-class
proxy as the fallback and the sanity reference.
"""

import numpy as np
import pytest
from vrgrid.perception import ground as ground_mod
from vrgrid.perception.ground import (
    _HAVE_PATCHWORKPP,
    GROUND_CLASSES,
    GROUND_METHOD_FALLBACK,
    GROUND_METHOD_PATCHWORKPP,
    ground_from_semantics,
    reset_fallback_warning,
    segment_ground,
    segment_ground_or_fallback,
)
from vrgrid.perception.loader import (
    _label_path,
    _velodyne_path,
    load_labels,
    load_velodyne_scan,
    verify_sequence_exists,
)
from vrgrid.perception.semantics import semantic_labels

_HAS_DATA = verify_sequence_exists("00") and _velodyne_path("00", 43).exists()
needs_pw = pytest.mark.skipif(not _HAVE_PATCHWORKPP, reason="pypatchworkpp not installed")
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="KITTI sequence 00 not present")


# --- semantic-class proxy (always available) -------------------------------


def test_ground_from_semantics_picks_the_ground_classes():
    labels = np.array([8, 9, 10, 11, 16, 0, 12, 17, -1], dtype=np.int32)
    #                  road park sidew og  terr car bld pole ign
    got = ground_from_semantics(labels)
    assert got.tolist() == [True, True, True, True, True, False, False, False, False]


def test_ground_classes_are_the_walkable_surfaces():
    assert GROUND_CLASSES == {8, 9, 10, 11, 16}


def test_ground_from_semantics_shape_and_dtype():
    out = ground_from_semantics(np.zeros(13, dtype=np.int32))
    assert out.shape == (13,) and out.dtype == bool


# --- segment_ground_or_fallback: the method tag + the loud warning ---------


def test_or_fallback_reports_which_method_ran():
    pts = np.random.default_rng(0).normal(size=(400, 4)).astype(np.float64)
    pts[:, 2] -= 1.6
    sem = np.full(400, 8, dtype=np.int32)  # all "road"
    reset_fallback_warning()
    _, method = segment_ground_or_fallback(pts, sem)
    expected = GROUND_METHOD_PATCHWORKPP if _HAVE_PATCHWORKPP else GROUND_METHOD_FALLBACK
    assert method == expected


def test_or_fallback_warns_loudly_once_when_patchworkpp_is_missing(monkeypatch):
    """The signal this whole helper exists to make unmissable."""
    monkeypatch.setattr(ground_mod, "_HAVE_PATCHWORKPP", False)
    reset_fallback_warning()
    sem = np.array([8, 8, 12, 0, 16], dtype=np.int32)
    pts = np.zeros((5, 4))

    with pytest.warns(RuntimeWarning, match="semantic-class FALLBACK"):
        mask, method = segment_ground_or_fallback(pts, sem, use_patchworkpp=True)

    assert method == GROUND_METHOD_FALLBACK
    assert mask.tolist() == [True, True, False, False, True]

    # once per process: the second call is silent
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        segment_ground_or_fallback(pts, sem, use_patchworkpp=True)


def test_or_fallback_deliberate_opt_out_is_still_flagged(monkeypatch):
    monkeypatch.setattr(ground_mod, "_HAVE_PATCHWORKPP", True)
    reset_fallback_warning()
    sem = np.array([8, 16, 12], dtype=np.int32)
    with pytest.warns(RuntimeWarning, match="opted out"):
        _, method = segment_ground_or_fallback(
            np.zeros((3, 4)), sem, use_patchworkpp=False)
    assert method == GROUND_METHOD_FALLBACK


# --- Patchwork++ ----------------------------------------------------------


@needs_pw
def test_segment_ground_returns_index_aligned_bool_mask():
    pts = np.random.default_rng(0).normal(size=(5000, 4)).astype(np.float32)
    pts[:, 2] -= 1.5  # push some below the sensor
    mask = segment_ground(pts)
    assert mask.shape == (5000,) and mask.dtype == bool


@needs_pw
@needs_data
def test_patchworkpp_split_agrees_with_semantic_classes():
    pts = load_velodyne_scan(_velodyne_path("00", 43))
    sem = semantic_labels(load_labels(_label_path("00", 43)))
    ground = segment_ground(pts)

    assert ground.shape == (len(pts),)
    assert 0.2 < ground.mean() < 0.7  # a street scene is part ground

    def frac_ground(cls):
        m = sem == cls
        return ground[m].mean() if m.sum() > 100 else None

    # walkable surfaces: almost all ground
    for cls in (8, 9, 10):  # road, parking, sidewalk
        f = frac_ground(cls)
        assert f is not None and f > 0.90, f"class {cls}: {f:.2%} ground"

    # vertical structure: almost none ground
    for cls in (0, 12, 15, 17):  # car, building, trunk, pole
        f = frac_ground(cls)
        assert f is not None and f < 0.15, f"class {cls}: {f:.2%} ground"

    # overall agreement with the semantic proxy, on labelled points
    valid = sem >= 0
    agreement = (ground[valid] == ground_from_semantics(sem)[valid]).mean()
    assert agreement > 0.90, f"agreement {agreement:.2%}"
