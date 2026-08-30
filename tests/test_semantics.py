"""Semantic label mapping. [JP]

Semantic class comes from the SemanticKITTI .label files, not from inference
(the FRNet port is non-functional -- see src/perception/frnet/). These tests
pin the raw-id -> 19-class map and the moving-* flag.
"""

import numpy as np
import pytest

from vrgrid.perception.loader import _label_path, load_labels
from vrgrid.perception.semantics import (
    SEMANTIC_KITTI_LABEL_MAP,
    is_moving,
    segment,
    semantic_labels,
)

_HAS_LABELS = _label_path("00", 43).exists()


def test_known_ids_map_to_expected_classes():
    raw = np.array([40, 50, 70, 10, 48, 0, 1, 99, 252], dtype=np.uint32)
    got = semantic_labels(raw)
    # road, building, vegetation, car, sidewalk, then three ignores, then
    # moving-car folded onto car
    assert got.tolist() == [8, 12, 14, 0, 10, -1, -1, -1, 0]


def test_class_19_and_unmapped_ids_become_minus_one():
    raw = np.array([0, 1, 52, 60, 300, 999, 65535], dtype=np.uint32)
    got = semantic_labels(raw)
    assert got.tolist() == [-1, -1, -1, 8, -1, -1, -1]  # 60 (lane-marking) -> road


def test_upper_bits_are_ignored():
    # the .label word packs instance id in the upper 16 bits
    raw = np.array([50 | (1234 << 16), 40 | (7 << 16)], dtype=np.uint32)
    assert semantic_labels(raw).tolist() == [12, 8]


def test_output_shape_and_dtype():
    raw = np.zeros(17, dtype=np.uint32)
    out = semantic_labels(raw)
    assert out.shape == (17,) and out.dtype == np.int32


def test_is_moving_flags_only_moving_ids():
    raw = np.array([10, 252, 253, 259, 260, 249, 50], dtype=np.uint32)
    assert is_moving(raw).tolist() == [False, True, True, True, False, False, False]


def test_map_covers_the_19_classes():
    classes = {c for c in SEMANTIC_KITTI_LABEL_MAP.values() if c != 19}
    assert classes == set(range(19))


def test_frnet_entry_points_raise_not_return_garbage():
    with pytest.raises(RuntimeError, match="FRNet inference is disabled"):
        segment(np.zeros((4, 4), dtype=np.float32))


@pytest.mark.skipif(not _HAS_LABELS, reason="SemanticKITTI .label files not present")
def test_real_frame_distribution_is_sane():
    labels = semantic_labels(load_labels(_label_path("00", 43)))
    assert labels.min() >= -1 and labels.max() <= 18
    frac_ignore = (labels == -1).mean()
    assert frac_ignore < 0.10  # frame 43 is ~1.9% unlabeled
    # building + road + car dominate this urban frame
    for cls in (12, 8, 0):
        assert (labels == cls).mean() > 0.05
