"""Semantic label mapping. [JP]

Semantic class comes from the SemanticKITTI .label files, not from inference
(the FRNet port works as of 2 Sep; the map does not use it, by choice -- see
src/perception/semantics.py). These tests
pin the raw-id -> 19-class map and the moving-* flag.
"""

import numpy as np
import pytest
from vrgrid.perception.loader import _label_path, load_labels, verify_sequence_exists
from vrgrid.perception.semantics import (
    SEMANTIC_KITTI_LABEL_MAP,
    is_moving,
    segment,
    semantic_labels,
)

_HAS_LABELS = verify_sequence_exists("00") and _label_path("00", 43).exists()


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


# --- learning_ids on real scans ---------------------------------------------

def test_unlabelled_points_get_a_packable_class_not_255():
    """Real SemanticKITTI scans contain `unlabeled` (raw 0) and ids outside the
    scheme. `semantic_labels` reports those as -1, and -1 through
    `astype(uint8)` is 255 -- which does not fit the 5-bit class field, so
    `scatter_sorted` rejected the very first real frame with "class ids must be
    < 32 to pack into the class key".

    The synthetic sequences write learning ids and never contain an unlabelled
    point, which is why nothing caught this until the loader was pointed at
    sequence 08. Mapped rather than dropped: an unlabelled return still has
    geometry, and a wall nobody labelled is still a wall.
    """
    import numpy as np
    from vrgrid.eval.harness import learning_ids
    from vrgrid.grid.fusion import CLASS_MAX, CLASS_UNLABELLED

    raw = np.array([0, 40, 48, 70, 252, 1, 99], dtype=np.uint32)   # 0 and 99 unmapped
    out = learning_ids(raw)

    assert out.dtype == np.uint8
    assert int(out.max()) <= CLASS_MAX, "must fit the 5-bit class field"
    assert out[0] == CLASS_UNLABELLED, "raw 0 is `unlabeled`"
    assert out[-1] == CLASS_UNLABELLED, "raw 99 is outside the scheme"
    assert out[1] != CLASS_UNLABELLED, "raw 40 is `road` and must survive"


def test_the_unlabelled_class_is_not_drivable():
    """It has to fail safe on §7.1 bit 4 -- an unknown class is not a licence
    to drive over it."""
    import numpy as np
    from vrgrid.grid.fusion import CLASS_UNLABELLED
    from vrgrid.grid.traversability import drivable_ids

    assert CLASS_UNLABELLED not in np.asarray(drivable_ids())


def test_instance_ids_in_the_upper_word_are_masked_off():
    """A `.label` word is 16 bits of semantics and 16 of instance id. Reading
    the whole word gives class ids in the thousands."""
    import numpy as np
    from vrgrid.eval.harness import learning_ids

    plain = np.array([40, 48, 70], dtype=np.uint32)
    with_instances = plain | (np.array([3, 17, 900], dtype=np.uint32) << 16)
    assert np.array_equal(learning_ids(plain), learning_ids(with_instances))
