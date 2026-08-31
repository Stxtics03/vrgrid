"""Ground segmentation. [JP]

Patchwork++ (pypatchworkpp, PyPI wheel -- not reimplemented), run on raw
sensor-frame points. Produces the ground points the elevation estimate is
built from and the non-ground points that become obstacle / ceiling evidence.

Patchwork++ works in the SENSOR frame: sensor at the origin, z up, ground near
z = -sensor_height. Pass the raw (N, 4) velodyne scan straight from the loader
-- do NOT lift into the vehicle frame first, and keep the intensity column
(Patchwork++'s Reflected-Noise-Removal step uses it).

Falls back to a semantic-class proxy (road / parking / sidewalk / other-ground
/ terrain are "ground"; lane-marking folds onto road in the label map) if
pypatchworkpp is not installed, so the pipeline still runs.
`ground_from_semantics()` is also the reference the Patchwork++ output is
sanity-checked against in tests/test_ground.py.
"""

import numpy as np

from .transforms import SENSOR_HEIGHT_M

try:
    import pypatchworkpp as _pw

    _HAVE_PATCHWORKPP = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_PATCHWORKPP = False

# 19-class indices that are walkable ground (semantics.FRNET_CLASS_NAMES order):
# 8 road, 9 parking, 10 sidewalk, 11 other-ground, 16 terrain.
GROUND_CLASSES = frozenset({8, 9, 10, 11, 16})

_estimator = None


def _get_estimator():
    """Lazily build one Patchwork++ estimator and reuse it across frames."""
    global _estimator
    if _estimator is None:
        params = _pw.Parameters()
        params.sensor_height = SENSOR_HEIGHT_M
        params.verbose = False
        _estimator = _pw.patchworkpp(params)
    return _estimator


def segment_ground(points: np.ndarray) -> np.ndarray:
    """Ground / non-ground split for one scan.

    Args:
        points: (N, 4) or (N, 3) float array -- raw SENSOR-frame [x, y, z, (intensity)].

    Returns:
        (N,) bool -- True where the point is ground. Index-aligned with `points`.

    Raises:
        RuntimeError: if pypatchworkpp is not installed. Use ground_from_semantics()
            as the fallback (the pipeline wiring does this automatically).
    """
    if not _HAVE_PATCHWORKPP:
        raise RuntimeError(
            "pypatchworkpp is not installed; call ground_from_semantics(labels) instead"
        )

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] == 3:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    elif pts.shape[1] != 4:
        raise ValueError(f"points must be (N, 3) or (N, 4), got {pts.shape}")

    est = _get_estimator()
    est.estimateGround(pts)

    mask = np.zeros(len(pts), dtype=bool)
    mask[np.asarray(est.getGroundIndices(), dtype=np.int64)] = True
    return mask


def ground_from_semantics(semantic_labels: np.ndarray) -> np.ndarray:
    """Ground mask from 19-class semantic labels (fallback / sanity reference).

    Args:
        semantic_labels: (N,) int -- 0-18 class index, -1 ignore
            (from semantics.semantic_labels()).

    Returns:
        (N,) bool -- True where the class is a walkable ground surface.
    """
    labels = np.asarray(semantic_labels)
    return np.isin(labels, list(GROUND_CLASSES))
