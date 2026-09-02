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

⚑ The fallback is NOT equivalent. It marks every point of a ground *class* as
  ground regardless of geometry, so on a sloped verge it admits ~4-12% of the
  `terrain` points (the raised part of an embankment) that the geometric
  segmenter rejects. Callers go through `segment_ground_or_fallback()`, which
  returns which method actually ran and warns once per process when the
  fallback stands in -- do not re-implement the `_HAVE_PATCHWORKPP` branch by
  hand, that is how it went silent.
"""

import warnings

import numpy as np

from .transforms import SENSOR_HEIGHT_M

try:
    import pypatchworkpp as _pw

    _HAVE_PATCHWORKPP = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_PATCHWORKPP = False

#: `PerceptionFrame.ground_method` / `segment_ground_or_fallback` return values.
GROUND_METHOD_PATCHWORKPP = "patchworkpp"
GROUND_METHOD_FALLBACK = "semantic_fallback"

_fallback_announced = False


def _announce_fallback(involuntary: bool) -> None:
    """Warn once per process that the semantic-class fallback is standing in.

    Kept to one warning per process (not per frame) so a whole sequence does
    not bury it, but loud enough that a run cannot quietly report
    fallback-derived ground numbers as Patchwork++ ones. Fires for BOTH the
    deliberate opt-out and the missing extension -- the wording says which --
    because a per-ring or curb table is equally not-comparable either way.
    """
    global _fallback_announced
    if _fallback_announced:
        return
    _fallback_announced = True
    reason = ("pypatchworkpp is not installed"
              if involuntary else
              "caller opted out (use_patchworkpp=False / --no-patchworkpp)")
    warnings.warn(
        "ground segmentation is using the semantic-class FALLBACK "
        f"(ground_from_semantics), not Patchwork++: {reason}. The fallback "
        "marks every point of a ground class as ground regardless of geometry, "
        "so it admits embankments and raised terrain the geometric segmenter "
        "rejects (~4-12% of `terrain` points on a sloped verge). Any per-ring / "
        "curb / plan-regret number from this run is on the fallback, not "
        "Patchwork++. Install the real segmenter with "
        "`pip install -e \".[perception]\"`; on a platform with no published "
        "wheel (Linux aarch64, Intel macOS) build from source -- see "
        "docs/perception-dashboard-summary.md.",
        RuntimeWarning,
        stacklevel=3,
    )


def reset_fallback_warning() -> None:
    """Re-arm the once-per-process fallback warning. For tests."""
    global _fallback_announced
    _fallback_announced = False


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


def segment_ground_or_fallback(points, semantic_labels, *, use_patchworkpp=True):
    """One ground call for the whole codebase: `(mask, method)`.

    Every consumer -- `run/__main__.iter_pipeline`, `eval/harness.real_scans`,
    `scripts/feature_report` -- must call this rather than branching on
    `_HAVE_PATCHWORKPP` itself. Each hand-rolled copy of that branch was a place
    the fallback could (and did) go unannounced.

    Args:
        points: (N, 4) or (N, 3) raw SENSOR-frame scan, for Patchwork++.
        semantic_labels: (N,) 19-class ids, for the fallback.
        use_patchworkpp: if False, the caller has deliberately opted out (the
            `--no-patchworkpp` flag). Still recorded in `method` and still
            warned about once -- the fallback's numbers are not comparable to
            Patchwork++'s either way -- but the warning text says it was a
            choice rather than a missing extension.

    Returns:
        (mask, method) -- `mask` is (N,) bool, `method` is
        ``GROUND_METHOD_PATCHWORKPP`` or ``GROUND_METHOD_FALLBACK``.
    """
    if use_patchworkpp and _HAVE_PATCHWORKPP:
        return segment_ground(points), GROUND_METHOD_PATCHWORKPP

    _announce_fallback(involuntary=use_patchworkpp)
    return ground_from_semantics(semantic_labels), GROUND_METHOD_FALLBACK
