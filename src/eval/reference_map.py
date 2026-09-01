"""Reference map — the ground truth every metric is measured against. [Aakash]

Master v4 §3.8, math §9. Built once per sequence and cached on disk: aggregate
all scans of a sequence with GT poses at 5 cm, static points only, then treat
that as truth.

Build this EARLY. Without it you are tuning the ring schedule by eye and have
no answer when a judge asks whether coarsening lost the kerb.

--- three properties it has to have, and one it must not -----------------

**Schedule-independent.** M* is rasterised on the base 5 cm lattice and
nothing else, so the same reference scores every schedule. A reference built
per-schedule would make cross-schedule comparison meaningless, which is the
one thing this file exists to make valid.

**On the same lattice as the map.** Indices here are `i_fine` from §2 —
literally the same integer division — so a ring-L cell's footprint is
`[i·k_L, (i+1)·k_L)` exactly, with no resampling and no epsilon. That is the
partition theorem of §2.2 being spent: the nesting is what makes `F(c)` in
§9.2 a slice rather than an interpolation.

**Static only.** `moving-*` (ids 250–259) come out, per §9.1. They are read
from the raw `.label` files; nothing is retrained and nothing is inferred.

**Not a Kalman map.** It is a plain aggregate — sum and count, no filter, no
LOD, no time limit. If the reference used the same fusion as the map under
test, the metric would be measuring agreement rather than accuracy.
"""

from pathlib import Path

import numpy as np

# §9.1's removal rule -- raw ids 250-259, straight from the `.label` file --
# is the same predicate the transient layer separates on, so it is imported
# rather than restated. It had been restated, and the reference map and the map
# under test disagreeing about which returns are dynamic is a way to score a
# ghost-removal metric against a reference that has the ghosts in it.
#
# ⚑ There are still two more copies: `perception/semantics.py: is_moving` and
#   the bare `MOVING_LABEL_IDS` in `perception/loader.py`. They agree with this
#   one today -- 250..259 inclusive, both spellings -- and consolidating them
#   is a cross-directory call, so it is proposed rather than done. See
#   `test_reference_map.py::test_one_definition_of_moving`.
from vrgrid.grid.transient import MOVING_ID_HI, MOVING_ID_LO, is_moving

__all__ = ["MOVING_ID_HI", "MOVING_ID_LO", "ReferenceMap", "is_moving"]


class ReferenceMap:
    """M*: mean static height per 5 cm cell, on the base lattice.

    Stored as dense 2D arrays over the bounding box actually observed, plus
    the lattice index of its corner. Dense because the metrics ask for block
    statistics over every ring cell, and a summed-area table over a dense
    array answers that in O(1) per block -- see `block_stats()`.
    """

    __slots__ = ("_sat", "cell_m", "class_id", "count", "height_cm", "i0", "j0")

    def __init__(self, cell_m, i0, j0, height_cm, count, class_id):
        self.cell_m = float(cell_m)
        self.i0 = int(i0)
        self.j0 = int(j0)
        self.height_cm = np.asarray(height_cm, dtype=np.float64)
        self.count = np.asarray(count, dtype=np.int64)
        self.class_id = np.asarray(class_id, dtype=np.uint8)
        self._sat = None

    @property
    def shape(self):
        return self.count.shape

    @property
    def observed(self) -> np.ndarray:
        return self.count > 0

    # --- block statistics, the machinery behind §9.2 and §9.3 ---------------

    def _tables(self):
        """Summed-area tables for count, height and height^2.

        Built once, lazily. Every per-ring metric asks for the mean and
        variance of the reference heights inside each coarse cell's footprint;
        with these that is four lookups per block instead of a k x k reduction
        per block, which at k = 8 over 745,000 cells is the difference between
        a metric you run every ablation and one you run once.
        """
        if self._sat is None:
            obs = self.observed
            h = np.where(obs, self.height_cm, 0.0)
            def sat(a):
                return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1),
                              ((1, 0), (1, 0)))
            self._sat = (sat(obs.astype(np.float64)), sat(h), sat(h * h))
        return self._sat

    def block_stats(self, i_lo, j_lo, k: int):
        """(n, mean, var) of reference heights in each k x k block.

        `i_lo`, `j_lo` are base-lattice indices of each block's low corner --
        `i_ring * k_L`, straight out of §2. Blocks that fall outside the
        reference's bounding box, wholly or partly, are answered on whatever
        part is inside; a block with no observed reference cell gets n = 0 and
        the caller must drop it rather than score it as a perfect match.

        `var` is the population variance -- the `spread²` of §9.3 eq. (27). It
        is the intrinsic sub-cell terrain variability, the part of the error
        that any single-value cell has to pay whatever the algorithm does.
        """
        cnt_t, h_t, h2_t = self._tables()
        H, W = self.shape

        r0 = np.clip(np.asarray(i_lo, dtype=np.int64) - self.i0, 0, H)
        c0 = np.clip(np.asarray(j_lo, dtype=np.int64) - self.j0, 0, W)
        r1 = np.clip(np.asarray(i_lo, dtype=np.int64) - self.i0 + k, 0, H)
        c1 = np.clip(np.asarray(j_lo, dtype=np.int64) - self.j0 + k, 0, W)

        def box(t):
            return t[r1, c1] - t[r0, c1] - t[r1, c0] + t[r0, c0]

        n = box(cnt_t)
        safe = np.maximum(n, 1.0)
        mean = box(h_t) / safe
        var = np.maximum(box(h2_t) / safe - mean * mean, 0.0)
        return n.astype(np.int64), mean, var

    # --- persistence --------------------------------------------------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, cell_m=self.cell_m, i0=self.i0, j0=self.j0,
            height_cm=self.height_cm.astype(np.float32),
            count=self.count.astype(np.int32),
            class_id=self.class_id,
        )

    def __repr__(self):
        return (f"ReferenceMap({self.shape[0]}x{self.shape[1]} @ {self.cell_m*100:.0f} cm, "
                f"{int(self.observed.sum()):,} observed cells)")


class _Builder:
    """Accumulates scans into M*. Two passes, because the extent is not known
    until the last scan and a dense array cannot be grown cheaply.

    Kept separate from `ReferenceMap` so the finished map is immutable-ish and
    cannot accidentally be added to after being measured against.
    """

    def __init__(self, cell_m: float):
        self.cell_m = cell_m
        self.keys = []          # flat lattice keys, per scan
        self.heights = []
        self.classes = []
        self.bounds = None

    def add(self, xyz_world, label_id):
        keep = ~is_moving(label_id)
        pts = np.asarray(xyz_world, dtype=np.float64)[keep]
        if pts.size == 0:
            return
        cls = (np.asarray(label_id, dtype=np.int64)[keep] & 0xFFFF).astype(np.uint8)

        i = np.floor(pts[:, 0] / self.cell_m).astype(np.int64)
        j = np.floor(pts[:, 1] / self.cell_m).astype(np.int64)
        self.keys.append(np.column_stack([i, j]))
        self.heights.append(pts[:, 2] * 100.0)
        self.classes.append(cls)

        lo = np.array([i.min(), j.min()])
        hi = np.array([i.max(), j.max()])
        self.bounds = (lo, hi) if self.bounds is None else (
            np.minimum(self.bounds[0], lo), np.maximum(self.bounds[1], hi))

    def finish(self) -> ReferenceMap:
        if self.bounds is None:
            raise ValueError("no static points -- the reference map would be empty")
        lo, hi = self.bounds
        H, W = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)

        h_sum = np.zeros(H * W, dtype=np.float64)
        count = np.zeros(H * W, dtype=np.int64)
        cls_first = np.zeros(H * W, dtype=np.uint8)

        for keys, heights, classes in zip(self.keys, self.heights, self.classes):
            flat = (keys[:, 0] - lo[0]) * W + (keys[:, 1] - lo[1])
            np.add.at(h_sum, flat, heights)
            np.add.at(count, flat, 1)
            # First writer wins, deterministically: the scans arrive in frame
            # order and `np.add.at`-style scatter has no defined order within a
            # scan, so "first" is resolved by only writing where still unset.
            unset = cls_first[flat] == 0
            cls_first[flat[unset]] = classes[unset]

        mean = np.where(count > 0, h_sum / np.maximum(count, 1), 0.0)
        return ReferenceMap(self.cell_m, lo[0], lo[1],
                            mean.reshape(H, W), count.reshape(H, W),
                            cls_first.reshape(H, W))


def build_from_scans(scans, out_path=None, cell_m: float = 0.05) -> ReferenceMap:
    """M* from an iterable of (points in VEHICLE frame, RAW label ids,
    vehicle -> world 4x4).

    This is the whole of §9.1 and it does not care where the scans came from,
    which is the point: the synthetic sequence and SemanticKITTI go through
    the same function, so the harness that works on one works on the other.

    ⚑ The frame in that first line used to read "sensor frame", and the body
      has always applied the pose and nothing else -- no `Tr`, no 1.73 m
      ground drop. Both callers pass the vehicle frame, so the code was right
      and the sentence was wrong, which is the more dangerous way round: the
      one caller that believed the docstring (`build()`, below) handed it raw
      sensor points and a raw `poses.txt` row. Same convention as
      `harness.run_sequence`, deliberately -- M* and M are built from the same
      scans and any disagreement between them is scored as map error.
    """
    b = _Builder(cell_m)
    for pts, labels, pose in scans:
        pts = np.asarray(pts, dtype=np.float64)
        world = pts @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]
        b.add(world, labels)
    ref = b.finish()
    if out_path is not None:
        ref.save(out_path)
    return ref


# Learning ids whose surface IS the ground: road, parking, sidewalk,
# other-ground, terrain. Used only to pick the returns the frame-convention
# guard is allowed to look at -- see `build`.
_GROUND_LEARNING_IDS = (8, 9, 10, 11, 16)


def build(sequence: str, out_path=None, cell_m: float = 0.05,
          max_frames: int | None = None, check_frame: bool = True):
    """M* for a SemanticKITTI sequence. Math §9.1.

    Reads through `perception.loader`, which is JP's and is where calibration
    and the velodyne-to-vehicle transform live.

    ⚑ **This had never been run.** It is the first step of any real-data
      evaluation -- no reference map, no metrics, no plan regret -- and it
      raised `ValueError: too many values to unpack` on its own first line,
      because `loader.scans` yields three things and this unpacked two. Behind
      that were two more, both of which would have produced a plausible map
      rather than an error:

        * it passed `poses[i]` straight through, and a `poses.txt` row is
          Camera-0 -> World_cam, not vehicle -> world. That is the 90 degree
          axis permutation `docs/frames.md` exists to prevent, applied to the
          artefact every metric is measured against.
        * it passed the loader's (N, 4) SENSOR-frame array to a function that
          wants (N, 3) in the VEHICLE frame -- an intensity column read as a
          coordinate, and every return 1.73 m under the road.

    So the chain is composed here, once, out of JP's transforms, and the first
    frame is checked rather than trusted: `check_frame` runs
    `harness.assert_world_is_z_up` over the ground-classed returns. Building a
    reference map from 40 GB with the wrong convention costs a day, and the
    result looks entirely plausible -- it is a complete map, in the wrong
    cells, and nothing downstream can tell.

    `max_frames` builds a subset: a first pass over 200 frames answers "is the
    frame right" in a minute instead of an hour, and it is what the demo path
    wants when the whole sequence is not needed.

    To build one from anything else -- the synthetic sequence, a recording --
    call `build_from_scans()` directly.
    """
    from vrgrid.eval.harness import FrameGuard
    from vrgrid.perception import loader
    from vrgrid.perception.semantics import semantic_labels
    from vrgrid.perception.transforms import T_S_V, vehicle_to_world

    def scans():
        guard = FrameGuard() if check_frame else None
        for pts, labels, pose in loader.scans(sequence, max_frames=max_frames):
            # Sensor -> vehicle: drop the origin to the road. `transform_points`
            # would do this too, but it also ignores the intensity column, and
            # being explicit about which three of the four go through is the
            # point of the bug above.
            xyz = np.asarray(pts, dtype=np.float64)[:, :3] + T_S_V[:3, 3]
            t_vw = vehicle_to_world(pose, sequence)
            if guard is not None and not guard.done:
                world = xyz @ t_vw[:3, :3].T + t_vw[:3, 3]
                ground = np.isin(semantic_labels(labels), _GROUND_LEARNING_IDS)
                guard.check(world, ground, t_vw[:3, 3])
            yield xyz, labels, t_vw

    return build_from_scans(scans(), out_path, cell_m)


def load(path) -> ReferenceMap:
    """Read a cached M* back. Cheap, so metrics can be re-run without
    rebuilding -- which matters because building is the expensive step and
    the ablation runs the metrics once per schedule."""
    z = np.load(Path(path), allow_pickle=False)
    return ReferenceMap(float(z["cell_m"]), int(z["i0"]), int(z["j0"]),
                        z["height_cm"], z["count"], z["class_id"])
