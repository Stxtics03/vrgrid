"""Information-loss metrics and the harness. Math §9. [Aakash]

The harness is the product, so the metrics get the same treatment as the
theorems: each one is checked against a case where the answer is known by
hand, and each one is checked against the way it could be gamed.
"""

import numpy as np
import pytest
from vrgrid.eval.harness import (
    build_gridmap,
    evaluate,
    format_result,
    memory_vs_regret_row,
    run_sequence,
)
from vrgrid.eval.metrics import (
    coarsening_ratio_per_ring,
    dynamic_removal,
    fill_rate_per_ring,
    height_rmse_per_ring,
    memory_bytes,
    occupancy_iou_per_ring,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, terrain_height_m, write_sequence
from vrgrid.grid.quantise import quantise_variance_cm2
from vrgrid.grid.query import slot_of
from vrgrid.grid.schedule import load

SCHEDULE = "5/10/20/40"


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """One synthetic sequence, its reference map, and a map built from it --
    module-scoped because building it is the expensive part and no test below
    mutates it destructively."""
    root = tmp_path_factory.mktemp("metrics")
    write_sequence(root, "99", n_frames=6)
    reference = build_from_scans(read_sequence(root, "99"))

    def scans():
        for pts, labels, pose in read_sequence(root, "99"):
            moving = (labels >= 250) & (labels <= 259)
            yield (pts[~moving], (labels[~moving] % 16).astype("uint8"),
                   np.ones(int((~moving).sum()), dtype=bool), pose)

    gm = build_gridmap(load(SCHEDULE))
    frames = run_sequence(gm, scans()).frames
    return gm, reference, frames


# --- §9.2: per-ring RMSE -----------------------------------------------------


def test_rmse_is_zero_when_the_map_holds_the_reference_mean(scene):
    """The definition, checked by construction: write each ring cell the exact
    mean of its reference footprint and RMSE must vanish. If it does not, the
    footprint `F(c)` is not the one eq. (26) means."""
    _gm, reference, _ = scene
    from vrgrid.eval.metrics import _ring_cells

    fresh = build_gridmap(load(SCHEDULE))
    for ring in range(len(fresh.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(fresh, ring)
        n_ref, mean, _ = reference.block_stats(i_lo, j_lo, fresh.schedule.k(ring))
        sel = n_ref > 0
        fresh.soa["ground_height"][slots[sel]] = np.rint(mean[sel]).astype(np.int16)
        fresh.soa["obs_count"][slots[sel]] = 5
        # A cell that HOLDS a height was fused, so it carries a variance.
        # Code 0 means "never fused" (fusion.initialise), and `_compared`
        # excludes those -- a hand-built fixture has to look like a real cell.
        fresh.soa["height_variance"][slots[sel]] = quantise_variance_cm2(1.0)

    rmse = height_rmse_per_ring(fresh, reference)
    for ring, v in rmse.items():
        assert v == pytest.approx(0.0, abs=0.5), f"ring {ring} RMSE {v}"


def test_rmse_grows_with_the_error_written_in(scene):
    """Monotone in the thing it measures -- the weakest possible sanity check,
    and the one that catches a metric wired to the wrong array."""
    _gm, reference, _ = scene
    from vrgrid.eval.metrics import _ring_cells

    seen = []
    for offset in (0, 5, 20):
        fresh = build_gridmap(load(SCHEDULE))
        for ring in range(len(fresh.schedule.rings)):
            slots, i_lo, j_lo = _ring_cells(fresh, ring)
            n_ref, mean, _ = reference.block_stats(i_lo, j_lo, fresh.schedule.k(ring))
            sel = n_ref > 0
            fresh.soa["ground_height"][slots[sel]] = np.rint(mean[sel] + offset)
            fresh.soa["obs_count"][slots[sel]] = 5
            # A cell that HOLDS a height was fused, so it carries a variance.
            # Code 0 means "never fused" (fusion.initialise) and `_compared`
            # excludes those -- a fixture has to look like a real cell.
            fresh.soa["height_variance"][slots[sel]] = quantise_variance_cm2(1.0)
        seen.append(height_rmse_per_ring(fresh, reference)[1])

    assert seen[0] < seen[1] < seen[2]
    assert seen[1] == pytest.approx(5.0, abs=0.6)
    assert seen[2] == pytest.approx(20.0, abs=0.6)


def test_a_ring_nobody_drove_through_reports_nan_not_zero(scene):
    """Zero error on a ring with nothing in it would read as a perfect score
    and would make every metric improve with range -- the exact opposite of
    the effect being measured."""
    reference = scene[1]
    empty = build_gridmap(load(SCHEDULE))
    rmse = height_rmse_per_ring(empty, reference)
    assert all(np.isnan(v) for v in rmse.values())


# --- §9.3: the coarsening ratio ----------------------------------------------


def test_the_bias_variance_decomposition_holds(scene):
    """IL^2 = bias^2 + spread^2, eq. (27). Not a property of the code -- it is
    the identity the whole ratio rests on -- so it is asserted on the numbers
    the code actually produces, in case the two terms ever drift apart."""
    gm, reference, _ = scene
    out = coarsening_ratio_per_ring(gm, reference)
    for ring, c in out.items():
        if c["n"] == 0:
            continue
        assert c["il_cm"] ** 2 == pytest.approx(
            c["bias_cm"] ** 2 + c["spread_cm"] ** 2, rel=1e-9), f"ring {ring}"


def test_rho_is_one_when_the_estimate_is_unbiased(scene):
    """rho ~ 1 means the coarsening cost only the terrain's own sub-cell
    variability -- "the saving was free". Constructed exactly: write every
    ring cell the mean of its footprint, so bias is 0 and IL is pure spread."""
    reference = scene[1]
    from vrgrid.eval.metrics import _ring_cells

    fresh = build_gridmap(load(SCHEDULE))
    for ring in range(len(fresh.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(fresh, ring)
        n_ref, mean, _ = reference.block_stats(i_lo, j_lo, fresh.schedule.k(ring))
        sel = n_ref > 0
        fresh.soa["ground_height"][slots[sel]] = np.rint(mean[sel]).astype(np.int16)
        fresh.soa["obs_count"][slots[sel]] = 5
        # A cell that HOLDS a height was fused, so it carries a variance.
        # Code 0 means "never fused" (fusion.initialise), and `_compared`
        # excludes those -- a hand-built fixture has to look like a real cell.
        fresh.soa["height_variance"][slots[sel]] = quantise_variance_cm2(1.0)

    out = coarsening_ratio_per_ring(fresh, reference)
    for ring, c in out.items():
        if c["n"] < 50:
            continue
        assert c["rho"] == pytest.approx(1.0, abs=0.25), f"ring {ring} rho {c['rho']}"


def test_rho_rises_when_the_estimate_is_biased_beyond_the_terrain(scene):
    """rho >> 1 is the diagnosis "the schedule is too aggressive, or the
    fusion is wrong". Bias the map by well over the terrain's own roughness
    and the ratio must say so."""
    reference = scene[1]
    from vrgrid.eval.metrics import _ring_cells

    fresh = build_gridmap(load(SCHEDULE))
    for ring in range(len(fresh.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(fresh, ring)
        n_ref, mean, _ = reference.block_stats(i_lo, j_lo, fresh.schedule.k(ring))
        sel = n_ref > 0
        fresh.soa["ground_height"][slots[sel]] = np.rint(mean[sel] + 30)
        fresh.soa["obs_count"][slots[sel]] = 5
        # A cell that HOLDS a height was fused, so it carries a variance.
        # Code 0 means "never fused" (fusion.initialise), and `_compared`
        # excludes those -- a hand-built fixture has to look like a real cell.
        fresh.soa["height_variance"][slots[sel]] = quantise_variance_cm2(1.0)

    out = coarsening_ratio_per_ring(fresh, reference)
    for ring, c in out.items():
        if c["n"] < 50:
            continue
        assert c["rho"] > 5.0, f"ring {ring} rho {c['rho']} did not notice a 30 cm bias"


def test_single_observation_footprints_are_excluded_from_rho(scene):
    """Their spread is 0 by construction, not by flatness. Dividing by it
    manufactures an infinite ratio out of one return -- which would then
    dominate the mean and make rho a measure of sampling density."""
    gm, reference, _ = scene
    out = coarsening_ratio_per_ring(gm, reference)
    for c in out.values():
        assert not np.isinf(c["rho"])


# --- occupancy and fill ------------------------------------------------------


def test_iou_not_accuracy(scene):
    """The map is mostly empty, so predicting FREE everywhere scores ~97%
    accuracy and 0 IoU. Only one of those two numbers notices."""
    gm, reference, _ = scene
    iou = occupancy_iou_per_ring(gm, reference)
    assert any(v > 0.05 for v in iou.values() if not np.isnan(v))

    empty = build_gridmap(load(SCHEDULE))
    for v in occupancy_iou_per_ring(empty, reference).values():
        assert np.isnan(v) or v == 0.0


def test_fill_rate_rises_with_frames_not_with_a_single_geometry(tmp_path):
    """⚑ §1.3, "ring-sweep filling": beyond ~25 m P_fill is under 2% per frame
    and the far field is populated by ego-motion sweeping the ring pattern
    across the ground. So the fill rate is a function of frame count, and a
    single-frame far-field number means nothing -- which is why this is
    measured over a growing sequence rather than asserted as a scalar.
    """
    write_sequence(tmp_path, "99", n_frames=8)
    reference = build_from_scans(read_sequence(tmp_path, "99"))

    seen = []
    for n_frames in (1, 4, 8):
        def scans(limit=n_frames):
            for i, (pts, labels, pose) in enumerate(read_sequence(tmp_path, "99")):
                if i >= limit:
                    return
                moving = (labels >= 250) & (labels <= 259)
                yield (pts[~moving], (labels[~moving] % 16).astype("uint8"),
                       np.ones(int((~moving).sum()), dtype=bool), pose)

        gm = build_gridmap(load(SCHEDULE))
        run_sequence(gm, scans())
        seen.append(fill_rate_per_ring(gm, reference))

    for ring in (1, 2):
        vals = [s[ring] for s in seen]
        assert vals[0] < vals[-1], f"ring {ring} fill did not grow with frames"


# --- §9.4: dynamic removal ---------------------------------------------------


def test_dynamic_removal_reports_both_directions():
    """DR alone is gameable -- delete the whole map and score 100% removal.
    The harmonic mean is what punishes trading one direction for the other,
    which is why §9.4 insists on all three numbers."""
    perfect = dynamic_removal(100, 100, 900, 900)
    assert perfect == {"DR": 1.0, "SP": 1.0, "F": 1.0}

    delete_everything = dynamic_removal(100, 100, 0, 900)
    assert delete_everything["DR"] == 1.0, "the gameable number is still 100%"
    assert delete_everything["F"] == 0.0, "the F-score did not catch it"

    keep_everything = dynamic_removal(0, 100, 900, 900)
    assert keep_everything["SP"] == 1.0
    assert keep_everything["F"] == 0.0


# --- memory and the table ----------------------------------------------------


def test_memory_defers_to_the_allocator(scene):
    """Two functions that both "know" the memory figure is how the report and
    the running system end up disagreeing -- and there is a live counter next
    to that number in the demo."""
    gm = scene[0]
    from vrgrid.gpu.allocators import bytes_allocated
    assert memory_bytes(gm.allocation) == bytes_allocated(gm.allocation)


def test_the_table_renders_every_ring_and_hides_nothing(scene):
    """A single aggregate number hides the entire claim: error is SUPPOSED to
    grow with range. So the table must have one row per ring, always."""
    gm, reference, frames = scene
    text = format_result(evaluate(gm, reference, frames), gm.schedule)

    for ring in gm.schedule.rings:
        assert f"{ring.half_width_m:.0f}m" in text
    assert "rho" in text and "RMSE" in text
    assert "--" in text or "nan" not in text.lower(), "nan leaked into the table"


def test_the_ablation_compares_schedules_against_one_reference(scene):
    """Flaw E6, structurally: both schedules are scored by the same M*, built
    once and knowing nothing about rings. A per-schedule reference would make
    the comparison meaningless and would still produce a plausible table."""
    _, reference, _ = scene
    rows = [memory_vs_regret_row(evaluate(build_gridmap(load(n)), reference))
            for n in ("5/10/20/40", "5/10/50")]

    assert rows[0]["megabytes"] > rows[1]["megabytes"]
    assert rows[0]["logical_cells"] == 745_000
    assert rows[1]["logical_cells"] == 520_000
    assert all(r["regret"] is None for r in rows), "regret is not wired up yet"


def test_the_synthetic_pipeline_recovers_the_surface_it_was_given(scene):
    """End to end, the claim the scaffold exists to support: scatter, fuse and
    coarsen a scan of an analytic surface, and the map agrees with that
    surface to within a centimetre or so at every ring.

    Not a reportable number -- no sensor noise, no occlusion, no registration
    error -- but if this is NOT true the pipeline is broken in a way that no
    amount of real data will fix.
    """
    gm, reference, frames = scene
    result = evaluate(gm, reference, frames)
    for ring, v in result.rmse_cm.items():
        if np.isnan(v):
            continue
        assert v < 3.0, f"ring {ring} is {v:.1f} cm off an analytic surface"


def test_query_agrees_with_the_surface_after_a_run(scene):
    """The other end of the same claim, through the public API rather than the
    arrays: ask `query()` about a place and get the terrain back."""
    gm = scene[0]
    for x, y in ((6.0, 0.0), (12.0, 1.0), (20.0, -2.0)):
        _ring, slot = slot_of(gm, x, y)
        if gm.soa["obs_count"][slot] == 0:
            continue
        from vrgrid.grid.query import query
        assert query(gm, x, y).ground_height == pytest.approx(
            terrain_height_m(x, y), abs=0.10)
