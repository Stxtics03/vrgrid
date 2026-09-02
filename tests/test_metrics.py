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
    footprint_coverage_per_ring,
    height_rmse_per_ring,
    memory_bytes,
    occupancy_iou_per_ring,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, terrain_height_m, write_sequence
from vrgrid.gpu.kernels import Z_MAX_CM
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


def test_a_ring_is_scored_only_where_it_still_answers(scene):
    """⚑ The migration confound. Ring L's buffer is a square of half-width
    `R_L`, so it physically covers the hole the finer rings serve, and every
    cell the vehicle migrates inward keeps its last far-range value there
    forever -- nothing clears it, and `query()` no longer reads it.

    Scoring those cells asks a value frozen at 60 m to match a reference that
    kept accumulating the close-range returns the cell never received. So a
    deliberately absurd height written into a migrated cell must not move
    RMSE_L by so much as a rounding error, while the same height written into
    a cell the ring still answers for must move it a lot. If the second
    assertion ever fails the filter has stopped letting anything through and
    the first one is passing for the wrong reason.
    """
    gm, reference, _ = scene
    from vrgrid.eval.metrics import _cell_centres_m, _ring_cells
    from vrgrid.grid.query import window_cells

    ring = 2
    ix, iy = window_cells(gm.buffers[ring])
    cx, cy = _cell_centres_m(gm, ring, ix, iy)
    served = set(_ring_cells(gm, ring)[0].tolist())
    all_slots = np.arange(gm.buffers[ring].slots) + gm.buffers[ring].offset

    inside = np.maximum(np.abs(cx), np.abs(cy)) < gm.schedule.rings[0].half_width_m
    migrated = [s for s in all_slots[inside].tolist() if s not in served]
    assert migrated, "no cell has migrated inward; the fixture drove too little"

    before = height_rmse_per_ring(gm, reference)[ring]

    # The fixture is module-scoped, so every write is put back -- obs_count and
    # height_variance included, because `_compared` reads both.
    #
    # ⚑ The nonsense must sit INSIDE the vertical band. `_compared` drops cells
    #   clamped at `Z_MIN_CM`/`Z_MAX_CM` (a clamped cell holds the band edge,
    #   not a measurement), so a value outside it is excluded for that reason
    #   instead and both assertions below would pass vacuously.
    absurd = Z_MAX_CM - 1                             # ~6 m of nonsense, in band

    def rmse_with_nonsense_in(cells):
        cells = list(cells)
        keep = (gm.soa["ground_height"][cells].copy(),
                gm.soa["obs_count"][cells].copy(),
                gm.soa["height_variance"][cells].copy())
        gm.soa["ground_height"][cells] = absurd
        gm.soa["obs_count"][cells] = np.maximum(gm.soa["obs_count"][cells], 5)
        gm.soa["height_variance"][cells] = np.maximum(
            gm.soa["height_variance"][cells], 1)
        try:
            return height_rmse_per_ring(gm, reference)[ring]
        finally:
            (gm.soa["ground_height"][cells], gm.soa["obs_count"][cells],
             gm.soa["height_variance"][cells]) = keep

    assert rmse_with_nonsense_in(migrated) == pytest.approx(before)
    assert rmse_with_nonsense_in(sorted(served)[: len(migrated)]) > before * 10


def test_the_scored_set_is_the_set_query_routes_to(scene):
    """The band predicate has to be the map's own routing rule, not a second
    opinion about it. `slot_of` is what `query()` uses; every cell scored for
    ring L must be the cell `slot_of` hands back at that place, or the metric
    is measuring memory the planner cannot reach."""
    gm = scene[0]
    rng = np.random.default_rng(0)
    from vrgrid.eval.metrics import _cell_centres_m, _ring_cells

    for ring in range(len(gm.schedule.rings)):
        slots, i_lo, j_lo = _ring_cells(gm, ring)
        if not slots.size:
            continue
        k = gm.schedule.k(ring)
        pick = rng.choice(slots.size, size=min(200, slots.size), replace=False)
        cx, cy = _cell_centres_m(gm, ring, i_lo[pick] // k, j_lo[pick] // k)
        for slot, x_m, y_m in zip(slots[pick], cx, cy):
            assert slot_of(gm, float(x_m), float(y_m)) == (ring, int(slot))


def test_cell_centres_agree_with_the_frame_path(scene):
    """Two spellings of a cell's position is how a metric ends up scoring the
    cell next door. `gate._cell_centre` is the frame loop's O(1) version and
    this is the vectorised one; they must agree exactly."""
    gm = scene[0]
    from vrgrid.eval.metrics import _cell_centres_m
    from vrgrid.grid.gate import _cell_centre
    from vrgrid.grid.query import window_cells

    rng = np.random.default_rng(1)
    for ring in range(len(gm.schedule.rings)):
        buf = gm.buffers[ring]
        ix, iy = window_cells(buf)
        cx, cy = _cell_centres_m(gm, ring, ix, iy)
        for local in rng.choice(buf.slots, size=50, replace=False):
            assert (cx[local], cy[local]) == _cell_centre(
                gm, ring, int(local) + buf.offset)


def test_a_single_ring_schedule_gives_up_nothing_to_the_band_filter(scene, tmp_path):
    """⚑ Why the confound was not merely noise: it was ASYMMETRIC across the
    schedules §8.2 compares. A uniform baseline has one ring, `ring_of` always
    answers 0, and no cell can ever migrate out from under it -- so the money
    plot charged the foveated schedules for stale memory and the uniform grids
    for none. The only cells a single-ring schedule loses here are the ones
    that fall outside the map altogether."""
    from vrgrid.eval.harness import uniform_schedule
    from vrgrid.eval.metrics import _ring_cells

    write_sequence(tmp_path, "99", n_frames=4)

    def scans():
        for pts, labels, pose in read_sequence(tmp_path, "99"):
            moving = (labels >= 250) & (labels <= 259)
            yield (pts[~moving], (labels[~moving] % 16).astype("uint8"),
                   np.ones(int((~moving).sum()), dtype=bool), pose)

    gm = build_gridmap(uniform_schedule(0.20, half_width_m=24.0))
    buf = gm.buffers[0]
    kept = np.isin(np.arange(buf.slots) + buf.offset, _ring_cells(gm, 0)[0])

    # ⚑ Asserted as a positive, not as "everything dropped was out of reach".
    #   Nothing IS dropped here -- that is the whole point -- so the negative
    #   form ran over an empty array and passed without testing anything. It
    #   did exactly that until 3 Sep.
    assert kept.all(), (
        f"a single-ring schedule lost {(~kept).sum():,} of {buf.slots:,} cells "
        "to the band filter; it has no finer ring to lose them to")

    # And the filter is not simply inert: the four-ring schedule DOES drop
    # cells on the same code path, which is the asymmetry this test is about.
    multi = build_gridmap(load(SCHEDULE))
    run_sequence(multi, scans())
    mbuf = multi.buffers[2]
    mkept = np.isin(np.arange(mbuf.slots) + mbuf.offset, _ring_cells(multi, 2)[0])
    assert not mkept.all(), "ring 2 dropped nothing; the filter is inert"


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


def test_coverage_says_how_little_of_a_coarse_footprint_M_star_saw(scene):
    """⚑ rho divides by a spread estimated from the reference cells M* actually
    observed inside `F(c)`. At k = 8 that is up to 64 cells and is typically a
    couple, so the denominator is thin exactly where the coarsening claim is
    loudest. Coverage has to fall with ring index -- if it ever comes out flat,
    `block_stats` is returning an observation count rather than a count of
    observed cells and every spread in the table is being read wrong."""
    gm, reference, _ = scene
    cov = footprint_coverage_per_ring(gm, reference)

    assert cov[0] == pytest.approx(1.0), "ring 0 is the base lattice; k = 1"
    seen = [cov[r] for r in sorted(cov) if not np.isnan(cov[r])]
    assert seen == sorted(seen, reverse=True), f"coverage did not fall: {cov}"
    assert all(0.0 <= v <= 1.0 for v in seen), cov
    assert cov[max(cov)] < 0.25, (
        "the coarsest ring's footprint is nearly covered, so either the "
        "sequence is far longer than this fixture or the block is wrong")


def test_the_sign_of_the_band_filter_follows_the_terrain_not_the_code():
    """⚑ Why two real-data measurements of this fix disagreed about its SIGN.

    The band filter drops ring L's stale interior. Whether that raises or lowers
    RMSE_L depends on which half sits on rougher ground -- the stale interior or
    the live annulus -- and that is a property of the scene, not of the metric.

    Three scenes, identical but for where the roughness contrast is placed
    relative to ring 2's inner boundary. Both halves are always rough; only the
    contrast moves, so neither RMSE can collapse to zero by construction. The
    third is a control with no contrast at all.

    Measured at 60,000 returns/frame, ring 2, and stable across 10/16/24/32
    frames: rough-near -37.5 to -48.9%, rough-far +6.1 to +17.4%, control +0.2
    to +0.9%. **The correction spans 55 percentage points on one codebase, one
    schedule and one frame count, purely from where the roughness sits.**

    ⚑ There is a SECOND driver and this test is deliberately sized to show it.
      At the reduced density here the control comes out near -10%, not near
      zero: the stale population was written at longer range from fewer
      returns, so it is the worse estimate even on statistically identical
      ground, and that pushes the correction negative independently of
      roughness. Its size falls as return density rises, which is why the
      60,000-point runs show a null control and this one does not.

    So the assertions are the ORDERING and the SPREAD, which hold at both
    densities, rather than any absolute figure -- the absolutes are exactly the
    thing that is scene- and sensor-dependent, and asserting one would be
    asserting the bug this test exists to explain.
    """
    from vrgrid.eval.metrics import _cell_centres_m
    from vrgrid.gpu.kernels import Z_MIN_CM
    from vrgrid.grid.lattice import ring_of
    from vrgrid.grid.query import window_cells

    n_frames, step_m, split_m, ring = 10, 2.0, 25.0, 2
    final_x = (n_frames - 1) * step_m
    scenes = {"rough_near": (0.15, 0.03), "rough_far": (0.03, 0.15),
              "control": (0.15, 0.15)}

    def scans(mode, rng):
        near_amp, far_amp = scenes[mode]
        for f in range(n_frames):
            vx = f * step_m
            r = rng.uniform(1.0, 55.0, 25_000)        # polar: density ~ 1/r
            th = rng.uniform(-np.pi, np.pi, 25_000)
            wx, wy = vx + r * np.cos(th), r * np.sin(th)
            cheb = np.maximum(np.abs(wx - final_x), np.abs(wy))
            amp = np.where(cheb < split_m, near_amp, far_amp)
            wz = amp * np.sin(3.1 * wx) * np.cos(2.7 * wy)
            pose = np.eye(4)
            pose[0, 3] = vx
            yield (np.column_stack([wx - vx, wy, wz]),
                   np.full(25_000, 40, dtype=np.uint32),      # raw `road`
                   np.ones(25_000, dtype=bool), pose)

    def rmse(gm, ref, banded):
        buf = gm.buffers[ring]
        ix, iy = window_cells(buf)
        slots = np.arange(buf.slots, dtype=np.int64) + buf.offset
        k = gm.schedule.k(ring)
        if banded:
            m = ring_of(*_cell_centres_m(gm, ring, ix, iy), gm.schedule,
                        gm.speed_ms) == ring
            ix, iy, slots = ix[m], iy[m], slots[m]
        n_ref, ref_mean, _ = ref.block_stats(ix * k, iy * k, k)
        g = gm.soa["ground_height"][slots]
        keep = ((n_ref > 0) & (gm.soa["obs_count"][slots] > 0)
                & (gm.soa["height_variance"][slots] > 0)
                & (g > Z_MIN_CM) & (g < Z_MAX_CM))
        mine = g[keep].astype(np.float64) + getattr(gm, "z_datum_m", 0.0) * 100.0
        return float(np.sqrt(np.mean((mine - ref_mean[keep]) ** 2)))

    change = {}
    for mode in scenes:
        ref = build_from_scans(
            (p, l, T) for p, l, _, T in scans(mode, np.random.default_rng(7)))
        gm = build_gridmap(load(SCHEDULE))
        run_sequence(gm, scans(mode, np.random.default_rng(7)))
        before = rmse(gm, ref, banded=False)
        change[mode] = (rmse(gm, ref, banded=True) - before) / before

    # Monotone in where the roughness was put -- that is the mechanism.
    assert change["rough_near"] < change["control"] < change["rough_far"], change

    # And the swing is large: moving the contrast alone moves the correction by
    # more than 15 points, which is why no single factor can post-hoc correct
    # the pre-fix numbers and the metric had to be fixed instead.
    assert change["rough_far"] - change["rough_near"] > 0.15, change


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
