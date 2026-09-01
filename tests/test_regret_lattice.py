"""Eq. (23) has to subtract like from like. Math §8.1. [Aakash]

Plan regret is `R(S) = J_M*(pi_S) - J_M*(pi*)`. Both terms are costs on M*, so
the only thing M_S is allowed to contribute is the PATH -- and a path is only
meaningful if the two costmaps describe the same world at the same scale.
They did not, in two independent ways, and each one had a number attached:

  the fill-rate confound   5/10/20/40 paid `w_unknown` on 100% of the
                           surviving window against uniform 20 cm's 4.2% --
                           a 4-unit handicap on every cell, for the crime of
                           resolving finely. The diagnostic built to catch
                           this read `.unknown` and reported 0.0%.

  the geometry mismatch    M_S OR-ed bitfields computed on the RING lattice
                           (5 cm neighbourhoods in ring 0) while M* computed
                           them at the 25 cm planning cell. The 12 cm kerb is
                           a step at 5 cm and smooth at 25 cm, so M_S invented
                           148 impassable cells M* did not have and missed 8
                           it did, against 12 real ones.

Both are fixed by the same principle -- evaluate the predicate once, on the
planning lattice, from summed evidence -- and this file is what keeps them so.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from vrgrid.eval.harness import build_gridmap, run_sequence, uniform_schedule
from vrgrid.eval.plan_regret import (
    common_support,
    costmap_from_gridmap,
    costmap_from_reference,
)
from vrgrid.eval.reference_map import build_from_scans
from vrgrid.eval.synthetic import read_sequence, write_sequence
from vrgrid.grid.schedule import load

FRAMES = 12
PLAN_N, PLAN_CELL_M = 44, 0.25


def _scans(root):
    for pts, labels, pose in read_sequence(root, "99"):
        yield pts, labels, np.ones(len(pts), dtype=bool), pose


@pytest.fixture(scope="module")
def scene():
    root = Path(tempfile.mkdtemp())
    write_sequence(root, "99", n_frames=FRAMES)
    reference = build_from_scans(read_sequence(root, "99"))
    vehicle_x = (FRAMES - 1) * 2.0
    x0, y0 = vehicle_x - 11.0, -5.5

    star = costmap_from_reference(reference, x0, y0, PLAN_N, PLAN_N, PLAN_CELL_M)
    mine = {}
    for name, sched in (("5/10/20/40", load("5/10/20/40")),
                        ("uniform_20cm", uniform_schedule(0.20, half_width_m=24.0))):
        gm = build_gridmap(sched)
        run_sequence(gm, _scans(root))
        mine[name] = costmap_from_gridmap(gm, x0, y0, PLAN_N, PLAN_N, PLAN_CELL_M,
                                          vehicle_xy_m=(vehicle_x, 0.0))
    return star, mine


def test_the_diagnostic_reads_the_array_the_cost_function_charges(scene):
    """⚑ `.unknown` is not who pays `w_unknown`.

    `_cost_from_bits` charges it for `unknown | TRAV_CONFIDENCE`, and the
    second term is almost all of it: a cell can have been observed -- so
    `unknown` is False -- and still sit below `n_min`. The confound diagnostic
    read `.unknown`, reported 0.0%, and hid a real 100.0%. A diagnostic that
    is wrong in the reassuring direction is worse than no diagnostic.
    """
    _star, mine = scene
    for name, m in mine.items():
        assert m.trav is not None, f"{name}: the bitfield is not retained"
        # low_confidence() is a superset of .unknown, by construction.
        assert np.all(m.low_confidence() | ~m.unknown), name
        assert m.low_confidence().sum() >= m.unknown.sum(), name


def test_a_fine_schedule_is_not_handicapped_for_being_fine(scene):
    """The fill-rate confound, closed.

    A 25 cm planning cell covers 25 map cells of a 5 cm ring. OR-ing the
    confidence bit over them set it whenever ANY sub-cell was thin, which at
    ring 0's fill rate is essentially always -- so the finer the schedule the
    larger its handicap, which is precisely backwards.

    Confidence is evidence, and evidence adds up: the observation counts over
    the footprint's DISTINCT cells are summed and compared against `n_min`
    once, which is what the reference side already did with `block_stats`.
    """
    _star, mine = scene
    mask = common_support(*mine.values())
    fine = mine["5/10/20/40"].low_confidence()[mask].mean()
    coarse = mine["uniform_20cm"].low_confidence()[mask].mean()

    assert fine < 0.10, (
        f"the variable schedule pays w_unknown on {fine:.1%} of the window; "
        "it was 100% when the confidence bit was OR-ed over sub-cells")
    assert fine - coarse < 0.10, (
        f"variable {fine:.1%} vs uniform {coarse:.1%} -- the fine schedule is "
        "still being penalised for resolving finely")


def test_the_two_sides_agree_about_where_the_walls_are(scene):
    """⚑ Eq. (23)'s premise, asserted directly.

    A path planned around walls that are not in M*, then scored on M*, is not
    a regret -- it is two different problems subtracted. The variable schedule
    invented 148 impassable cells and missed 8, against 12 real ones, because
    its bits came from the ring lattice and M*'s from the planning lattice.
    """
    star, mine = scene
    mask = common_support(*mine.values())
    blocked_star = ~np.isfinite(star.cost) & mask

    m = mine["5/10/20/40"]
    blocked_mine = ~np.isfinite(m.cost) & mask
    invented = int((blocked_mine & ~blocked_star).sum())
    missed = int((blocked_star & ~blocked_mine).sum())

    assert invented == 0, f"{invented} impassable cells M* does not have"
    assert missed == 0, f"{missed} of M*'s impassable cells are absent from M_S"


def test_the_scene_actually_contains_walls(scene):
    """Non-vacuity, and it is not a formality here: with zero impassable cells
    on either side, "the two agree" is true and means nothing. The synthetic
    scene's 12 cm kerbs are what M* is supposed to resolve."""
    star, _ = scene
    assert int((~np.isfinite(star.cost)).sum()) > 0, (
        "M* has no impassable cells, so the agreement test above is vacuous")


def test_a_coarse_uniform_grid_is_the_one_that_smooths_walls_away(scene):
    """The result the ablation exists to show, and the direction matters.

    Once both sides are on the planning lattice, the schedule that MISSES M*'s
    hazards is the coarse uniform one -- it averages a 12 cm kerb into its
    20 cm cells until the step falls under `s_max`. That is the physically
    correct answer and it used to be inverted: the fine map was the one
    reported as disagreeing with the reference, because it was the only one
    resolving the kerb at all.
    """
    star, mine = scene
    mask = common_support(*mine.values())
    blocked_star = ~np.isfinite(star.cost) & mask

    fine_missed = int((blocked_star & np.isfinite(mine["5/10/20/40"].cost)).sum())
    coarse_missed = int((blocked_star & np.isfinite(mine["uniform_20cm"].cost)).sum())
    assert coarse_missed > fine_missed, (
        f"uniform 20 cm missed {coarse_missed} of M*'s walls and the variable "
        f"schedule missed {fine_missed} -- the coarse grid is supposed to be "
        "the one that cannot see them")


def test_the_predicate_is_the_same_on_both_sides(scene):
    """Clearance is evaluated on neither side.

    M* is 2.5D ground and has no ceiling, so it cannot set the clearance bit.
    Leaving it on the M_S side alone means M_S blocking cells M* is
    structurally unable to block, and eq. (23) scoring a difference in map
    CONTENTS as a cost of coarsening.
    """
    from vrgrid.cell import TRAV_CLEARANCE

    star, mine = scene
    assert not (star.trav & TRAV_CLEARANCE).any()
    for name, m in mine.items():
        assert not (m.trav & TRAV_CLEARANCE).any(), (
            f"{name} sets a clearance bit the reference cannot answer")
