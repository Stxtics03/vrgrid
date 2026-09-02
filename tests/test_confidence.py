"""Per-cell confidence in the traversability verdict. Math §7.5. [Shrestha]

Every number here is derived from fields the map already carries, so the first
thing these tests pin is that nothing is stored: the cell struct is frozen at
12 B and this feature must not cost a byte of it.
"""

import numpy as np
from vrgrid.cell import CELL_BYTES, CELL_FIELDS, FLAG_BLIND, alloc_soa
from vrgrid.grid.confidence import (
    class_vote_share,
    drivable_confidence,
    margins,
    saturated,
    summarise,
)
from vrgrid.grid.fusion import COUNTER_MAX, initialise, pack_class
from vrgrid.grid.quantise import quantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import class_ids

SIDE = 16
CELL_M = 0.20


def _grid(class_name="road", n=8, counter=COUNTER_MAX, sigma_cm=1.0):
    soa = alloc_soa(SIDE * SIDE)
    initialise(soa)
    soa["ground_height"][:] = 0
    soa["obs_count"][:] = n
    soa["semantic_class"][:] = pack_class(class_ids()[class_name], counter)
    soa["height_variance"][:] = quantise_variance_cm2(sigma_cm ** 2)
    soa["flags"][:] = 0
    return soa


def _in(a):
    return np.asarray(a).reshape(SIDE, SIDE)[1:-1, 1:-1].reshape(-1)


def test_the_cell_struct_is_untouched():
    """The headline memory figures are computed from CELL_BYTES. If confidence
    ever needs a stored field, every number in the report moves -- so this is
    the test that has to fail first."""
    assert CELL_BYTES == 12
    assert len(CELL_FIELDS) == 10
    assert not any("confid" in name for name, _ in CELL_FIELDS)


# --- the class channel -------------------------------------------------------

def test_vote_share_is_a_lower_bound_from_the_boyer_moore_counter():
    """share >= (n + counter) / 2n. n=8, counter=7 -> 0.9375, exactly."""
    assert np.allclose(class_vote_share(_grid(n=8, counter=7), slice(None)), 0.9375)
    assert np.allclose(class_vote_share(_grid(n=4, counter=2), slice(None)), 0.75)


def test_a_just_replaced_candidate_is_a_coin_flip():
    """counter == 0 means the candidate was replaced this frame and has no
    margin over the runner-up. 0.5, not 1.0."""
    assert np.allclose(class_vote_share(_grid(counter=0), slice(None)), 0.5)


def test_an_unobserved_cell_has_no_class_confidence():
    """Not 0.5. A cell with no observations has no candidate to be confident
    in, and the byte it holds is a default rather than a vote."""
    assert np.allclose(class_vote_share(_grid(n=0), slice(None)), 0.0)


def test_the_counter_saturates_and_the_share_says_so():
    """A 3-bit counter stops counting at 7, so a cell seen 200 times reports a
    LOWER share than one seen 8 times -- not because it agrees less but
    because the register ran out. `saturated()` is what stops that being read
    as disagreement."""
    few, many = _grid(n=8, counter=7), _grid(n=200, counter=7)
    s_few = class_vote_share(few, slice(None))[0]
    s_many = class_vote_share(many, slice(None))[0]
    assert s_many < s_few, "the bound should loosen, not tighten"
    assert s_many > 0.5
    assert saturated(many, slice(None)).all()
    assert not saturated(_grid(counter=3), slice(None)).any()


# --- the composite -----------------------------------------------------------

def test_flat_well_observed_road_is_confident():
    conf = _in(drivable_confidence(_grid(), slice(None), SIDE, CELL_M))
    assert conf.min() > 0.8, f"clean road came out at {conf.min():.2f}"


def test_a_non_drivable_class_has_no_drivability_confidence():
    """Confident that it is a building is not confidence that it is drivable.
    Reported as 0, not as a high number about a different question."""
    soa = _grid(class_name="building")
    assert np.allclose(_in(drivable_confidence(soa, slice(None), SIDE, CELL_M)), 0.0)


def test_confidence_is_the_weakest_channel_not_the_average():
    """One bad channel must dominate. A cell that is flat, well observed and
    unanimous but ROUGH is not 75% trustworthy."""
    soa = _grid(sigma_cm=10.0)              # sigma^2 = 100 cm^2, over the 25 cm^2 max
    m = margins(soa, slice(None), SIDE, CELL_M)
    assert _in(m.surface).max() == 0.0
    assert _in(m.class_share).min() > 0.8, "only the surface channel should be bad"
    assert np.allclose(_in(drivable_confidence(soa, slice(None), SIDE, CELL_M)), 0.0)


def test_thin_evidence_caps_confidence_below_a_confident_label():
    """One observation of a cell is one observation, however unanimous."""
    th = load_thresholds()["traversability"]
    soa = _grid(n=1, counter=1)
    conf = _in(drivable_confidence(soa, slice(None), SIDE, CELL_M))
    assert np.allclose(conf, 1.0 / th["n_min"], atol=1e-9)


def test_the_blind_cone_has_zero_confidence_whatever_it_holds():
    """FLAG_BLIND cells are unknown by construction (master v4 §3.6). Filling
    them with a plausible class and a full counter must not buy confidence."""
    soa = _grid()
    soa["flags"][:] = FLAG_BLIND
    assert np.allclose(drivable_confidence(soa, slice(None), SIDE, CELL_M), 0.0)


def test_a_slope_at_the_threshold_has_no_geometric_margin():
    """The margin is distance from the threshold, so a cell sitting exactly on
    tan(theta_max) reads 0 -- traversable by §7.1 and not to be trusted."""
    th = load_thresholds()["traversability"]
    soa = _grid()
    span_m = 2 * CELL_M                       # baseline_m 0.50 <= 2c, so k = 1
    rise_cm = np.tan(np.radians(th["theta_max_deg"])) * span_m * 100.0
    ramp = np.arange(SIDE) * (rise_cm / 2.0)
    soa["ground_height"][:] = np.round(
        ramp[None, :] * np.ones((SIDE, 1))).reshape(-1).astype(np.int16)

    m = margins(soa, slice(None), SIDE, CELL_M)
    assert _in(m.geometry).max() < 0.05, f"margin {_in(m.geometry).max():.3f}"


def test_summarise_reports_a_distribution_not_just_a_mean():
    """A map that is uniformly 0.5 and one that is half 0.9 half 0.1 have the
    same mean and are not the same map."""
    soa = _grid()
    rings = [(slice(None), SIDE)]

    class _S:
        """The one thing `summarise` needs off a Schedule: each ring's size."""

        def __init__(self):
            self.rings = [type("R", (), {"cell_m": CELL_M})()]

    rows = summarise(soa, _S(), rings)
    level, cell_m, n_seen, mean, hi, lo, binding = rows[0]
    assert level == 0 and cell_m == CELL_M and n_seen == SIDE * SIDE
    assert mean > 0.8 and hi > 0.8 and lo == 0.0
    # clean drivable road: nothing is holding the verdict down but the
    # 3-bit counter's own ceiling, so the label channel binds.
    assert binding == "label", binding
