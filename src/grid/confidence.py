"""Per-cell confidence in the traversability verdict. Math §7.5. [Shrestha]

§7.1 returns six bits and no idea how sure it is. A planner that cannot tell
"probably drivable" from "definitely drivable" has to treat both as the same
fact, which means either trusting a cell it should slow down for or refusing
one it could have used. This reports the margin behind each verdict.

WHERE THE NUMBERS COME FROM -- nothing new is stored
    The cell struct is frozen at 12 B and adding a field means recomputing
    every memory figure in the report. Every quantity here is DERIVED from
    fields the map already carries:

      class      the Boyer-Moore counter, which §10.2 already calls "a
                 confidence readout" and which nothing had yet read
      evidence   `obs_count` against `n_min`
      geometry   how far the slope and step sit below their thresholds
      surface    `height_variance` against `sigma2_max`

    So the memory bound is untouched and this can be switched on and off
    without changing a single stored byte.

WHAT THIS IS NOT
    It is not a calibrated probability. Nothing here has been fitted against
    outcomes, and calling 0.6 a 60% chance of anything would be inventing
    precision the map cannot support. Each channel is a MARGIN in [0, 1] with
    a stated meaning, and `drivable_confidence` combines them by taking the
    WEAKEST -- a cell is exactly as trustworthy as the least trustworthy thing
    known about it. A product would read lower and look more sophisticated,
    and would be asserting an independence between slope error and label error
    that nobody has checked.
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.cell import FLAG_BLIND
from vrgrid.grid.fusion import COUNTER_MAX, unpack_class
from vrgrid.grid.quantise import dequantise_variance_cm2
from vrgrid.grid.schedule import load_thresholds
from vrgrid.grid.traversability import (
    drivable_ids,
    gradient,
    max_step_cm,
)


@dataclass(frozen=True)
class Margins:
    """One array per channel, all in [0, 1], all "1 = as good as it gets".

    Kept separate rather than pre-combined because they fail for different
    reasons and a planner may weigh them differently: a low `surface` margin
    is rough ground it can crawl over, a low `evidence` margin is ground it
    has barely seen, and those are not the same problem.
    """
    class_share: np.ndarray   # lower bound on the winning label's vote share
    evidence: np.ndarray      # observation count against n_min
    geometry: np.ndarray      # slope and step against their thresholds
    surface: np.ndarray       # variance against sigma2_max


def class_vote_share(soa, ring_slice) -> np.ndarray:
    """Lower bound on the winning label's share of the votes, in [0.5, 1].

    Boyer-Moore keeps `counter = votes_for - votes_against` for the surviving
    candidate (§10.2), so with `n` observations

        votes_for >= (n + counter) / 2   ->   share >= (n + counter) / (2n)

    The counter saturates at COUNTER_MAX = 7, which makes it an UNDERESTIMATE
    of the true margin once a cell has been seen more than seven times -- so
    this is a genuine lower bound and never an optimistic one. That is the
    right direction to be wrong in, and it is also why a heavily observed
    unanimous cell reports about 0.5 rather than 1.0: the 3-bit counter simply
    does not retain the evidence, and claiming otherwise would be inventing
    it. `saturated` in `margins()` is what flags cells in that regime.

    A counter of 0 means the candidate was just replaced and is no better than
    a coin flip between the two leading labels. Unobserved cells return 0.
    """
    candidate, counter = unpack_class(soa["semantic_class"][ring_slice])
    n = soa["obs_count"][ring_slice].astype(np.float64)
    del candidate
    share = np.zeros_like(n)
    seen = n >= 1
    share[seen] = (n[seen] + counter[seen]) / (2.0 * n[seen])
    return np.clip(share, 0.0, 1.0)


def margins(soa, ring_slice, side: int, cell_m: float, thresholds=None) -> Margins:
    """The four channels, for one ring. All derived, nothing stored."""
    th = thresholds if thresholds is not None else load_thresholds()
    t = th["traversability"]

    n = soa["obs_count"][ring_slice].astype(np.float64)
    ground = soa["ground_height"][ring_slice].astype(np.int32)
    blind = (soa["flags"][ring_slice] & FLAG_BLIND) != 0

    evidence = np.clip(n / float(t["n_min"]), 0.0, 1.0)
    evidence[blind] = 0.0        # unknown by construction, not merely thin

    baseline_m = t.get("baseline_m")
    dzdx, dzdy = gradient(ground, side, cell_m, baseline_m)
    slope = np.hypot(dzdx, dzdy)
    step_m = max_step_cm(ground, side, cell_m, baseline_m) / 100.0
    slope_margin = 1.0 - slope / np.tan(np.radians(t["theta_max_deg"]))
    step_margin = 1.0 - step_m / t["s_max_m"]
    geometry = np.clip(np.minimum(slope_margin, step_margin), 0.0, 1.0)

    sigma2_m2 = dequantise_variance_cm2(soa["height_variance"][ring_slice]) * 1e-4
    surface = np.clip(1.0 - sigma2_m2 / t["sigma2_max_m2"], 0.0, 1.0)

    return Margins(class_share=class_vote_share(soa, ring_slice),
                   evidence=evidence, geometry=geometry, surface=surface)


def drivable_confidence(soa, ring_slice, side: int, cell_m: float,
                        thresholds=None) -> np.ndarray:
    """How far to trust "this cell is drivable", in [0, 1].

    The WEAKEST of the four margins, and 0 outright if the winning label is
    not in the drivable set -- geometry decides and semantics filters (§7.1),
    so a confident verdict about a non-drivable class is a confident NO, which
    this reports as no confidence in drivability rather than as high
    confidence in something else.

    ⚑ Read this WITH the verdict, never instead of it. A cell can be
      traversable under §7.1 and carry a confidence of 0.1 -- that is the
      whole point, and it is the case the bitfield alone could not express.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    m = margins(soa, ring_slice, side, cell_m, th)
    candidate, _ = unpack_class(soa["semantic_class"][ring_slice])
    drivable = np.isin(candidate.astype(np.int32), drivable_ids(th))

    worst = np.minimum(np.minimum(m.class_share, m.evidence),
                       np.minimum(m.geometry, m.surface))
    return np.where(drivable, worst, 0.0)


def saturated(soa, ring_slice) -> np.ndarray:
    """Cells whose Boyer-Moore counter has hit its ceiling.

    Their `class_share` is a floor and nothing more: the counter stopped
    counting, so a cell seen 200 times unanimously and one seen 8 times
    unanimously report the same number. Worth surfacing next to the share so
    a low value is not read as disagreement when it is really just a 3-bit
    register having run out of room.
    """
    _, counter = unpack_class(soa["semantic_class"][ring_slice])
    return counter >= COUNTER_MAX


#: The channels `binding` can name, in the order `drivable_confidence` applies
#: them. "class" here is the drivable-set gate, not the vote share.
CHANNELS = ("not-drivable", "label", "evidence", "geometry", "surface")


def summarise(soa, schedule, rings, thresholds=None):
    """Per ring: (level, cell_m, cells, mean, frac>=0.8, frac<0.2, binding).

    Reported as a distribution rather than a mean alone -- a map where every
    cell is 0.5 and a map that is half 0.9 and half 0.1 have the same mean and
    are not the same map, and only one is safe to plan through.

    `binding` is the channel that most often held the verdict down. Without it
    a ring reading 0.00 is unreadable: on the synthetic scene ring 3 comes out
    at 0.00 not because the far field is rough or thin but because most of it
    is the vegetation verge, and a confident "not drivable" is reported as no
    confidence in drivability. That is the correct answer and it looks exactly
    like a broken one, so the reason is printed next to the number.
    """
    th = thresholds if thresholds is not None else load_thresholds()
    out = []
    for level, (sl, side) in enumerate(rings):
        cell_m = schedule.rings[level].cell_m
        conf = drivable_confidence(soa, sl, side, cell_m, th)
        m = margins(soa, sl, side, cell_m, th)
        candidate, _ = unpack_class(soa["semantic_class"][sl])
        drivable = np.isin(candidate.astype(np.int32), drivable_ids(th))

        seen = soa["obs_count"][sl] >= 1
        c = conf[seen]
        if not c.size:
            out.append((level, cell_m, 0, float("nan"), float("nan"),
                        float("nan"), "--"))
            continue
        stack = np.stack([np.where(drivable, np.inf, -1.0)[seen],
                          m.class_share[seen], m.evidence[seen],
                          m.geometry[seen], m.surface[seen]])
        binding = CHANNELS[int(np.bincount(np.argmin(stack, axis=0),
                                           minlength=len(CHANNELS)).argmax())]
        out.append((level, cell_m, int(c.size), float(c.mean()),
                    float((c >= 0.8).mean()), float((c < 0.2).mean()), binding))
    return out
