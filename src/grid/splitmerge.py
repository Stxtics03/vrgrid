"""Split and merge with honest uncertainty. Math §4–5. [Aakash — Day 2]

The two rules that are easy to get wrong and impossible to see going wrong:

MERGE is marginalisation over a footprint, so it obeys the law of total
variance -- NOT inverse-variance fusion, which is the rule for repeated
measurements of one quantity. Four children measure four different places.

    mu_p     = sum(w_i mu_i)
    sigma2_p = sum(w_i sigma_i^2)  +  sum(w_i (mu_i - mu_p)^2)
               within-cell            between-cell, the spread you just erased

Drop the second term and merged cells come out most confident exactly where
they straddle a kerb. It compiles fine. It produces a map that looks right.

SPLIT inflates variance and sets the `derived` bit. Children inherit mu_p with
a strictly larger variance; the bit records that the value was not measured.
That bit is what makes merge(split(c)) == c exact (Theorem 2). Without it, a
cell oscillating across a ring boundary as the vehicle changes speed inflates
its variance every frame with no physical cause, and the map drifts toward
uncertainty.

--- how Theorem 2 is made exact, and what it asks of the map-level code ------

Theorem 2 wants merge(split(c)) == c *bit for bit*, in mean and variance. It
cannot be had by deflating: sigma2_p = sigma2_child - Delta is exact in real
arithmetic and not in float64, and the error is worst exactly where it matters
most -- a confident cell (sigma2_p ~ 1e-6 m^2) split on a slope (Delta ~ 1e-2)
does not survive the round trip through that subtraction.

So the restore path does not compute anything. It returns the parent value,
which still exists: split does not destroy the parent, it writes children into
the finer ring / refinement pool while the ring-L cell stays resident in its
own buffer. `CellValue.derived_from` is that cell, at value level.

    ⚑ The one requirement this puts on the map-level implementation: split
      must not overwrite the parent cell. If a future SoA split reuses the
      parent's slot, Theorem 2 stops being exact and this module is lying.

--- two things §4–5 say that do not follow, both pinned in tests -------------

(1) kappa. §5.2 states "kappa = 1/16 from the offset geometry (d^2 =
    c_p^2/16)". The offset geometry is right -- a child centre of a 2x2 split
    sits c_p/4 off the parent centre on each axis -- but (17) multiplies
    kappa by (c_p^2 - c_c^2), not by c_p^2. At c_c = c_p/2 that factor is
    (3/4)c_p^2, so kappa = 1/16 yields 3c_p^2/64 where the stated geometry
    gives 4c_p^2/64: a uniform 25% under-inflation.

    The value that reproduces the geometry is kappa = 1/12, and -- this is the
    useful part -- it is 1/12 for every refinement ratio m, because the
    mean-square child-centre offset per axis is exactly

        c_p^2 (m^2 - 1) / (12 m^2)  ==  (c_p^2 - c_c^2) / 12

    So (17)'s FORM generalises to m != 2 for free, which the project needs:
    the 5/10/50 ablation refines 5x between rings 1 and 2, i.e. 25 children,
    not 4. Only the constant is off, and by the same factor 3/4 at every m.

    The default stays 1/16, read from configs/thresholds.yaml. Changing a
    frozen constant is a room decision, not mine. `test_kappa_from_geometry`
    keeps both numbers visible until it is made.

(2) alpha. (17) adds a roughness term alpha unconditionally, but §5.3's "on a
    perfectly flat road grad z = 0 and splitting costs nothing" and §5.4's
    unit test (c) "split on flat ground -> sigma2_child = sigma2_parent" are
    both false for any alpha > 0. Theorem 1 itself survives -- alpha > 0 only
    strengthens it -- the remark and test (c) do not.

    alpha is 0 in config, which is the honest value and not a convenient one:
    §5.2 calibrates it against the reference map (§9), and the reference map
    is blocked on the download. Whoever calibrates it has to restate §5.3 and
    rewrite §5.4(c) in the same commit, which is what
    `test_alpha_would_break_the_flat_ground_remark` exists to say.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from vrgrid.cell import FLAG_BLIND, FLAG_DERIVED, FLAG_DYNAMIC, FLAG_REFINED
from vrgrid.grid.schedule import CONFIG_DIR

# The kappa that §5.2's own offset geometry gives, once (17)'s (c_p^2 - c_c^2)
# normalisation is accounted for. NOT the default -- see the module docstring.
# Exported so the test can pin the discrepancy rather than describe it.
KAPPA_FROM_GEOMETRY = 1.0 / 12.0


@dataclass(frozen=True)
class SplitParams:
    """(17)'s two calibration constants. Never inline -- they live in
    configs/, and they are frozen before schedules are compared (flaw E6)."""

    kappa: float = 1.0 / 16.0   # §5.2 as written
    alpha_m2: float = 0.0       # uncalibrated: §9 needs the reference map


_PARAMS_CACHE: dict = {}


def load_params(path=None) -> SplitParams:
    """Read the `split_merge` block of configs/thresholds.yaml.

    Cached: this is startup configuration, not a per-cell lookup, and the
    frame loop must not touch the filesystem.
    """
    p = Path(path) if path is not None else CONFIG_DIR / "thresholds.yaml"
    key = str(p.resolve())
    if key not in _PARAMS_CACHE:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        _PARAMS_CACHE[key] = SplitParams(**raw.get("split_merge", {}))
    return _PARAMS_CACHE[key]


@dataclass(frozen=True)
class CellValue:
    """One cell's height estimate in physical units, at value level.

    The stored form is `include/vrgrid/cell.py`: int16 centimetres and a
    log-quantised uint8 variance. This module works in float64 metres on
    purpose -- §4–5 are statements about the mathematics, and the storage
    codec is not defined anywhere yet.

    ⚑ That codec is a real gap and it lands on this module. Theorem 1 says
      sigma2_child > sigma2_parent *strictly*, and a shallow enough slope
      inflates the variance by less than one uint8 log bucket, so the two
      stored values come out equal. Nobody has written quantise()/
      dequantise() and nobody should invent the scheme alone: it decides how
      much of §5 is observable in the map at all. Raised, not fixed here.

    Frozen on purpose. `derived_from` is only sound if a child cannot be
    mutated behind the flag's back: the only way to change a value is to build
    a new one, and `clear_derived()` is the way §5.4 says to do that.
    """

    mu_m: float
    sigma2_m2: float
    n: int = 1                      # obs_count; the counts behind w_i, eq. (15)
    flags: int = 0
    derived_from: "CellValue | None" = None

    def __post_init__(self):
        if self.sigma2_m2 < 0.0:
            raise ValueError(f"negative variance: {self.sigma2_m2!r}")
        if self.n < 0:
            raise ValueError(f"negative observation count: {self.n!r}")

    @property
    def derived(self) -> bool:
        """§5.4's one bit: this value was split, not measured."""
        return bool(self.flags & FLAG_DERIVED)

    @property
    def sigma_cm(self) -> float:
        """Standard deviation in centimetres -- the unit every §4–5 worked
        example is quoted in, so a test can be read against the document."""
        return 100.0 * float(np.sqrt(self.sigma2_m2))


def clear_derived(cell: CellValue) -> CellValue:
    """Drop the `derived` bit. §5.4: set on split, cleared by any measurement.

    fusion.py calls this on every cell it writes a measurement into. It also
    drops `derived_from`, which is what takes a subsequent merge off the
    restore path -- exactly §5.4's "no observations since split" clause.
    """
    return CellValue(cell.mu_m, cell.sigma2_m2, cell.n,
                     cell.flags & ~FLAG_DERIVED, None)


# --- split, math §5 ----------------------------------------------------------


def inflate(sigma2_m2, grad_z, c_parent_m: float, c_child_m: float,
            params: SplitParams | None = None):
    """Equation (17). Scalar in -> float out; ndarray in -> ndarray out.

        sigma2_child = sigma2_parent + kappa ||grad z||^2 (c_p^2 - c_c^2) + alpha

    `grad_z` is the ground gradient at the parent, dimensionless (m/m),
    estimated by finite differences over the parent's 4-neighbours per §7.1.
    Pass the magnitude; only the slope can matter, never its sign, which is
    why it appears squared.

    The (c_p^2 - c_c^2) factor is what makes splitting a flat surface free and
    what makes the formula scale-correct at any refinement ratio -- see the
    module docstring on kappa.
    """
    p = params or load_params()
    g = np.asarray(grad_z, dtype=np.float64)
    s2 = np.asarray(sigma2_m2, dtype=np.float64)
    if c_child_m >= c_parent_m:
        raise ValueError(
            f"split must go finer: c_child {c_child_m} m >= c_parent {c_parent_m} m"
        )
    out = s2 + p.kappa * g * g * (c_parent_m**2 - c_child_m**2) + p.alpha_m2
    return float(out) if np.ndim(out) == 0 else out


def split(parent: CellValue, schedule, ring: int, grad_z: float = 0.0,
          params: SplitParams | None = None) -> tuple:
    """One ring-`ring` cell -> its children on ring `ring - 1`. Math §5.

    Returns m^2 children, where m = k_ring / k_(ring-1) is the schedule's
    refinement ratio at that boundary -- 2 everywhere in 5/10/20/40, but 5
    across the 10->50 boundary of the ablation, so this is deliberately not
    hardwired to four. `validate()` has already guaranteed m is an integer.

    Every child gets:

    * mu_i = mu_p, forced. With nothing to distinguish the children, any other
      assignment invents structure it never measured (§5.1).
    * sigma2_i by (17), strictly larger whenever the ground is not flat.
    * FLAG_DERIVED, and a link to this exact parent value, which is what makes
      Theorem 2 exact rather than nearly exact.
    * n_i = n_p. The child's footprint really was observed n_p times -- at the
      parent's resolution, and FLAG_DERIVED is the field that records that.
      Splitting must not make a cell look unobserved: n is what
      TRAV_CONFIDENCE fails on, so zeroing it would mark a known-good patch of
      road untraversable the moment the vehicle slowed down and refined it.

    Ring 0 is the base 5 cm lattice and cannot be split: semantic refinement
    goes into the pool at ring-0 resolution, never below c0 (master v4 §3.4).
    """
    ring = int(ring)
    if ring < 1:
        raise ValueError(
            f"ring {ring} is the base lattice ({schedule.base_cell_m} m) and has "
            "nothing finer to split into (master v4 §3.4)"
        )
    c_p = schedule.rings[ring].cell_m
    c_c = schedule.rings[ring - 1].cell_m
    m = schedule.k(ring) // schedule.k(ring - 1)

    child = CellValue(
        mu_m=parent.mu_m,
        sigma2_m2=inflate(parent.sigma2_m2, grad_z, c_p, c_c, params),
        n=parent.n,
        flags=(parent.flags | FLAG_DERIVED) & ~FLAG_REFINED,
        derived_from=parent,
    )
    # All m^2 children are identical by construction and CellValue is frozen,
    # so one object shared m^2 times is the same thing as m^2 copies of it.
    return (child,) * (m * m)


# --- merge, math §4 ----------------------------------------------------------


def law_of_total_variance(mu, sigma2, weights=None):
    """Equations (15) and (16), over the last axis. Arrays, no CellValue.

        mu_p     = sum(w_i mu_i)
        sigma2_p = sum(w_i sigma2_i) + sum(w_i (mu_i - mu_p)^2)

    Split out so that when the allocator lands the SoA path can merge a whole
    ring in one call through the same operator `merge()` uses -- the same
    reason lattice.py routes query() and scatter() through one floor divide.

    Returns (mu_p, sigma2_p).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if mu.shape != sigma2.shape:
        raise ValueError(f"mu {mu.shape} and sigma2 {sigma2.shape} disagree")

    if weights is None:
        w = np.full(mu.shape, 1.0 / mu.shape[-1])
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), mu.shape)
        if np.any(w < 0.0):
            raise ValueError("negative merge weight")
        total = w.sum(axis=-1, keepdims=True)
        # All-zero counts is "no child observed yet", which is equal counts,
        # which §4.2 sends to uniform. It is not a division by zero.
        w = np.where(total > 0.0,
                     w / np.where(total > 0.0, total, 1.0),
                     1.0 / mu.shape[-1])

    mu_p = (w * mu).sum(axis=-1)
    within = (w * sigma2).sum(axis=-1)
    between = (w * (mu - mu_p[..., None]) ** 2).sum(axis=-1)
    return mu_p, within + between


def merge(children, weights=None) -> CellValue:
    """m^2 children -> one parent. See math §4.2 for the exact rule.

    Two branches, and §5.4 is explicit that the first is not an optimisation --
    it is what makes split and merge an exact inverse pair:

        all children derived from the same split, none measured since
            -> return that parent value, untouched
        otherwise
            -> equations (15) and (16), genuine marginalisation

    Weights are w_i = n_i / sum(n_j) per §4.2, or uniform when the counts are
    equal (which includes all-zero). Pass `weights` to override.
    """
    children = tuple(children)
    if not children:
        raise ValueError("merge of no children")
    root = round(len(children) ** 0.5)
    if root * root != len(children) or root < 2:
        raise ValueError(
            f"merge takes m^2 children of one parent, got {len(children)}. m is "
            "the schedule's refinement ratio: 2 throughout 5/10/20/40, 5 across "
            "the 10->50 boundary of the ablation."
        )

    # --- the restore branch, §5.4 -------------------------------------------
    p = children[0].derived_from
    if (p is not None
            and all(c.derived for c in children)
            and all(c.derived_from is p for c in children)
            # "no observations since split", literally. A child whose count
            # moved without its bit being cleared means fusion forgot
            # clear_derived(); fall through to (16) rather than restore a
            # value that is no longer the marginal of what is stored.
            and all(c.n == p.n for c in children)):
        return p

    # --- the marginalisation branch, §4.2 -----------------------------------
    mu = np.array([c.mu_m for c in children], dtype=np.float64)
    s2 = np.array([c.sigma2_m2 for c in children], dtype=np.float64)
    counts = np.array([c.n for c in children], dtype=np.float64)
    mu_p, sigma2_p = law_of_total_variance(
        mu, s2, counts if weights is None else weights
    )

    return CellValue(
        mu_m=float(mu_p),
        sigma2_m2=float(sigma2_p),
        # Conservative, and it is what keeps §7.2 Theorem 3 true through a
        # merge: a parent is only as confident as its least-observed child, so
        # a child failing TRAV_CONFIDENCE cannot be averaged into a parent that
        # reports SAFE. Summing the counts would do exactly that.
        n=min(c.n for c in children),
        flags=_merge_flags(children),
        derived_from=None,
    )


def _merge_flags(children) -> int:
    """Bitfield of the merged cell.

    DERIVED and BLIND are properties of the whole footprint -- one measured
    child makes the parent measured, one seen child makes it seen -- so they
    survive only unanimously. DYNAMIC is safety-relevant in the positive
    direction, so any child carries it up. REFINED describes a cell finer than
    range alone asked for, which a merged parent is not, by definition.
    """
    flags = 0
    if all(c.flags & FLAG_DERIVED for c in children):
        flags |= FLAG_DERIVED
    if all(c.flags & FLAG_BLIND for c in children):
        flags |= FLAG_BLIND
    if any(c.flags & FLAG_DYNAMIC for c in children):
        flags |= FLAG_DYNAMIC
    return flags
