"""Ring schedule loading and validation. Master v4 §3.1.

This is the one piece of the grid that had to exist on Day 0, because the
config files are frozen against it and CI checks them. Aakash owns it from
here; rewrite freely, keep `validate()` rejecting the same two things.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Ring:
    ring: int
    half_width_m: float
    cell_m: float
    cells: int
    s_az_cm: float


@dataclass
class Anisotropy:
    """Speed-scaled foveation parameters. Math §6.2 eq. (20).

    Defaults are the isotropic case: at v = 0 every stretch is 1 and (20)
    collapses to the plain Chebyshev norm of (18), which is what makes the
    anisotropic path safe to run always rather than as a special case.
    """

    kappa_forward: float = 1.0
    kappa_side: float = 0.5
    v_ref_ms: float = 15.0
    rear_stretch: float = 1.0          # a_r: the rear is never stretched
    rear_floor_cell_m: float = 0.20    # hard floor within 50 m behind


@dataclass
class Schedule:
    name: str
    base_cell_m: float
    rings: list
    total_cells: int
    vertical_extent_m: tuple
    hysteresis_eps: float
    anisotropy: Anisotropy = field(default_factory=Anisotropy)

    def k(self, ring: int) -> int:
        """Integer divisor from the base lattice to ring `ring` (math §2.1)."""
        return round(self.rings[ring].cell_m / self.base_cell_m)


class ScheduleError(ValueError):
    pass


def load(name_or_path) -> Schedule:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"schedule_{str(name_or_path).replace('/', '_')}.yaml"
    with open(p) as f:
        raw = yaml.safe_load(f)
    s = Schedule(
        name=raw["name"],
        base_cell_m=raw["base_cell_m"],
        rings=[Ring(**r) for r in raw["rings"]],
        total_cells=raw["total_cells"],
        vertical_extent_m=tuple(raw["vertical_extent_m"]),
        hysteresis_eps=raw["hysteresis_eps"],
        anisotropy=Anisotropy(**raw.get("anisotropy", {})),
    )
    validate(s)
    return s


def validate(s: Schedule) -> None:
    """Reject a schedule that would break the lattice, and warn about one that
    would merely be nonsense.

    Two checks, both from master v4 §3.1:

    1. HARD -- every consecutive ratio must be an integer. Powers of two are a
       convenience (bit shift), not a requirement: 5/10/50 is legal because
       10/5 = 2 and 50/10 = 5. Non-integer ratios make the lattices drift apart
       in floating point and produce gaps and double-counts at ring boundaries.
    2. SOFT -- cell size must stay within 2x of the sensor's azimuthal spacing
       s_az at that range. 5 cm cells out to 100 m passes check 1 and is still
       nonsense (flaw E4).
    """
    if not s.rings:
        raise ScheduleError("schedule has no rings")

    for r in s.rings:
        ratio = r.cell_m / s.base_cell_m
        if abs(ratio - round(ratio)) > 1e-9:
            raise ScheduleError(
                f"ring {r.ring}: cell {r.cell_m} m is not an integer multiple "
                f"of the base lattice {s.base_cell_m} m"
            )

    for a, b in zip(s.rings, s.rings[1:]):
        ratio = b.cell_m / a.cell_m
        if abs(ratio - round(ratio)) > 1e-9:
            raise ScheduleError(
                f"non-integer ratio between rings {a.ring} and {b.ring}: "
                f"{b.cell_m} / {a.cell_m} = {ratio:.4f}"
            )
        if b.half_width_m <= a.half_width_m:
            raise ScheduleError("ring half-widths must strictly increase")

    warnings = []
    for r in s.rings:
        if r.s_az_cm and not (0.5 <= (r.cell_m * 100) / r.s_az_cm <= 2.0):
            warnings.append(
                f"ring {r.ring}: cell {r.cell_m * 100:.0f} cm diverges more "
                f"than 2x from s_az = {r.s_az_cm:.1f} cm at that range"
            )
    if warnings:
        import warnings as _w

        _w.warn("; ".join(warnings), stacklevel=2)
