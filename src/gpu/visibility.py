"""Visibility cleanup — O(1) per cell, no ray casting. [Shrestha]

Math §10.4. For a map cell at `p`, project it into the current range image and
compare ranges:

    (u, v) = proj(p),   r_expected = ||p - p_sensor||
    see-through   iff   R_current(u, v) > r_expected + delta          (32)

If this frame's beam came back from *further away* than the cell, the beam went
through where the cell claims something is, so the cell is empty. That is a
range comparison rather than a 3D traversal: one texture read and one subtract
per cell, fully parallel, no ray marching, and the cost does not grow with map
size the way DDA traversal does.

**The guard is mandatory, not an optimisation.** A cell with a return in the
current scan is never cleared, whatever the range image says. Without it the
cleanup eats fences, poles and sign posts within a few frames: a pole is one
cell wide, the beam that hits it and the beam that passes beside it land in
adjacent range-image columns, and at range they quantise into the same column.
The far beam then "proves" the pole is not there. `test_a_pole_survives_a_
hundred_frames` is the test; it fails within four frames without the guard.

**What this file does not do.** It produces a see-through *mask*. Turning that
into log-odds and a three-state occupancy decision is fusion (math §10.1,
Aakash) -- semantics live in one place, and this file only makes them fast.
`apply_miss()` is offered as the obvious consumer and takes its constants from
`configs/thresholds.yaml` rather than defining any.

⚑ **delta: the doc and the config disagree, and the doc is right.** §10.4 says
"set delta = 3*sigma_r(r) from (12), so the guard band widens with distance
automatically rather than being a hand-tuned constant." `visibility.range_
tolerance_m` in the config is a hand-tuned constant, 0.30 m. Measured against
eq (12) it is 26.9 sigma at 5 m and 1.7 sigma at 100 m -- so it almost never
clears in the near field, which is where ghosts are most dangerous, and it
clears real structure at range, which is what the guard exists to prevent. We
implement 3*sigma(r), and keep the config value as a FLOOR, because pose and
registration error do not shrink with range the way sensor noise does and
something has to cover them. Flagged for the room; nobody's threshold was
changed, one was given a new job.
"""

from dataclasses import dataclass

import numpy as np
from vrgrid.gpu.kernels import measurement_variance_cm2

NO_RETURN = np.inf  # range-image pixels where the beam came back from nothing


@dataclass(frozen=True)
class Sensor:
    """Projection geometry, from `configs/thresholds.yaml: sensor`."""

    height_m: float = 1.73
    phi_min_deg: float = -24.8
    phi_max_deg: float = 2.0
    sigma_r_m: float = 0.02
    sigma_phi_deg: float = 0.1

    @classmethod
    def from_config(cls, thresholds: dict) -> "Sensor":
        s = thresholds["sensor"]
        return cls(height_m=s["height_m"], phi_min_deg=s["phi_min_deg"],
                   phi_max_deg=s.get("phi_max_deg", 2.0),
                   sigma_r_m=s["sigma_r_m"], sigma_phi_deg=s["sigma_phi_deg"])


# The default sensor, as one module-level instance rather than a fresh object
# per call. Callers with a different sensor pass `Sensor.from_config(...)`.
HDL64E = Sensor()


def clear_tolerance_m(range_m, sensor: "Sensor | None" = None, floor_m: float = 0.30):
    """delta for eq (32): three sigma of the height model, floored.

    math §10.4 wants the band to widen with distance on its own. The floor
    covers pose and registration error, which does not. Kept as the readable
    definition; `_delta_into()` is the same arithmetic in preallocated buffers,
    and `test_the_scratch_path_matches_the_readable_one` holds them together.
    """
    sensor = sensor or HDL64E
    sigma_m = np.sqrt(measurement_variance_cm2(
        range_m, sigma_r_m=sensor.sigma_r_m,
        sigma_phi_rad=np.radians(sensor.sigma_phi_deg),
        h_s_m=sensor.height_m)) / 100.0
    return np.maximum(3.0 * sigma_m, floor_m)


def _delta_into(out, r, t, sensor: Sensor, floor_m: float) -> None:
    """`clear_tolerance_m` with every intermediate written through an `out=`.

    sigma_z^2 = (h/r)^2 sigma_r^2 + r^2 sigma_phi^2, eq (12) at normal
    incidence. Done in metres throughout -- the cm round trip in
    `measurement_variance_cm2` cancels exactly (sqrt(v*1e4)/100 == sqrt(v)).
    """
    sigma_phi = np.radians(sensor.sigma_phi_deg)
    np.maximum(r, 1e-3, out=t)
    np.divide(sensor.height_m, t, out=t)
    np.multiply(t, t, out=t)
    np.multiply(t, sensor.sigma_r_m ** 2, out=t)      # near-field floor term
    np.multiply(r, r, out=out)
    np.multiply(out, sigma_phi ** 2, out=out)          # the r^2 term, dominant
    np.add(out, t, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, 3.0, out=out)
    np.maximum(out, floor_m, out=out)


def spherical_project(x_m, y_m, z_m, shape, sensor: "Sensor | None" = None, xp=np,
                      scratch: dict | None = None):
    """Vehicle-frame points -> range-image (u, v) and true range from the sensor.

    Vehicle frame is x forward, y left, z up (docs/frames.md); the sensor sits
    at (0, 0, height_m), so the range this returns is measured FROM THE SENSOR
    and not from the vehicle origin. Getting that wrong biases every comparison
    by 1.73 m and produces a map that looks entirely plausible.

    The image shape is authoritative for the azimuth resolution -- the range
    image is JP's artefact and this reads its geometry rather than assuming it.

    ⚑ When `perception/range_image.py` lands, this should be replaced by a call
    into JP's `project()`. Two projections in one system is exactly the drift
    that `docs/frames.md` exists to prevent; this one is here so the kernel can
    be built and tested before his exists, not to compete with it.
    """
    sensor = sensor or HDL64E
    height, width = shape
    x = xp.asarray(x_m)
    y = xp.asarray(y_m)
    n = len(x)
    if scratch is None:
        scratch = new_visibility_scratch(max(n, 1))
    dz, r = scratch["dz"][:n], scratch["r"][:n]
    t1, t2 = scratch["t1"][:n], scratch["t2"][:n]
    u, v, in_view = scratch["u"][:n], scratch["v"][:n], scratch["in_view"][:n]

    xp.subtract(xp.asarray(z_m), sensor.height_m, out=dz)
    xp.multiply(x, x, out=r)
    xp.multiply(y, y, out=t1)
    xp.add(r, t1, out=r)
    xp.multiply(dz, dz, out=t1)
    xp.add(r, t1, out=r)
    xp.sqrt(r, out=r)

    # Azimuth wraps, so u is always in view; only elevation can leave the image.
    xp.arctan2(y, x, out=t1)
    xp.divide(t1, xp.pi, out=t1)
    xp.subtract(1.0, t1, out=t1)
    xp.multiply(t1, 0.5 * width, out=t1)
    xp.floor(t1, out=t1)
    xp.clip(t1, 0, width - 1, out=t1)
    xp.copyto(u, t1, casting="unsafe")

    phi_min = xp.radians(sensor.phi_min_deg)
    phi_max = xp.radians(sensor.phi_max_deg)
    xp.maximum(r, 1e-9, out=t1)
    xp.divide(dz, t1, out=t2)
    xp.clip(t2, -1.0, 1.0, out=t2)
    xp.arcsin(t2, out=t2)
    xp.subtract(phi_max, t2, out=t2)
    xp.divide(t2, phi_max - phi_min, out=t2)
    xp.multiply(t2, height, out=t2)
    xp.floor(t2, out=t2)

    xp.greater_equal(t2, 0.0, out=in_view)
    xp.less(t2, height, out=scratch["mask"][:n])
    xp.logical_and(in_view, scratch["mask"][:n], out=in_view)
    xp.greater(r, 1e-6, out=scratch["mask"][:n])
    xp.logical_and(in_view, scratch["mask"][:n], out=in_view)

    xp.clip(t2, 0, height - 1, out=t2)
    xp.copyto(v, t2, casting="unsafe")
    return u, v, r, in_view


@dataclass
class CleanupResult:
    """Counts, because "the cleanup works" is not a claim anyone can check."""

    see_through: np.ndarray  # bool per candidate cell -- the kernel's product
    tested: int
    out_of_view: int
    protected: int           # would have been cleared; had a return this scan
    cleared: int

    @property
    def protected_fraction(self) -> float:
        would_clear = self.protected + self.cleared
        return self.protected / would_clear if would_clear else 0.0


def new_visibility_scratch(max_cells: int) -> dict:
    """Preallocated working set, sized by candidate cells. Same contract as the
    scatter scratch: pass it every frame, never grow it inside the loop."""
    f64 = ("dz", "r", "t1", "t2", "observed", "delta")
    i32 = ("u", "v")
    b = ("in_view", "mask", "see_through", "guard")
    s = {name: np.zeros(max_cells, np.float64) for name in f64}
    s.update({name: np.zeros(max_cells, np.int32) for name in i32})
    # `flat` is intp, not int32, and the gather below asks for mode="clip".
    # Measured at 200,000 cells: intp + clip allocates 1 KB a frame, intp +
    # the default "raise" allocates 1.6 MB, and int32 + "raise" allocates 3.2
    # MB -- numpy copies the index array to bounds-check it, and copies it
    # again to widen it. The indices are already clamped in view, so "clip"
    # removes a check that cannot fire.
    s["flat"] = np.zeros(max_cells, np.intp)
    s.update({name: np.zeros(max_cells, np.bool_) for name in b})
    return s


def visibility_cleanup(x_m, y_m, z_m, range_image, has_return_now=None,
                       sensor: "Sensor | None" = None, floor_m: float = 0.30,
                       protect_current_returns: bool = True,
                       scratch: dict | None = None) -> CleanupResult:
    """Eq (32) with the mandatory guard. One pass, no traversal, no allocation
    when handed a scratch.

    `x_m, y_m, z_m` are candidate cell centres in the vehicle frame -- normally
    the currently-OCCUPIED cells, not the whole map, since a free cell has
    nothing to clear. `has_return_now` is the guard: True for cells that
    received a return in this scan, which `scatter()` already knows.
    """
    sensor = sensor or HDL64E
    n = len(np.asarray(x_m))
    if scratch is None:
        scratch = new_visibility_scratch(max(n, 1))
    cap = len(scratch["u"])
    if n > cap:
        raise ValueError(f"{n:,} candidate cells exceeds the visibility scratch "
                         f"capacity of {cap:,}; size it at startup, not here")
    if n == 0:
        return CleanupResult(scratch["see_through"][:0], 0, 0, 0, 0)

    u, v, r, in_view = spherical_project(x_m, y_m, z_m, range_image.shape,
                                         sensor, scratch=scratch)
    observed, delta = scratch["observed"][:n], scratch["delta"][:n]
    out, t1, flat = scratch["see_through"][:n], scratch["t1"][:n], scratch["flat"][:n]

    # One gather out of the flattened image. `range_image[v, u]` would build a
    # fresh array every frame; so, more quietly, would np.take with int32
    # indices or the default bounds-checking mode -- see new_visibility_scratch.
    width = range_image.shape[1]
    np.multiply(v, width, out=flat)
    np.add(flat, u, out=flat)
    np.take(range_image.reshape(-1), flat, out=observed, mode="clip")

    _delta_into(delta, r, t1, sensor, floor_m)

    # (32): the beam returned from beyond the cell, so it passed through it.
    # A pixel with no return proves nothing -- the beam may have been absorbed,
    # hit glass, or gone to the sky -- so NO_RETURN must not clear anything.
    # Comparing against inf would clear the entire map, which is the failure
    # mode this comment exists to prevent.
    mask = scratch["mask"][:n]
    np.add(r, delta, out=delta)
    np.greater(observed, delta, out=out)
    np.logical_and(out, in_view, out=out)
    np.isfinite(observed, out=mask)
    np.logical_and(out, mask, out=out)

    out_of_view = n - int(in_view.sum())
    protected = 0
    if protect_current_returns and has_return_now is not None:
        guard = scratch["guard"][:n]
        np.copyto(guard, np.asarray(has_return_now, dtype=bool))
        np.logical_and(out, guard, out=mask)
        protected = int(np.count_nonzero(mask))
        np.logical_not(guard, out=guard)
        np.logical_and(out, guard, out=out)

    return CleanupResult(out, n, out_of_view, protected, int(np.count_nonzero(out)))


def apply_miss(log_odds: np.ndarray, cells: np.ndarray, see_through: np.ndarray,
               log_odds_miss: int, clamp: tuple) -> int:
    """Fold a see-through mask into occupancy. Thin on purpose -- the occupancy
    model is math §10.1 and belongs to fusion, not here.

    Clamping is what makes a map able to change its mind: an unclamped cell with
    500 free observations needs 500 occupied ones to register a new obstacle.
    """
    hit = cells[see_through]
    if hit.size:
        np.clip(log_odds[hit].astype(np.int32) + log_odds_miss, clamp[0], clamp[1],
                out := np.empty(hit.size, np.int32))
        log_odds[hit] = out
    return int(hit.size)


def visibility_scratch_bytes(max_cells: int) -> int:
    """What `new_visibility_scratch` costs, for the memory bound.

    Not folded into `allocate()` yet: doing so means fixing a cap on candidate
    cells per frame, which moves the headline total. That is a team decision of
    the same kind as the transient-layer line, not one to make quietly inside
    src/gpu. 68 B per candidate cell -- 13.6 MB at 200,000.
    """
    return sum(a.nbytes for a in new_visibility_scratch(max_cells).values())
