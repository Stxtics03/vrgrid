"""The height-variance codec: float cm² <-> the cell's one log-quantised byte.

[Aakash — Day 1, forced by §3.3]

`cell.height_variance` is a uint8 and the struct comments say "log-quantised"
without saying how. Nothing could be built on §3 until the scheme existed, so
it is defined here. Three decisions, all of which change behaviour:

**Code 0 is MAXIMUM variance, not minimum.** `allocate()` zeros every field and
the ego-motion shift zeros each newly exposed strip, so a cell that has never
been looked at — and every cell that just scrolled into view — arrives holding
0. If 0 decoded to the smallest variance, the map would come up claiming
millimetre certainty about ground it has never seen, and the first Kalman gain
would be ~0, so it would never recover: the cell would ignore every subsequent
measurement. Inverting the scale makes zeroed memory mean "I know nothing",
which is the same reason `OCC_UNKNOWN` is 0 and the blind cone is unknown
rather than free.

**The floor is the 1 cm quantisation step, not zero.** Heights are stored to
1 cm, so σ² below q²/12 = 0.083 cm² (§3.4's σ = 2.9 mm) is a claim the storage
cannot express. A filter allowed to drive σ² to zero also locks: `K = 0` and
the cell stops responding to evidence. §3.3 adds process noise for the same
reason; this is the storage-side floor of the same argument.

**The ceiling is the vertical extent, (8 m)².** The map spans −2 to +6 m, so a
cell with no information has a height uncertainty of that order. Beyond it
there is nothing to distinguish.

Resolution that gives: 255 codes over a dynamic range of 7.7e6 in variance, so
one code is a factor of 1.064 — **6.4% in variance, 3.2% in σ**. Two
consequences worth knowing rather than discovering:

- Nothing finer than 6.4% is representable, so §5's Theorem 1 (split strictly
  inflates variance) is only *observable* in the stored map when the slope
  term clears 6.4%. On a 20 cm ring at ‖∇z‖ = 0.2 it does not. The theorem is
  unaffected; its visibility in the map is. This is the gap flagged in
  `splitmerge.CellValue`, now measured rather than asserted.
- Rounding is toward the LARGER variance (floor in code space), so the stored
  value is always ≥ the true one. The map never claims more certainty than it
  earned, and a decreasing variance can never round back up — which is what
  keeps §3.4's monotonicity test true through the codec.
"""

import numpy as np

# (1 cm)^2 / 12 -- uniform quantisation noise of the 1 cm height step, math
# §3.4. Nothing the map stores can be more certain than the step it stores in.
SIGMA2_MIN_CM2 = 1.0 / 12.0

# (8 m)^2 in cm^2: the -2..+6 m vertical extent, i.e. total ignorance about a
# height that is nonetheless known to be inside the map.
SIGMA2_MAX_CM2 = 800.0**2

CODES = 256
_MAX_CODE = CODES - 1
_LOG_RANGE = np.log(SIGMA2_MAX_CM2 / SIGMA2_MIN_CM2)

# Factor between adjacent codes: 1.064, i.e. 6.4% in variance, 3.2% in sigma.
CODE_RATIO = float(np.exp(_LOG_RANGE / _MAX_CODE))


def dequantise_variance_cm2(code):
    """Code -> variance in cm². Scalar in -> float out, array in -> array out.

        sigma2(code) = SIGMA2_MAX * (SIGMA2_MIN/SIGMA2_MAX)^(code/255)

    Monotonically DECREASING in `code`: 0 is "no information", 255 is the
    quantisation floor. See the module docstring for why that way round.
    """
    c = np.asarray(code, dtype=np.float64)
    out = SIGMA2_MAX_CM2 * np.exp(-_LOG_RANGE * c / _MAX_CODE)
    return float(out) if np.ndim(out) == 0 else out


def quantise_variance_cm2(sigma2_cm2):
    """Variance in cm² -> uint8 code, rounded toward LARGER variance.

    Floor in code space, so `dequantise(quantise(v)) >= v` always, up to the
    clamp at the floor. Conservative in the direction the whole project is
    conservative in: an overstated variance costs a cell some sharpness, an
    understated one makes the filter ignore real evidence.
    """
    v = np.maximum(np.asarray(sigma2_cm2, dtype=np.float64), 1e-12)
    exact = _MAX_CODE * np.log(SIGMA2_MAX_CM2 / v) / _LOG_RANGE
    code = np.clip(np.floor(exact), 0, _MAX_CODE)
    out = code.astype(np.uint8)
    return np.uint8(out) if np.ndim(out) == 0 else out


def sigma_cm(code):
    """Standard deviation in cm, the unit the §3.2 table is quoted in."""
    return np.sqrt(dequantise_variance_cm2(code))
