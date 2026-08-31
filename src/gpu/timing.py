"""Per-stage latency instrumentation. [Shrestha]

Two rules this file exists to enforce.

**p50 AND p99, never the mean.** A system averaging 40 FPS with 200 ms spikes
is unsafe, and the spikes almost always come from allocation or the map-shift
path -- both mine. A mean hides exactly the failure the timing harness is for.

**No allocation in the frame loop.** A timing harness that appends to a list
every frame is itself an allocation in the loop, which would make the harness a
source of the jitter it is meant to measure. Samples go into a preallocated
circular buffer per stage, sized at construction.

Headroom against the 10 Hz sensor rate is FPS / 10 -- 40 FPS is 4x headroom,
not 3x. `summary()` computes it so nobody does that subtraction by hand.

Percentiles use nearest-rank (`method="higher"`), not numpy's default linear
interpolation. Interpolation reports a latency that never occurred and rounds
the tail DOWN: with 100 frames at 10 ms and one at 500 ms, the default returns
a p99 of 14.9 ms -- a number no frame ever took, and a 33x under-statement of
the spike. Under-reporting the tail is precisely the failure this harness
exists to prevent, so every percentile here is a real observed sample.
"""

import time
from contextlib import contextmanager

import numpy as np

SENSOR_HZ = 10.0

# The pipeline levels of master v4 §3.5, in order. Fixing the names here keeps
# the dashboard's stage list stable and stops two modules inventing two
# spellings of "range image".
# `ground`, `reflectivity`, `bin` and `shift` were missing from the original
# list, which was written before those stages existed as separately timeable
# things. Adding a name is additive -- `summary()` omits stages with no
# samples, and nothing outside this file reads the tuple -- but RENAMING one
# would break the dashboard's stage list, which is what fixing the spellings
# here was for.
#
# `bin` is the point-to-slot step. It has no owning module (see the Gate 3
# review); the name exists here so the thing can at least be measured under
# one spelling while that is settled.
STAGES = (
    "load", "transform", "range_image", "semantics", "motion",
    "ground", "reflectivity",
    "bin", "scatter", "fuse", "split_merge", "cleanup", "pyramid", "shift",
    "total",
)


class Timer:
    """Fixed-capacity per-stage timing. Allocates once, in __init__."""

    def __init__(self, stages=STAGES, capacity: int = 4096):
        self.capacity = capacity
        self._names = tuple(stages)
        self._buf = np.zeros((len(self._names), capacity), dtype=np.float64)
        self._n = np.zeros(len(self._names), dtype=np.int64)   # total ever recorded
        self._index = {name: i for i, name in enumerate(self._names)}

    def record(self, stage: str, dt_ms: float) -> None:
        i = self._index[stage]
        self._buf[i, self._n[i] % self.capacity] = dt_ms
        self._n[i] += 1

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - t0) * 1e3)

    def _samples(self, stage: str):
        i = self._index[stage]
        n = min(int(self._n[i]), self.capacity)
        return self._buf[i, :n]

    def summary(self) -> dict:
        """p50, p99, max and count per stage. Empty stages are omitted."""
        out = {}
        for name in self._names:
            s = self._samples(name)
            if s.size == 0:
                continue
            out[name] = {
                "p50_ms": float(np.percentile(s, 50, method="higher")),
                "p99_ms": float(np.percentile(s, 99, method="higher")),
                "max_ms": float(s.max()),
                "n": int(self._n[self._index[name]]),
            }
        return out

    def headroom(self, stage: str = "total") -> dict | None:
        """FPS and headroom against the sensor rate, at p50 and at p99.

        Report the p99 headroom too. A pipeline that clears 10 Hz on the median
        and misses it one frame in a hundred has dropped a frame of obstacles.
        """
        s = self._samples(stage)
        if s.size == 0:
            return None
        p50 = float(np.percentile(s, 50, method="higher"))
        p99 = float(np.percentile(s, 99, method="higher"))
        return {
            "fps_p50": 1e3 / p50,
            "fps_p99": 1e3 / p99,
            "headroom_p50": (1e3 / p50) / SENSOR_HZ,
            "headroom_p99": (1e3 / p99) / SENSOR_HZ,
            "meets_sensor_rate": p99 <= 1e3 / SENSOR_HZ,
        }

    def snapshot(self) -> dict:
        """A detached copy for the dashboard.

        Rendering is decoupled from processing: JP's dashboard reads a snapshot
        at its own rate and can never throttle the pipeline. It must not hold a
        reference into the live buffer.
        """
        return {"stages": self.summary(), "headroom": self.headroom()}

    def reset(self) -> None:
        """Zero the counts without reallocating. For warm-up discard."""
        self._n[:] = 0

    def table(self) -> str:
        rows = [f"{'stage':<14}{'p50 ms':>9}{'p99 ms':>9}{'max ms':>9}{'n':>7}"]
        rows.append("-" * 48)
        for name, s in self.summary().items():
            rows.append(f"{name:<14}{s['p50_ms']:>9.2f}{s['p99_ms']:>9.2f}"
                        f"{s['max_ms']:>9.2f}{s['n']:>7}")
        h = self.headroom()
        if h:
            rows.append("")
            rows.append(f"{h['fps_p50']:.1f} FPS p50 ({h['headroom_p50']:.1f}x headroom), "
                        f"{h['fps_p99']:.1f} FPS p99 ({h['headroom_p99']:.1f}x)")
            rows.append("meets 10 Hz at p99" if h["meets_sensor_rate"]
                        else "MISSES 10 Hz at p99")
        return "\n".join(rows)


# Module-level default so a stage can be timed without threading a Timer
# through every call. Tests and the pipeline pass their own.
default_timer = Timer()


@contextmanager
def stage(name: str, timer: Timer | None = None):
    with (timer or default_timer).stage(name):
        yield
