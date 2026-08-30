"""Rerun dashboard — `python -m vrgrid.dash`. [JP]

Runs as a SEPARATE PROCESS from the pipeline. If the dashboard falls over two
days before submission the framework must still run and still produce numbers.

Build it against a mock grid on Day 0, before the real grid exists. It is one
of the two things judges actually see.

Shows: the map coloured by ring, the schedule selector, per-stage timings
(median and p95), live memory against the preallocated bound, and the
persistent-unknown fraction.
"""

from .demo_synthetic import main


if __name__ == "__main__":
    main()