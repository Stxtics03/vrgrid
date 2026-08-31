# Agent Instructions — JP's scope only

This repo is a shared team project (Stxtics03/vrgrid). I am JP and I only own
the perception front-end + dashboard, per docs/jp-working-brief.md Part 2 and Part 3.

## Files I own — edit these freely
- loader.py
- transforms.py
- dashboard.py
- docs/frames.md

## Files I do NOT own — never modify, refactor, or "fix" these
- Grid engine / evaluation code (Aakash's work: lattice, rings, toroidal shift,
  split/merge, memory bound, reference map, plan-regret study)
- GPU kernels / memory code (Shrestha's work: scatter(), shift kernel, visibility
  cleanup, preallocation, allocating 3D baseline)
- Prior-art docs (Srinivas)
- Segmentation/dynamics code beyond what I explicitly call (Hriday)
- Traversability/evaluation code (Pratyushi)
- docs/sih-math.md (append-only reference, not to be edited)

If you notice an issue in code outside my scope, mention it in your response —
do not edit it.

## Conventions to follow everywhere
- Ranges/gradients are float metres, suffix `_m`. Heights are int16 1cm, suffix `_cm`.
  Never mix silently.
- Checkpoint and dataset paths always go in configs/, never inline.
- Every coordinate transform must be documented in docs/frames.md in the same commit.
- Vehicle frame: x forward, y left, z up.

## Testing
- Run `pytest -q` before considering any task done.
- New tests for my modules go under tests/, committed — not run manually and discarded.