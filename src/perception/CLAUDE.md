# src/perception — JP

- **Frames.** Vehicle frame is x forward, y left, z up. Every transform you
  write goes in `docs/frames.md` in the same commit — origin, axes,
  handedness, units. Frame confusion is the most common silent bug here: the
  map looks plausible and slowly rotates. Run the static-wall test on Day 0.
- **Wire things in, do not rebuild them.** Patchwork++ for ground, KISS-ICP
  for odometry. The standalone FRNet port in `frnet/` **works as of 2 Sep** —
  98.3% point accuracy, 69.8% mIoU against the paper's 73.3%. It is still not
  the map's semantic source, and that is now a choice rather than a defect: GT
  `.label` files isolate the mapping contribution from segmentation quality.
  Run it with `scripts/frnet_eval.py`; report it alongside the map, never
  swapped into it.
- **The FOV bug was in `semantics.py`, not in `frnet/`.** `configs/frnet.yaml`
  carries the HDL-64E's *physical* vertical FOV (2.0 / −24.8) for the range
  image; the checkpoint learned a *fixed* spherical projection (3.0 / −25.0).
  They are different quantities and feeding the first to the model was one of
  the three things that held it at ~15%. The training values are pinned as
  `FRNET_TRAIN_FOV_UP_DEG` / `_DOWN_DEG` where a sensor config cannot reach
  them — do not re-plumb them from a config.
- **Both semantic class and motion are ground truth**, read from the raw
  `.label` files: 19-class semantic via `semantics.semantic_labels()`, motion
  (`moving-*`, IDs 250-259) via `semantics.is_moving()`. Disclose it; it
  isolates the mapping contribution from segmentation error, which is what a
  careful evaluator wants. Zero training, zero inference.
- **Dataset:** SemanticKITTI sequences 00, 07, 08 only (~40 GB). Cache format
  is yours to choose, but it must be deterministic — the same sequence must
  produce byte-identical inputs twice.
- **Units.** Ranges and gradients are float metres, heights are int16 in 1 cm.
  Suffix every variable `_m` or `_cm`. Never mix silently.
- **Checkpoint and dataset paths live in `configs/`**, never inline.
