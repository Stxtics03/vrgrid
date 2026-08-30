# Research Log

Append-only. Newest entries at the bottom. One entry per finding, with who and when.

Format:

```
## YYYY-MM-DD — <name>
**Module:** <research module>
**Finding:**
**Source:**
**So what:** what this changes in the build, if anything.
```

---

## 2026-08-31 — JP
**Module:** β — dynamics & segmentation
**Finding:** The standalone FRNet implementation in `src/perception/frnet/` (a
hand port with no mmcv/mmdet3d dependency) does not reproduce the trained
network. The published checkpoint loads cleanly — 413/413 tensors, no shape
mismatch — but the forward pass is wrong: the backbone uses `nn.LeakyReLU`
where `configs/_base_/models/frnet.py` in the FRNet repo trains with
`act_cfg=dict(type='HSwish')`; the FOV is fed as `fov_up=2.0 / fov_down=-24.8`
where training used `3.0 / -25.0`; and the test-time `RangeInterpolation`
densification (H=64, W=2048) is absent. Measured point accuracy vs GT on
SemanticKITTI seq 00: **16.3% (frame 43), 15.3% (frame 100)** — worse than
predicting "all road". Both frames collapse to ~50–58% *other-ground* where GT
has ~0%. Not frame-specific; systemic.
**Source:** FRNet-master repo configs (`configs/_base_/models/frnet.py`,
`configs/_base_/datasets/semantickitti_seg.py`); direct inference runs on the
pretrained checkpoint `frnet-semantickitti_seg.pth`.
**So what:** FRNet is dropped from the pipeline. Semantic class (19-class) now
comes straight from the SemanticKITTI raw `.label` files via
`semantics.semantic_labels()`, the same source and 19-class scheme as the
`moving-*` motion flag. Both semantic and motion labels are therefore ground
truth — disclose it plainly; it isolates the mapping contribution from
segmentation error, which is what a careful evaluator wants (master v4 §3.6,
risk register). The broken port is kept, flagged non-functional, so a proper
`mmengine`/`mmcv`/`mmdet`/`mmdet3d` install can swap real FRNet back in later if
time allows.
