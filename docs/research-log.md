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
---
---
## 2026-08-28 - Hriday
**Module:** R2 (Dynamics & Segmentation)
**Finding:** FRNet (19-class SemanticKITTI checkpoint) requires a 64x512 spherical projection. Standard model is 10M params (73.3% mIoU). Fast-FRNet fallback is 7.5M params. Dynamic classes are explicitly mapped to IDs 252 (moving-car), 253 (moving-bicyclist), 254 (moving-person), 255 (moving-motorcyclist).
**Source:** FRNet GitHub (Xiangxu-0103/FRNet) & paper (arXiv:2312.04484).
**So what:** D2 (JP) must build a projection with an inverse index using exact spherical math ($r, \theta, \phi \rightarrow u, v$). **Use Velodyne FOV bounds for the projection: +2° (top) to -24.9° (bottom).** If the main pipeline OOMs (runs out of memory), swap to the 7.5M Fast-FRNet checkpoint immediately. These four moving IDs are the sole triggers for ghost removal.
---