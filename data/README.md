# data/

Gitignored. Nothing in here is ever committed.

SemanticKITTI, **sequences 00, 07 and 08 only** — about 40 GB, not the full
200 GB. JP starts this download before anything else on Day 0: it is the one
item on the critical path that neither cleverness nor effort can accelerate.

```
data/
└── sequences/
    ├── 00/{velodyne/*.bin, labels/*.label, poses.txt, calib.txt}
    ├── 07/   ← mapping-parameter tuning
    └── 08/   ← reporting, as the community uses it
```

Tune on 07, report on 08. Never both on the same sequence.

Both the semantic class (19-class) and the motion flag (`moving-*`, IDs
250–259) are read straight from the raw `.label` files. Nothing is inferred
and nothing is retrained — FRNet is not used (the only standalone port
available does not reproduce the trained network). Disclose this: the mapping
contribution is evaluated independently of segmentation quality.
