#!/usr/bin/env python3
"""FRNet semantic segmentation on real SemanticKITTI. Point accuracy and mIoU.

    python scripts/frnet_eval.py [--seq 08] [--frames 200] [--checkpoint PATH]

Gate 6 says every number on a slide comes from a script. This is the script for
the deep-learning half of the problem statement: the pretrained FRNet
checkpoint, through our standalone port, scored per point against the
SemanticKITTI `.label` ground truth.

⚑ THIS IS NOT THE MAP'S SEMANTIC SOURCE. The pipeline takes semantics from the
  `.label` files on purpose -- that isolates the mapping contribution from
  segmentation quality, which is what §9's evaluation is for. This model is
  reported ALONGSIDE the map, never swapped into it, so no mapping number
  depends on it.

⚑ mIoU IS A DATASET METRIC. Intersections and unions accumulate across frames
  and are divided once at the end. Averaging per-frame mIoU inflates it,
  because a frame holding three easy classes outscores one holding nineteen.

⚑ AND "PRESENT" MEANS PRESENT IN THE GROUND TRUTH. An earlier version counted a
  class whenever `union > 0`, which is true if the model hallucinates a single
  point of a class that never occurs. Over 200 frames of seq 08, truck,
  motorcycle, motorcyclist and other-vehicle have ZERO ground-truth points;
  averaging four guaranteed zeros dragged the reported mIoU from 69.8% to
  51.5%. Ground-truth support is printed beside every class so the reader can
  see what each number rests on.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

CLASSES = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
           "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
           "other-ground", "building", "fence", "vegetation", "trunk",
           "terrain", "pole", "traffic-sign"]
#: The five §7.1 consults on every cell. The map never distinguishes the rest
#: from each other -- they are all simply "not drivable".
DRIVABLE = ("road", "parking", "sidewalk", "other-ground", "terrain")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default="08",
                    help="SemanticKITTI's official validation sequence")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--checkpoint", default="checkpoints/frnet-semantickitti_seg.pth")
    ap.add_argument("--fast-scatter", action="store_true",
                    help="swap the frustum reductions for torch.scatter_reduce via "
                         "scripts/frnet_fast_scatter.py -- ~35 min becomes ~1 min. "
                         "Verifies equivalence before patching; JP's port is not edited")
    args = ap.parse_args()

    import torch
    from vrgrid.perception import loader, semantics
    from vrgrid.perception.frnet import FRNet

    if args.fast_scatter:
        sys.path.insert(0, str(Path(__file__).parent))
        from frnet_fast_scatter import enable
        enable(verify=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FRNet(num_classes=20, ignore_index=19, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                  fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        blob.get("state_dict", blob), strict=False)
    model.to(dev).eval()
    print(f"{ckpt.name} on {dev}: {len(missing)} missing, {len(unexpected)} "
          f"unexpected tensors (auxiliary heads are training-only)")

    inter = np.zeros(19, np.int64)
    union = np.zeros(19, np.int64)
    support = np.zeros(19, np.int64)
    correct = total = frames = 0

    root = Path(loader.DATA_ROOT) / "sequences" / args.seq
    for i in range(args.frames):
        scan = root / "velodyne" / f"{i:06d}.bin"
        label = root / "labels" / f"{i:06d}.label"
        if not scan.exists() or not label.exists():
            break
        pts = loader.load_velodyne_scan(scan)
        gt = semantics.semantic_labels(loader.load_labels(label))
        with torch.no_grad():
            pred = model.predict(
                [torch.from_numpy(pts).float().to(dev)])[0].cpu().numpy()
        ok = gt >= 0
        correct += int((pred[ok] == gt[ok]).sum())
        total += int(ok.sum())
        for c in range(19):
            p, g = pred[ok] == c, gt[ok] == c
            inter[c] += int((p & g).sum())
            union[c] += int((p | g).sum())
            support[c] += int(g.sum())
        frames += 1

    if not total:
        print("no labelled frames found", file=sys.stderr)
        return 1

    seen = support > 0
    iou = np.zeros(19)
    iou[seen] = inter[seen] / union[seen]

    print(f"\nsequence {args.seq}, {frames} frames, {total:,} labelled points")
    # Held in a name rather than inlined: a backslash escape inside an f-string
    # is a syntax error before Python 3.12 and this project targets >=3.10.
    reductions = ("torch.scatter_reduce (--fast-scatter)" if args.fast_scatter
                  else "the port's own loops")
    print(f"  reductions                {reductions}")
    print(f"  point accuracy            {correct / total:>6.1%}")
    print(f"  mIoU over {int(seen.sum()):>2} present classes  {iou[seen].mean():>6.1%}"
          f"   (paper: 73.3% over all 4,071 frames)")

    print(f"\n  {'class':<16}{'IoU':>8}{'gt points':>14}")
    print("  " + "-" * 38)
    for c in np.argsort(-iou):
        if seen[c]:
            print(f"  {CLASSES[c]:<16}{iou[c]:>7.1%}{support[c]:>14,}")
    absent = [CLASSES[c] for c in range(19) if not seen[c]]
    if absent:
        print(f"\n  no ground truth in this slice, excluded: {', '.join(absent)}")

    drive = [c for c in range(19) if CLASSES[c] in DRIVABLE and seen[c]]
    if drive:
        print(f"\n  §7.1 drivable set only: mIoU {iou[drive].mean():>5.1%} "
              f"over {len(drive)} classes -- the only ones the map consults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
