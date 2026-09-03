#!/usr/bin/env python3
"""Fine-tune the FRNet decode head on SemanticKITTI, reproducibly.

    python scripts/frnet_finetune.py --steps 600 --out checkpoints/frnet-tuned.pth
    python scripts/frnet_eval.py --checkpoint checkpoints/frnet-tuned.pth

Gate 6 says every number on a slide comes from a script in `scripts/`. The
-0.5 mIoU fine-tuning result in `docs/handover-2026-09-02.md` is on a slide and
had **no script behind it** -- the run was done from a scratch file that is
gone, and `checkpoints/frnet-finetuned-terrain.pth` holds a bare `state_dict`
with no optimiser, no step count and no record of the recipe. This is that
script, written so the negative result can be re-derived and so the next run
can actually resume.

⚑ TRAINING LOSS IS NOT A RESULT, and this recipe is the reason. A class-weighted
  loss falls when the head grows more CONFIDENT on the weighted classes whether
  or not it grows more CORRECT: the 2 Sep run fell 0.167 -> 0.1425 and lost
  0.5 mIoU. The only honest read is `scripts/frnet_eval.py` on the held-out
  sequence, and this script refuses to train on that sequence.

⚑ THIS IS NOT THE MAP'S SEMANTIC SOURCE. The pipeline takes semantics from the
  `.label` files on purpose, which isolates the mapping contribution from
  segmentation quality. A checkpoint out of this script is reported alongside
  the map and never swapped into it, so no mapping number depends on it.
  CLAUDE.md's "don't retrain anything" governs that pipeline; this is the
  separately-reported DL half of the problem statement.

⚑ FREEZING WEIGHTS IS NOT FREEZING A MODULE. `model.train()` puts every
  BatchNorm in the network into training mode, so a "frozen" backbone still
  drifts its running mean and variance on every forward pass -- the weights
  stay put and the function computed does not. Frozen modules are held in
  `.eval()` here for exactly that reason, and `--no-freeze-bn` exists to
  reproduce the other behaviour rather than to hide it.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from vrgrid.perception import loader, semantics
from vrgrid.perception.frnet import FRNet

#: SemanticKITTI's official split. 08 is validation and is what frnet_eval.py
#: scores, so it is never trained on -- asserted below, not merely intended.
TRAIN_SEQUENCES = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
HOLDOUT_SEQUENCE = "08"

CLASSES = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
           "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
           "other-ground", "building", "fence", "vegetation", "trunk",
           "terrain", "pole", "traffic-sign"]
#: The model's own ignore slot. `semantics.semantic_labels` returns -1 for
#: unlabeled; the loss wants that as index 19, which FRNet declares as its
#: ignore_index and which frnet_eval.py's `gt >= 0` mask excludes from scoring.
IGNORE_INDEX = 19


def build_model(checkpoint: Path, device: str) -> FRNet:
    """Exactly the construction `scripts/frnet_eval.py` uses.

    The FOV constants are the projection the checkpoint was TRAINED with, not
    the HDL-64E's physical field of view -- overriding them with the sensor
    config was one of the three divergences that collapsed this port to ~15%
    point accuracy. A fine-tune built on a different projection than the eval
    would produce a number that cannot be compared to the baseline at all.
    """
    model = FRNet(num_classes=20, ignore_index=IGNORE_INDEX, output_shape=(64, 512),
                  fov_up=semantics.FRNET_TRAIN_FOV_UP_DEG,
                  fov_down=semantics.FRNET_TRAIN_FOV_DOWN_DEG)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob.get("state_dict", blob), strict=False)
    print(f"init from {checkpoint.name}: {len(missing)} missing, {len(unexpected)} "
          f"unexpected (auxiliary heads are training-only and correctly unused)")
    return model.to(device)


def frame_index(sequences) -> list[tuple[str, int]]:
    """Every (sequence, frame) with both a scan and a label, as one flat list."""
    index = []
    for seq in sequences:
        root = Path(loader.DATA_ROOT) / "sequences" / seq
        velo, lab = root / "velodyne", root / "labels"
        if not velo.is_dir() or not lab.is_dir():
            print(f"  sequence {seq}: no velodyne/ or labels/, skipped")
            continue
        n = 0
        for scan in sorted(velo.glob("*.bin")):
            if (lab / f"{scan.stem}.label").exists():
                index.append((seq, int(scan.stem)))
                n += 1
        print(f"  sequence {seq}: {n:>5} labelled frames")
    return index


def load_frame(seq: str, i: int, device: str):
    """One frame as (points, labels) on device, densified as the head sees it.

    `range_interpolation` APPENDS synthetic returns to fill isolated holes in
    the range image, so the first `n_real` rows stay aligned with the labels
    and the appended ones have no ground truth. They are marked ignore rather
    than dropped: the network must see the same densified cloud it is evaluated
    on, and the loss must not invent targets for points nobody measured.
    """
    pts = loader.load_velodyne_scan(loader._velodyne_path(seq, i))
    gt = semantics.semantic_labels(loader.load_labels(loader._label_path(seq, i)))
    x = torch.from_numpy(pts).float().to(device)
    y = torch.from_numpy(np.where(gt < 0, IGNORE_INDEX, gt)).long().to(device)
    return x, y


def class_weights(targets: str, weight: float, device: str) -> torch.Tensor:
    """1.0 everywhere, `weight` on the named classes, 0.0 on the ignore slot."""
    w = torch.ones(20, device=device)
    w[IGNORE_INDEX] = 0.0
    named = [c.strip() for c in targets.split(",") if c.strip()]
    for name in named:
        if name not in CLASSES:
            sys.exit(f"unknown class {name!r}; choose from: {', '.join(CLASSES)}")
        w[CLASSES.index(name)] = weight
    print(f"class weights: {weight}x on {', '.join(named) or '(none)'}, 1.0 elsewhere")
    return w


def freeze(model: FRNet, scope: str, freeze_bn: bool) -> list[nn.Module]:
    """Freeze everything outside `scope`. Returns the modules the caller must .eval().

    The trainable parameter count is printed, not returned -- the return value
    is the frozen-module list, and the caller MUST put those in `.eval()` and
    keep them there. A frozen BatchNorm left in training mode still updates its
    running statistics: the weights hold still while the function the module
    computes moves. The annotation said `-> int` and the docstring promised the
    parameter count, so anyone coding to either would have dropped the list and
    silently lost exactly the guarantee this file exists to make.
    """
    if scope == "all":
        trainable = list(model.parameters())
        frozen_modules = []
    elif scope == "head":
        trainable = list(model.decode_head.parameters())
        frozen_modules = [model.voxel_encoder, model.backbone]
    elif scope == "head+backbone":
        trainable = list(model.decode_head.parameters()) + list(model.backbone.parameters())
        frozen_modules = [model.voxel_encoder]
    else:
        sys.exit(f"unknown --unfreeze {scope!r}: choose head, head+backbone or all")

    trainable_ids = {id(p) for p in trainable}
    for p in model.parameters():
        p.requires_grad_(id(p) in trainable_ids)

    n = sum(p.numel() for p in trainable)
    total = sum(p.numel() for p in model.parameters())
    print(f"unfreeze={scope}: {n:,} of {total:,} parameters trainable ({n / total:.1%})"
          + ("" if freeze_bn else "  [--no-freeze-bn: frozen BN stats WILL drift]"))
    return frozen_modules


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=600,
                    help="optimiser steps; the 2 Sep run that lost 0.5 mIoU used 600")
    ap.add_argument("--batch", type=int, default=2, help="frames per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--unfreeze", default="head", choices=["head", "head+backbone", "all"],
                    help="what to train; the 2 Sep run trained the decode head only")
    ap.add_argument("--no-freeze-bn", action="store_true",
                    help="let frozen modules stay in train() so their BatchNorm "
                         "running stats drift -- reproduces the naive recipe")
    ap.add_argument("--weight-classes", default="terrain,vegetation",
                    help="comma-separated classes to upweight; empty for none")
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--init", default="checkpoints/frnet-semantickitti_seg.pth",
                    help="checkpoint to start from -- pass a tuned one to CONTINUE it")
    ap.add_argument("--resume", action="store_true",
                    help="also restore optimiser state and step count from --init, "
                         "which only works if --init was written by this script")
    ap.add_argument("--out", default="checkpoints/frnet-finetuned.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--holdout", default=HOLDOUT_SEQUENCE,
                    help="sequence kept out of training; frnet_eval.py scores it")
    ap.add_argument("--fast-scatter", action="store_true",
                    help="swap the frustum reductions for torch.scatter_reduce via "
                         "scripts/frnet_fast_scatter.py -- 3.3 h becomes minutes. "
                         "Verifies forward AND backward before patching; the frozen "
                         "port in src/perception/frnet is not edited")
    args = ap.parse_args()

    # ⚑ Not cosmetic. Python block-buffers stdout the moment it is piped rather
    #   than attached to a terminal, and a run this long is always piped into a
    #   log -- so without this the first progress line lands near the last, and
    #   a multi-hour job is indistinguishable from a hung one for its whole
    #   duration. Found the only way it gets found: watching a live run.
    sys.stdout.reconfigure(line_buffering=True)

    # Opt-in and verified: see the header of scripts/frnet_fast_scatter.py. The
    # recipe recorded beside the weights says which reduction path produced them,
    # because a checkpoint is not reproducible without that.
    if args.fast_scatter:
        sys.path.insert(0, str(Path(__file__).parent))
        from frnet_fast_scatter import enable
        enable(verify=True)

    out = Path(args.out)
    reported = Path("checkpoints/frnet-semantickitti_seg.pth")
    if out.resolve() == reported.resolve():
        sys.exit(f"refusing to overwrite {reported} -- it is the REPORTED model "
                 f"(90.3% point accuracy, 65.2% mIoU over the 15 classes present "
                 f"in 200 frames of seq 08). Write somewhere else.")

    # ⚑ The guard the 2 Sep run had and the reason its -0.5 mIoU is believable:
    #   the scored sequence is asserted out of training rather than assumed out.
    train_seqs = [s for s in TRAIN_SEQUENCES if s != args.holdout]
    assert args.holdout not in train_seqs, (
        f"sequence {args.holdout} is scored by frnet_eval.py and must never be trained on")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init = Path(args.init)
    if not init.exists():
        print(f"checkpoint not found: {init}", file=sys.stderr)
        return 2

    print(f"device {device}, holding out sequence {args.holdout}")
    print("indexing training frames:")
    frames = frame_index(train_seqs)
    if not frames:
        print(f"no labelled frames under {loader.DATA_ROOT}. Set VRGRID_DATA_ROOT to the "
              f"directory holding poses/ and sequences/.", file=sys.stderr)
        return 2
    print(f"  {len(frames):,} frames over {len(train_seqs)} sequences\n")

    model = build_model(init, device)
    frozen_modules = freeze(model, args.unfreeze, not args.no_freeze_bn)
    weights = class_weights(args.weight_classes, args.weight, device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    step0 = 0
    if args.resume:
        blob = torch.load(init, map_location=device, weights_only=False)
        if "optimizer" not in blob:
            sys.exit(f"--resume needs a checkpoint written by this script; {init.name} "
                     f"holds only {sorted(blob)}. Drop --resume to start a fresh run "
                     f"from these weights.")
        opt.load_state_dict(blob["optimizer"])
        step0 = int(blob.get("step", 0))
        print(f"resumed optimiser state at step {step0}")

    print(f"\ntraining {args.steps} steps, batch {args.batch}, lr {args.lr}\n")
    model.train()
    for m in frozen_modules:          # see the BatchNorm note in the docstring
        m.eval()

    losses = []
    first = last = None
    t0 = time.perf_counter()
    for step in range(step0, step0 + args.steps):
        picks = [frames[random.randrange(len(frames))] for _ in range(args.batch)]
        batch = [load_frame(seq, i, device) for seq, i in picks]
        points = [model.range_interpolation(x) for x, _ in batch]
        n_real = [y.shape[0] for _, y in batch]

        voxel = model(points)
        logits = voxel["seg_logit"]
        coors = voxel["coors"]

        # Line the targets up with the batch the network actually returned:
        # `coors[:, 0]` is the batch index, and the appended interpolated points
        # trail each item's real ones and are scored as ignore.
        target = torch.full((logits.shape[0],), IGNORE_INDEX,
                            dtype=torch.long, device=device)
        for b, (_, y) in enumerate(batch):
            rows = (coors[:, 0] == b).nonzero(as_tuple=True)[0][:n_real[b]]
            target[rows] = y[:rows.shape[0]]

        loss = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last = loss.detach().item()
        losses.append(last)
        if first is None:
            first = last
        if (step - step0 + 1) % args.log_every == 0:
            window = losses[-args.log_every:]
            done = step - step0 + 1
            rate = done / (time.perf_counter() - t0)
            print(f"  step {step + 1:>5}  loss {sum(window) / len(window):.4f}"
                  f"   {rate:4.2f} steps/s  eta {(args.steps - done) / rate / 60:5.1f} min")

    dt = time.perf_counter() - t0
    step_end = step0 + args.steps

    # Everything needed to re-derive or continue this run travels WITH the
    # weights. The 2 Sep checkpoint carried a bare state_dict, which is why the
    # run behind the reported table could not be resumed or even re-read.
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step_end,
        "recipe": {
            "init": str(init), "steps": args.steps, "batch": args.batch, "lr": args.lr,
            "weight_decay": args.weight_decay, "unfreeze": args.unfreeze,
            "freeze_bn": not args.no_freeze_bn, "weight_classes": args.weight_classes,
            "weight": args.weight, "seed": args.seed,
            "fast_scatter": args.fast_scatter,
            "train_sequences": train_seqs, "holdout": args.holdout,
            "loss_first": first, "loss_last": last,
        },
    }, out)

    print(f"\n{args.steps} steps in {dt / 60:.1f} min, now at step {step_end}")
    print(f"training loss {first:.4f} -> {last:.4f}")
    print("⚑ that number is NOT a result -- a class-weighted loss falls when the head "
          "gets more\n  confident on the weighted classes, correct or not. Score it:")
    print(f"\n  python scripts/frnet_eval.py --seq {args.holdout} --frames 200 "
          f"--checkpoint {out}")
    print(f"  python scripts/frnet_eval.py --seq {args.holdout} --frames 200"
          f"  # the baseline, same 200 frames\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
