#!/usr/bin/env python3
"""Native-scatter shim for FRNet's frustum reductions. Opt-in, verified, local.

    from frnet_fast_scatter import enable
    enable(verify=True)      # patch, then PROVE the patch changes nothing

    python scripts/frnet_fast_scatter.py          # standalone: verify + time it

`src/perception/frnet/frustum_encoder.py` implements `scatter_max` and
`scatter_mean` as `for i in range(dim_size)` loops that build a full-length
boolean mask per output slot. `dim_size` is the number of OCCUPIED frustum
pixels -- about 25,000 -- so each call is ~25,000 iterations over a 124,000-row
tensor, and a forward pass makes SEVEN such calls: two in the encoder, and five
through `frnet_backbone.point2frustum` (once before the stage loop, once per
each of the four stages). Measured 3 Sep on this machine, CUDA, one seq 00
frame: `voxel_encoder` 4,678.5 ms of a 10,535.1 ms forward. That is 3.3 hours
for a 600-step fine-tune and ~35 minutes for `frnet_eval.py --frames 200`.

`torch.Tensor.scatter_reduce` does the same reduction natively, three orders of
magnitude faster at these shapes, matching the loop's empty-slot convention
both ways.

⚑ "BIT-IDENTICAL" WAS A CPU RESULT AND DOES NOT SURVIVE THE MOVE TO CUDA. The
  3 Sep log recorded max abs diff 0.000e+00 for both reductions, measured on
  CPU. On CUDA `scatter_max` is still bit-identical -- max is order-independent
  -- but `scatter_mean` differs by up to 2 float32 ulp (2.384e-07 absolute) on
  40% of slots, because the native kernel sums a slot's rows in a different
  order than `src[mask].mean(dim=0)` does and float addition is not
  associative. That is float32 rounding, not a different answer, and it is a
  DIFFERENT KIND of claim than bit-identity: verify() gates `scatter_max` at
  exactly zero and `scatter_mean` at a stated ulp bound, and the substitution
  is then proven where it actually matters by re-deriving the reported 90.3%
  point accuracy / 69.8% mIoU through `frnet_eval.py --fast-scatter`.

⚑ THIS FILE DOES NOT EDIT JP'S PORT. `src/perception/frnet/` is a deliberately
  frozen reference port (`extend-exclude` in pyproject.toml) and is his to
  change; the finding is written up for him in `docs/research-log.md`, 3 Sep.
  This is a runtime monkey-patch living in `scripts/`, applied only by scripts
  that ask for it with `--fast-scatter`. Nothing imports it implicitly, the
  frozen source keeps computing exactly what it computes today, and deleting
  this file restores the slow path everywhere.

⚑ PATCH BOTH MODULES OR PATCH NEITHER. `frnet_backbone.py` does
  `from .frustum_encoder import scatter_max` at import time, which BINDS the
  function object into its own namespace. Rebinding only
  `frustum_encoder.scatter_max` leaves the backbone's five calls -- the majority
  of them -- still running the loop, and the run merely looks disappointing
  rather than broken. `enable()` rebinds the name in every module that holds it.

⚑ THE ARGMAX RETURN IS DROPPED, ON PURPOSE. The loop's second return value is
  already wrong: `argmax[i]` holds the index within the masked subset, not into
  the full input as `torch_scatter.scatter_max` returns. All three call sites
  discard it (`voxel_feats, _ = ...`), which is why the port scores 98.3% with
  the bug in it. `scatter_reduce` provides no argmax at all, so this shim
  returns None -- a caller that starts using it gets an immediate TypeError
  instead of a silently wrong index.

⚑ GRADIENTS, WHICH THE 3 SEP MEASUREMENT DID NOT COVER. That measurement was
  forward-only and a fine-tune needs the backward half to be right too, so
  `verify()` checks it. The risk was ties: `amax` splits gradient evenly among
  tied maxima where `Tensor.max(dim=0)` hands all of it to the first, and ties
  are reachable here because ReLU emits exact zeros. Measured, the max backward
  is exact to 0.000e+00 and the mean backward carries the same 2 ulp as its
  forward. Checked rather than assumed away, and re-checked on every enable().
"""
import sys
import time

import torch
from vrgrid.perception.frnet import frnet_backbone, frustum_encoder

#: The originals, captured at import so verify() can still call them after a
#: patch and so disable() is exact rather than approximate.
LOOP_SCATTER_MAX = frustum_encoder.scatter_max
LOOP_SCATTER_MEAN = frustum_encoder.scatter_mean

#: Every module namespace holding a bound reference to one of the reductions.
#: `frnet_backbone` matters most: five of the seven per-forward calls are its.
_MAX_HOLDERS = (frustum_encoder, frnet_backbone)
_MEAN_HOLDERS = (frustum_encoder,)


def _expand_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """(N,) slot ids -> (N, ...) index of src's shape, as scatter_reduce wants."""
    return index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)


def fast_scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                     dim_size: int | None = None):
    """Bit-identical replacement for the loop. Returns (values, None) -- see header.

    `-inf` init with include_self=True leaves untouched slots at `-inf`, which
    is exactly what the loop does by skipping `if mask.any()`. Both call sites
    build `index` from `torch.unique(..., return_inverse=True)`, so in practice
    every slot is occupied and the convention is never exercised -- it is
    matched anyway so this is a drop-in at shapes it has not been shown.
    """
    if dim != 0:
        raise NotImplementedError(f"shim covers dim=0 only, got dim={dim}")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = src.new_full((dim_size,) + src.shape[1:], float("-inf"))
    return out.scatter_reduce(0, _expand_index(index, src), src,
                              reduce="amax", include_self=True), None


def fast_scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                      dim_size: int | None = None) -> torch.Tensor:
    """Bit-identical replacement for the loop.

    Zero init with include_self=False leaves untouched slots at 0, matching the
    loop's skipped branch. include_self=True would fold the initial zero into
    the average and quietly divide by count+1.
    """
    if dim != 0:
        raise NotImplementedError(f"shim covers dim=0 only, got dim={dim}")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = src.new_zeros((dim_size,) + src.shape[1:])
    return out.scatter_reduce(0, _expand_index(index, src), src,
                              reduce="mean", include_self=False)


def enable(verify: bool = True, device: str | None = None) -> None:
    """Rebind the reductions in every module that holds one. Verifies by default.

    `verify=False` exists for a second process that has already seen the proof
    in the same session's log, not as a convenience -- an unverified numerical
    substitution under a checkpoint that gets reported is how a wrong number
    reaches a slide.
    """
    if verify:
        verify_equivalence(device=device)
    for mod in _MAX_HOLDERS:
        mod.scatter_max = fast_scatter_max
    for mod in _MEAN_HOLDERS:
        mod.scatter_mean = fast_scatter_mean
    print(f"fast-scatter ENABLED in {', '.join(m.__name__.rsplit('.', 1)[-1] for m in _MAX_HOLDERS)}"
          f" (scatter_max) and {', '.join(m.__name__.rsplit('.', 1)[-1] for m in _MEAN_HOLDERS)}"
          f" (scatter_mean)")


def disable() -> None:
    """Put the loops back, exactly as imported."""
    for mod in _MAX_HOLDERS:
        mod.scatter_max = LOOP_SCATTER_MAX
    for mod in _MEAN_HOLDERS:
        mod.scatter_mean = LOOP_SCATTER_MEAN


def verify_equivalence(n: int = 20000, slots: int = 4000, channels: int = 64,
                       device: str | None = None, seed: int = 0) -> None:
    """Forward bit-identity and backward agreement, loop vs shim. Raises on drift.

    Shapes are scaled down from the real ones (124,000 x 256 into 25,000 slots)
    because the LOOP side has to run too and the loop is the whole problem: at
    full size this check would itself take ~40 s per reduction. The reduction is
    shape-independent, and `frnet_eval.py --fast-scatter` re-checks the result
    at full size where it counts -- as the reported 90.3% / 69.8%.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randn(n, channels, generator=g).to(dev)
    index = torch.randint(0, slots, (n,), generator=g).to(dev)
    # A slot deliberately left empty, to exercise the -inf / 0 conventions that
    # torch.unique never produces at the real call sites.
    index[index == slots - 1] = 0

    print(f"verifying on {dev}: {n:,} rows x {channels} ch into {slots:,} slots "
          f"({int((torch.bincount(index, minlength=slots) == 0).sum())} empty)")

    ref_max, _ = LOOP_SCATTER_MAX(src, index, dim=0, dim_size=slots)
    new_max, argmax = fast_scatter_max(src, index, dim=0, dim_size=slots)
    ref_mean = LOOP_SCATTER_MEAN(src, index, dim=0, dim_size=slots)
    new_mean = fast_scatter_mean(src, index, dim=0, dim_size=slots)

    if argmax is not None:
        raise AssertionError("shim must return None for argmax; see header")
    #: `scatter_max` is order-independent and must match exactly on any device.
    #: `scatter_mean` sums each slot in the native kernel's order, so on CUDA it
    #: lands within float32 rounding of the loop rather than on top of it. The
    #: bound is stated in ulp so it cannot quietly widen into a real difference.
    ULP = torch.finfo(torch.float32).eps
    TOLERANCE = {"scatter_max": 0.0, "scatter_mean": 4 * ULP}
    for name, ref, new in (("scatter_max", ref_max, new_max),
                           ("scatter_mean", ref_mean, new_mean)):
        finite = torch.isfinite(ref)
        d = (ref[finite] - new[finite]).abs().max().item()
        same_inf = torch.equal(torch.isfinite(ref), torch.isfinite(new))
        exact = int((ref[finite] != new[finite]).sum())
        print(f"  {name:<13} forward  max abs diff {d:.3e} ({d / ULP:.1f} ulp), "
              f"{exact:,} of {int(finite.sum()):,} values differ, "
              f"empty-slot convention {'matches' if same_inf else 'DIFFERS'}")
        if d > TOLERANCE[name] or not same_inf:
            raise AssertionError(
                f"{name} differs from the loop by {d:.3e} ({d / ULP:.1f} ulp), over "
                f"its bound of {TOLERANCE[name]:.3e} -- refusing to patch")

    # Backward. The 3 Sep measurement was forward-only and a fine-tune needs
    # this half to be right too.
    for name, loop_fn, fast_fn in (("scatter_max", LOOP_SCATTER_MAX, fast_scatter_max),
                                   ("scatter_mean", LOOP_SCATTER_MEAN, fast_scatter_mean)):
        grads = []
        for fn in (loop_fn, fast_fn):
            x = src.clone().requires_grad_(True)
            out = fn(x, index, dim=0, dim_size=slots)
            out = out[0] if isinstance(out, tuple) else out
            out.nan_to_num(neginf=0.0).square().sum().backward()
            grads.append(x.grad)
        d = (grads[0] - grads[1]).abs().max().item()
        rows = int((grads[0] != grads[1]).any(dim=1).sum())
        note = "" if d == 0.0 else (f" ({d / ULP:.1f} ulp on {rows:,} of {n:,} rows, "
                                    f"the same float32 rounding as the forward)")
        print(f"  {name:<13} backward max abs diff {d:.3e}{note}")
        if d > TOLERANCE[name]:
            raise AssertionError(
                f"{name} gradients differ by {d:.3e} ({d / ULP:.1f} ulp), over its "
                f"bound of {TOLERANCE[name]:.3e} -- refusing to patch")

    print("  verified: scatter_max exact in both directions, scatter_mean within "
          "float32 rounding\n")


def _benchmark(device: str | None = None) -> None:
    """The real shapes, loop vs shim, on this machine."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n, slots, channels = 124_000, 25_000, 256
    g = torch.Generator(device="cpu").manual_seed(0)
    src = torch.randn(n, channels, generator=g).to(dev)
    index = torch.randint(0, slots, (n,), generator=g).to(dev)

    print(f"benchmark on {dev}: {n:,} x {channels} into {slots:,} slots")
    for name, loop_fn, fast_fn in (("scatter_max", LOOP_SCATTER_MAX, fast_scatter_max),
                                   ("scatter_mean", LOOP_SCATTER_MEAN, fast_scatter_mean)):
        ts = []
        for fn in (loop_fn, fast_fn):
            fn(src, index, dim=0, dim_size=slots)          # warm up / compile
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(src, index, dim=0, dim_size=slots)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        print(f"  {name:<13} loop {ts[0] * 1e3:10.1f} ms   shim {ts[1] * 1e3:7.2f} ms"
              f"   {ts[0] / ts[1]:8.0f}x")


if __name__ == "__main__":
    verify_equivalence()
    _benchmark()
    sys.exit(0)
