#!/usr/bin/env python3
"""The Gate 3 figure: the same map, with and without §10.4. [Shrestha]

    python scripts/ghost_removal_figure.py [--frames 14] [--out docs/figures]

Two panels of OCCUPIED CELLS, not of returns. That distinction is the whole
point of the figure: `dashboard/pipeline_view.py` can already hide the moving
points, and hiding the input demonstrates nothing about the mapping engine.
What a viewer actually sees behind a moving car in a 2.5D map is a streak of
*cells* that were fused from it and never cleared, and removing those is what
§10.4 does. So the left panel is the map with the cleanup off -- the trail --
and the right panel is the same frames with it on.

The scene is a car driving away down the road while the static world stays
put, which is the shape of the thing the demo claims. It is deliberately not
the scene `tests/test_engine.py` uses: that one teleports the car away after
three frames, which is the sharpest possible test and the least interesting
picture. A trail has to be swept out to be seen.

Every number printed and drawn comes from `MapEngine`'s own counters, and the
two runs share one generated sequence, so the panels differ in exactly one
thing: `ghost_removal`.

**The trail does not vanish, it decays, and the residual is the point.** With
the cleanup on, the surviving ghost cells are the last few metres immediately
behind the car -- 22.8 to 25.5 m of a sweep that ran 13.1 to 25.5 m -- and
never the far end. That is §10.1 working as specified rather than a partial
failure: `log_odds_hit` is +4 and `log_odds_miss` is -2, so a cell the car sat
in for k frames needs about 2k clear observations before it drops below
`log_odds_occupied`. A cleanup that erased the whole trail in one frame would
be one that erases a stopped car, and a stopped car is not a ghost. The script
prints the surviving extent so the shape of the residual is checkable and not
just visible.

⚑ Synthetic. Analytic terrain, no sensor noise, no occlusion, no registration
  error, and ground-truth motion. The figure is stamped with that inside the
  image so a screenshot cannot lose it. It shows that the mechanism works; it
  is not evidence about how well it works on SemanticKITTI.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from vrgrid.grid.schedule import load
from vrgrid.perception import range_image as ri_mod
from vrgrid.run.engine import MapEngine

SENSOR_H = 1.73
GROUND_R = 11.0          # the ground disc stops short of the car's lane
WALL_X = 34.0            # a static wall the beams return from
CAR_START_X, CAR_END_X = 14.0, 27.0
CAR_Y = 0.0
CAR_HALF = 1.0

# ⚑ The caveat has to follow the DATA, not the script. Printed and stamped
#   unconditionally it put "SYNTHETIC: analytic terrain, no sensor noise" on a
#   figure drawn from 200 real frames of sequence 08 -- a false label on a
#   report figure, and one that understates the result rather than overstating
#   it, which is exactly the kind that survives review.
def caveat_for(seq):
    if seq:
        return (f"REAL: SemanticKITTI sequence {seq}. Ground-truth semantics "
                "and motion from the .label files; no learned segmentation.")
    return CAVEAT


CAVEAT = ("SYNTHETIC: analytic terrain, no sensor noise, occlusion or "
          "registration error, ground-truth motion")


class _Frame:
    """The fields `MapEngine.step` reads off a `PerceptionFrame`."""

    def __init__(self, index, points, ground_mask):
        p4 = np.column_stack([points, np.full(len(points), 0.5)])
        image, inverse = ri_mod.project(p4)
        self.index = index
        self.points_sensor = p4
        self.points_world = points + np.array([0.0, 0.0, SENSOR_H])
        self.pose = np.eye(4)[:3]
        self.vehicle_xyz_world = np.zeros(3)
        self.semantic = np.zeros(len(points), np.int8)
        self.moving = np.zeros(len(points), bool)
        self.ground = ground_mask
        self.reflectivity8 = np.full(len(points), 100, np.uint8)
        self.range_image = image
        self.inverse_index = inverse


def build_scene(rng):
    """The static world, generated once and reused every frame.

    The ground stops at 11 m and the car drives from 14 m out. A cell holding
    both car returns and ground returns stays occupied after the car passes --
    correctly, the ground is really there -- and it would blunt the picture
    for a reason that has nothing to do with the cleanup.

    The wall spans -4 to +2 m so it subtends every elevation the car does. A
    wall that stopped at the car's lowest beam would leave those beams
    returning from nothing, and NO_RETURN must never clear anything, so the
    trail would survive for a correct reason and the figure would libel the
    kernel.
    """
    n_g, n_w = 6000, 11000
    r = rng.uniform(4.0, GROUND_R, n_g)
    a = rng.uniform(-np.pi, np.pi, n_g)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n_g, -SENSOR_H)])
    wall = np.column_stack([np.full(n_w, WALL_X),
                            rng.uniform(-9.0, 9.0, n_w),
                            rng.uniform(-4.0, 2.0, n_w)])
    return ground, wall


def car_at(rng, x_m, n=1500):
    return np.column_stack([rng.uniform(x_m - CAR_HALF, x_m + CAR_HALF, n),
                            rng.uniform(CAR_Y - CAR_HALF, CAR_Y + CAR_HALF, n),
                            rng.uniform(-1.5, -0.2, n)])


def sequence(rng, frames):
    """The car drives away from the sensor; everything else stays put."""
    ground, wall = build_scene(rng)
    for i in range(frames):
        t = i / max(frames - 1, 1)
        x = CAR_START_X + t * (CAR_END_X - CAR_START_X)
        points = np.vstack([ground, wall, car_at(rng, x)])
        mask = np.zeros(len(points), bool)
        mask[:len(ground)] = True
        yield _Frame(i, points, mask), x


def run(frames, ghost_removal, seed=0):
    rng = np.random.default_rng(seed)
    engine = MapEngine(load("5/10/20/40"), max_points=40_000,
                       max_candidates=80_000, ghost_removal=ghost_removal)
    counters, track = [], []
    for frame, car_x in sequence(rng, frames):
        counters.append(engine.step(frame))
        track.append(car_x)
    _, x, y, _ = engine.occupied_cells()
    return engine, (x, y), counters, track


def trail_mask(x, y, car_x_now):
    """Occupied cells inside the lane the car swept, BEHIND where it is now.

    Nothing in the static scene occupies that strip -- the ground stops at
    11 m and the wall is at 34 m -- so a cell there is either the car or a
    ghost of it, and counting them is what "the trail" means as a number.
    """
    return ((np.abs(y - CAR_Y) <= CAR_HALF + 0.3)
            & (x >= CAR_START_X - CAR_HALF - 0.3)
            & (x <= car_x_now - CAR_HALF - 0.3))


def draw(panels, counts, path, frames, car_x_now, caveat=CAVEAT):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharex=True, sharey=True)
    for ax, (title, (x, y), ghost_mask), n_trail in zip(axes, panels, counts):
        keep = np.ones(len(x), bool) if car_x_now is None else (
            (x > -2.0) & (x < WALL_X + 2.0) & (np.abs(y) < 10.0))
        ghost = ghost_mask & keep
        ax.scatter(x[keep & ~ghost], y[keep & ~ghost], s=1.4, c="#8A8F98",
                   linewidths=0, label="static map")
        if ghost.any():
            ax.scatter(x[ghost], y[ghost], s=3.2, c="#C44E52", linewidths=0,
                       label=f"ghost cells ({n_trail:,})")
        if car_x_now is not None:
            ax.add_patch(Rectangle((car_x_now - CAR_HALF, CAR_Y - CAR_HALF),
                                   2 * CAR_HALF, 2 * CAR_HALF, fill=False,
                                   edgecolor="#3B7DD8", linewidth=1.4, zorder=5))
            ax.annotate("car now", (car_x_now, CAR_Y + CAR_HALF),
                        textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8, color="#3B7DD8")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("x (m, world)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=3)
    axes[0].set_ylabel("y (m, world)")
    fig.suptitle("Ghost removal in the MAP, not the point cloud (math §10.4)",
                 fontsize=12)
    fig.text(0.5, 0.015, caveat, ha="center", fontsize=7.5, color="#B4342F")
    tail = (f"car {CAR_START_X:.0f} m -> {CAR_END_X:.0f} m" if car_x_now is not None
            else "ghosts identified by the GT moving-* label")
    # Inside the right-hand axes rather than in the figure footer: at this
    # aspect ratio two footer lines and the x-labels all land on top of each
    # other.
    axes[1].text(0.995, 0.02, f"{frames} frames, {tail}",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="0.5")
    fig.tight_layout(rect=(0, 0.07, 1, 0.99))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def run_real(seq, frames, ghost_removal, clip_class_ids, no_patchworkpp):
    """Replay a real sequence and return the map plus the cells moving returns
    ever touched.

    On real data there is no known car lane to count, so a ghost is defined the
    only way ground truth allows: a cell some `moving-*` return was binned into.
    Whether it is STILL occupied at the end is the question the figure asks.
    """
    from vrgrid.run.__main__ import iter_pipeline

    engine = MapEngine(load("5/10/20/40"), ghost_removal=ghost_removal,
                       clip_class_ids=clip_class_ids)
    counters, ever, current = [], set(), set()
    for frame in iter_pipeline(seq, frames, use_patchworkpp=not no_patchworkpp):
        counters.append(engine.step(frame))

        # After step(), because step() shifts the ring windows before it bins;
        # binning against the pre-shift windows would name different slots.
        moving = np.asarray(frame.moving)[:engine.max_points]
        current = set()
        if moving.any():
            pts = frame.points_sensor[:engine.max_points]
            world = frame.points_world[:engine.max_points]
            idx = engine.bin(pts[moving, 0], pts[moving, 1],
                             world[moving, 0], world[moving, 1])
            current = {int(i) for i in idx[idx >= 0]}
            ever |= current

    # A ghost is a cell some moving return passed through and then LEFT. Cells
    # the object still occupies in the final frame are not a trail, they are an
    # object, and the current-return guard is supposed to protect them.
    trail = ever - current
    slots, x, y, _ = engine.occupied_cells()
    ghost = np.fromiter((int(s) in trail for s in slots), bool, len(slots))
    return engine, (x, y), ghost, counters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--seq", default=None,
                    help="draw a REAL sequence (needs $VRGRID_DATA_ROOT); "
                         "ghosts are the cells moving-* returns touched")
    ap.add_argument("--no-patchworkpp", action="store_true")
    ap.add_argument("--clip-class-ids", action="store_true",
                    help="--seq only: clip semantic ids to 15 (math §10.2)")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.seq:
        off_engine, off_xy, off_ghost, off_counters = run_real(
            args.seq, args.frames, False, args.clip_class_ids, args.no_patchworkpp)
        on_engine, on_xy, on_ghost, on_counters = run_real(
            args.seq, args.frames, True, args.clip_class_ids, args.no_patchworkpp)
        car_now, track = None, []
    else:
        off_engine, off_xy, off_counters, track = run(args.frames, ghost_removal=False)
        on_engine, on_xy, on_counters, _ = run(args.frames, ghost_removal=True)
        car_now = track[-1]
        off_ghost = trail_mask(*off_xy, car_now)
        on_ghost = trail_mask(*on_xy, car_now)

    n_off, n_on = int(off_ghost.sum()), int(on_ghost.sum())

    print(f"{'':<22}{'occupied':>12}{'ghost cells':>14}{'cleared':>13}"
          f"{'protected':>13}")
    print("-" * 74)
    for label, engine, n, counters in (("cleanup OFF", off_engine, n_off, off_counters),
                                       ("cleanup ON", on_engine, n_on, on_counters)):
        print(f"{label:<22}{len(engine.occupied_slots()):>12,}{n:>14,}"
              f"{sum(c.cleared for c in counters):>13,}"
              f"{sum(c.protected for c in counters):>13,}")
    removed = 1.0 - (n_on / n_off) if n_off else float("nan")
    print(f"\n{removed:.1%} of the trail removed. {sum(c.protected for c in on_counters):,} cells "
          f"were spared by the current-return\nguard -- the wall and the ground, which the "
          f"cleanup must not eat.")

    off_m, on_m = off_ghost, on_ghost
    if car_now is not None and on_m.any() and off_m.any():
        ox, mx = off_xy[0][off_m], on_xy[0][on_m]
        print(f"\nThe residual is the FRESH end of the trail: with the cleanup on it "
              f"spans\n{mx.min():.1f}-{mx.max():.1f} m against {ox.min():.1f}-{ox.max():.1f} m "
              f"with it off, and the car is at {car_now:.1f} m. Occupancy\nis log-odds "
              f"(§10.1): +4 a hit, -2 a miss, so a cell the car sat in for k frames\nneeds "
              f"~2k clear looks to fall below the threshold. A cleanup that erased the\nwhole "
              f"trail in one frame would also erase a stopped car.")

    with (out / "ghost_removal.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "car_x_m", "mode", "occupied", "tested",
                    "cleared", "protected"])
        for mode, counters in (("off", off_counters), ("on", on_counters)):
            for c, car_x in zip(counters, track):
                w.writerow([c.index, f"{car_x:.2f}", mode, c.occupied, c.tested,
                            c.cleared, c.protected])

    panels = [(f"ghost removal OFF  --  {n_off:,} ghost cells", off_xy, off_ghost),
              (f"ghost removal ON  --  {n_on:,} ghost cells", on_xy, on_ghost)]
    if draw(panels, (n_off, n_on), out / "ghost_removal.svg", args.frames,
            car_now, caveat_for(args.seq)):
        print(f"\nwrote {out / 'ghost_removal.csv'}, {out / 'ghost_removal.svg'} "
              f"and {out / 'ghost_removal.png'}")
    else:
        print(f"\nwrote {out / 'ghost_removal.csv'}. No figure: matplotlib is not "
              f"installed -- `pip install -e \".[report]\"`.")
    print(f"\n{caveat_for(args.seq)}")


if __name__ == "__main__":
    main()
