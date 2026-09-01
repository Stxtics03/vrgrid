"""A synthetic sequence, in the real layout. [Aakash]

The reference map is the long pole and it is blocked on a 40 GB download. What
is NOT blocked is everything downstream of it, provided something produces the
same shapes. So this generates a miniature sequence on disk in exactly the
layout of `data/README.md`, and `reference_map.build()` reads it through the
same code path it will read SemanticKITTI with. The hour the download lands,
one argument changes.

⚑ This is a scaffold for the harness, not a substitute for data. Nothing
  measured on it goes on a slide. Its terrain is analytic, so "RMSE against
  the reference" here measures the *pipeline* — lattice, fusion, coarsening —
  against a surface with no sensor noise, no occlusion and no registration
  error. That is the right thing to develop against and the wrong thing to
  report.

The surface is chosen to exercise the things the metrics are supposed to
catch, and nothing else:

    a crowned road          the ordinary case, gentle curvature
    kerbs at |y| = 3 m      a 12 cm step -- §4.1's worked example, the thing
                            merge is supposed to widen the variance across
    a verge beyond          rougher, with sub-cell noise, so `spread` in §9.3
                            is non-zero and rho is not degenerate
    a ramp from x = 30 m    a real slope, so §7.1 bit 1 has something to fire
                            on and Theorem 1's inflation has a gradient
    a pothole at (18, 0)    40 cm deep, 60 cm wide -- around the §1.4 negative
                            obstacle limit (r_max is 10.8 m at 50 cm and
                            15.2 m at 1 m), so it is *supposed* to be
                            invisible far out. If a metric ever says we found
                            it at 40 m, the metric is wrong.

                            ⚑ Until 2026-09-01 it was invisible EVERYWHERE:
                            across a 12-frame sequence the old sampler
                            returned not one point below -30 cm, at any range.
                            The scene's only negative obstacle had never once
                            been observed as a hole, and R(S) was being read
                            off a lane with no hazard in it. With the
                            intersection solved properly the hole is resolved
                            at 7.9 m and not at 11.7 m, which is the §1.4
                            crossover doing what it says.

`structure=True` adds a facade, a pole and a sign to that. They are off by
default because every reference number in `docs/` was measured on the ground
scene, and because a wall is not a surface the elevation metrics have any
opinion about. What they are for is the *label set*: pole and traffic-sign are
learning ids 17 and 18, so a sequence written with them is the one that proves
the 5|3 class byte (§10.2) survives the real path.

On what "the real layout" means
-------------------------------
This used to write three things the loader does not read, which is a
different claim from the one the docstring made:

    it wrote                            `perception.loader` reads
      sequences/<seq>/poses.txt           poses/<seq>.txt          (moved)
      (no calib.txt)                      sequences/<seq>/calib.txt (added)
      vehicle-frame .bin                  SENSOR-frame .bin        (dropped 1.73 m)
      19-class learning ids               RAW SemanticKITTI ids    (remapped)
      vehicle->world poses                Camera-0 -> World_cam    (recomposed)

Shrestha found the first two by reading `loader.py`'s header, which says in as
many words that `sequences/<seq>/poses.txt` is the SemanticKITTI internal SLAM
file and that we deliberately use the official GT poses instead -- so the file
this module produced was precisely the one the loader is built to ignore. The
other three are the same class of mistake and were only invisible because
`read_sequence` was the sole reader and shared every one of the assumptions.

The last three are worth being explicit about, because "two lines" was the
estimate and each of them is a silent wrong answer rather than a crash:

  frame   A `.bin` holds SENSOR-frame points; the vehicle origin is 1.73 m
          below the laser (`transforms.T_S_V`). Writing vehicle-frame points
          into one puts every road return 1.73 m underground, and the map
          still looks entirely plausible.

  labels  A `.label` holds RAW ids, which `semantics.semantic_labels` maps.
          Learning ids written into one are read as raw: 9 is unmapped, 10 is
          `car` and 11 is `bicycle`, so the road came back as ignore and the
          parking as car. `read_sequence` returns raw ids, as it always
          documented, and now they are actually raw.

  poses   A `poses/<seq>.txt` row is Camera-0 -> World_cam, not vehicle ->
          world; `transforms.vehicle_to_world` is the composition. The old
          identity-rotation rows were right for `read_sequence` and would be
          wrong by a 90 degree axis permutation for the loader -- the exact
          failure `harness.assert_world_is_z_up` was added to catch.

The poses are DERIVED from that composition rather than written by hand:
`pose_for` inverts it. A hand-written pose would be a second opinion about the
frame convention, and the point of `docs/frames.md` is that there is one.
"""

from pathlib import Path

import numpy as np

# Raw SemanticKITTI ids -- what a `.label` file holds. `semantics.py` maps them
# to the 19-class scheme (road 40 -> 8, parking 44 -> 9, sidewalk 48 -> 10).
ROAD, PARKING, SIDEWALK = 40, 44, 48
BUILDING, VEGETATION, POLE, TRAFFIC_SIGN = 50, 70, 80, 81
MOVING_CAR = 252  # raw moving-* id, stripped by the reference map
MOVING_PERSON = 254  # raw moving-person, the crowd's label

KERB_Y_M = 3.0
KERB_HEIGHT_M = 0.12
RAMP_START_X_M = 30.0
RAMP_SLOPE = 0.06
POTHOLE_XY_M = (18.0, 0.0)
POTHOLE_RADIUS_M = 0.30
POTHOLE_DEPTH_M = 0.40

# HDL-64E vertical FOV, as configs/frnet.yaml has it. `range_image.project`
# warns and clamps outside this, and a fixture that trips a real warning
# teaches everyone to ignore it.
PHI_MIN_DEG, PHI_MAX_DEG = -24.8, 2.0

# Beams are fired a thousandth of a degree inside that, and the reason is
# storage, not optics. A `.bin` is float32, so a return generated at exactly
# -24.8 deg rounds to either side of the floor when it is written; the whole
# bottom beam -- 386 returns a sweep -- came back as `n_clamped_below` from
# `range_image.project`. 0.001 deg is 1.7 mm at 100 m, so nothing measurable
# moves, and no beam sits on the knife edge any more.
FOV_EPS_DEG = 1e-3


def terrain_height_m(x, y, seed: int = 0):
    """Analytic ground elevation. Deterministic given (x, y) -- it is a
    surface, not a sample, so the same place must give the same height on
    every frame or the reference map is meaningless."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    z = -0.02 * np.minimum(np.abs(y), KERB_Y_M) ** 2 / KERB_Y_M      # road crown
    z += np.where(np.abs(y) > KERB_Y_M, KERB_HEIGHT_M, 0.0)          # kerb
    z += np.where(x > RAMP_START_X_M, RAMP_SLOPE * (x - RAMP_START_X_M), 0.0)

    # verge roughness: deterministic hash-noise, so it is part of the surface
    # rather than sensor noise. This is what `spread` in §9.3 measures.
    verge = np.abs(y) > KERB_Y_M + 1.0
    grain = np.sin(12.7 * x + seed) * np.cos(11.3 * y - seed)
    z += np.where(verge, 0.02 * grain, 0.0)

    d = np.hypot(x - POTHOLE_XY_M[0], y - POTHOLE_XY_M[1])
    z -= np.where(d < POTHOLE_RADIUS_M, POTHOLE_DEPTH_M, 0.0)
    return z


def class_at(x, y):
    """Raw SemanticKITTI id of the ground surface at (x, y)."""
    y = np.abs(np.asarray(y, dtype=np.float64))
    return np.where(y <= KERB_Y_M, ROAD,
                    np.where(y <= KERB_Y_M + 1.0, SIDEWALK, PARKING)).astype(np.uint16)


def _beam_range(phi, theta, pose_x_m, sensor_height_m, seed, max_range_m,
                iters: int = 48):
    """Where each beam meets the ground. Vectorised bisection on

        f(r) = h_s + r*tan(phi) - z(r cos(theta), r sin(theta))

    which is beam height minus surface height: positive while the beam is
    still above the ground, negative once it is under. f(0) = h_s - z(0) > 0
    and f decreases as the beam descends onto a surface that does not fall
    away faster, so the root is bracketed and unique.

    ⚑ This replaced `r = h_s/tan|phi|` followed by ONE correction step, which
      was wrong twice over.

      The correction used `(h_s + z)` where the sensor's height above a
      surface at elevation z is `(h_s - z)`. On flat ground the two agree, so
      it survived every test the scene had; on the 12 cm kerb and the 40 cm
      pothole it put the return on the wrong side of the feature, radially, by
      about `2z/tan|phi|` -- 1.7 m at the steepest beam.

      Fixing the sign turned the step into a fixed-point iteration whose
      multiplier is `(dz/dr)/tan|phi|`. On the 6% ramp that is 2.0 for a
      shallow beam: it diverges, and 532 returns/sweep came out at elevations
      down to -85 degrees, which `range_image.project` correctly reported as
      clamped below the -24.8 degree floor. Bisection has no such condition.

    Returns (r, hit) -- range along the ground plane, and whether a beam met
    the surface within `max_range_m` at all.
    """
    lo = np.zeros_like(phi)
    hi = np.full_like(phi, max_range_m)
    slope = np.tan(phi)                      # negative: these beams point down

    def f(r):
        z = terrain_height_m(r * np.cos(theta) + pose_x_m, r * np.sin(theta), seed)
        return sensor_height_m + r * slope - z

    # A beam that is still above the surface at max_range never lands inside
    # the scene -- the ramp climbs without bound, so this is also what stops a
    # near-horizontal beam being chased to infinity.
    hit = f(hi) <= 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = f(mid) > 0.0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi), hit


def _in_sensor_fov(xyz_vehicle, sensor_height_m: float):
    """Drop returns the sensor could not physically have made.

    A real HDL-64E fires 64 beams between -24.8 and +2 degrees and nothing
    outside that comes back, and `range_image.project` warns and clamps to an
    edge ring for anything that does. A fixture that trips a real warning
    teaches everyone to ignore it.

    `structure` returns are sampled from a shape rather than fired, so they
    need this. Ground returns are fired and land on the surface, so they are
    inside by construction -- see the residual test in `scan`.

    Measured from the LASER, which is `sensor_height_m` above the vehicle
    origin; doing it in the vehicle frame would tilt the cone by the mount
    height. `horiz > 1.0` drops what is inside the sensor housing.
    """
    xyz = np.asarray(xyz_vehicle, dtype=np.float64)
    horiz = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.degrees(np.arctan2(xyz[:, 2] - sensor_height_m, np.maximum(horiz, 1e-9)))
    return (phi >= PHI_MIN_DEG) & (phi <= PHI_MAX_DEG) & (horiz > 1.0)


def _structure(rng, pose_x_m, sensor_height_m, n_wall=6000, n_pole=900,
               n_sign=400, n_tree=1200):
    """Off-ground returns: a facade each side, a pole, a sign, a canopy.

    Returns (points in VEHICLE frame, raw ids). Geometry only has to be
    plausible -- what this is for is the classes. Pole and traffic-sign are
    learning ids 17 and 18, above the old 4-bit candidate, so a sequence with
    them in it exercises the 5|3 byte end to end instead of in the abstract.
    """
    parts, labels = [], []

    y = np.where(rng.random(n_wall) < 0.5, -7.0, 7.0)
    parts.append(np.column_stack([rng.uniform(6.0, 45.0, n_wall), y,
                                  rng.uniform(0.0, 3.0, n_wall)]))
    labels.append(np.full(n_wall, BUILDING))

    parts.append(np.column_stack([rng.uniform(12.5, 13.5, n_pole),
                                  rng.uniform(4.4, 4.6, n_pole),
                                  rng.uniform(0.2, 2.5, n_pole)]))
    labels.append(np.full(n_pole, POLE))

    parts.append(np.column_stack([rng.uniform(12.4, 13.6, n_sign),
                                  rng.uniform(4.2, 4.8, n_sign),
                                  rng.uniform(2.2, 2.9, n_sign)]))
    labels.append(np.full(n_sign, TRAFFIC_SIGN))

    parts.append(np.column_stack([rng.uniform(20.0, 44.0, n_tree),
                                  rng.uniform(5.0, 6.0, n_tree) *
                                  rng.choice([-1.0, 1.0], n_tree),
                                  rng.uniform(1.5, 3.4, n_tree)]))
    labels.append(np.full(n_tree, VEGETATION))

    xyz = np.vstack(parts)
    lab = np.concatenate(labels)
    keep = _in_sensor_fov(xyz, sensor_height_m)
    return xyz[keep], lab[keep].astype(np.uint16)


def _crowd(rng, n_people: int, sensor_height_m: float):
    """`n_people` pedestrians in the near field, on the raw `moving-person` id.

    The worst case this project has, and it is worst in four ways at once
    (Day 6 D1, "confirm the memory bound holds under a dense-crowd scene"):

      * every return is dynamic, so the transient layer takes all of them and
        the tracked-object list is pushed at its `max_tracks` cap;
      * `person` is in `refine_classes`, so the semantic gate fires on every
        one of them and the refinement pool is pushed at its 512-block cap;
      * they are small and close, so they occupy many distinct fine cells
        rather than a few coarse ones, pushing `max_candidates`;
      * they are separate objects a metre apart, which is the clustering
        worst case -- one big blob is cheaper than 200 small ones.

    Spread over a 30 x 16 m patch ahead of the vehicle: dense enough to be a
    crowd, not so dense that they merge into one cluster and stop being 200
    tracked objects.
    """
    per = 24                                   # returns per person
    cx = rng.uniform(4.0, 34.0, n_people).repeat(per)
    cy = rng.uniform(-8.0, 8.0, n_people).repeat(per)
    pts = np.column_stack([
        cx + rng.uniform(-0.25, 0.25, n_people * per),
        cy + rng.uniform(-0.25, 0.25, n_people * per),
        rng.uniform(0.0, 1.8, n_people * per),
    ])
    keep = _in_sensor_fov(pts, sensor_height_m)
    return pts[keep], np.full(int(keep.sum()), MOVING_PERSON, dtype=np.uint16)


def scan(pose_x_m: float = 0.0, sensor_height_m: float = 1.73,
         n_azimuth: int = 720, n_beams: int = 64, max_range_m: float = 100.0,
         moving_car: bool = True, structure: bool = False, crowd: int = 0,
         seed: int = 0):
    """One HDL-64E-shaped sweep of the surface, in VEHICLE frame.

    Beams are fired on the real angular grid and intersected with the ground
    (`_beam_range`). The intersection is the §1.2 model itself -- on flat
    ground it reduces to `r = h_s/tan|phi|` -- which is exactly the geometry
    the ring schedule is derived from, so the sampling density this produces
    has the right *shape*: quadratic radial spacing, linear azimuthal spacing,
    and a blind cone.

    `crowd=N` adds N pedestrians on the raw `moving-person` id -- the
    worst-case scene for the memory bound. Off by default; see `_crowd`.

    Returns (points (N,3), raw label id (N,) uint16, is_ground (N,)).
    """
    phi = np.radians(np.linspace(PHI_MIN_DEG + FOV_EPS_DEG,
                                PHI_MAX_DEG - FOV_EPS_DEG, n_beams))
    theta = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    phi, theta = phi.reshape(-1), theta.reshape(-1)

    down = phi < np.radians(-0.2)                       # beams that reach ground
    phi, theta = phi[down], theta[down]

    r, hit = _beam_range(phi, theta, pose_x_m, sensor_height_m, seed, max_range_m)
    x, y = r * np.cos(theta), r * np.sin(theta)
    z = terrain_height_m(x + pose_x_m, y, seed)

    # ⚑ Keep only beams that actually MEET the surface.
    #
    #   `terrain_height_m` has two step discontinuities -- the 18 cm kerb and
    #   the 40 cm pothole rim -- and a beam aimed at one has no intersection
    #   at all: f jumps across zero instead of crossing it. Bisection is
    #   perfectly happy to return the edge, and those returns are wrong in two
    #   visible ways. Their realised elevation is off by the angle the step
    #   subtends (2.6 deg at the near kerb), which put ~0.6% of every sweep
    #   outside the sensor's own -24.8 deg floor for `range_image.project` to
    #   clamp and warn about. And they sit within float32 rounding of the
    #   discontinuity, so re-evaluating the surface at the stored coordinates
    #   lands on the far side: a 40 cm error on a scene whose whole point is
    #   that returns are EXACTLY on the analytic surface.
    #
    #   The residual test is the general form of both -- it needs no list of
    #   which features are steps -- and it makes the FOV property true by
    #   construction, since a return with zero residual is at exactly the
    #   elevation its beam was fired at.
    #
    #   Dropped rather than placed on the riser: a return on a vertical kerb
    #   face is not on `terrain_height_m(x, y)`, which is single-valued and is
    #   what every metric compares against, so keeping them would make the
    #   REFERENCE wrong -- worse than a thin gap along the kerb line. If §4.1
    #   ever wants real riser returns they belong in `_structure`, with the
    #   rest of the off-ground geometry.
    on_surface = np.abs(sensor_height_m + r * np.tan(phi) - z) < 1e-6

    keep = hit & on_surface & (r > 0) & (r < max_range_m)
    x, y, z = x[keep], y[keep], z[keep]
    pts = np.column_stack([x, y, z])
    cls = class_at(x + pose_x_m, y)
    ground = np.ones(pts.shape[0], dtype=bool)

    rng = np.random.default_rng(seed)

    if structure:
        s_pts, s_cls = _structure(rng, pose_x_m, sensor_height_m)
        pts = np.vstack([pts, s_pts])
        cls = np.concatenate([cls, s_cls])
        ground = np.concatenate([ground, np.zeros(len(s_pts), dtype=bool)])

    if crowd:
        c_pts, c_cls = _crowd(rng, crowd, sensor_height_m)
        pts = np.vstack([pts, c_pts])
        cls = np.concatenate([cls, c_cls])
        ground = np.concatenate([ground, np.zeros(len(c_pts), dtype=bool)])

    if moving_car:
        # A car 12 m ahead, 2 m wide, 1.5 m tall, on the raw moving-* label so
        # the reference map has something to strip and DR/SP (§9.4) has both
        # directions to score. It pulls away at 0.5 m/frame in the vehicle
        # frame, so at the default 2 m/frame step it moves in the WORLD too --
        # a `moving-car` label on a car that never moves is worse than no car.
        n = 400
        cx = 12.0 + 0.5 * seed
        car = np.column_stack([
            cx + rng.uniform(-2.0, 2.0, n),
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(0.0, 1.5, n),
        ])
        pts = np.vstack([pts, car])
        cls = np.concatenate([cls, np.full(n, MOVING_CAR, dtype=np.uint16)])
        ground = np.concatenate([ground, np.zeros(n, dtype=bool)])

    return pts, cls, ground


# --- the frame convention, once ----------------------------------------------

def _tr_velo_to_cam0() -> np.ndarray:
    """Velodyne -> Camera-0, as the exact inverse of the axis permutation.

    KITTI's real `Tr` also carries a few centimetres of translation and a
    fraction of a degree of rotation. Using the exact inverse instead keeps
    this sequence's world coordinates equal to the ones it was asked for, so a
    test can assert where the pothole ended up rather than only that nothing
    crashed. The chain being exercised is identical either way -- it is the
    same four matrices multiplied in the same order.
    """
    from vrgrid.perception.transforms import R_CAM0_TO_VEH

    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = R_CAM0_TO_VEH.T
    return tr


def pose_for(vehicle_xyz_world, tr=None) -> np.ndarray:
    """The `poses/<seq>.txt` row that puts the vehicle at `vehicle_xyz_world`.

    Inverts `transforms.vehicle_to_world`'s composition:

        T_VW = R_flip @ T_pose @ Tr @ T_V_S
        =>  T_pose = R_flip^-1 @ T_VW @ T_V_S^-1 @ Tr^-1

    Rotation is identity: the vehicle drives straight along +x. That is
    deliberate for a first pass -- it exercises ego-motion filling and the
    toroidal shift without also exercising rotation, so when the numbers are
    wrong there is one fewer place for them to be wrong. Note that the row
    written to disk is NOT the identity even so, because the composition puts
    a 90 degree axis permutation and the sensor mount between the two.
    """
    from vrgrid.perception.transforms import _T_V_S, R_CAM0_TO_VEH

    tr = _tr_velo_to_cam0() if tr is None else np.asarray(tr, dtype=np.float64)
    r_flip = np.eye(4, dtype=np.float64)
    r_flip[:3, :3] = R_CAM0_TO_VEH

    t_vw = np.eye(4, dtype=np.float64)
    t_vw[:3, 3] = np.asarray(vehicle_xyz_world, dtype=np.float64)

    t_pose = np.linalg.inv(r_flip) @ t_vw @ np.linalg.inv(_T_V_S) @ np.linalg.inv(tr)
    return t_pose[:3, :4]


def write_sequence(root, sequence: str = "99", n_frames: int = 12,
                   step_m: float = 2.0, **kw) -> Path:
    """Write a sequence to disk in the layout `perception.loader` reads.

        <root>/poses/<seq>.txt                  official-GT-shaped poses
        <root>/sequences/<seq>/calib.txt        the `Tr:` line
        <root>/sequences/<seq>/velodyne/*.bin   float32 x,y,z,intensity, SENSOR frame
        <root>/sequences/<seq>/labels/*.label   uint32 raw SemanticKITTI ids

    Poses are pure forward translation at `step_m` per frame. Returns `root`,
    which is what `loader.DATA_ROOT` wants pointing at it.
    """
    from vrgrid.perception.transforms import T_S_V

    root = Path(root)
    seq_dir = root / "sequences" / sequence
    (seq_dir / "velodyne").mkdir(parents=True, exist_ok=True)
    (seq_dir / "labels").mkdir(parents=True, exist_ok=True)
    (root / "poses").mkdir(parents=True, exist_ok=True)

    tr = _tr_velo_to_cam0()
    row = " ".join(f"{v:.12e}" for v in tr[:3, :4].reshape(-1))
    (seq_dir / "calib.txt").write_text(
        "P0: " + " ".join(["0.0"] * 12) + "\n"
        f"Tr: {row}\n")

    # Vehicle -> Sensor is the inverse of the 1.73 m ground drop. A `.bin` is
    # in the sensor frame; `scan` produces the vehicle frame.
    dz = -T_S_V[2, 3]

    poses = []
    for i in range(n_frames):
        pose_x = i * step_m
        pts, cls, _ = scan(pose_x_m=pose_x, seed=i, **kw)

        buf = np.column_stack([pts[:, 0], pts[:, 1], pts[:, 2] + dz,
                               np.full(len(pts), 0.5)]).astype(np.float32)
        buf.tofile(seq_dir / "velodyne" / f"{i:06d}.bin")
        cls.astype(np.uint32).tofile(seq_dir / "labels" / f"{i:06d}.label")

        poses.append(pose_for((pose_x, 0.0, 0.0), tr).reshape(-1))

    np.savetxt(root / "poses" / f"{sequence}.txt", np.array(poses), fmt="%.12e")
    return root


def read_sequence(root, sequence: str = "99"):
    """Iterate a sequence from disk: (points in VEHICLE frame, RAW label ids,
    vehicle -> world 4x4).

    Reads the documented layout through the same two functions the real path
    uses -- `loader.read_calib` for `Tr` and `transforms.vehicle_to_world` for
    the composition -- so this is a second *caller* of the convention, not a
    second copy of it. What it is not is a second loader: `perception.loader`
    handles KITTI's own quirks and is the thing that reads real data. This
    exists so the harness is not blocked on it, and returns the vehicle frame
    because that is what `run_sequence` consumes.
    """
    from vrgrid.perception.loader import read_calib
    from vrgrid.perception.transforms import T_S_V, vehicle_to_world

    root = Path(root)
    seq_dir = root / "sequences" / sequence
    tr = read_calib(seq_dir / "calib.txt")["Tr_velo_to_cam0"]
    poses = np.loadtxt(root / "poses" / f"{sequence}.txt").reshape(-1, 3, 4)

    for i, row in enumerate(poses):
        pts = np.fromfile(seq_dir / "velodyne" / f"{i:06d}.bin", dtype=np.float32)
        pts = pts.reshape(-1, 4)[:, :3].astype(np.float64)
        pts[:, 2] += T_S_V[2, 3]                       # sensor -> vehicle
        lbl = np.fromfile(seq_dir / "labels" / f"{i:06d}.label", dtype=np.uint32)
        yield pts, (lbl & 0xFFFF).astype(np.uint32), vehicle_to_world(row, tr=tr)
