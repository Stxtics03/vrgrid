"""A synthetic sequence, in the real layout. [Aakash]

The reference map is the long pole and it is blocked on a 40 GB download. What
is NOT blocked is everything downstream of it, provided something produces the
same shapes. So this generates a miniature sequence on disk in exactly the
`data/sequences/NN/{velodyne,labels,poses.txt}` layout of `data/README.md`,
and `reference_map.build()` reads it through the same code path it will read
SemanticKITTI with. The hour the download lands, one argument changes.

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
    a pothole at (18, 0)    40 cm deep, 60 cm wide -- inside the §1.4 negative
                            obstacle limit (r_max = 10.8 m for 50 cm), so it
                            is *supposed* to be invisible past ~11 m. If a
                            metric ever says we found it at 40 m, the metric
                            is wrong.

Classes are `road` / `sidewalk` / `parking` only. `terrain` would be the
natural label for the verge; it is learning id 17 and did not fit the cell's
4-bit class nibble, which is where the §10.2 conflict was first met in
practice. The byte was re-split 5 | 3 on 1 Sep and `terrain` fits now, so this
scene can grow a verge whenever the metrics want one -- it has not been
changed yet, because every reference number in `docs/` was measured against
these three classes.
"""

from pathlib import Path

import numpy as np

# Class ids used here, all of which fit in 4 bits. See the note above.
ROAD, PARKING, SIDEWALK = 9, 10, 11
MOVING_CAR = 252  # raw SemanticKITTI moving-* id, stripped by the reference map

KERB_Y_M = 3.0
KERB_HEIGHT_M = 0.12
RAMP_START_X_M = 30.0
RAMP_SLOPE = 0.06
POTHOLE_XY_M = (18.0, 0.0)
POTHOLE_RADIUS_M = 0.30
POTHOLE_DEPTH_M = 0.40


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
    y = np.abs(np.asarray(y, dtype=np.float64))
    return np.where(y <= KERB_Y_M, ROAD,
                    np.where(y <= KERB_Y_M + 1.0, SIDEWALK, PARKING)).astype(np.uint8)


def scan(pose_x_m: float = 0.0, sensor_height_m: float = 1.73,
         n_azimuth: int = 720, n_beams: int = 64, max_range_m: float = 100.0,
         moving_car: bool = True, seed: int = 0):
    """One HDL-64E-shaped sweep of the surface, in VEHICLE frame.

    Beams are fired on the real angular grid and intersected with the ground.
    The intersection is the §1.2 model itself -- `r = h_s/tan|phi|` on flat
    ground, then one correction step for the actual surface height -- which is
    exactly the geometry the ring schedule is derived from, so the sampling
    density this produces has the right *shape*: quadratic radial spacing,
    linear azimuthal spacing, and a blind cone.

    Returns (points (N,3), class_id (N,), is_ground (N,)).
    """
    phi = np.radians(np.linspace(-24.8, 2.0, n_beams))
    theta = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
    phi, theta = np.meshgrid(phi, theta, indexing="ij")
    phi, theta = phi.reshape(-1), theta.reshape(-1)

    down = phi < np.radians(-0.2)                       # beams that reach ground
    phi, theta = phi[down], theta[down]

    r = sensor_height_m / np.tan(-phi)                  # flat-ground first guess
    x, y = r * np.cos(theta), r * np.sin(theta)
    z = terrain_height_m(x + pose_x_m, y, seed)
    r = (sensor_height_m + z) / np.tan(-phi)            # one correction step
    x, y = r * np.cos(theta), r * np.sin(theta)
    z = terrain_height_m(x + pose_x_m, y, seed)

    keep = (r > 0) & (r < max_range_m)
    x, y, z = x[keep], y[keep], z[keep]
    pts = np.column_stack([x, y, z])
    cls = class_at(x + pose_x_m, y)
    ground = np.ones(pts.shape[0], dtype=bool)

    if moving_car:
        # A car 12 m ahead, 2 m wide, 1.5 m tall, on the raw moving-* label so
        # the reference map has something to strip and DR/SP (§9.4) has both
        # directions to score.
        rng = np.random.default_rng(seed)
        n = 400
        cx = 12.0 + 0.5 * seed
        car = np.column_stack([
            cx + rng.uniform(-2.0, 2.0, n),
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(0.0, 1.5, n),
        ])
        pts = np.vstack([pts, car])
        cls = np.concatenate([cls, np.full(n, MOVING_CAR, dtype=np.uint8)])
        ground = np.concatenate([ground, np.zeros(n, dtype=bool)])

    return pts, cls, ground


def write_sequence(root, sequence: str = "99", n_frames: int = 12,
                   step_m: float = 2.0, **kw) -> Path:
    """Write a sequence to disk in the layout of `data/README.md`.

    Poses are pure forward translation. That is deliberate for a first pass:
    it exercises ego-motion filling and the toroidal shift without also
    exercising rotation, so when the numbers are wrong there is one fewer
    place for them to be wrong. Rotation goes in when JP's real poses do.
    """
    root = Path(root)
    seq = root / "sequences" / sequence
    (seq / "velodyne").mkdir(parents=True, exist_ok=True)
    (seq / "labels").mkdir(parents=True, exist_ok=True)

    poses = []
    for i in range(n_frames):
        pose_x = i * step_m
        pts, cls, _ = scan(pose_x_m=pose_x, seed=i, **kw)

        # KITTI .bin is float32 x,y,z,intensity
        buf = np.column_stack([pts, np.full(len(pts), 0.5)]).astype(np.float32)
        buf.tofile(seq / "velodyne" / f"{i:06d}.bin")
        # .label is uint32, lower 16 bits semantic
        cls.astype(np.uint32).tofile(seq / "labels" / f"{i:06d}.label")

        T = np.eye(4)
        T[0, 3] = pose_x
        poses.append(T[:3, :].reshape(-1))

    np.savetxt(seq / "poses.txt", np.array(poses), fmt="%.6e")
    return seq


def read_sequence(root, sequence: str = "99"):
    """Iterate a sequence from disk: (points, class_id, pose 4x4).

    Reads the documented layout, so it is the same reader for synthetic and
    real data. `perception.loader` will do this against SemanticKITTI with its
    calibration handling; this is the minimal version that needs no calib and
    exists so the harness is not blocked on it.
    """
    seq = Path(root) / "sequences" / sequence
    poses = np.loadtxt(seq / "poses.txt").reshape(-1, 3, 4)
    for i, row in enumerate(poses):
        pts = np.fromfile(seq / "velodyne" / f"{i:06d}.bin", dtype=np.float32)
        pts = pts.reshape(-1, 4)[:, :3].astype(np.float64)
        lbl = np.fromfile(seq / "labels" / f"{i:06d}.label", dtype=np.uint32)
        T = np.eye(4)
        T[:3, :] = row
        yield pts, (lbl & 0xFFFF).astype(np.uint32), T
