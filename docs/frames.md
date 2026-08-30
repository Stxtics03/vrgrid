# Frames

Every coordinate frame in the system, and every transform between them. Written Day 0
so nobody has to guess handedness, units, or origin at integration time.

Per jp-working-brief.md: **Vehicle frame is x forward, y left, z up.**
Ranges and gradients are float metres (suffix `_m`). Heights are int16 in 1 cm (suffix `_cm`).

---

## Frames

| Frame | Symbol | Origin | Axes / Handedness | Units | Owner |
|-------|--------|--------|-------------------|-------|-------|
| **World** | W | Coincides with the Vehicle frame at frame 0 of the sequence; fixed thereafter | x **Forward**, y **Left**, z **Up** **Right-handed** | metres (`_m`) | JP |
| **Vehicle** | V | On the road surface directly below the Velodyne | x **Forward**, y **Left**, z **Up** **Right-handed** | metres (`_m`) | JP |
| **Sensor (Velodyne HDL-64E)** | S | Velodyne optical center, ~1.73 m above the road | x **Forward**, y **Left**, z **Up** **Right-handed** | metres (`_m`) | JP |

**Note:** The HDL-64E sensor frame in KITTI convention matches the vehicle frame exactly (x forward, y left, z up); they differ only by the 1.73 m mounting height. The `Tr` matrix in `calib.txt` is Velodyne → Camera-0 (x right, y down, z forward); `poses.txt` is Camera-0 → World_cam in that same camera convention. Both are used inside `T_V_W` and the result is rotated once into the z-up World frame above, so **every downstream consumer sees x-forward, y-left, z-up**.

---

## Transforms

| From → To | Notation | Defined by | Notes |
|-----------|----------|------------|-------|
| Sensor → Vehicle | `T_S_V` | **KITTI documented convention** (1.73 m HDL-64E mount height, identity rotation) | **Identity rotation**, translation only: sensor at (0, 0, 1.73) m in vehicle frame. `T_S_V = [I \| [0, 0, 1.73]^T]`. **NOT from calib.txt** — see note below. |
| Vehicle → World | `T_V_W(k)` | `R_flip · pose(k) · Tr · T_V_S` | Per-frame. `T_V_S` undoes the 1.73 m ground drop, `Tr` (calib.txt) is Velodyne→Camera-0, `pose(k)` (poses.txt) is Camera-0→World_cam, `R_flip` (`R_CAM0_TO_VEH`) rotates World_cam into the z-up World frame. See Transform Details. |
| Sensor → World | `T_S_W(k)` | `T_S_W = T_V_W(k) @ T_S_V` | Composed per frame. The 1.73 m drop cancels, so this equals the textbook KITTI chain `R_flip · pose(k) · Tr` acting on raw Velodyne points. |

---

## Transform Details

### Sensor → Vehicle (`T_S_V`)

The HDL-64E is rigidly mounted on the roof.

**Source of parameters (STATED ASSUMPTION, NOT MEASURED DATA):**

- **1.73 m height**: From KITTI spec sheet / HDL-64E documentation ("sensor height 1.73 m"). Universal convention in literature.
- **Identity rotation**: Assumes Velodyne axes align with vehicle axes (x forward, y left, z up). Standard KITTI convention.

**Why NOT from calib.txt:**
- `calib.txt` Tr matrix is **Velodyne → Camera 0** (x right, y down, z forward), NOT Velodyne → Vehicle.
- Its translation is `[-0.012, -0.054, -0.292]` m — clearly a camera offset, not vehicle.
- The KITTI **odometry benchmark does not publish** a Velodyne→Vehicle extrinsic.
- The separate KITTI **raw-data release** includes `calib_imu_to_velo.txt` (IMU→Velodyne), but we don't have that file.

```
T_S_V = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 1.73],
         [0, 0, 0, 1]]
```

**Units:** metres. `T_S_V` is constant for all frames.

**Validation:** If this assumption is wrong, the static-wall test (Gate Item 1) will catch it — a known flat wall will show systematic rotation/translation drift across frames.

### Vehicle → World (`T_V_W(k)`)

**KITTI `poses.txt` provides Camera-0 (left camera) → World_cam**, NOT Vehicle → World, and
`calib.txt` `Tr` provides Velodyne → Camera-0. `T_V_W` chains through both and then
rotates the camera-convention world into the z-up World frame.

Constant axis permutation **Camera-0 → World (z-up)**:
```
R_CAM0_TO_VEH = [[ 0,  0,  1],
                 [-1,  0,  0],
                 [ 0, -1,  0]]   # world_x = cam_z, world_y = -cam_x, world_z = -cam_y
```

Constant Vehicle → Velodyne (undo the ground drop): `T_V_S = [I | [0, 0, -1.73]^T]`.

Per-frame composition (right-to-left on a Vehicle-frame point):
```
T_V_W(k) = R_flip · pose(k) · Tr · T_V_S

  T_V_S     Vehicle  → Velodyne      (constant, +1.73 m removed)
  Tr        Velodyne → Camera-0      (constant, calib.txt `Tr:`)
  pose(k)   Camera-0 → World_cam     (poses.txt line k, [R|t] row-major, frame 0 = identity)
  R_flip    World_cam → World (z-up) (constant, R_CAM0_TO_VEH as a 4×4)
```

**Units:** metres. One matrix per frame k. Implemented in `perception/transforms.py::vehicle_to_world`.

**Note on GT poses:** `poses.txt` is the OXTS RTK-GPS/INS trajectory. It carries a few
cm of its own drift over ~100 frames, so the static-wall test targets sub-degree /
sub-decimetre stability, not zero. The offset *slope* across 100 frames (not its
span) is the sensitive check for a `T_V_W` translation error — a bumpy facade
contributes bounded span but no slope.

### Sensor → World (`T_S_W(k)`)

Composed per frame:
```
T_S_W(k) = T_V_W(k) @ T_S_V
```

Used to transform raw sensor points to world coordinates for mapping.

---

## Point Transform Pipeline

```
Raw points (Sensor frame, N×4: x, y, z, intensity)
    │
    ├─► T_S_V (constant) ──► Vehicle frame (N×3)
    │
    ├─► Range image projection (64×512) using vehicle-frame points
    │     Spherical projection: yaw = atan2(y, x), pitch = asin(z / r)
    │     Inverse index stored for reversibility.
    │
    ├─► Semantic + motion labels (from the raw .label file, no inference)
    │     semantic_labels(): per-point 19-class (0-18, -1=ignore)
    │     is_moving():       per-point moving-* flag
    │
    └─► T_V_W(k) (per-frame) ──► World frame (for mapping)
          Scatter to variable-resolution grid.
```

---

## Static-Wall Test (Gate Item 1)

**Purpose:** Verify `T_S_V` and `T_V_W` are correct before any mapping runs.

**Procedure** (`tests/test_static_wall.py`, three 100-frame segments of seq 00):
1. Select wall points per frame by the SemanticKITTI `building` label (raw id 50)
   inside a vehicle-relative lateral band, so the selection slides with the vehicle.
2. Transform to World; accumulate all 100 frames and fit ONE verticality-constrained
   RANSAC plane as the common reference.
3. Per frame, check: normal drift vs the global normal; mean signed distance to the
   global plane (its span **and its linear slope** across the 100 frames); plane RMS;
   point count.

Per-frame the wall normal is re-fitted and split, relative to frame 0's normal, into
signed **yaw** (rotation in the horizontal plane) and signed **pitch** (tilt from
vertical). A linear trend is fitted to yaw, pitch and offset vs frame index; the
total change over the 100-frame window is the drift number.

**Failure modes:**
- **yaw trend** in the normal → rotation error in `R_flip` / `Tr` / `T_S_V`
- **offset trend** in the plane → `pose(k)` parsing or composition in `T_V_W`

**Strict gates (all three segments 3150 / 2550 / 0600):** global plane vertical to
`|n·up| < 0.12`; `|yaw trend| < 1.0°`; `|offset trend| < 0.03 m`. Measured:
`|n·up|` 0.023 / 0.044 / 0.081; yaw `−0.08 / −0.22 / +0.27°`;
offset `+0.45 / −0.88 / +2.51 cm`.

**Loose regression guards (not tuning targets):** unsigned normal drift mean < 3°,
max < 6°; plane RMS max < 0.25 m.

**Pitch trend is reported, not gated.** `turning_2550` shows +3.8° pitch while its
yaw and offset are clean; it does not track vehicle pitch (flat road here) and steps
mid-segment — a set-back upper storey entering the vehicle-relative selection band,
not a transform error. See the pitch note in `tests/test_static_wall.py`.

---

## Units Checklist

| Quantity | Frame | Type | Suffix | Example |
|----------|-------|------|--------|---------|
| Range / distance | Sensor, Vehicle, World | float32 | `_m` | `range_m`, `half_width_m` |
| Point coordinates (x, y, z) | Sensor, Vehicle, World | float32 | `_m` | `x_m`, `y_m`, `z_m` |
| Ground/ceiling height | Vehicle, World | int16 (1 cm) | `_cm` | `ground_height_cm` |
| Height variance | Vehicle, World | uint8 (log) | — | `height_var` |
| Cell size | Vehicle, World | float32 | `_m` | `cell_m` |
| Pose translation | World | float64 | `_m` | `t_x`, `t_y`, `t_z` |

**Never mix `_m` and `_cm` silently.** Convert explicitly.