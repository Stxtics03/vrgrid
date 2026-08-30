# Frames

Every coordinate frame in the system, and every transform between them. Written Day 0
so nobody has to guess handedness, units, or origin at integration time.

Per jp-working-brief.md: **Vehicle frame is x forward, y left, z up.**
Ranges and gradients are float metres (suffix `_m`). Heights are int16 in 1 cm (suffix `_cm`).

---

## Frames

| Frame | Symbol | Origin | Axes / Handedness | Units | Owner |
|-------|--------|--------|-------------------|-------|-------|
| **World** | W | First pose of sequence (sequence 00, frame 0), ENU | x East, y North, z Up **Right-handed** | metres (`_m`) | JP |
| **Vehicle** | V | Vehicle center of gravity (ground projection), at sensor height | x **Forward**, y **Left**, z **Up** **Right-handed** | metres (`_m`) | JP |
| **Sensor (Velodyne HDL-64E)** | S | Velodyne optical center | x **Forward**, y **Left**, z **Up** **Right-handed** | metres (`_m`) | JP |

**Note:** The HDL-64E sensor frame in KITTI convention matches the vehicle frame exactly (x forward, y left, z up). The `Tr` matrix in `calib.txt` transforms from Velodyne → Camera 0 (x right, y down, z forward), not to vehicle frame.

---

## Transforms

| From → To | Notation | Defined by | Notes |
|-----------|----------|------------|-------|
| Sensor → Vehicle | `T_S_V` | `calib.txt` (Tr_velo_to_cam) + known Camera 0 ↔ Vehicle | **Identity rotation**, translation only: sensor mounted at (0, 0, 1.73) m in vehicle frame. `T_S_V = [I | [0, 0, 1.73]^T]`. |
| Vehicle → World | `T_V_W(k)` | `poses.txt` (KITTI poses) | Per-frame 4×4 pose from `poses.txt` (row-major 3×4, bottom row implied [0,0,0,1]). Pose is **Vehicle in World** (i.e., `T_V_W`). |
| Sensor → World | `T_S_W(k)` | `T_S_W = T_V_W @ T_S_V` | Composed for each frame k. Used to transform points to world for mapping. |

---

## Transform Details

### Sensor → Vehicle (`T_S_V`)

The HDL-64E is rigidly mounted on the roof. From KITTI spec and `calib.txt`:

- **Rotation**: Identity (sensor axes align with vehicle axes: x forward, y left, z up)
- **Translation**: `[0, 0, 1.73]^T` metres (sensor height from ground)

```
T_S_V = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 1.73],
         [0, 0, 0, 1]]
```

**Units:** metres. `T_S_V` is constant for all frames.

### Vehicle → World (`T_V_W(k)`)

From `poses.txt` (KITTI odometry ground truth). Each line is a 3×4 matrix in row-major order:

```
[R_00 R_01 R_02 t_x
 R_10 R_11 R_12 t_y
 R_20 R_21 R_22 t_z]
```

Bottom row is `[0, 0, 0, 1]`. This is the **Vehicle pose in World frame** (i.e., transforms a point from Vehicle → World).

**Units:** metres. One per frame k.

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
    ├─► FRNet inference (uses raw Sensor-frame points + frustum coords)
    │     Output: per-point semantic labels (0-18, -1=ignore)
    │
    └─► T_V_W(k) (per-frame) ──► World frame (for mapping)
          Scatter to variable-resolution grid.
```

---

## Static-Wall Test (Gate Item 1)

**Purpose:** Verify `T_S_V` and `T_V_W` are correct before any mapping runs.

**Procedure:**
1. Pick a flat building face in sequence 00 (e.g., frames 100–200).
2. Transform points from each frame to World using `T_S_W(k)`.
3. Fit a plane to wall points per frame.
4. Assert plane normal and offset **do not drift** across frames.

**Failure modes:**
- Slow **rotation** of wall normal → `T_S_V` (or sensor axes) is wrong
- Slow **translation** of wall plane → `T_V_W` (or pose parsing) is wrong

**Pass criterion:** Wall plane parameters stable to < 1 cm / 0.1° over 100 frames.

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