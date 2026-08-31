#!/usr/bin/env python
"""
Minimal Rerun dashboard — synthetic 2.5D map with ring boundaries.

Renders:
- Flat ground plane (class: road)
- Two boxes (class: car)
- A slope (class: terrain)
- Ring boundaries from configs/schedule_5_10_20_40.yaml
"""

import numpy as np
import rerun as rr
import yaml

# The 19-class colour table lives in `palettes.py`, which imports no rerun, so
# the CVD audit can read it without the viewer installed. Re-exported here
# because this module has always been its import path.
from .palettes import class_to_color


def load_schedule(config_path: str = "configs/schedule_5_10_20_40.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_synthetic_map() -> dict:
    """
    Create synthetic 2.5D map elements.

    Returns dict with:
    - ground_height: (H, W) float32 — height in meters
    - semantic_class: (H, W) int32 — class indices
    """
    # Use a small grid for synthetic demo
    H, W = 200, 200
    scale = 0.5  # meters per pixel

    ground_height = np.zeros((H, W), dtype=np.float32)
    semantic_class = np.full((H, W), -1, dtype=np.int32)

    # Class indices (SemanticKITTI 19-class)
    ROAD = 8
    CAR = 0
    TERRAIN = 16

    # Center of image = vehicle position
    cx, cy = W // 2, H // 2

    # 1. Flat ground plane (road) — everywhere initially
    semantic_class[:, :] = ROAD

    # 2. Slope: ramp rising from y=150 to y=180 (south of vehicle)
    # Height goes from 0m to 2m over 30 pixels
    slope_start = 150
    slope_end = 180
    for y in range(slope_start, slope_end + 1):
        frac = (y - slope_start) / (slope_end - slope_start)
        ground_height[y, :] = frac * 2.0
        semantic_class[y, :] = TERRAIN

    # 3. Box 1: car at (x=+10m, y=+5m) -> pixels
    box1_cx = cx + int(10 / scale)
    box1_cy = cy - int(5 / scale)
    box_size = int(3 / scale)  # 3m box
    ground_height[box1_cy-box_size:box1_cy+box_size, box1_cx-box_size:box1_cx+box_size] = 1.5
    semantic_class[box1_cy-box_size:box1_cy+box_size, box1_cx-box_size:box1_cx+box_size] = CAR

    # 4. Box 2: car at (x=-8m, y=-12m)
    box2_cx = cx - int(8 / scale)
    box2_cy = cy + int(12 / scale)
    ground_height[box2_cy-box_size:box2_cy+box_size, box2_cx-box_size:box2_cx+box_size] = 1.5
    semantic_class[box2_cy-box_size:box2_cy+box_size, box2_cx-box_size:box2_cx+box_size] = CAR

    return {
        "ground_height": ground_height,
        "semantic_class": semantic_class,
        "scale": scale,
        "center": (cx, cy),
    }


def log_synthetic_map(map_data: dict):
    """Log the synthetic map to Rerun."""
    H, W = map_data["ground_height"].shape
    scale = map_data["scale"]
    cx, cy = map_data["center"]

    # Class color image (top-down)
    class_img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            class_img[y, x] = class_to_color(map_data["semantic_class"][y, x])

    # Log as Image (top-down view)
    rr.log("map/classes", rr.Image(class_img))

    # Log height as depth image for 3D view
    height_normalized = (map_data["ground_height"] - map_data["ground_height"].min()) / \
                        (map_data["ground_height"].max() - map_data["ground_height"].min() + 1e-6)
    height_img = (height_normalized * 255).astype(np.uint8)
    rr.log("map/height", rr.Image(height_img))

    # Log ground points for 3D view
    points_3d = []
    colors_3d = []
    stride = 2  # subsample for performance
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Convert pixel to world coords (meters)
            wx = (x - cx) * scale
            wy = (cy - y) * scale  # y flipped: image y down = world y forward
            wz = map_data["ground_height"][y, x]
            points_3d.append([wx, wy, wz])
            colors_3d.append(class_to_color(map_data["semantic_class"][y, x]))

    points_3d = np.array(points_3d, dtype=np.float32)
    colors_3d = np.array(colors_3d, dtype=np.uint8)
    rr.log("map/points", rr.Points3D(points_3d, colors=colors_3d, radii=0.1))


def log_ring_boundaries(schedule: dict, map_data: dict):
    """Log ring boundaries as circles in Rerun."""
    scale = map_data["scale"]
    cx, cy = map_data["center"]

    rings = schedule.get("rings", [])
    for ring in rings:
        half_width = ring["half_width_m"]
        # Convert to pixels
        radius_px = int(half_width / scale)

        # Log as circle (series of line segments)
        num_segments = 100
        angles = np.linspace(0, 2 * np.pi, num_segments + 1)
        xs = cx + radius_px * np.cos(angles)
        ys = cy - radius_px * np.sin(angles)  # flip y

        # Convert to world coords for logging
        strip = []
        for i in range(num_segments):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i + 1], ys[i + 1]
            # World coords
            wx1, wy1 = (x1 - cx) * scale, (cy - y1) * scale
            wx2, wy2 = (x2 - cx) * scale, (cy - y2) * scale
            strip.append([[wx1, wy1, 0], [wx2, wy2, 0]])

        if strip:
            strip = np.array(strip, dtype=np.float32)
            color = [255, 255, 255]  # white boundaries
            rr.log(f"rings/ring_{ring['ring']}", rr.LineStrips3D(strip, colors=color, radii=0.05))

        # Also log ring label
        label_pos = [half_width, 0, 0.5]
        rr.log(f"rings/label_{ring['ring']}",
               rr.Points3D([label_pos], colors=[255, 255, 0], radii=0.2,
                           labels=[f"Ring {ring['ring']}: {ring['cell_m']*100:.0f}cm"]))


def main():
    # Initialize Rerun
    rr.init("vrgrid_synthetic_demo", spawn=True)

    # Load schedule
    schedule = load_schedule()
    print(f"Loaded schedule: {schedule['name']}")
    for r in schedule["rings"]:
        print(f"  Ring {r['ring']}: half_width={r['half_width_m']}m, cell={r['cell_m']*100:.0f}cm")

    # Create synthetic map
    map_data = make_synthetic_map()

    # Log map
    log_synthetic_map(map_data)

    # Log ring boundaries
    log_ring_boundaries(schedule, map_data)

    # Log vehicle position
    rr.log("vehicle", rr.Points3D([[0, 0, 1.73]], colors=[0, 255, 0], radii=0.3,
                                   labels=["Vehicle (HDL-64E @ 1.73m)"]))

    # Log blind cone (from config)
    blind_cone = 3.74
    theta = np.linspace(-np.pi, np.pi, 50)
    xs = blind_cone * np.cos(theta)
    ys = blind_cone * np.sin(theta)
    cone_strip = []
    for i in range(len(theta) - 1):
        cone_strip.append([[xs[i], ys[i], 0], [xs[i+1], ys[i+1], 0]])
    cone_strip = np.array(cone_strip, dtype=np.float32)
    rr.log("sensor/blind_cone", rr.LineStrips3D(cone_strip, colors=[255, 0, 0], radii=0.05,
                                                 labels=["Blind cone (3.74m)"]))

    print("Dashboard running. Check Rerun viewer.")
    print("Views: 'map/classes' (top-down), 'map/points' (3D), 'rings/*' (boundaries)")


if __name__ == "__main__":
    main()