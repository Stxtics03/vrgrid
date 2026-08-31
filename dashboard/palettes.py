"""Colour tables for the dashboard -- no Rerun dependency. [JP]

Every colour the dashboard renders lives here and only here: the SemanticKITTI
19-class map (`class_to_color`, `_CLASS_LUT`), the 7-group colourblind-safe map
(`GROUP_*`) and the ghost highlight (`GHOST_RGB`). `demo_synthetic.py` and
`pipeline_view.py` import from this module instead of defining colours of their
own, so there is one source of truth for what the audit in `cvd.py` checks.

Deliberately free of `import rerun`: the CVD audit and `tests/test_cvd.py` need
the numbers, not the viewer, and CI installs only `[dev]` -- rerun-sdk is the
optional `[dash]` extra. Keep it that way; a top-level rerun import here turns
the whole palette suite red on a machine without the viewer installed.
"""

import numpy as np


def class_to_color(class_idx: int) -> list:
    """Map class index to RGB color (rerun format)."""
    colors = {
        -1: [50, 50, 50],      # unknown
        0: [245, 150, 100],    # car
        1: [245, 230, 100],    # bicycle
        2: [150, 60, 30],      # motorcycle
        3: [180, 30, 80],      # truck
        4: [255, 0, 0],        # other-vehicle
        5: [30, 30, 255],      # person
        6: [200, 40, 255],     # bicyclist
        7: [90, 30, 150],      # motorcyclist
        8: [255, 0, 255],      # road
        9: [255, 150, 255],    # parking
        10: [75, 0, 75],       # sidewalk
        11: [75, 0, 175],      # other-ground
        12: [0, 200, 255],     # building
        13: [50, 120, 255],    # fence
        14: [0, 175, 0],       # vegetation
        15: [0, 60, 135],      # trunk
        16: [80, 240, 150],    # terrain
        17: [150, 240, 255],   # pole
        18: [0, 0, 255],       # traffic-sign
    }
    return colors.get(class_idx, [128, 128, 128])


_CLASS_LUT = np.array([class_to_color(c) for c in range(-1, 19)], dtype=np.uint8)  # index c+1

# Highlight for moving points (motion layer + the world/ghosts overlay).
# Okabe & Ito (2008) "reddish purple" -- chosen because it is the one hue that
# stays >= Delta-E 16 from every colour in the SemanticKITTI class map AND from
# the motion-static grey, under normal vision and simulated protanopia,
# deuteranopia and tritanopia (see dashboard/cvd.py). Red would collide with
# `vegetation` under deuteranopia (Delta-E 3) and with `other-vehicle`; this
# does not. The 3x point radius is the redundant, colour-independent cue.
GHOST_RGB = (204, 121, 167)  # #CC79A7

# --- `groups` palette: 19 SemanticKITTI classes -> 7 colourblind-safe groups --
#
# Colours are Okabe & Ito (2008) plus two greys, with #CC79A7 held back for the
# ghost highlight. Every pair clears Delta-E 16 under normal vision and
# simulated protanopia / deuteranopia / tritanopia (dashboard/cvd.py).
GROUP_NAMES = (
    "drivable-ground", "structure", "vegetation", "vehicle",
    "vulnerable-road-user", "pole-signage", "unknown",
)
GROUP_RGB = np.array([
    (187, 187, 187),  # drivable-ground  -- neutral grey
    (0, 114, 178),    # structure        -- Okabe-Ito blue
    (0, 158, 115),    # vegetation       -- Okabe-Ito bluish-green
    (230, 159, 0),    # vehicle          -- Okabe-Ito orange
    (213, 94, 0),     # vulnerable-road-user -- Okabe-Ito vermillion (warning)
    (86, 180, 233),   # pole-signage     -- Okabe-Ito sky-blue
    (85, 85, 85),     # unknown          -- dark grey
], dtype=np.uint8)

# raw class index -> group index (index this by `semantic + 1`; row 0 is class -1)
GROUP_MEMBERS = {
    "drivable-ground": ("road", "parking", "sidewalk", "other-ground"),
    "structure": ("building", "fence"),
    "vegetation": ("vegetation", "trunk", "terrain"),
    "vehicle": ("car", "bicycle", "motorcycle", "truck", "other-vehicle"),
    "vulnerable-road-user": ("person", "bicyclist", "motorcyclist"),
    "pole-signage": ("pole", "traffic-sign"),
    "unknown": ("unlabeled",),
}
_CLASS_TO_GROUP = {
    -1: 6, 0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 0, 9: 0,
    10: 0, 11: 0, 12: 1, 13: 1, 14: 2, 15: 2, 16: 2, 17: 5, 18: 5,
}
_GROUP_LUT = np.array([_CLASS_TO_GROUP[c] for c in range(-1, 19)], dtype=np.uint8)  # index c+1
