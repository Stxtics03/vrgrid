"""Colour-vision-deficiency check for the dashboard palettes. [JP]

    python -m vrgrid.dash.cvd                 # text report: colliding pairs per palette
    python -m vrgrid.dash.cvd --png out.png   # swatch grid: every palette x {normal,protan,deutan,tritan}

Simulation uses the Machado, Oliveira & Fernandes (2009) matrices at severity
1.0, applied in LINEAR sRGB (the DaltonLens convention). Collision test is
Delta-E 1976 (Euclidean in CIELAB) between the simulated colours of every pair
in a palette; below ~12 the two are hard to tell apart, below ~6 they are
effectively the same.

Source of the palettes under test: dashboard/palettes.py -- `class_to_color`
(the SemanticKITTI standard 19-class map), the `GROUP_*` tables and the ghost
highlight, plus the motion / ground pairs hard-coded in `_frame_colors`. That
module imports no rerun, so this audit and its tests run in CI, where the
viewer (the optional `[dash]` extra) is not installed.
"""

import numpy as np

from .palettes import GHOST_RGB, GROUP_NAMES, GROUP_RGB, class_to_color

# Machado et al. 2009, severity 1.0. Applied to linear RGB.
SIM = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritanopia": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}

DELTA_E_HARD = 12.0  # below this, two colours are hard to distinguish
DELTA_E_SAME = 6.0   # below this, effectively identical


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def simulate(rgb, kind: str) -> np.ndarray:
    """Simulate CVD on an (..., 3) uint8/array of sRGB colours. `kind` in SIM, or
    "normal" for a pass-through."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if kind == "normal":
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    lin = _srgb_to_linear(rgb)
    sim = lin @ SIM[kind].T
    return np.clip(np.round(_linear_to_srgb(sim)), 0, 255).astype(np.uint8)


# --- CIELAB (D65) for Delta-E ---------------------------------------------

_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _to_lab(rgb):
    xyz = _srgb_to_linear(rgb) @ _M_RGB2XYZ.T / _WHITE

    def f(t):
        d = 6 / 29
        return np.where(t > d**3, np.cbrt(t), t / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def delta_e(a, b) -> float:
    return float(np.linalg.norm(_to_lab(a) - _to_lab(b)))


# --- the palettes under test --------------------------------------------------


CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]


def _palettes() -> dict[str, dict[str, list[int]]]:
    return {
        "class (semantickitti)": {n: class_to_color(i) for i, n in enumerate(CLASS_NAMES)}
        | {"unknown": class_to_color(-1)},
        "class (groups)": {n: list(GROUP_RGB[i]) for i, n in enumerate(GROUP_NAMES)},
        "motion": {"static": [90, 90, 90], "moving": list(GHOST_RGB)},
        "ground": {"ground": [170, 130, 90], "non-ground": [70, 130, 180]},
        "intensity/reflectivity": {f"v={v}": [v, v, v] for v in (0, 64, 128, 192, 255)},
    }


def ghost_vs_class_min_delta_e() -> tuple[float, str]:
    """Min Delta-E of the ghost highlight against every class colour + the
    motion-static grey, across all four simulations. Returns (min, closest)."""
    targets = {n: class_to_color(i) for i, n in enumerate(CLASS_NAMES)}
    targets["unknown"] = class_to_color(-1)
    targets["static-grey"] = [90, 90, 90]
    best = (1e9, "")
    for name, rgb in targets.items():
        d = min(delta_e(simulate(GHOST_RGB, k), simulate(rgb, k)) for k in ["normal", *SIM])
        if d < best[0]:
            best = (d, name)
    return best


def min_delta_e(palette: dict[str, list[int]]) -> tuple[float, tuple[str, str, str]]:
    """Smallest CIELAB Delta-E between any two colours of `palette`, taken over
    normal vision and all three CVD simulations. Returns (min, (kind, a, b))."""
    names = list(palette)
    best = (1e9, ("", "", ""))
    for kind in ["normal", *SIM]:
        sim = {n: simulate(palette[n], kind) for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                de = delta_e(sim[names[i]], sim[names[j]])
                if de < best[0]:
                    best = (de, (kind, names[i], names[j]))
    return best


def report() -> str:
    out = []
    for pname, pal in _palettes().items():
        out.append(f"\n=== {pname} ({len(pal)} colours) ===")
        names = list(pal)
        worst_any = 1e9
        collisions = []
        for kind in ["normal", *SIM]:
            sim = {n: simulate(pal[n], kind) for n in names}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    de = delta_e(sim[names[i]], sim[names[j]])
                    worst_any = min(worst_any, de)
                    if de < DELTA_E_HARD:
                        collisions.append((kind, names[i], names[j], de))
        if not collisions:
            out.append(f"  OK -- min pairwise Delta-E across all sims = {worst_any:.1f}")
            if pname == "motion":
                de, closest = ghost_vs_class_min_delta_e()
                out.append(f"  ghost highlight vs class map + static grey: "
                           f"min Delta-E {de:.1f} (closest: {closest})")
        else:
            out.append(f"  {len(collisions)} hard-to-distinguish pair(s) (Delta-E < {DELTA_E_HARD}):")
            for kind, a, b, de in sorted(collisions, key=lambda x: x[3]):
                tag = "  SAME" if de < DELTA_E_SAME else ""
                out.append(f"    [{kind:12}] {a:16} vs {b:16}  Delta-E {de:4.1f}{tag}")
    return "\n".join(out)


def save_png(path: str):
    import matplotlib.pyplot as plt

    pals = _palettes()
    kinds = ["normal", *SIM]
    fig, axes = plt.subplots(len(pals), len(kinds), figsize=(3.4 * len(kinds), 2.6 * len(pals)))
    for r, (pname, pal) in enumerate(pals.items()):
        names, cols = list(pal), np.array(list(pal.values()))
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            sim = simulate(cols, kind) / 255.0
            ax.imshow(sim[None, :, :], aspect="auto")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(pname, fontsize=9)
            if r == 0:
                ax.set_title(kind, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="vrgrid.dash.cvd")
    p.add_argument("--png", default=None, help="write a swatch-grid PNG here")
    args = p.parse_args(argv)
    print(report())
    if args.png:
        save_png(args.png)


if __name__ == "__main__":
    main()
