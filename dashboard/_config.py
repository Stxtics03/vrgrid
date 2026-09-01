"""Config- and schedule-derived values for the dashboard. [JP]

Imports no rerun. These read the frozen config files (`configs/thresholds.yaml`,
`configs/schedule_*.yaml`) via `vrgrid.grid.schedule`, and `tests/test_dashboard.py`
exercises them in CI where the optional `[dash]` extra (rerun-sdk) is not
installed. `pipeline_view.py` imports from here; `palettes.py` stays
colours-only, so the split is: colours in `palettes`, config/schedule wiring
here, rerun logging in `pipeline_view`.

Nothing in this project may hardcode a threshold or a ring size inline
(CLAUDE.md, "Don't"); this module is where the dashboard reads them instead.
"""

from vrgrid.grid.schedule import CONFIG_DIR, load_thresholds
from vrgrid.grid.schedule import load as _load_schedule


def blind_cone_radius_m() -> float:
    """Blind-cone radius from `configs/thresholds.yaml` `sensor.blind_cone_m`.

    Math §1.4 eq (5): r_blind = h_s / tan|phi_min| = 1.73 / tan(24.8 deg) =
    3.74 m. The earlier plan assumed 1-2 m; master v4 corrected it. Read from
    the frozen config through the one cached reader (`load_thresholds`), never
    hardcoded -- a second literal is how the file and the running system drift.
    """
    return float(load_thresholds()["sensor"]["blind_cone_m"])


def available_schedules() -> list[str]:
    """Schedule names discovered from `configs/schedule_*.yaml` -- no hardcoding.

    Returns the bare names (`5_10_20_40`, `5_10_50`) as `grid.schedule.load`
    accepts them.
    """
    return sorted(
        p.stem.removeprefix("schedule_") for p in CONFIG_DIR.glob("schedule_*.yaml")
    )


def schedule_legend_markdown(active: str) -> str:
    """Markdown table of every available schedule and its ring boundaries, read
    from the config files, with the active one marked.

    This is the (display-only) schedule selector: switching schedules mid-run is
    not wired -- `--schedule <name>` on `vrgrid.dash` / `vrgrid.run` picks one at
    startup. Ring half-widths and cell sizes come straight from the yaml via
    `grid.schedule.load`, so this cannot drift from what the engine uses.
    """
    lines = [
        "**Schedule selector** (display only -- `--schedule <name>` picks it at startup)",
        "",
        "| schedule | rings: half-width m / cell cm | cells | MB |",
        "|---|---|---|---|",
    ]
    for name in available_schedules():
        s = _load_schedule(name)
        rings = ", ".join(f"{r.half_width_m:g}/{r.cell_m * 100:g}" for r in s.rings)
        mark = " **(active)**" if name == active else ""
        lines.append(
            f"| `{name}`{mark} | {rings} | {s.total_cells:,} | {s.total_cells * 12 / 1e6:.2f} |"
        )
    return "\n".join(lines)
