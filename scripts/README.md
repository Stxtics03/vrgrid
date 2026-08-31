# scripts/

**Gate 6: every number on a slide comes from a script in here.** If a figure
is not produced by something in this directory, it does not go on a slide.

| Script | Produces |
| --- | --- |
| `memory_table.py` | The memory comparison table — reads `CELL_BYTES` from the frozen struct, so the report cannot drift from the code |
| `sampling_table.py` | `s_az` / `s_rad` tables, single-frame fill rates, blind cone, pothole range limit, pedestrian crossover |
| `bench_scatter.py` | Scatter p50/p99 and scratch cost for both paths, and the hash proving they agree |
| `bench_pyramid.py` | Conservative-pyramid rebuild p50/p99, node and scratch cost, and the SAFE/BLOCKED/MIXED split per level. Also prints the corrected §7.2 memory figure |
| `memory_bound.py` | The itemised preallocated bound, read off the real allocation -- with and without the pyramid |
| `timing_table.py` | The Day-6 per-stage latency table: p50/p99/max and headroom at p99, machine named, `--alloc` for transient bytes per frame. Drives the real lattice, scatter, fuse, cleanup, pyramid and shift off a synthetic sweep. Covers the mapping back end only, and prints the unmeasured stages with their owner rather than dropping them |
| `regret_plot.py` | The Day-4 headline figure -- memory against plan regret (§8.2) -- as `regret.csv` plus `regret.svg`/`.png`. Runs `eval_synthetic`'s sweep by importing it, and refuses to draw a monotone story over non-monotone rows: it prints which step violates the claim |
| `baseline_demo.py` | Our map beside the uniform 2.5D and dense 3D baselines, all three counters resident and ticking. `--dense` allocates the full 2.56 GB |

To add: `ablation_table.py` (schedule comparison, thresholds frozen first),
`ghost_removal_figure.py`.

Not a numbers script, but it lives here because it is run once and checked in:

| Script | Does |
| --- | --- |
| `create-research-issues.sh` | Creates the labels, milestones and 18 research issues from `docs/github-issues.md`. Needs `gh auth login`. Run once, from the repo root. |
