# vrgrid — Adaptive Variable-Resolution 2.5D LiDAR Mapping

**SIH26053.** A drop-in replacement for a uniform 2.5D occupancy grid. Cell size
adapts to distance, semantics and direction of travel, under a memory bound
fixed at startup: **8.94 MB**, ~21.5× less than a uniform 5 cm 2.5D grid and
~286× less than a dense 5 cm 3D voxel grid, at the same near-field accuracy.
It removes dynamic ghosts. And it proves the compression is free by showing it
does not change the plan a robot would make.

That last clause is the contribution. Everything before it is engineering.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate   # Debian/Kali: required, PEP 668
pip install -e ".[dev]"            # or: make setup
make test                          # all unit tests
python -m vrgrid.run --seq 08 --schedule 5/10/20/40
python -m vrgrid.dash              # Rerun dashboard, separate process
python scripts/sampling_table.py   # the numbers behind the ring schedule
```

The dataset is **not** in this repo. SemanticKITTI sequences 00, 07, 08 only
(~40 GB) into `data/`, which is gitignored.

## Who owns what

Six people, three code owners. Directory ownership is the integration defence:
structured so two people rarely touch the same file.

| Directory | Owner | Contents |
| --- | --- | --- |
| `include/vrgrid/` | ⚠️ **all three** | FROZEN interfaces — cell struct, the five signatures |
| `src/grid/` | Aakash | lattice, rings, split/merge, fusion, refinement pool |
| `src/eval/` | Aakash | reference map, metrics, plan regret |
| `src/gpu/` | Shrestha | kernels, allocators, timing |
| `src/perception/` | JP | loader, transforms, range image, semantic + motion labels (GT), ground |
| `dashboard/` | JP | Rerun app |
| `configs/` | all three | schedules + thresholds, frozen before the ablation |
| `docs/research-log.md` | Srinivas, Hriday, Pratyushi | findings, append-only |

Research is three modules that feed the code, not a separate track:
**α** representation and prior art (Srinivas → Aakash), **β** dynamics and
segmentation (Hriday → JP), **γ** traversability and evaluation (Pratyushi →
the evaluation side). Every paper read produces a line in `docs/research-log.md`;
if it produced no line, it should not have been read.

**Stay inside your own directory.** Cross-directory changes go through a
same-day PR. `include/vrgrid/` changes need all three devs in the same room.

## Branching

Trunk-based, short-lived branches. Nothing lives longer than 24 hours.

| Situation | What to do |
| --- | --- |
| Only your own directory | Commit straight to `main` |
| Touching `include/vrgrid/` | Branch + all three review, same day. Rare by design |
| Someone else's directory | Branch `<name>/<thing>`, PR, merge same day |
| Risky experiment | Branch `<name>/spike-<thing>`, merge or **delete** within 24 h |

`main` is always green — if CI is red, fixing it outranks whatever you were
doing. Merge to `main` at least once a day, before the 20:00 gate review. Tag
`day3-ghosts`, `day4-regret`, `day6-freeze`. After the Day 6 freeze, bug fixes
only. Each dev works in their own clone or `git worktree` — three people
driving Claude Code in one working tree will produce conflicting edits.

## CI

Unit tests, plus the two that catch what is expensive to find late:
**determinism** (same input twice → identical map hash) and **partition**
(10⁶ random points, exactly one cell per ring). Both are blocking.

Theorem tests are proofs, not tuning targets. If one fails, the implementation
is wrong — do not weaken the test.

## Docs

`CLAUDE.md` holds the invariants. Everything else loads on demand:

| File | Contents |
| --- | --- |
| [`docs/sih-math.md`](docs/sih-math.md) | Every formula, theorem and unit test. Cited by § from code |
| [`docs/master-v4.md`](docs/master-v4.md) | Architecture and scope decisions |
| [`docs/execution-plan.md`](docs/execution-plan.md) | Day-by-day plan and gates |
| [`docs/team-assignments.md`](docs/team-assignments.md) | Named ownership |
| [`docs/research-modules.md`](docs/research-modules.md) | Reading assignments |
| [`docs/repo-setup.md`](docs/repo-setup.md) | This layout, branching, Day-0 start order |
| [`docs/frames.md`](docs/frames.md) | Every coordinate transform — **write it Day 0** |
| [`docs/research-log.md`](docs/research-log.md) | Append-only findings |

## Honest novelty statement

Foveated grids, elevation maps and dynamic-point removal are all published.
What is contributed here is (a) a resolution schedule driven by **both range
and semantics** under a **hard preallocated memory bound**, (b)
**uncertainty-honest split/merge** with a provable round-trip property, and
(c) a **plan-sensitivity evaluation** — coarsening measured in units of
planner regret rather than reconstruction error.

Cite Triebel 2006 (multi-level surface maps) and Droeschel 2014 (nested
ego-centric multi-resolution grids) next to the ring diagram. A panel that
sees you cite the thing you resemble concludes you know the field.
