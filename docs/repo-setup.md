# Repo Setup, Branching, and Who Starts When

---

## Part 1 — What to give Claude Code

### The mistake to avoid

Do not paste the four planning docs into `CLAUDE.md`, and **do not `@import` them either.** `@path` imports load at launch, so splitting a file across imports does not reduce context — it just spreads the same cost across more files. Those four docs are roughly 30,000 words. Loaded every session, they crowd out the code Claude actually needs to read and dilute the rules that matter.

`CLAUDE.md` is working memory, not documentation. Keep it under ~200 lines.

### The structure that works

```
vrgrid/
├── CLAUDE.md              ← lean, invariants only (drafted, drop it in)
├── docs/
│   ├── sih-math.md        ← read on demand
│   ├── master-v4.md
│   ├── execution-plan.md
│   ├── team-assignments.md
│   ├── research-modules.md
│   ├── research-log.md    ← Srinivas/Hriday/Pratyushi append here
│   └── frames.md          ← every coordinate transform, written Day 0
```

Everything in `docs/` lives in the repo but loads only when asked. In a session: *"read docs/sih-math.md §5 and implement split() with the derived bit"* — Claude pulls just that section.

The `CLAUDE.md` I drafted contains only things that cannot be derived from the codebase: the invariants that are silently violable. Float atomics compile fine and produce a map that looks right. Inverse-variance merge compiles fine and produces a map that is confidently wrong at kerbs. Those are exactly what belongs in persistent memory — a rule earns its place by being non-obvious and expensive to get wrong.

### Nested CLAUDE.md per directory

Add small, local files where the rules are specific to a subtree:

- `src/gpu/CLAUDE.md` — fixed-point conventions, SoA layout, no allocation in loop, which kernels exist
- `src/grid/CLAUDE.md` — pointer to math §2, §4, §5; the frozen cell struct; ring migration rules
- `src/perception/CLAUDE.md` — frame conventions, FRNet config path, dataset cache format

Keep each under 30 lines. They load when Claude works in that subtree, so Shrestha's kernel rules don't consume context while JP works on the loader.

### Practical notes

- Launch Claude Code from the **repo root**, not a subdirectory, so the project `CLAUDE.md` loads.
- Run `/memory` to see what's actually loaded if behaviour seems off.
- Each dev should work in their own clone or `git worktree`. Three people driving Claude Code in one working tree will produce conflicting edits with no merge step.
- Claude Code also writes its own auto-memory notes across sessions. Leave it alone; it's separate from your `CLAUDE.md`.
- Update `CLAUDE.md` when you correct Claude twice about the same fact. Not after one bad session — that's how these files fill up with exceptions.

Docs: https://code.claude.com/docs/en/memory

---

## Part 2 — Repo layout as conflict prevention

**The core idea: structure the repo so that branching barely matters.** With three devs and eight days, integration failure is the most likely way this project dies. The cheapest insurance is not a clever branching model — it's directory ownership, so two people rarely touch the same file.

```
vrgrid/
├── CLAUDE.md
├── Makefile
├── configs/
│   ├── schedule_5_10_20_40.yaml      default
│   ├── schedule_5_10_50.yaml         ablation
│   └── thresholds.yaml               FROZEN before schedule comparison
├── include/vrgrid/                   ⚠ FROZEN Day 0 — whole-team change only
│   ├── cell.py                       12-byte struct / SoA dtype
│   └── api.py                        scatter, fuse, split, merge, query
├── src/
│   ├── grid/          [Aakash]   lattice, rings, migration, split/merge, fusion, pool
│   ├── eval/          [Aakash]   reference map, metrics, plan regret
│   ├── gpu/           [Shrestha] kernels, allocators, timing, 3D baseline stub
│   └── perception/    [JP]       loader, transforms, range image, semantic+motion labels (GT), ground, reflectivity
├── dashboard/         [JP]
├── tests/                            one file per module, name matches source
├── scripts/                          every number on a slide comes from a script here
└── docs/
```

**`include/vrgrid/` is the only genuinely shared surface.** Freeze it in the first two hours of Day 0 and treat it as immutable. Any change requires all three devs to agree, in the same room. This one rule prevents most of the integration risk.

**`scripts/` matters more than it looks.** Gate 6 says every number on a slide must be reproducible. If a figure isn't produced by something in `scripts/`, it doesn't go on a slide.

---

## Part 3 — Branching

**Trunk-based, short-lived branches. Nothing lives longer than 24 hours.**

Long-lived per-person feature branches are the classic way an 8-day project fails: three people work happily in isolation for five days, integrate on Day 6, discover the interfaces drifted, and have no recovery time. You do not have the schedule to absorb that.

| Situation | What to do |
|---|---|
| Editing only files in your own directory | **Commit straight to `main`.** Ceremony costs more than it saves here. |
| Touching `include/vrgrid/` | Branch + all three devs review, same day. Rare by design. |
| Touching someone else's directory | Branch `<name>/<thing>`, PR, merge same day. |
| Risky experiment | Branch `<name>/spike-<thing>`, and either merge or **delete** it within 24 h. |

Rules:
- `main` is always green. If CI is red, fixing it outranks whatever you were doing.
- **Merge to `main` at least once per day**, before the 20:00 gate review. A day's work sitting unmerged is a day of integration risk.
- Never merge a red branch "to fix later."
- Tag `day3-ghosts`, `day4-regret`, `day6-freeze`. If Day 7 goes wrong you can ship a tag.
- After the Day 6 freeze: bug-fix commits only, each referencing the bug.

**CI must run:** unit tests, the determinism test (same input twice → identical hash), and the partition test. Those three catch the failures that are expensive to find late. Everything else can be manual.

---

## Part 4 — Start order for Day 0

There is a real dependency chain in the first few hours. Getting it wrong costs half a day.

**Hour 0 — before the meeting**

**JP starts the SemanticKITTI download** (sequences 00, 07, 08 only — ~40 GB, not 200 GB). This is the single item on the critical path that neither cleverness nor effort can accelerate. Start it, then walk into the meeting.

**Hours 0–2 — all six, together, no code**

Freeze, in this order:
1. **Cell struct** — 12 bytes, master v4 §3.3. Aakash writes it, Shrestha confirms the SoA layout works for his kernels, everyone signs off.
2. **The five function signatures** + `CellQuery`.
3. Vertical extent (−2 to +6 m), ring schedule (default + ablation), GPU name, 1 cm quantisation.
4. `docs/frames.md` — every coordinate transform written down. Yes, before any code. This is the bug that costs three days if you defer it.
5. Repo scaffolded, CI running one stub test, everyone can push.

**Hour 3 — Aakash commits stubs of all five functions** that compile and return dummy data. From this moment the other two build against real signatures, not imagined ones. This is the single highest-leverage thirty minutes of the whole project.

**Hours 3+ — parallel, in this order of urgency**

| Who | First task | Why first |
|---|---|---|
| **Aakash** | Lattice + indexing + partition test | Everything downstream indexes through it |
| **Shrestha** | SoA allocator + timing harness against the frozen struct | Aakash's `scatter()` needs somewhere to write |
| **JP** | Static-wall test, then dashboard against a mock grid | Frame bugs found now cost minutes; found Day 4 they cost days |
| **Hriday** | ⚑ **Verify `moving-*` IDs 250–259 in raw `.label` files** | The no-retraining plan depends on it. 4-hour deadline. |
| **Srinivas** | Triebel 2006, Droeschel 2014 | Memo tonight; may reword our novelty claim |
| **Pratyushi** | Psomiadis 2024 | Starts the Day-3 novelty verdict |

**The two who genuinely block others:** Aakash (stubs, hour 3) and Hriday (label verdict, hour 4). If either slips, say so in the room immediately — both have clean fallbacks, but only if the fallback starts the same day.

**Shrestha's dependency shape is worth naming.** His Day 0–1 work blocks both other devs; from Day 2 onward it reverses and he's optimising what they built. So his early items are non-negotiable and his later ones are flexible — which is also why he's the right person to pull onto split/merge if Aakash is behind at the Day 2 gate.

---

## Part 5 — Day 0 checklist

```
[ ] Download started (JP) — before anything else
[ ] Cell struct frozen and committed
[ ] Five signatures + CellQuery frozen
[ ] docs/frames.md written
[ ] Repo scaffolded, CI green on a stub test, all six can push
[ ] CLAUDE.md at root; docs/ populated with all planning files
[ ] Nested CLAUDE.md in src/gpu/, src/grid/, src/perception/
[ ] Stub implementations committed (Aakash, hour 3)
[ ] Partition test passing on 10⁶ points
[ ] Dashboard renders a mock map
[ ] Hriday's label verdict delivered
[ ] Each dev on their own clone or worktree
```

**GATE 0 passes when a mock map renders and the partition test is green.** Not before.
