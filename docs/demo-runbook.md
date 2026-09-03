# Demo runbook — how to start it, and how to show it

*Companion to `docs/demo-safe-ranges.md` (which frames hold what) and
`docs/defense-rehearsal-playbook.md` (what to say when they push back). This
file is the operational one: the commands, the order, and the failure modes.*

Everything here was run end-to-end on this machine on 2026-09-03. One launcher
drives all of it:

```bash
./scripts/demo.sh check     # preflight — do this first, every time
./scripts/demo.sh bake      # pre-render every scene (~3 min, ~1 GB) — do it BEFORE you present
./scripts/demo.sh foveation # play a scene
./scripts/demo.sh numbers   # the tables, in the terminal
```

---

## 0. The one thing that will break the demo

The loader needs the directory holding `poses/` and `sequences/`. In this clone
that is **`data/dataset`**, not `data/`. Point it one level up and every scene
dies with `FileNotFoundError: GT poses not found: data/poses/00.txt` — the
default `VRGRID_DATA_ROOT` is wrong for this checkout.

`scripts/demo.sh` resolves it for you and never lets you type it. If you run a
raw command instead, export it yourself:

```bash
export VRGRID_DATA_ROOT=$PWD/data/dataset
```

---

## 1. Which way to show it — baked recordings, with one live run

**Show pre-baked `.rrd` recordings. Keep exactly one live run, at the start, to
prove they are real.**

The recordings are not a video and not a mock: `./scripts/demo.sh bake` runs the
identical pipeline the live command runs — real SemanticKITTI scans, real
Patchwork++ ground segmentation, the real `MapEngine` — and writes what it drew
to a Rerun file. Playing one back gives you every entity, every toggle and the
whole timeline, instantly and scrubbable.

Why not present live:

- A live scene computes for 13–35 s before the picture is complete, with a
  Patchwork++ init banner in front of it. That is a long silence on stage.
- Live means you cannot scrub. If a panelist says *"go back to when the
  motorcyclist passed"*, a baked recording drags a slider; a live run reruns.
- Every live scene is a fresh chance for the environment to be wrong.

And why keep one live run anyway: a baked file invites *"so it's a video."*
`./scripts/demo.sh check` answers that in about ten seconds — it runs the real
pipeline on two real frames in front of them and prints live counters
(`3,099 cells cleared, 11,056 spared by the current-return guard`). Open with
it. Then everything after it is credibly the same system.

If the machine is not yours, or the projector is HDMI-hostile: bake at home,
copy `demo/*.rrd`, and play them with `rerun demo/<scene>.rrd`. The Rerun viewer
is the only thing the playback needs — no venv, no dataset, no 90 GB.

---

## 2. The arc — six minutes

The contribution is the composition under a hard bound, not any one picture.
Order the scenes so each one answers the question the last one raises.

### ① The claim, in the terminal — 30 s

```bash
./scripts/demo.sh check      # live proof, on real frames
./scripts/demo.sh numbers    # 8.94 MB vs 192 MB vs 2.56 GB
```

> "The map is 8.94 MB, and that is fixed at startup — every array is allocated
> before the first scan and nothing in the frame loop grows. A uniform 5 cm
> 2.5D grid over the same footprint is 192 MB; a dense 5 cm 3D voxel grid is
> 2.56 GB."

Volunteer the sparse-3D row (~130–240 MB, ~15–27×) yourself. Leading with 286×
alone reads as cherry-picking; giving them the unflattering baseline reads as
good faith — and it is the number they were about to ask for.

### ② Foveation — where the memory went — 90 s

```bash
./scripts/demo.sh foveation
```

Ring boundary circles and the red blind cone track the vehicle; the occupied
surface fills in as it drives; cell size steps 5 → 10 → 20 → 40 cm outward.

> "Resolution is not uniform and it is not arbitrary. The rings are where the
> sensor's returns actually are: at 50 m consecutive laser rings hit the road
> 10.8 m apart, so a 5 cm cell out there is storing an interpolation, not a
> measurement."

Point at the **red circle**: the 3.74 m blind cone. Say it out loud —
`unknown`, never `free`. Three occupancy states, and `world/map/free` and
`world/map/unknown` are separate entities on purpose. A panel that hears you
distinguish those two knows you have thought about what the robot does not know.

### ③ Ghosts — the toggle — 2 min, this is the moment

```bash
./scripts/demo.sh ghosts-off   # trails stay in the map
./scripts/demo.sh ghosts-on    # same 60 frames, trails gone
```

Two windows side by side beats flipping between them. Same sequence, same
frames, same schedule — the only difference is whether §10.4 runs.

Inside one recording you can also toggle at the point-cloud level: the eye icon
on `world/ghosts` in the entity panel shows and hides the moving returns.

> "A moving car writes occupancy into every cell it passes through. Visibility
> cleanup removes what the current scan can see through — but it never clears a
> cell that has a return in the current scan, which is what stops it eating
> fences, poles and sign posts."

The measured number: **13.5 % of the trail removed, 4.96 M cells cleared** on
sequence 08 (`scripts/ghost_removal_figure.py --seq 08`, figure already in
`docs/figures/ghost_removal.png`), and **429,012 cells spared by the
current-return guard** in the 60-frame seq-00 run above. Quote the guard number
too — it is the evidence the cleanup is conservative rather than aggressive.

⚠️ **Do not put the `--show-ghosts` terminal output on screen.** It prints
`0 occupied cells, 0 cleared, 0 protected`, because the occupied counter is
only computed inside the ghost-removal branch. The map is fine and the picture
is correct — the counter simply is not filled on that path. Nothing is wrong,
but you do not want to explain that live.

### ④ Against a uniform grid — 60 s

```bash
./scripts/demo.sh dense3d
```

Our grid beside a uniform 5 cm dense voxel grid, same frames, box counts in the
overlay.

Measured on the baked run: **1,544 variable-resolution boxes against 290,448
dense voxels — 188.1x — at a 20 m footprint**.

Say the disclaimer before they read it: this render uses a **reduced 20 m
footprint** because that is what a dev machine can allocate. The ratio you see
here is a local illustration; the 286× is the full-grid byte ratio from
`memory_table.py`. The script's own docstring says so — being the one who says
it first costs nothing and buys everything.

### ⑤ Optional, if they are engaged and time is there

```bash
./scripts/demo.sh features      # curbs, potholes, per-cell confidence
./scripts/demo.sh traffic       # seq 07, dense moving traffic
./scripts/demo.sh reflectivity  # seq 00, lane paint
```

`features` is new as of 3 September — it draws the curb/pothole layer (§7.4) and
per-cell confidence (§7.5), which until now existed in the code and rendered
nowhere. It is the answer to *"what does the map actually know?"*. Two guards
when you show it: confidence is **a margin, not a probability**
(`known-limitations` §4), and the curb/pothole counts are a demonstration with
no ground truth to score against — see below.

---

## 3. What you must not claim

The gap between what the README says and what the evidence supports is where
this demo can be lost. From `docs/handover-2026-09-02.md`:

**The plan-regret result is not proven — do not present it as the headline.**
The README's framing ("proves the compression is free by showing it does not
change the plan") is ahead of the data.

⚠️ **Regenerate these numbers before you present.** The table below is read from
a `docs/figures/regret.csv` written 2026-09-02 18:46 — *nine hours before* the
§9.2 eval merge landed on `main`. That merge changed `reference_map`, `metrics`
and `plan_regret`, so the figures may have moved. Rebuild with
`.venv/bin/python scripts/regret_plot.py --seq 08` and check the rows against
what is printed here; the shape of the argument below holds either way, but the
digits are Aakash's lane and his to confirm.

| schedule | MB | regret |
|---|---|---|
| 5/10/20/40 | 29.06 | 0.6038 |
| 5/10/50 | 23.62 | 0.6038 |
| uniform 10 cm | 78.50 | **0.5553** |
| uniform 20 cm | 30.50 | 0.6455 |
| uniform 40 cm | 18.50 | 0.7613 |

*(as of 2026-09-02, pre-§9.2-merge — see the warning above)*

What is defensible: *our two schedules produce the same plan despite a 5.4 MB
difference, and we beat a uniform 20 cm map of comparable size.* What is not:
any claim of a knee, or that regret is flat under compression — a uniform 10 cm
map at 2.7× the memory still plans measurably better, and on the longitudinal
query the frozen schedules lose to uniform outright. `regret_plot.py` refuses
to draw a monotone story over these rows, by design. **If asked, say the
evaluation is comparable at a fixed window and not across windows, and that
§9.2's per-ring comparison is open work.**

**Potholes are a demonstration, not a rate.** 56–551 cells per sequence, 10×
spread, no pattern. SemanticKITTI has no curb or pothole ground truth, so there
is nothing to score against. Say that before you are asked — playbook Q5.

**Ring 0 has no coarsening ratio ρ on any sequence — and know the *current*
reason, because it was re-diagnosed on 3 September and the older explanation is
false.** It is not that ring-0 footprints hold a single reference *return*.
`block_stats` counts observed **cells**, and a ring-0 footprint is `k = 1` —
exactly one cell — so `n_ref` can never exceed 1 and the `n_ref > 1` guard in
`coarsening_ratio_per_ring` drops ring 0 **by arithmetic, on every sequence**.
The data is there: M\* holds more than one return in **92.8 / 97.0 / 97.4 %** of
the ring-0 cells §9.2 scores on 00 / 07 / 08.

Closing it would score ring 0 at **ρ 1.01–1.24, the best of any ring** — so this
is a fixable gap with a measured cost, deliberately deferred to Day 7, not a
permanent limitation. Say it that way. It is deferred because `block_stats` is
shared: the fix moves ring 1 by −21.5 % on 07, invalidates every cached M\*
`.npz`, and forces the eleven-sequence table and the "ρ median 1.45" headline to
be regenerated. Full diagnosis in `docs/known-limitations.md` §2b.

**Semantics come from the `.label` files, not from a network.** Nothing is
retrained and no inference runs for labels. FRNet is reported alongside the map
(90.3 % point accuracy, 69.8 % mIoU) and never swapped into it, so the mapping
contribution is evaluated independently of segmentation quality. Disclose that
plainly — it is a strength, not an admission.

---

## 4. Failure modes, and what to do

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: GT poses not found: data/poses/00.txt` | `VRGRID_DATA_ROOT` one level too high | `export VRGRID_DATA_ROOT=$PWD/data/dataset`, or just use `demo.sh` |
| Viewer never appears | no display / remote session | bake instead and play the `.rrd`; `DISPLAY` is `:0.0` on this box |
| `PatchWorkpp` banner then a long pause | normal — Patchwork++ init | it is ~4 s for 20 frames; baked scenes skip it |
| `[!] ground: SEMANTIC-CLASS FALLBACK` printed | `pypatchworkpp` not installed | as of 3 Sep the fallback is **loud** rather than silent. The proxy **includes terrain and admits embankments** — do not demo the ground layer on it. Rebuild from the git clone, not PyPI (`pip install ./patchwork-plusplus/python`) — handover JP item 3 |
| Ghost scene shows nothing moving | wrong frames | seq 00 needs frames ~0–60; single best frame is 10 (motorcyclist + pedestrian) |
| `0 occupied cells` printed | `--show-ghosts` counter artifact | expected, harmless, keep it off screen (see ③) |
| Scene is slow live | 160 frames ≈ 35 s | play the baked file |

---

## 5. Before you leave for the venue

```bash
./scripts/demo.sh check          # must end "OK -- the pipeline runs on real data."
./scripts/demo.sh bake           # ~3 min, writes demo/*.rrd (1.2 GB, gitignored)
ls -lh demo/                     # six recordings
make test                        # main must be green
```

Copy `demo/` onto the presenting machine along with the Rerun viewer. Nothing
else is required to play the scenes back — not the venv, not the 84.8 GB.
