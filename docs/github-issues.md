# GitHub Issues — Research Team

*18 issues for Srinivas (R1), Hriday (R2), Pratyushi (R3). Copy-paste ready, or run `scripts/create-research-issues.sh`.*

**The rule behind every issue here:** it closes on a decision or an artifact, never on "read the papers." A research issue that closes with a bibliography has failed. If an issue's "Done when" doesn't name something a developer can act on or something that goes in the report, it shouldn't exist.

---

## Setup

**Labels**
```
research            r1-representation    P0-blocking
writing             r2-dynamics          P1
gate                r3-evaluation        P2
```

**Milestones**
```
Day 0  — 28 Aug     Day 3 — 31 Aug (gates close)
Day 1–2 — 29–30 Aug Day 5 — 2 Sep (research stops)
                    Day 6 — 3 Sep (writing done)
```

**Two issues gate the whole project** — #1 and #6. Both are P0. If either comes back negative, the whole team hears it the same hour, not in a memo.

---

# P0 — Blocking

## #1 · Verify `moving-*` labels exist in raw SemanticKITTI `.label` files
**Hriday · `r2-dynamics` `P0-blocking` `gate` · Day 0, 4-hour deadline**

Our entire no-retraining plan rests on this. Master v4 §3.6 claims `moving-*` IDs 250–259 are present in the raw `.label` files and the 19-class collapse happens only in the `learning_map` config. If that's true we skip several days of GPU training. If it's false, JP falls back to residual-image MOS **the same day**.

**Do:** download one sequence's labels, load the `.label` file directly, print `np.unique()` of the raw values. Cross-check against the SemanticKITTI paper (Behley, ICCV 2019) and the `semantic-kitti.yaml` config.

**Done when:** a one-line verdict is posted to the team channel with the actual unique-value output pasted in. Not "I think so" — the array.

**Blocks:** JP's Day 1–2 motion pipeline, and decision C in the execution plan.

---

## #6 · Novelty verdict — is plan-regret evaluation already published?
**Pratyushi · `r3-evaluation` `P0-blocking` `gate` · Day 3, hard deadline**

This gates our headline slide. We claim nobody measures map coarsening in units of *planner regret*. The closest prior art is Psomiadis, Maity & Tsiotras (ICRA 2024, arXiv:2309.13451), which selects map compression guided by the robot's path — but on an **information-theoretic** objective, not plan cost.

**Do:** read Psomiadis 2024, the iterative version (arXiv:2503.10843), and Larsson 2021 (RA-L 6(4):7651). Search "task-aware perception," "task-driven map compression," "perception-planning co-design," "planner-aware resolution."

**Done when** you can state in three sentences exactly how our metric differs from theirs — or you escalate that it doesn't. If you can't articulate the difference cleanly, the claim is in trouble and we need to know on Day 3, not on the 5th.

**Fallback if it's published:** reframe as validation in a new domain. Keep the curve — it's still the best figure in the deck.

---

# P1 — Feeds a developer directly

## #2 · Prior art we must cite — the cite-or-get-caught list
**Srinivas · `r1-representation` `P1` · Day 2**

Three known landmines. A panel member who recognises our ring diagram as a 2014 paper we didn't cite will discount everything else we say.

- **Triebel, Pfaff & Burgard, IROS 2006 (MLS maps)** — our ground+ceiling cell is a two-layer MLS map. Twenty years old.
- **Droeschel, Stückler & Behnke, ICRA 2014 + JFR 33(4) 2016** — our foveated ring grid, with interlaced ring buffers, in 2014.
- **Losasso & Hoppe, SIGGRAPH 2004 (geometry clipmaps)** — the graphics origin of nested toroidal LOD.

**Done when:** a memo lists each with one sentence on what they did and one on what we do differently, *and* master v4's novelty wording is edited where it overclaims.

**Bonus:** extract Droeschel's ring-buffer implementation approach for Aakash — we may be able to reuse the structure.

---

## #3 · Metric definitions as pseudocode
**Pratyushi · `r3-evaluation` `P1` · Day 1**

Aakash implements the metrics; he shouldn't have to read four papers to do it.

**Done when** `docs/metric-definitions.md` gives implementable pseudocode for: plan regret `R(S)` (§8.1, both paths scored on the reference map), discrete Fréchet distance, `ρ = IL/spread` (§9.3), and DR/SP/F (§9.4).

**Blocks:** Aakash's Day 4 plan-regret study.

---

## #4 · FRNet setup memo
**Hriday · `r2-dynamics` `P1` · Day 1**

**Done when** JP has: exact config file name, checkpoint URL, confirmation the checkpoints are 19-class, the class-index mapping, MMDetection3D version constraints, and any known gotchas. Also check whether Fast-FRNet (~7.5M params) is worth having in reserve.

Verify while you're there: Apache 2.0 licence, and the 73.3% SemanticKITTI mIoU figure.

**Blocks:** JP's Day 1 inference task.

---

## #5 · Confirm the measurement-variance model
**Srinivas · `r1-representation` `P1` · Day 1**

Math §3.2 derives `σ²_z = [sin²φ σ²_r + r² cos²φ σ²_φ] / cos²θ_inc` from Fankhauser et al. (CLAWAR 2014).

**Done when** you've checked our derivation against theirs and confirmed the incidence-angle term is handled the same way. If we've got it wrong, Aakash needs to know before Day 2's fusion work.

---

## #8 · DynamicMap Benchmark metric definitions
**Hriday · `r2-dynamics` `P1` · Day 3**

Zhang et al., ITSC 2023. **Critical nuance:** it evaluates *offline whole-map* cleaning; we run an *online rolling local map*. Our numbers are not directly comparable unless we match their definitions exactly and say what differs.

**Done when** the exact DR/SP definitions are written down for D3, plus a note on what makes our setting different.

---

# P2 — Positioning and numbers

## #7 · Rings vs octree — the one-paragraph justification
**Srinivas · `r1-representation` `P2` · Day 4**

Why 2.5D rings and not wavemap's wavelet octree? Our answer is dense arrays, coalesced GPU access, planner-native queries. **Verify that's actually true** rather than assuming it.

**Done when:** one citeable paragraph exists, and we can answer the question live.

---

## #9 · Has max-mipmaps already been imported to traversability?
**Srinivas · `r1-representation` `P2` · Day 3**

Our conservative pyramid (§7) borrows maximum mipmaps (Tevs, Ihrke & Seidel, I3D 2008) and Hi-Z (Greene 1993). Check whether robotics already did this — arXiv:2010.07929 uses multi-resolution *maximum occupancy* queries for coarse-to-fine collision checking, which is close.

**Done when:** verdict on whether we claim the 2.5D-traversability specialisation or cite someone else's. Affects Shrestha's Day 5 stretch item.

---

## #10 · Baseline number table
**Hriday · `r2-dynamics` `P2` · Day 5**

Every published number we can sit next to, **with its conditions** — dataset, sequence, hardware. Numbers without conditions are useless and will get us caught.

Methods: Removert, ERASOR, Octomap, Dynablox, DUFOMap, BeautyMap.

**Done when:** table committed to `docs/baselines.md`, ready to paste into the deck.

---

## #11 · Comparison table skeleton — resolution schedules
**Pratyushi · `r3-evaluation` `P2` · Day 4**

RoadRunner M&M (RA-L 2024, arXiv:2409.10940) is the most directly comparable: ±50 m at 0.2 m, ±100 m at 0.8 m, learned end-to-end. Get their exact schedule and reported improvements.

**Done when:** table skeleton exists with their schedule next to ours, and the differentiator is stated (theirs is a learned map, ours is an online-refined data structure).

---

## #12 · Traversability decomposition — what conditions do others use?
**Pratyushi · `r3-evaluation` `P2` · Day 4**

Our bitfield has six bits (§7.1). Check SALON (ICRA 2025) and EVORA for what conditions they decompose into and whether we're missing one that matters.

Also produce the one-sentence "why not DOGMa" answer (Nuss et al., IJRR 2018) — we'll be asked.

---

## #13 · Citation verification sweep
**All three · `research` `P1` · Day 5**

Plan v2 already contained one hallucinated citation: **"RangeBlock," 74.5% mIoU — no such paper.** The number matches SphereFormer (74.8%, CVPR 2023), a sparse-voxel transformer, not a range-view method.

**Do:** every citation destined for a slide or the report gets checked against a real PDF. Flag these specifically:
- ❌ RangeBlock — confirm removed everywhere
- ⚠ PCT "three orders of magnitude" — author-reported, own dataset. Quote with conditions.
- ⚠ ML-SkiMap 9.6% retention — author-reported, one cloud
- ⚠ Verti-Bench venue (RSS 2025?) — confirm
- ⚠ Any arXiv preprint stamped 2026 — check for a published version

**Done when:** every citation has a verified PDF, or it's deleted. If you can't find the PDF, the paper doesn't exist.

---

# Writing — Day 5–6

*Research stops Day 5. All three move to writing. These three issues become the related-work section.*

## #14 · Related-work: representation
**Srinivas · `writing` · Day 6** — ~250 words. MLS → multi-level 2.5D → PCT/APGM → wavemap → clipmaps. Ends by stating what's ours.

## #15 · Related-work: segmentation and dynamic removal
**Hriday · `writing` · Day 6** — ~200 words. Range-view segmentation, FLARES, MOS, the removal lineage.

## #16 · Related-work: traversability and evaluation
**Pratyushi · `writing` · Day 6** — ~250 words. Traversability learning, off-road mapping, information-theoretic map compression, and how plan-regret differs. Plus the "why not DOGMa" paragraph.

## #17 · Hard-question rehearsal answers
**All three · `writing` · Day 7** — one written answer each, assigned by owner:

| Question | Owner |
|---|---|
| "Isn't this just a clipmap / Droeschel 2014?" | Srinivas |
| "Why not full 3D / wavemap?" | Srinivas |
| "Planners want uniform grids — you give the savings back." | Pratyushi |
| "Your Ring 0 shows no improvement." | Pratyushi |
| "Can you detect a pothole at 50 m?" | Hriday |
| "Did you tune on your test set?" | Hriday |

**Done when** each has a written answer under 60 seconds spoken, rehearsed once out loud.

## #18 · research-log.md discipline
**All three · `research` · standing, closes Day 6**

One shared append-only file, format in the research-modules doc. Every paper read produces a line: DECISION, COST, POSITIONING. **If a paper produced no line, you shouldn't have read it.**

**Done when** the log is complete and the three related-work sections are traceable to entries in it.

---

## Suggested board columns

```
Blocked  │  Today  │  In progress  │  Needs team decision  │  Done
```

**"Needs team decision"** is the column that matters. When Hriday's #1 or Pratyushi's #6 lands negative, it moves there and gets raised the same hour — not saved for the evening gate. Those two are the only issues on this board that can change the shape of the project.
