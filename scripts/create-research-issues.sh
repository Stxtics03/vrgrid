#!/usr/bin/env bash
# Creates labels, milestones and the 18 research issues.
#
#   gh auth login
#   cd <repo>
#   bash scripts/create-research-issues.sh
#
# Set GH_USER_* to the GitHub usernames, or leave blank to create unassigned.

set -euo pipefail

GH_USER_SRINIVAS="${GH_USER_SRINIVAS:-}"
GH_USER_HRIDAY="${GH_USER_HRIDAY:-}"
GH_USER_PRATYUSHI="${GH_USER_PRATYUSHI:-}"

a() { [ -n "$1" ] && echo "--assignee $1" || echo ""; }

echo "==> labels"
gh label create research          --color 0E8A16 --description "Research track"                --force
gh label create writing           --color 5319E7 --description "Report / deck writing"         --force
gh label create gate              --color B60205 --description "Decides a project claim"        --force
gh label create r1-representation --color 1D76DB --description "Representation & prior art"     --force
gh label create r2-dynamics       --color 1D76DB --description "Dynamics & segmentation"        --force
gh label create r3-evaluation     --color 1D76DB --description "Traversability & evaluation"    --force
gh label create P0-blocking       --color D93F0B --description "Someone is waiting on this"     --force
gh label create P1                --color FBCA04 --description "Feeds a developer directly"     --force
gh label create P2                --color C2E0C6 --description "Positioning and numbers"        --force

echo "==> milestones"
for m in "Day 0 - 28 Aug" "Day 1-2 - 29-30 Aug" "Day 3 - 31 Aug" "Day 5 - 2 Sep" "Day 6-7 - 3-4 Sep"; do
  gh api "repos/{owner}/{repo}/milestones" -f title="$m" >/dev/null 2>&1 || echo "   (exists: $m)"
done

echo "==> issues"

gh issue create --title "[R2] Verify moving-* labels exist in raw SemanticKITTI .label files" \
  --label research,r2-dynamics,P0-blocking,gate --milestone "Day 0 - 28 Aug" $(a "$GH_USER_HRIDAY") \
  --body 'Our entire no-retraining plan rests on this. Master v4 §3.6 claims `moving-*` IDs 250-259 are present in the raw `.label` files and the 19-class collapse happens only in the `learning_map` config.

If true, we skip several days of GPU training. If false, JP falls back to residual-image MOS **the same day**.

**Do:** load one `.label` file directly, print `np.unique()` of the raw values. Cross-check against Behley et al. ICCV 2019 and `semantic-kitti.yaml`.

**Done when:** one-line verdict in the team channel with the actual unique-value array pasted in. Not "I think so" — the array.

**Deadline: 4 hours. Blocks JP Day 1-2.**'

gh issue create --title "[R3] Novelty verdict — is plan-regret evaluation already published?" \
  --label research,r3-evaluation,P0-blocking,gate --milestone "Day 3 - 31 Aug" $(a "$GH_USER_PRATYUSHI") \
  --body 'Gates our headline slide. We claim nobody measures coarsening in units of planner regret. Closest prior art: Psomiadis, Maity & Tsiotras, ICRA 2024 (arXiv:2309.13451) — selects map compression guided by the path, but on an **information-theoretic** objective, not plan cost.

**Read:** Psomiadis 2024; iterative version arXiv:2503.10843; Larsson 2021 RA-L 6(4):7651. Search "task-aware perception", "task-driven map compression", "perception-planning co-design", "planner-aware resolution".

**Done when** you can state in three sentences exactly how our metric differs — or you escalate that it does not.

**Fallback if published:** reframe as validation in a new domain, keep the curve.

**Hard deadline Day 3.**'

gh issue create --title "[R1] Prior art we must cite — the cite-or-get-caught list" \
  --label research,r1-representation,P1 --milestone "Day 1-2 - 29-30 Aug" $(a "$GH_USER_SRINIVAS") \
  --body 'Three known landmines. A panel member who recognises our ring diagram as a 2014 paper we did not cite will discount everything else we say.

- **Triebel, Pfaff & Burgard, IROS 2006** — our ground+ceiling cell is a two-layer MLS map.
- **Droeschel, Stückler & Behnke, ICRA 2014 + JFR 33(4) 2016** — our foveated ring grid, with interlaced ring buffers, in 2014.
- **Losasso & Hoppe, SIGGRAPH 2004** — graphics origin of nested toroidal LOD.

**Done when:** memo lists each with one sentence on what they did and one on what we do differently, AND master v4 wording is edited where it overclaims.

**Bonus:** extract Droeschel ring-buffer implementation notes for Aakash.'

gh issue create --title "[R3] Metric definitions as pseudocode for Aakash" \
  --label research,r3-evaluation,P1 --milestone "Day 1-2 - 29-30 Aug" $(a "$GH_USER_PRATYUSHI") \
  --body 'Aakash implements the metrics; he should not have to read four papers to do it.

**Done when** `docs/metric-definitions.md` gives implementable pseudocode for:
- plan regret R(S) — math §8.1, both paths scored on the reference map
- discrete Fréchet distance
- ρ = IL/spread — math §9.3
- DR / SP / F — math §9.4

**Blocks Aakash Day 4.**'

gh issue create --title "[R2] FRNet setup memo" \
  --label research,r2-dynamics,P1 --milestone "Day 1-2 - 29-30 Aug" $(a "$GH_USER_HRIDAY") \
  --body '**Done when** JP has: exact config filename, checkpoint URL, confirmation checkpoints are 19-class, class-index mapping, MMDetection3D version constraints, known gotchas.

Also check whether Fast-FRNet (~7.5M params) is worth holding in reserve.

Verify: Apache 2.0 licence, 73.3% SemanticKITTI mIoU.

**Blocks JP Day 1.**'

gh issue create --title "[R1] Confirm measurement-variance model against Fankhauser 2014" \
  --label research,r1-representation,P1 --milestone "Day 1-2 - 29-30 Aug" $(a "$GH_USER_SRINIVAS") \
  --body 'Math §3.2 derives σ²_z = [sin²φ σ²_r + r² cos²φ σ²_φ] / cos²θ_inc from Fankhauser et al., CLAWAR 2014.

**Done when** our derivation is checked against theirs, especially the incidence-angle term. If we have it wrong, Aakash needs to know before Day 2 fusion work.'

gh issue create --title "[R2] DynamicMap Benchmark metric definitions" \
  --label research,r2-dynamics,P1 --milestone "Day 3 - 31 Aug" $(a "$GH_USER_HRIDAY") \
  --body 'Zhang et al., ITSC 2023.

**Critical nuance:** it evaluates offline whole-map cleaning; we run an online rolling local map. Our numbers are not directly comparable unless we match definitions exactly and state what differs.

**Done when:** exact DR/SP definitions written down for D3, plus a note on what makes our setting different.'

gh issue create --title "[R1] Rings vs octree — the one-paragraph justification" \
  --label research,r1-representation,P2 --milestone "Day 5 - 2 Sep" $(a "$GH_USER_SRINIVAS") \
  --body 'Why 2.5D rings and not wavemap wavelet octree? Our answer: dense arrays, coalesced GPU access, planner-native queries. **Verify that is actually true** rather than assuming.

**Done when:** one citeable paragraph exists and we can answer live.'

gh issue create --title "[R1] Has max-mipmaps already been imported to traversability?" \
  --label research,r1-representation,P2 --milestone "Day 3 - 31 Aug" $(a "$GH_USER_SRINIVAS") \
  --body 'Our conservative pyramid (math §7) borrows maximum mipmaps (Tevs, Ihrke & Seidel, I3D 2008) and Hi-Z (Greene 1993).

Check whether robotics already did this — arXiv:2010.07929 uses multi-resolution **maximum occupancy** queries for coarse-to-fine collision checking, which is close.

**Done when:** verdict on whether we claim the 2.5D-traversability specialisation or cite someone else. Affects Shrestha Day 5 stretch item.'

gh issue create --title "[R2] Baseline number table" \
  --label research,r2-dynamics,P2 --milestone "Day 5 - 2 Sep" $(a "$GH_USER_HRIDAY") \
  --body 'Every published number we can sit next to, **with its conditions** — dataset, sequence, hardware. Numbers without conditions are useless and will get us caught.

Methods: Removert, ERASOR, Octomap, Dynablox, DUFOMap, BeautyMap.

**Done when:** committed to `docs/baselines.md`, ready to paste into the deck.'

gh issue create --title "[R3] Comparison table skeleton — resolution schedules" \
  --label research,r3-evaluation,P2 --milestone "Day 5 - 2 Sep" $(a "$GH_USER_PRATYUSHI") \
  --body 'RoadRunner M&M (RA-L 2024, arXiv:2409.10940) is most directly comparable: ±50 m at 0.2 m, ±100 m at 0.8 m, learned end-to-end.

**Done when:** table skeleton with their schedule next to ours, and the differentiator stated — theirs is a learned map, ours is an online-refined data structure.'

gh issue create --title "[R3] Traversability decomposition — what conditions do others use?" \
  --label research,r3-evaluation,P2 --milestone "Day 5 - 2 Sep" $(a "$GH_USER_PRATYUSHI") \
  --body 'Our bitfield has six bits (math §7.1). Check SALON (ICRA 2025) and EVORA for what conditions they decompose into and whether we are missing one that matters.

Also produce the one-sentence "why not DOGMa" answer (Nuss et al., IJRR 2018) — we will be asked.'

gh issue create --title "[ALL] Citation verification sweep" \
  --label research,P1 --milestone "Day 5 - 2 Sep" \
  --body 'Plan v2 contained one hallucinated citation: **"RangeBlock", 74.5% mIoU — no such paper.** The number matches SphereFormer (74.8%, CVPR 2023), a sparse-voxel transformer, not range-view.

Every citation destined for a slide or the report gets checked against a real PDF. Flag specifically:

- [ ] RangeBlock — confirm removed everywhere
- [ ] PCT "three orders of magnitude" — author-reported, own dataset; quote with conditions
- [ ] ML-SkiMap 9.6% retention — author-reported, one cloud
- [ ] Verti-Bench venue (RSS 2025?) — confirm
- [ ] Any arXiv preprint stamped 2026 — check for a published version

**Done when:** every citation has a verified PDF, or it is deleted. If you cannot find the PDF, the paper does not exist.'

gh issue create --title "[R1] Related-work: representation (~250 words)" \
  --label research,writing,r1-representation --milestone "Day 6-7 - 3-4 Sep" $(a "$GH_USER_SRINIVAS") \
  --body 'MLS → multi-level 2.5D → PCT / APGM → wavemap → clipmaps. Ends by stating what is ours.'

gh issue create --title "[R2] Related-work: segmentation and dynamic removal (~200 words)" \
  --label research,writing,r2-dynamics --milestone "Day 6-7 - 3-4 Sep" $(a "$GH_USER_HRIDAY") \
  --body 'Range-view segmentation, FLARES, MOS, the removal lineage.'

gh issue create --title "[R3] Related-work: traversability and evaluation (~250 words)" \
  --label research,writing,r3-evaluation --milestone "Day 6-7 - 3-4 Sep" $(a "$GH_USER_PRATYUSHI") \
  --body 'Traversability learning, off-road mapping, information-theoretic map compression, and how plan-regret differs. Plus the "why not DOGMa" paragraph.'

gh issue create --title "[ALL] Hard-question rehearsal answers" \
  --label research,writing --milestone "Day 6-7 - 3-4 Sep" \
  --body 'One written answer each, assigned by owner:

| Question | Owner |
|---|---|
| "Isn'"'"'t this just a clipmap / Droeschel 2014?" | Srinivas |
| "Why not full 3D / wavemap?" | Srinivas |
| "Planners want uniform grids — you give the savings back." | Pratyushi |
| "Your Ring 0 shows no improvement." | Pratyushi |
| "Can you detect a pothole at 50 m?" | Hriday |
| "Did you tune on your test set?" | Hriday |

**Done when** each has a written answer under 60 seconds spoken, rehearsed once out loud.'

gh issue create --title "[ALL] research-log.md discipline" \
  --label research --milestone "Day 6-7 - 3-4 Sep" \
  --body 'One shared append-only file, format in `docs/research-modules.md`. Every paper read produces a line: DECISION, COST, POSITIONING.

**If a paper produced no line, you should not have read it.**

**Done when** the log is complete and the three related-work sections are traceable to entries in it.'

echo
echo "Done. 18 issues created."
echo "Watch #1 (Hriday, 4h) and #2 (Pratyushi, Day 3) — those two can change the shape of the project."
