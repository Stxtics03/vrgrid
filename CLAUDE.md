# vrgrid — Adaptive Variable-Resolution 2.5D LiDAR Map

SIH26053. Foveated 2.5D elevation grid: cell size grows with range, semantics can force local refinement, memory bounded at startup.

## Commands
```bash
make test              # all unit tests — must pass before any merge
make test-determinism  # same input twice -> identical map hash
python -m vrgrid.run --seq 08 --schedule 5/10/20/40
python -m vrgrid.dash  # Rerun dashboard, separate process
```

## Reference docs — read on demand, do not assume
- `docs/sih-math.md` — **every formula, theorem and unit test.** Read the relevant section before touching fusion, split/merge, traversability, or metrics. Section numbers are cited in code comments.
- `docs/master-v4.md` — architecture and scope decisions.
- `docs/execution-plan.md`, `docs/team-assignments.md` — who owns what.

## Hard invariants — violating these silently breaks correctness

**Integer lattice only.** Cell indices are `i_L = i_fine // k_L` where `i_fine = floor(x/0.05)` and `k_L` is an integer. Never compute `floor(x/0.20)` directly — float lattices drift apart and produce gaps and double-counts at ring boundaries. See math §2.

**Fixed-point accumulation, never float atomics.** Heights accumulate as int32 in 1 cm units. Float atomic adds are non-associative, so results differ run to run and bugs move when you look at them. See math §3.4.

**Ring ratios must be integers.** `validate()` rejects non-integer ratios between consecutive rings. Powers of two are a convenience (bit-shift), not a requirement — 5/10/50 is legal because 10/5=2 and 50/10=5.

**Merge uses the law of total variance, not inverse-variance fusion.** Children measure different *places*, not the same quantity. `σ²_p = Σwᵢσᵢ² + Σwᵢ(μᵢ−μ_p)²`. Dropping the second term makes merged cells most confident exactly where they straddle a kerb. See math §4.

**Split inflates variance and sets the `derived` bit.** Children inherit `μ_p` and a strictly larger variance. The `derived` bit is what makes `merge(split(c)) == c` exact. Without it, a cell oscillating across a ring boundary inflates variance every frame with no physical cause. See math §5.

**Visibility cleanup never clears a cell with a return in the current scan.** Without this guard it eats fences, poles and sign posts within a few frames. See math §10.4.

**No allocation inside the frame loop.** Grid arrays, refinement pool, transient layer and tracked-object list are all preallocated at startup. The compile-time memory bound is a headline claim and allocation in the loop makes it false.

**Structure-of-arrays, not array-of-structs.** One array per field. Coalesced GPU access.

**Core has no ROS dependency.** ROS adapter is an optional module under `adapters/`. If ROS breaks two days before submission the framework must still run.

**Unknown ≠ free.** Three occupancy states. Unknown is decided by observation count, not by log-odds near zero. The blind cone (3.74 m radius) is unknown, never free.

## Layout and ownership
```
include/vrgrid/     FROZEN interfaces — whole-team change only, never edit unilaterally
src/grid/           lattice, rings, split/merge, fusion, refinement pool   [Aakash]
src/eval/           reference map, metrics, plan regret                    [Aakash]
src/gpu/            kernels, allocators, timing                            [Shrestha]
src/perception/     loader, transforms, range image, FRNet, ground         [JP]
dashboard/          Rerun app                                             [JP]
tests/              one file per module
docs/               planning docs + research-log.md
```
Stay inside your own directory. Cross-directory changes go through a same-day PR.

## Conventions
- Heights are int16 in **1 cm** units. Ranges and gradients are float metres. Never mix silently — suffix variables `_cm` or `_m`.
- Vehicle frame: x forward, y left, z up. Every transform is written down in `docs/frames.md`. Frame confusion is the most common silent bug in this project — the map looks plausible and slowly rotates.
- Cell struct is **12 bytes, frozen Day 0**. Adding a field means recomputing every memory figure in the report.
- Math section numbers go in docstrings: `"""Merge four children. See math §4.2."""`

## Testing rules
- Every formula in `docs/sih-math.md` has a named unit test. If you implement a formula, implement its test in the same commit.
- The determinism test and the partition test (10⁶ random points, exactly one cell per ring) are CI-blocking.
- Do not weaken a test to make it pass. If a theorem test fails, the implementation is wrong — those are proofs, not tuning targets.

## Don't
- Don't reimplement Patchwork++ or KISS-ICP. Wire them in.
- Don't retrain anything, and don't run inference for labels. Both the 19-class semantic label and the `moving-*` motion flag come straight from the SemanticKITTI raw `.label` files (`src/perception/semantics.py`). FRNet is not used — the only standalone port available does not reproduce the trained network and is flagged non-functional in `src/perception/frnet/`. Disclose plainly: the mapping contribution is evaluated independently of segmentation quality.
- Don't add features after the Day 6 freeze.
- Don't hardcode thresholds inline — they live in `configs/`, and they are frozen before schedules are compared.
