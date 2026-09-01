"""The world-frame contract at the harness seam. Math §2.1, docs/frames.md.

Frame confusion is the most common silent bug in this project (CLAUDE.md) and
`run_sequence` is where it enters: it hands world coordinates to `i_ring`,
which decides cell identity, and a wrong cell is a perfectly valid cell. There
is nothing downstream that would raise.

JP found the seam by reading rather than by running -- the harness composes
`pts @ pose[:3,:3].T + pose[:3,3]` itself, and a raw KITTI `poses.txt` row is
Camera-0 -> World_cam, not vehicle -> world. It has never bitten because the
synthetic sequences write identity-rotation poses ("rotation goes in when JP's
real poses do", `synthetic.write_sequence`), so the naive composition is
exactly right for them and would be exactly wrong on the first real frame.
"""

import numpy as np
import pytest
from vrgrid.eval.harness import (
    GUARD_BASELINE_M,
    FrameConventionError,
    FrameGuard,
    assert_world_is_z_up,
)

# KITTI camera convention: x right, y down, z forward. `poses.txt` is in it.
R_VEH_TO_CAM = np.array([[0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, 0.0]])


def _ground_sweep(n=4000, seed=0):
    """A flat-ish ground patch in the VEHICLE frame: z ~ 0, x forward."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-40.0, 60.0, n)
    y = rng.uniform(-30.0, 30.0, n)
    z = rng.normal(0.0, 0.05, n)          # centimetres of roughness
    return np.column_stack([x, y, z])


def test_a_z_up_world_passes():
    """The convention the lattice needs: ground near z = 0."""
    pts = _ground_sweep()
    assert_world_is_z_up(pts, np.ones(len(pts), bool))


def test_a_translated_z_up_world_still_passes():
    """Driving 300 m does not move the ground plane, so the check does not
    drift out of tolerance over a sequence."""
    pts = _ground_sweep() + np.array([300.0, -12.0, 0.0])
    assert_world_is_z_up(pts, np.ones(len(pts), bool))


def test_a_camera_convention_pose_is_caught():
    """⚑ The bug this file exists for.

    A raw `poses.txt` row applied to z-up vehicle points puts the ground on a
    plane of constant *y*, and world z becomes forward distance -- tens of
    metres, spread over the whole sweep. Silent everywhere downstream.
    """
    pts = _ground_sweep() @ R_VEH_TO_CAM.T
    with pytest.raises(FrameConventionError, match="not z-up"):
        assert_world_is_z_up(pts, np.ones(len(pts), bool))


def test_the_message_names_the_fix():
    """A frame bug found at 2 a.m. should not also be a scavenger hunt."""
    pts = _ground_sweep() @ R_VEH_TO_CAM.T
    with pytest.raises(FrameConventionError) as e:
        assert_world_is_z_up(pts, np.ones(len(pts), bool))
    assert "vehicle_to_world" in str(e.value)
    assert "calib.txt" in str(e.value)


def test_only_ground_returns_are_consulted():
    """A wall, a car flank and a tree canopy are legitimately far off the
    datum. Checking every return would make the tolerance meaningless, so the
    check is restricted to what the ground segmenter flagged."""
    ground = _ground_sweep(n=2000)
    canopy = ground + np.array([0.0, 0.0, 8.0])
    pts = np.vstack([ground, canopy])
    mask = np.zeros(len(pts), bool)
    mask[:len(ground)] = True

    assert_world_is_z_up(pts, mask)                    # ground only: fine
    with pytest.raises(FrameConventionError):
        assert_world_is_z_up(pts, np.ones(len(pts), bool))


def test_no_ground_returns_is_not_an_error():
    """A sweep with nothing flagged as ground has nothing to say about the
    convention. Silence, not a failure -- an indoor or heavily occluded frame
    must not take the run down."""
    pts = _ground_sweep()
    assert_world_is_z_up(pts, np.zeros(len(pts), bool))


def test_the_harness_checks_its_first_frame():
    """End to end: the guard is actually wired into `run_sequence`, not merely
    importable from it."""
    from vrgrid.eval.harness import build_gridmap, run_sequence
    from vrgrid.grid.schedule import load

    gm = build_gridmap(load("5/10/20/40"))
    pts = _ground_sweep()
    labels = np.zeros(len(pts), np.uint32)
    ground = np.ones(len(pts), bool)
    pose = np.zeros((3, 4))
    pose[:3, :3] = R_VEH_TO_CAM          # the wrong convention, as a pose

    with pytest.raises(FrameConventionError):
        run_sequence(gm, [(pts, labels, ground, pose)])


# --- FrameGuard: when the check is actually able to fire ---------------------

def test_frame_zero_cannot_tell_the_conventions_apart():
    """⚑ The reason `FrameGuard` exists, stated as the property it defends.

    A KITTI `poses.txt` starts at the identity by construction. Feeding raw
    SENSOR-frame points through an identity pose -- exactly what the broken
    `reference_map.build()` did -- leaves them in the sensor frame, which is
    x-forward, y-left, z-up with the ground at -1.73 m. That is inside the
    2 m tolerance, so a single-frame check passes a sequence that is wrong
    from frame 1 onward.

    This asserts the weakness directly rather than leaving it implied, because
    the obvious `assert_world_is_z_up(frame0)` reads like sufficient cover and
    is not.
    """
    ground = _ground_sweep() - np.array([0.0, 0.0, 1.73])   # sensor frame
    assert_world_is_z_up(ground, np.ones(len(ground), bool))  # no raise


def test_the_guard_looks_again_once_the_vehicle_has_moved():
    """One look at frame 0, one after GUARD_BASELINE_M, then it stops."""
    guard = FrameGuard()
    pts = _ground_sweep()
    ok = np.ones(len(pts), bool)

    guard.check(pts, ok, np.zeros(3))                 # frame 0, fine
    assert not guard.done, "the guard stopped before it had seen any motion"

    # Still parked: not the second look, so a bad frame here is not caught.
    guard.check(pts, ok, np.array([1.0, 0.0, 0.0]))
    assert not guard.done

    bad = _ground_sweep() @ R_VEH_TO_CAM.T
    with pytest.raises(FrameConventionError):
        guard.check(bad, ok, np.array([GUARD_BASELINE_M + 1.0, 0.0, 0.0]))


def test_the_guard_stops_costing_anything_after_the_second_look():
    """It runs over a whole sequence, so it has to fall out of the way. Once
    the second look has happened the guard is done and never looks again --
    including at a frame that would fail."""
    guard = FrameGuard()
    pts = _ground_sweep()
    ok = np.ones(len(pts), bool)

    guard.check(pts, ok, np.zeros(3))
    guard.check(pts, ok, np.array([0.0, GUARD_BASELINE_M + 1.0, 0.0]))
    assert guard.done

    guard.check(_ground_sweep() @ R_VEH_TO_CAM.T, ok, np.array([50.0, 0.0, 0.0]))


def test_a_caller_with_no_pose_gets_a_single_unconditional_check():
    """Not every caller has a translation to offer. Passing None means "check
    now and be done", which is the old behaviour, kept deliberately rather
    than left as an accident."""
    guard = FrameGuard()
    with pytest.raises(FrameConventionError):
        guard.check(_ground_sweep() @ R_VEH_TO_CAM.T,
                    np.ones(4000, bool), None)
