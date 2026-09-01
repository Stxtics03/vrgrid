"""`scripts/build_reference_map.py` -- M* for a real sequence. [Aakash]

Gate 6: every number on a slide comes from a script. M* is upstream of every
number there will be, so the script that builds it is the one that most needs
to work before the 40 GB lands rather than after.

It does not need the 40 GB to be tested. `eval/synthetic.write_sequence`
writes the layout `perception.loader` reads, so pointing the loader at one
exercises the whole real path -- calib, GT poses, sensor -> vehicle -> world,
`FrameGuard`, the cache round trip -- on a scene whose surface is known
analytically.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from vrgrid.eval.synthetic import write_sequence

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

build_reference_map = pytest.importorskip("build_reference_map")


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    """A written sequence with the loader pointed at it.

    `loader.DATA_ROOT` is resolved from `$VRGRID_DATA_ROOT` at IMPORT time, so
    setting the variable in a test does nothing once the module is loaded --
    the attributes have to be patched. The script holds the module rather than
    the values, so patching reaches it.
    """
    from vrgrid.perception import loader, transforms

    write_sequence(tmp_path, "08", n_frames=8, structure=True)
    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(loader, "GT_POSES_DIR", tmp_path / "poses")
    monkeypatch.setattr(loader, "VELODYNE_DIR", tmp_path / "sequences")
    monkeypatch.setattr(loader, "LABELS_DIR", tmp_path / "sequences")
    transforms._TR_CACHE.pop("08", None)
    yield tmp_path
    transforms._TR_CACHE.pop("08", None)


def _run(*argv):
    monkey = ["build_reference_map.py", *argv]
    old, sys.argv = sys.argv, monkey
    try:
        return build_reference_map.main()
    finally:
        sys.argv = old


def test_it_builds_and_caches_a_reference_map(sequence, tmp_path, capsys):
    """The end-to-end run, through `perception.loader` and nothing else.

    ⚑ `reference_map.build()` -- the only path from SemanticKITTI to M* --
      had never been executed by anything before 2 Sep. It raised
      `ValueError: too many values to unpack` on its own first line, because
      `loader.scans` yields three things and it unpacked two, and behind that
      it passed a raw `poses.txt` row and a 4-column sensor-frame array to a
      function wanting a vehicle -> world matrix and 3 columns. That is why no
      real number existed: not that the evaluation was hard, that its first
      step could not run.
    """
    out = tmp_path / "mstar-08.npz"
    assert _run("08", "--out", str(out)) == 0
    assert out.exists()

    printed = capsys.readouterr().out
    assert "reloaded OK" in printed
    assert "trajectory:" in printed, "the frame line is the point of the output"

    from vrgrid.eval.reference_map import load
    ref = load(out)
    assert ref.cell_m == 0.05
    assert ref.count.sum() > 10_000


def test_the_ground_lands_where_the_surface_is(sequence, tmp_path):
    """M* is what everything is measured against, so it is the one map whose
    heights must be checked against something external -- here, the analytic
    surface the sweep was generated from. A frame error survives every other
    assertion in this file and dies on this one."""
    from vrgrid.eval.reference_map import load
    from vrgrid.eval.synthetic import terrain_height_m

    out = tmp_path / "mstar-08.npz"
    assert _run("08", "--out", str(out)) == 0
    ref = load(out)

    obs = np.argwhere(ref.count > 0)
    x = (obs[:, 0] + ref.i0 + 0.5) * ref.cell_m
    y = (obs[:, 1] + ref.j0 + 0.5) * ref.cell_m
    got = ref.height_cm[obs[:, 0], obs[:, 1]] / 100.0
    err = np.abs(got - terrain_height_m(x, y, 0))
    assert np.median(err) < 0.02


def test_a_first_pass_can_stop_early(sequence, tmp_path):
    """`--max-frames` is what makes checking the frame convention on a real
    sequence a minute rather than an hour."""
    short, full = tmp_path / "a.npz", tmp_path / "b.npz"
    assert _run("08", "--out", str(short), "--max-frames", "2") == 0
    assert _run("08", "--out", str(full)) == 0

    from vrgrid.eval.reference_map import load
    assert load(short).count.sum() < load(full).count.sum()


def test_missing_data_is_a_clean_exit_not_a_traceback(tmp_path, monkeypatch,
                                                      capsys):
    """The common case for the next six days: no download yet. It must say
    which path it looked in, because "set $VRGRID_DATA_ROOT" is the answer
    almost every time and a stack trace from inside the loader does not say
    that."""
    from vrgrid.perception import loader

    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(loader, "GT_POSES_DIR", tmp_path / "poses")
    monkeypatch.setattr(loader, "VELODYNE_DIR", tmp_path / "sequences")

    assert _run("08") == 2
    err = capsys.readouterr().err
    assert "VRGRID_DATA_ROOT" in err and "not present" in err


def test_the_script_is_runnable_as_a_script():
    """Gate 6 means someone types this at a shell. Import-time errors --
    a bad sys.path insert, a missing dependency -- are found here rather than
    on the evening the download lands."""
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(SCRIPTS / "build_reference_map.py"),
                       run_name="__main__")
    assert e.value.code == 2, "with no data configured it should exit 2"
