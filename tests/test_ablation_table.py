"""The threshold freeze behind the schedule ablation. [Shrestha]

The sweep is minutes long and belongs in `scripts/`, not in CI. What is worth
pinning here is the property the ablation rests on: CLAUDE.md says thresholds
are frozen BEFORE schedules are compared, and a comparison run under two
different threshold sets is not a comparison. The digest is how that stops
being a promise.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "thresholds.yaml"


@pytest.fixture(scope="module")
def digest_fn():
    pytest.importorskip("vrgrid.eval.harness")
    from ablation_table import thresholds_digest
    return thresholds_digest


def test_the_digest_is_of_the_file_bytes(digest_fn):
    """Of the bytes, not of the parsed dict, so a reader can check it with
    `sha256sum configs/thresholds.yaml` from a shell. A hash of the dict would
    depend on Python's iteration order and be unverifiable outside Python."""
    got, parsed = digest_fn(CONFIG)
    assert got == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert "visibility" in parsed and "scatter" in parsed


def test_the_digest_moves_when_a_threshold_moves(tmp_path, digest_fn):
    """The whole point: an edit anywhere in the file has to change the digest,
    or the freeze proves nothing."""
    before, _ = digest_fn(CONFIG)
    edited = tmp_path / "thresholds.yaml"
    edited.write_text(CONFIG.read_text().replace("range_tolerance_m: 0.30",
                                                 "range_tolerance_m: 0.31"))
    after, _ = digest_fn(edited)
    assert after != before


def test_a_matching_digest_is_reproducible_from_the_shell(digest_fn):
    """If `sha256sum` and this disagree, the number printed beside the table
    cannot be checked by anyone who did not run the script."""
    if not Path("/usr/bin/sha256sum").exists():
        pytest.skip("sha256sum not available")
    got, _ = digest_fn(CONFIG)
    out = subprocess.run(["/usr/bin/sha256sum", str(CONFIG)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split()[0] == got


def test_expect_thresholds_refuses_a_moved_config(tmp_path, monkeypatch):
    """`--expect-thresholds` is the assertion form. A run under the wrong
    config must fail loudly rather than produce a table nobody can trace."""
    pytest.importorskip("vrgrid.eval.harness")
    import ablation_table

    monkeypatch.setattr(ablation_table, "CONFIG", CONFIG)
    monkeypatch.setattr(sys, "argv",
                        ["ablation_table.py", "--expect-thresholds", "0" * 64])
    with pytest.raises(SystemExit) as excinfo:
        ablation_table.main()
    assert "thresholds have moved" in str(excinfo.value)
