#!/usr/bin/env python3
"""Is the dataset complete? One table, no network. [Shrestha]

    python scripts/data_status.py            # every sequence
    python scripts/data_status.py 07 08      # just these

Answers the question "can I start yet", per sequence, against the counts the
KITTI archive itself declares. `VRGRID_DATA_ROOT` decides where it looks, and
defaults to ./data/dataset -- the same rule `perception.loader` follows, so if
this says a sequence is READY the loader can read it.

⚑ A count is not an integrity check. Every scan fetched by the ranged
  downloader was verified against the zip's own CRC-32 as it landed, so a file
  that exists here is byte-identical to what unzipping the 80 GB archive would
  have produced. What this checks is whether they are all PRESENT.
"""

import os
import sys
from pathlib import Path

# Scans per sequence, from the central directory of data_odometry_velodyne.zip
# (43,574 entries, read over HTTP range requests). Sequences 00-10 are the
# training half and carry ground-truth poses; 11-21 are the test half.
EXPECTED = {
    "00": 4541, "01": 1101, "02": 4661, "03": 801, "04": 271, "05": 2761,
    "06": 1101, "07": 1101, "08": 4071, "09": 1591, "10": 1201, "11": 921,
    "12": 1061, "13": 3281, "14": 631, "15": 1901, "16": 1731, "17": 491,
    "18": 1801, "19": 4981, "20": 831, "21": 2721,
}
LABELLED = {f"{i:02d}" for i in range(11)}   # SemanticKITTI ships labels for these


def main(argv):
    root = Path(os.environ.get("VRGRID_DATA_ROOT", "data/dataset")).expanduser()
    want = [s.zfill(2) for s in argv[1:]] or sorted(EXPECTED)

    print(f"root: {root}{'' if root.is_dir() else '   ⚑ NOT A DIRECTORY'}\n")
    print(f"{'seq':>4} {'scans':>16} {'GB':>7}  {'labels':>7} {'poses':>6}  status")
    print("-" * 66)

    done = total_have = total_want = 0
    gb = 0.0
    for seq in want:
        exp = EXPECTED.get(seq)
        if exp is None:
            print(f"{seq:>4}   not a KITTI odometry sequence")
            continue
        velo = root / "sequences" / seq / "velodyne"
        bins = sorted(velo.glob("*.bin")) if velo.is_dir() else []
        have = len(bins)
        size = sum(b.stat().st_size for b in bins) / 1e9
        labels = (root / "sequences" / seq / "labels")
        n_lab = len(list(labels.glob("*.label"))) if labels.is_dir() else 0
        pose = (root / "poses" / f"{seq}.txt").exists()

        # A sequence is only usable if the scans are all there AND, for 00-10,
        # the labels match them one for one -- semantics come from the .label
        # files, so a scan without its label is a frame the pipeline must skip.
        ok = have == exp and (seq not in LABELLED or n_lab == exp)
        status = "READY" if ok else f"{have/exp:5.1%}"
        if have and not ok and have == exp:
            status = "scans ok, labels missing"
        lab = f"{n_lab:,}" if n_lab else ("-" if seq not in LABELLED else "none")
        print(f"{seq:>4} {have:>7,}/{exp:<8,} {size:>7.2f}  {lab:>7} "
              f"{'yes' if pose else '-':>6}  {status}")
        done += ok
        total_have += have
        total_want += exp
        gb += size

    print("-" * 66)
    print(f"{done}/{len(want)} sequences ready   "
          f"{total_have:,}/{total_want:,} scans   {gb:.1f} GB")
    if total_have < total_want:
        print(f"\nstill downloading: {total_want - total_have:,} scans to go.")
    else:
        print("\nComplete. Every scan the archive declares is on disk.")
    return 0 if done == len(want) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
