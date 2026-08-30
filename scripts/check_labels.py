import os
import glob
import numpy as np

# Path to the SemanticKITTI sequence labels (JP is downloading this to data/)
LABELS_DIR = "data/sequences/00/labels"

# These are the standard SemanticKITTI moving classes
MOVING_IDS = {
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

def verify_moving_labels():
    if not os.path.exists(LABELS_DIR):
        print(f"Waiting for JP... Directory '{LABELS_DIR}' not found yet.")
        return

    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.label")))[:30]
    if not label_files:
        print("Directory exists, but no .label files found yet. Download might still be running.")
        return

    found = set()
    for f in label_files:
        # The label is a 32-bit unsigned integer, lower 16 bits correspond to the semantic label
        raw_labels = np.fromfile(f, dtype=np.uint32)
        semantic = raw_labels & 0xFFFF
        found.update(set(np.unique(semantic)).intersection(MOVING_IDS.keys()))

    if found:
        print("\n" + "="*45)
        print("R2 GATE 0 VERDICT: CONFIRMED moving-* IDs exist!")
        for id_val in sorted(found):
            print(f" - ID {id_val}: {MOVING_IDS[id_val]}")
        print("="*45)
    else:
        print("VERDICT: No moving-* IDs detected in scanned frames. We may need LMNet residuals.")

if __name__ == "__main__":
    verify_moving_labels()