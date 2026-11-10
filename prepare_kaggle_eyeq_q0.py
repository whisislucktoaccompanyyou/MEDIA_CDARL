"""Build a quality==0 Kaggle manifest without touching the all-Kaggle experiment.

Keeps the patient split already written in data/Dataset_Kaggle/split_manifest.json
(seed 2026). Re-splitting would put the same eye in train here and test there.
Does not start training and does not rewrite Dataset_Kaggle.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ROOT.parent / "data" / "kaggle" / "Label_EyeQ_train.csv"
SOURCE = ROOT.parent / "data" / "kaggle" / "extracted" / "train"
BASE = ROOT / "data" / "Dataset_Kaggle" / "split_manifest.json"
DEST = ROOT / "data" / "Dataset_Kaggle_EyeQ_q0"


def main():
    good = set()
    with LABELS.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0] == "":
                continue
            if row[2].strip() == "0":
                good.add(row[1].strip())
    print("quality==0 images in CSV:", len(good), flush=True)

    missing = sorted(name for name in good if not (SOURCE / name).is_file())
    if missing:
        raise SystemExit("%d quality==0 images are missing, first: %s" % (len(missing), missing[0]))

    base = json.loads(BASE.read_text())
    manifest = {split: [name for name in names if name in good] for split, names in base.items()}
    assigned = {name for names in manifest.values() for name in names}
    unseen = sorted(good - assigned)
    if unseen:
        raise SystemExit("%d quality==0 images are outside the saved patient split" % len(unseen))

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    link = DEST / "train" / "trainA"
    link.parent.mkdir(parents=True, exist_ok=True)
    target = (ROOT / "data" / "Dataset_Kaggle" / "train" / "trainA").resolve()
    if not target.is_dir():
        raise SystemExit("fractal labels missing: %s" % target)
    if link.is_symlink():
        if link.resolve() != target:
            raise SystemExit("trainA link points elsewhere: %s" % link.resolve())
    elif link.exists():
        raise SystemExit("refusing to replace existing %s" % link)
    else:
        link.symlink_to(target, target_is_directory=True)
    print("EyeQ q0 splits:", {key: len(value) for key, value in manifest.items()}, flush=True)
    print("manifest:", DEST / "split_manifest.json", flush=True)


if __name__ == "__main__":
    main()
