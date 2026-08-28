"""MotionSense: 24 subjects, phone in a front trouser pocket, 50 Hz.

Continuous CSV trials, so this loader windows them itself. `userAcceleration.*`
is already gravity-free (iOS Core Motion separates it on device), and
`rotationRate.*` is already rad/s, so the channels line up with UCI's without
conversion.

Mount: front trouser pocket. Axis orientation therefore does *not* correspond to
UCI's waist mounting, which is exactly the domain gap we want measured rather
than hidden -- see `splits.leave_dataset_out`.

Hosted in the author's own GitHub repo, so no Kaggle account or API token.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CLS, UNKNOWN, WindowSet
from .download import extract, fetch, find_root
from ..windows import sliding_windows

TAG = "ms"
FS = 50

URLS = ["https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"]
MARKER = "wlk_15"
SUBDIR = "motionsense"

# Folder-name prefix -> our label. MotionSense has no 'lay' trials at all, so
# unlike UCI nothing is dropped here.
LABEL_MAP = {
    "wlk": CLS["walk"],
    "ups": CLS["upstairs"],
    "dws": CLS["downstairs"],
    "sit": CLS["sit"],
    "std": CLS["stand"],
    "jog": CLS["jog"],
}

# Column names in the unified channel order. Read by name, never by position:
# the CSVs carry an unnamed index column and extra attitude/gravity columns, and
# their order has differed between releases of the archive.
COLUMNS = [
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
]


def download(data_dir: str | Path) -> Path:
    """Fetch and extract MotionSense, returning its root directory."""
    data_dir = Path(data_dir)
    zip_path = data_dir / "motionsense.zip"
    fetch(URLS, zip_path, min_mb=50)
    extract(zip_path, data_dir / SUBDIR, expect=None)
    return find_root(data_dir / SUBDIR, MARKER)


def load(root: str | Path, win: int = 128, stride: int = 64) -> WindowSet:
    """Load MotionSense, windowing the continuous trials.

    Parameters
    ----------
    win, stride : int
        Window length and hop in samples. Defaults match UCI's native 128 steps
        at 50% overlap so the two corpora can be concatenated.
    """
    root = Path(root)
    trial_dirs = sorted(glob.glob(str(root / "*_*")))
    if not trial_dirs:
        raise FileNotFoundError(f"no trial folders (like 'wlk_15') found under {root}")

    parts: list[WindowSet] = []
    skipped_short = 0
    per_class: dict[str, int] = {}

    for folder in trial_dirs:
        folder = Path(folder)
        activity = folder.name.split("_")[0]
        if activity not in LABEL_MAP:
            continue

        for csv in sorted(folder.glob("sub_*.csv")):
            df = pd.read_csv(csv)
            missing = [c for c in COLUMNS if c not in df.columns]
            if missing:
                print(f"  !! {csv.name} in {folder.name} lacks {missing}, skipping")
                continue

            sig = df[COLUMNS].to_numpy(dtype=np.float32)
            if len(sig) < win:
                skipped_short += 1
                continue

            X = sliding_windows(sig, win, stride)
            if not len(X):
                continue

            subject = csv.stem  # 'sub_1' ... 'sub_24'
            parts.append(
                WindowSet(
                    X=X,
                    y_act=np.full(len(X), LABEL_MAP[activity], dtype=np.int64),
                    # No falls, and no evidence either way -- see uci.py.
                    y_fall=np.full(len(X), UNKNOWN, dtype=np.int8),
                    subject=np.full(len(X), f"{TAG}_{subject}"),
                    dataset=np.full(len(X), TAG),
                )
            )
            per_class[activity] = per_class.get(activity, 0) + len(X)

    if not parts:
        raise RuntimeError(
            f"no usable MotionSense CSVs under {root}. Expected folders like "
            f"'wlk_15' containing 'sub_1.csv'."
        )
    if skipped_short:
        print(f"  MotionSense: skipped {skipped_short} trials shorter than one window")

    out = WindowSet.concat(parts)
    print(f"  MotionSense: {out.X.shape} | {len(set(out.subject.tolist()))} subjects")
    return out
