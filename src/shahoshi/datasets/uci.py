"""UCI HAR: 30 subjects, waist-mounted smartphone, 50 Hz.

Already windowed at 128 steps with 50% overlap by the dataset authors, and the
`body_acc_*` channels already have gravity removed by the filter chain that
`shahoshi.signal` replicates for other corpora. So this loader is the simple
one: stack channels, map labels, prefix subject ids.

`LAYING` (label 6) is dropped -- see the note in `datasets.base`.

Mount: waist. Our device is worn on the wrist, so expect a domain gap that no
amount of data from this corpus closes; `splits.leave_dataset_out` is how we
measure it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CLS, UNKNOWN, WindowSet
from .download import extract, fetch, find_root

TAG = "uci"
FS = 50
WIN = 128  # native windowing; do not change without re-windowing from raw

# UCI's own host has intermittent TLS and rate-limit trouble; the Coursera
# CloudFront mirror has been stable for a decade and carries identical data.
URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://d396qusza40orc.cloudfront.net/getdata/projectfiles/UCI%20HAR%20Dataset.zip",
]

MARKER = "train/y_train.txt"

# UCI activity id -> our label. 6 (LAYING) is deliberately absent.
LABEL_MAP = {
    1: CLS["walk"],
    2: CLS["upstairs"],
    3: CLS["downstairs"],
    4: CLS["sit"],
    5: CLS["stand"],
}
DROPPED = {6: "LAYING"}

# Unified channel order: acc xyz then gyr xyz.
_SIGNALS = ("body_acc", "body_gyro")
_AXES = ("x", "y", "z")


def download(data_dir: str | Path) -> Path:
    """Fetch and extract UCI HAR, returning its root directory."""
    data_dir = Path(data_dir)
    zip_path = data_dir / "uci_har.zip"

    # The UCI archive nests a second zip inside the first on some mirrors.
    fetch(URLS, zip_path, min_mb=50)
    extract(zip_path, data_dir / "uci_har", expect=None)
    try:
        return find_root(data_dir / "uci_har", MARKER)
    except FileNotFoundError:
        inner = list((data_dir / "uci_har").rglob("UCI HAR Dataset.zip"))
        if not inner:
            raise
        extract(inner[0], data_dir / "uci_har")
        return find_root(data_dir / "uci_har", MARKER)


def load(root: str | Path) -> WindowSet:
    """Load UCI HAR into the unified WindowSet contract."""
    root = Path(root)
    if not (root / MARKER).exists():
        raise FileNotFoundError(f"{root} does not look like UCI HAR (no {MARKER})")

    parts: list[WindowSet] = []
    for split in ("train", "test"):
        sig_dir = root / split / "Inertial Signals"

        chans = []
        for sig in _SIGNALS:
            for ax in _AXES:
                path = sig_dir / f"{sig}_{ax}_{split}.txt"
                if not path.exists():
                    raise FileNotFoundError(f"missing channel file {path}")
                chans.append(np.loadtxt(path, dtype=np.float32))  # (n, 128)

        X = np.stack(chans, axis=-1)  # (n, 128, 6)
        if X.shape[1] != WIN:
            raise ValueError(f"expected {WIN}-step windows, got {X.shape[1]}")

        raw_y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
        subj = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        if not (len(raw_y) == len(subj) == len(X)):
            raise ValueError(
                f"{split}: {len(X)} windows but {len(raw_y)} labels and "
                f"{len(subj)} subject ids"
            )

        keep = np.isin(raw_y, list(LABEL_MAP))
        n_dropped = int((~keep).sum())
        if n_dropped:
            names = ", ".join(sorted({DROPPED.get(v, str(v)) for v in raw_y[~keep]}))
            print(f"  UCI {split:5s}: dropping {n_dropped:,} windows ({names})")

        X, raw_y, subj = X[keep], raw_y[keep], subj[keep]
        y_act = np.array([LABEL_MAP[v] for v in raw_y], dtype=np.int64)

        parts.append(
            WindowSet(
                X=X,
                y_act=y_act,
                # UCI contains no falls, and -- more importantly -- no evidence
                # either way about violent events. UNKNOWN, not 0, so these
                # windows are masked out of the fall loss rather than teaching
                # the fall head that ordinary walking is a confirmed non-fall.
                y_fall=np.full(len(X), UNKNOWN, dtype=np.int8),
                subject=np.array([f"{TAG}_{s:02d}" for s in subj]),
                dataset=np.full(len(X), TAG),
            )
        )
        print(f"  UCI {split:5s}: {X.shape}")

    return WindowSet.concat(parts)
