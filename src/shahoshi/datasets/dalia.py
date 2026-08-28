"""PPG-DaLiA: 15 subjects, E4 wrist PPG across eight everyday activities.

The negatives corpus. WESAD says what a stress response looks like; this says
what the rest of life looks like, which is where a false-alarm rate in alarms
per hour actually comes from. It is also where motion artefacts stop being a
caveat and become measurable: cycling, stair climbing and table football are
exactly the conditions under which a wrist PPG degrades.

**Every window here is labelled non-stress, and that is an assumption.** DaLiA
runs no stress protocol and ships no affect labels, so `y_stress` is 0 for all
of it by construction rather than by observation. Someone stuck in traffic
during the driving block may well have been stressed. The assumption is exactly
what makes the corpus usable as a false-alarm denominator, so it is stated here
rather than buried: a "false" alarm on DaLiA means an alarm during ordinary
activity, not an alarm verified against a calm subject.

Ground-truth HR comes free. DaLiA ships an ECG-derived heart rate computed over
8 s windows shifted by 2 s -- which is why `e4.WIN_SECONDS` and
`e4.STRIDE_SECONDS` are 8 and 2. A window here lines up with the corpus's own
estimate exactly, so `hr_ref` is the reference value for that span rather than
something interpolated onto it. No R-peak detection runs for this corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cache, e4
from .base import HR_FS, HRWindowSet
from .download import extract, fetch, find_root

TAG = "dalia"

SUBJECTS: tuple[str, ...] = tuple(f"S{i}" for i in range(1, 16))

URLS = ["https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"]
MARKER = "S1/S1.pkl"

# See the note on wesad.SUBDIR: SN/SN.pkl is ambiguous between the two corpora.
SUBDIR = "dalia"

FS_ACTIVITY = 4         # the activity track's own rate

# Ground truth is stamped every 2 s and describes the 8 s window that *starts*
# there, so its timestamp is that window's centre.
FS_LABEL_HZ = 0.5
LABEL_WINDOW_S = 8.0

# DaLiA's activity ids. 0 is the transition between protocol blocks and is
# dropped: it is neither one activity nor another.
ACTIVITIES: dict[int, str] = {
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}


def download(data_dir: str | Path) -> Path:
    """Fetch and extract PPG-DaLiA, returning its root directory."""
    data_dir = Path(data_dir)
    zip_path = data_dir / "ppg_dalia.zip"
    fetch(URLS, zip_path, min_mb=500)
    extract(zip_path, data_dir / SUBDIR, expect=None)
    return find_root(data_dir / SUBDIR, MARKER)


def distill(pkl_path: str | Path) -> dict[str, np.ndarray]:
    """Reduce one subject's pickle to the arrays the HR branch needs."""
    data = e4.read_pickle(pkl_path)
    bvp, acc = e4.wrist_channels(data, str(pkl_path))
    sig = e4.to_device_rate(bvp, acc, fs=HR_FS)

    activity = np.asarray(data["activity"]).reshape(-1)
    n = min(len(sig), int(len(activity) * HR_FS / FS_ACTIVITY))
    if n <= 0:
        raise RuntimeError(f"{pkl_path}: no overlap between signal and activity track")
    sig = sig[:n]
    code = e4.resample_labels(activity, FS_ACTIVITY, n, HR_FS).astype(np.int16)

    hr = np.asarray(data["label"], dtype=np.float64).reshape(-1)
    ref_t = (np.arange(len(hr)) / FS_LABEL_HZ + LABEL_WINDOW_S / 2.0).astype(np.float32)

    return {
        "sig": sig,
        "code": code,
        "ref_t": ref_t,
        "ref_bpm": hr.astype(np.float32),
    }


def load(
    root: str | Path,
    win: int | None = None,
    stride: int | None = None,
    cache_dir: str | Path | None = None,
    subjects: list[str] | None = None,
) -> HRWindowSet:
    """Load PPG-DaLiA as windowed wrist PPG with ECG-derived reference HR.

    Parameters
    ----------
    root : path
        Directory containing ``S1/S1.pkl`` etc.
    win, stride : int, optional
        Window and hop in samples at `HR_FS`. Changing these away from the 8 s /
        2 s default breaks the exact alignment with the corpus's own ground
        truth, which is then matched by nearest-centre instead.
    cache_dir : path, optional
        Where distilled per-subject arrays live.
    subjects : list of str, optional
        Subset of `SUBJECTS`, for a fast smoke test.
    """
    root = Path(root)
    win = e4.win_samples(HR_FS) if win is None else int(win)
    stride = e4.stride_samples(HR_FS) if stride is None else int(stride)
    subjects = list(SUBJECTS) if subjects is None else list(subjects)

    parts: list[HRWindowSet] = []
    dropped_total = 0

    for sid in subjects:
        payload = None
        cpath = None
        if cache_dir is not None:
            cpath = cache.subject_path(cache_dir, TAG, sid)
            payload = cache.read(cpath)

        if payload is None:
            pkl = root / sid / f"{sid}.pkl"
            if not pkl.exists():
                print(f"  {sid}: {pkl} missing, skipping")
                continue
            payload = distill(pkl)
            if cpath is not None:
                cache.write(cpath, payload)

        sig, code = payload["sig"], payload["code"]
        starts, wcode = e4.homogeneous_windows(code, win, stride)
        keep = np.isin(wcode, list(ACTIVITIES))
        dropped_total += int((~keep).sum())
        starts, wcode = starts[keep], wcode[keep]
        if len(starts) == 0:
            print(f"  {sid}: no windows survived activity filtering")
            continue

        idx = starts[:, None] + np.arange(win)[None, :]
        parts.append(
            HRWindowSet(
                X=sig[idx],
                # See the module docstring: assumed, not observed.
                y_stress=np.zeros(len(starts), dtype=np.int8),
                condition=np.array(
                    [ACTIVITIES[int(c)] for c in wcode], dtype=object
                ),
                hr_ref=e4.reference_for_windows(
                    starts, win, HR_FS, payload["ref_t"], payload["ref_bpm"], "window"
                ),
                subject=np.full(len(starts), f"{TAG}_{sid}"),
                dataset=np.full(len(starts), TAG),
                fs=HR_FS,
            )
        )

    if not parts:
        raise RuntimeError(
            f"no usable PPG-DaLiA subjects under {root}. Expected folders like "
            f"'S1' containing 'S1.pkl'."
        )
    if dropped_total:
        print(f"  DaLiA: dropped {dropped_total:,} windows in transitions (activity 0)")
    out = HRWindowSet.concat(parts)
    print(f"  DaLiA: {out.X.shape} | {len(out.subjects)} subjects")
    return out
