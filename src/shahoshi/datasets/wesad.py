"""WESAD: 15 subjects, Empatica E4 on the wrist, with a stress protocol.

The only public corpus carrying wrist PPG, wrist accelerometry and a stress
label on one clock -- and the first wrist-mounted data this project has used at
all, every Stage 0 corpus being waist- or pocket-mounted.

Caveats that belong on every number derived from this corpus
------------------------------------------------------------
**The stressor is a job interview, not an assault.** WESAD's stress condition is
the Trier Social Stress Test: public speaking and mental arithmetic in front of
a panel. It produces a real, large sympathetic response, and it is the closest
public proxy available. It is not the thing this device is meant to detect, and
a recall number measured here is a number about public-speaking stress.

**An E4 is not a MAX30102.** Different LED wavelength, adaptive gain, a properly
tensioned strap, and a device designed to keep skin contact. The target is a
breakout board on hand-cut veroboard in a hand-made wrist mount. This gap is the
heart-rate equivalent of Stage 0's wrist/waist gap, and it is measured and
reported rather than augmented away.

**Amusement is kept as a negative, deliberately.** Watching funny clips raises
heart rate without being distress, so folding it into the negatives is what
stops the branch from learning "elevated HR" and calling it stress. It stays
named in `condition` so Stage C can report it as its own hard-negative slice
instead of hiding inside an aggregate.

Layout and provenance
---------------------
One pickle per subject at ``WESAD/SN/SN.pkl``. Subjects run S2-S17 with S1 and
S12 absent from the release, which is why `SUBJECTS` is written out rather than
generated from a range -- a range would silently look for two subjects that have
never existed and report the corpus as damaged.

The archive is ~18 GB, almost all of it chest signals this branch does not use.
The distilled per-subject cache in `cache.py` is what makes a second run cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cache, e4, ecg
from .base import HR_FS, UNKNOWN, HRWindowSet
from .download import extract, fetch, find_root

TAG = "wesad"

# S1 and S12 are not in the release.
SUBJECTS: tuple[str, ...] = tuple(
    f"S{i}" for i in list(range(2, 12)) + list(range(13, 18))
)

# Uni Siegen's own sciebo share. If this dies, `fetch` prints exactly where to
# put a hand-downloaded copy, which on Colab is the realistic fallback.
URLS = ["https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"]
MARKER = "S2/S2.pkl"

# Extraction directory under the data root. WESAD and PPG-DaLiA both name their
# files SN/SN.pkl, so MARKER alone cannot tell the two corpora apart; the search
# has to be scoped to this subdirectory or a re-run finds whichever extracted
# first and loads it as the other one.
SUBDIR = "wesad"

FS_CHEST = 700          # chest ECG and the label track share this clock
FS_LABEL = 700

# The protocol conditions. 0 is 'not defined / transient' and 5-7 are marked in
# the dataset's own README as to be ignored; both are dropped rather than
# labelled, because a transition between conditions is neither stress nor calm.
CONDITIONS: dict[int, str] = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}
STRESS_CONDITIONS = frozenset({"stress"})


def download(data_dir: str | Path) -> Path:
    """Fetch and extract WESAD, returning its root directory."""
    data_dir = Path(data_dir)
    zip_path = data_dir / "wesad.zip"
    fetch(URLS, zip_path, min_mb=1000)
    extract(zip_path, data_dir / SUBDIR, expect=None)
    return find_root(data_dir / SUBDIR, MARKER)


def distill(pkl_path: str | Path, with_reference: bool = True) -> dict[str, np.ndarray]:
    """Reduce one subject's pickle to the arrays the HR branch actually needs.

    Returns a dict of ``sig`` (n, 4) at `HR_FS`, ``code`` (n,) condition ids,
    and ``ref_t`` / ``ref_bpm``, the ECG-derived beat series. This is exactly
    what gets cached, so it is also the unit that `cache.CACHE_VERSION` guards.
    """
    data = e4.read_pickle(pkl_path)
    bvp, acc = e4.wrist_channels(data, str(pkl_path))
    sig = e4.to_device_rate(bvp, acc, fs=HR_FS)

    labels = np.asarray(data["label"]).reshape(-1)
    # Never extrapolate the label track past its own end: truncate the signal to
    # the labelled duration instead, so no window is labelled by index clamping.
    n = min(len(sig), int(len(labels) * HR_FS / FS_LABEL))
    if n <= 0:
        raise RuntimeError(f"{pkl_path}: no overlap between signal and label track")
    sig = sig[:n]
    code = e4.resample_labels(labels, FS_LABEL, n, HR_FS).astype(np.int16)

    ref_t = np.zeros(0, dtype=np.float32)
    ref_bpm = np.zeros(0, dtype=np.float32)
    if with_reference:
        chest_ecg = data.get("signal", {}).get("chest", {}).get("ECG")
        if chest_ecg is None:
            print(f"  {Path(pkl_path).name}: no chest ECG, hr_ref will be NaN")
        else:
            t, bpm = ecg.instantaneous_hr(np.asarray(chest_ecg).reshape(-1), FS_CHEST)
            ref_t, ref_bpm = t.astype(np.float32), bpm.astype(np.float32)
            if len(t) == 0:
                print(f"  {Path(pkl_path).name}: ECG yielded no usable beats")

    return {"sig": sig, "code": code, "ref_t": ref_t, "ref_bpm": ref_bpm}


def load(
    root: str | Path,
    win: int | None = None,
    stride: int | None = None,
    cache_dir: str | Path | None = None,
    subjects: list[str] | None = None,
    with_reference: bool = True,
) -> HRWindowSet:
    """Load WESAD as windowed wrist PPG with stress labels and ECG ground truth.

    Parameters
    ----------
    root : path
        Directory containing ``S2/S2.pkl`` etc.
    win, stride : int, optional
        Window and hop in samples at `HR_FS`. Defaults to the 8 s / 2 s
        convention documented in `e4`.
    cache_dir : path, optional
        Where distilled per-subject arrays live. Pass a Drive path on Colab.
    subjects : list of str, optional
        Subset of `SUBJECTS`, for a fast smoke test.
    with_reference : bool
        Derive ground-truth HR from the chest ECG. Reading and processing a
        700 Hz ECG per subject is the slow half of a cold run.
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
            payload = distill(pkl, with_reference=with_reference)
            if cpath is not None:
                cache.write(cpath, payload)

        sig, code = payload["sig"], payload["code"]
        starts, wcode = e4.homogeneous_windows(code, win, stride)
        keep = np.isin(wcode, list(CONDITIONS))
        dropped_total += int((~keep).sum())
        starts, wcode = starts[keep], wcode[keep]
        if len(starts) == 0:
            print(f"  {sid}: no windows survived condition filtering")
            continue

        idx = starts[:, None] + np.arange(win)[None, :]
        names = np.array([CONDITIONS[int(c)] for c in wcode], dtype=object)
        parts.append(
            HRWindowSet(
                X=sig[idx],
                y_stress=np.isin(names, list(STRESS_CONDITIONS)).astype(np.int8),
                condition=names,
                hr_ref=e4.reference_for_windows(
                    starts, win, HR_FS, payload["ref_t"], payload["ref_bpm"], "beat"
                ),
                subject=np.full(len(starts), f"{TAG}_{sid}"),
                dataset=np.full(len(starts), TAG),
                fs=HR_FS,
            )
        )

    if not parts:
        raise RuntimeError(
            f"no usable WESAD subjects under {root}. Expected folders like "
            f"'S2' containing 'S2.pkl'."
        )
    if dropped_total:
        print(
            f"  WESAD: dropped {dropped_total:,} windows in transient/ignored "
            f"conditions (labels 0 and 5-7)"
        )
    out = HRWindowSet.concat(parts)
    print(f"  WESAD: {out.X.shape} | {len(out.subjects)} subjects")
    return out
