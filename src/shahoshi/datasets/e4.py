"""Shared parsing for the two Empatica E4 corpora, WESAD and PPG-DaLiA.

Both ship one pickle per subject with the same nesting -- ``signal.wrist.BVP``
at 64 Hz and ``signal.wrist.ACC`` at 32 Hz -- so the reading, rate conversion
and windowing live here once and each loader supplies only what differs: where
the archive comes from, what the condition labels mean, and where ground-truth
HR comes from.

Three things this module decides, all of which are easy to get silently wrong.

**Pickle encoding.** Both corpora's pickles were written under Python 2. Loading
them under Python 3 without ``encoding='latin1'`` raises `UnicodeDecodeError` on
the byte-string keys, and the workaround people reach for first -- ``encoding=
'bytes'`` -- succeeds but returns ``b'signal'`` style keys, so every subsequent
lookup by ``'signal'`` fails with a KeyError that looks like a corrupt file.

**ACC scaling.** The E4 stores acceleration in units of 1/64 g. That is an
assumption about the corpus, not a fact this code can prove, so rather than
assert it the loaders keep gravity in the channel and the availability gate
reports median |acc|. At rest that must land near 1.0 g; a reading near 64 means
`E4_ACC_LSB_PER_G` is wrong for this release of the corpus, and it is visible in
one line instead of surviving as a silently mis-scaled artefact gate.

**Condition labels are categorical.** They are resampled by index lookup, never
by `signal.resample`. Polyphase-filtering a label array interpolates *between
class ids* and produces windows labelled with conditions that never occurred.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .base import HR_FS, N_HR_CHANNELS
from ..signal import resample
from ..windows import n_windows

# Native Empatica E4 rates, identical across both corpora.
FS_BVP = 64
FS_ACC = 32

# See the module docstring: an assumption, made visible by the availability gate
# rather than asserted here.
E4_ACC_LSB_PER_G = 64.0

# Windowing defaults, in seconds rather than samples so they survive a change of
# HR_FS. 8 s with a 2 s hop is not an arbitrary pick: it is exactly PPG-DaLiA's
# own ground-truth convention (HR estimated over 8 s, shifted by 2 s), so a
# window's reference HR is the corpus's own estimate for that same span rather
# than something interpolated onto it. It is also the conventional window for
# PPG heart-rate estimation -- long enough for several beats at any plausible
# rate, short enough that the rate is roughly stationary across it.
WIN_SECONDS = 8.0
STRIDE_SECONDS = 2.0


def win_samples(fs: int = HR_FS) -> int:
    return int(round(WIN_SECONDS * fs))


def stride_samples(fs: int = HR_FS) -> int:
    return int(round(STRIDE_SECONDS * fs))


def read_pickle(path: str | Path) -> dict:
    """Load one subject pickle, handling its Python 2 provenance."""
    path = Path(path)
    with open(path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")
    if not isinstance(data, dict) or "signal" not in data:
        raise RuntimeError(
            f"{path} did not unpickle to the expected dict with a 'signal' key; "
            f"got {type(data).__name__} with keys "
            f"{sorted(data)[:8] if isinstance(data, dict) else 'n/a'}"
        )
    return data


def wrist_channels(data: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract wrist BVP (64 Hz) and ACC (32 Hz, in g) from a subject pickle.

    Returns
    -------
    bvp : (n,) float64
    acc : (m, 3) float64, gravity included, in g
    """
    try:
        wrist = data["signal"]["wrist"]
        bvp = np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1)
        acc = np.asarray(wrist["ACC"], dtype=np.float64).reshape(-1, 3)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"{source}: could not read signal.wrist.BVP / .ACC ({exc}). "
            f"Available wrist keys: "
            f"{sorted(data.get('signal', {}).get('wrist', {}))}"
        ) from exc
    if bvp.size == 0 or acc.size == 0:
        raise RuntimeError(f"{source}: wrist BVP or ACC is empty")
    return bvp, acc / E4_ACC_LSB_PER_G


def to_device_rate(
    bvp: np.ndarray, acc: np.ndarray, fs: int = HR_FS
) -> np.ndarray:
    """Resample BVP and ACC onto one array at the device rate.

    Returns
    -------
    (n, 4) float32 in `HR_CHANNELS` order, truncated to the shorter of the two
    resampled streams. The truncation matters: the E4's BVP and ACC streams do
    not end on the same sample, and zero-padding the shorter one would append a
    stretch of impossible data -- a flat PPG under a still accelerometer -- that
    reads downstream as a perfectly calm subject.
    """
    bvp_r = resample(bvp, FS_BVP, fs)
    acc_r = resample(acc, FS_ACC, fs, axis=0)
    n = min(len(bvp_r), len(acc_r))
    if n == 0:
        raise RuntimeError("resampling produced an empty signal")
    out = np.empty((n, N_HR_CHANNELS), dtype=np.float32)
    out[:, 0] = bvp_r[:n]
    out[:, 1:] = acc_r[:n]
    return out


def resample_labels(labels: np.ndarray, fs_in: float, n_out: int, fs_out: int) -> np.ndarray:
    """Rate-convert a categorical label track by nearest-index lookup.

    Never `signal.resample`: filtering a label array blends neighbouring class
    ids into values that correspond to no class at all.
    """
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("empty label track")
    idx = np.floor(np.arange(n_out) * (fs_in / float(fs_out))).astype(np.int64)
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx]


def homogeneous_windows(
    codes: np.ndarray, win: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Window starts whose whole span carries a single condition code.

    Returns
    -------
    starts : (k,) int64
        Start index of each accepted window.
    code : (k,) int64
        The single condition code covering that window.

    Windows spanning a condition boundary are dropped rather than given a
    majority label. Both corpora hold each condition for minutes at a time, so
    the loss is a couple of windows per transition, and the alternative is a
    window labelled 'stress' that is three-quarters baseline -- which inflates
    apparent separability in exactly the direction that flatters the result.
    """
    codes = np.asarray(codes).reshape(-1)
    m = n_windows(len(codes), win, stride)
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    starts = np.arange(m, dtype=np.int64) * stride
    idx = starts[:, None] + np.arange(win)[None, :]
    block = codes[idx]
    first = block[:, 0]
    keep = (block == first[:, None]).all(axis=1)
    return starts[keep], first[keep].astype(np.int64)


def reference_for_windows(
    starts: np.ndarray,
    win: int,
    fs: int,
    ref_t: np.ndarray,
    ref_bpm: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Ground-truth HR per window, or NaN where the corpus supplies none.

    Parameters
    ----------
    kind : {'beat', 'window'}
        How to reduce the reference series onto a window, which differs by
        corpus and must not be conflated:

        ``'beat'`` -- `ref_t`/`ref_bpm` are per-heartbeat instantaneous rates
        (WESAD, derived from chest ECG). The window's truth is the **mean over
        the beats inside it**; a single beat's instantaneous rate is dominated
        by respiratory sinus arrhythmia and is not what an 8 s PPG estimate is
        trying to match.

        ``'window'`` -- `ref_bpm` are already per-window estimates stamped at
        their window centres (PPG-DaLiA). The truth is the **single nearest
        sample to this window's centre**; averaging several of those would
        average overlapping estimates of overlapping spans and smooth away the
        very variation being measured.
    """
    if kind not in ("beat", "window"):
        raise ValueError(f"kind must be 'beat' or 'window'; got {kind!r}")

    out = np.full(len(starts), np.nan, dtype=np.float64)
    ref_t = np.asarray(ref_t, dtype=np.float64).reshape(-1)
    ref_bpm = np.asarray(ref_bpm, dtype=np.float64).reshape(-1)
    if len(ref_t) != len(ref_bpm):
        raise ValueError(f"ref_t has {len(ref_t)} entries, ref_bpm has {len(ref_bpm)}")
    if len(ref_t) == 0 or len(starts) == 0:
        return out

    order = np.argsort(ref_t)
    ref_t, ref_bpm = ref_t[order], ref_bpm[order]
    win_s = win / float(fs)
    t0 = np.asarray(starts, dtype=np.float64) / float(fs)

    if kind == "beat":
        lo = np.searchsorted(ref_t, t0, side="left")
        hi = np.searchsorted(ref_t, t0 + win_s, side="right")
        # At least two beats, or the "mean rate" is one interval's worth of noise.
        enough = (hi - lo) >= 2
        csum = np.concatenate([[0.0], np.cumsum(ref_bpm)])
        out[enough] = (csum[hi[enough]] - csum[lo[enough]]) / (hi[enough] - lo[enough])
        return out

    centres = t0 + win_s / 2.0
    j = np.clip(np.searchsorted(ref_t, centres), 1, len(ref_t) - 1)
    left_closer = (centres - ref_t[j - 1]) <= (ref_t[j] - centres)
    nearest = np.where(left_closer, j - 1, j)
    # Only accept a stamp that actually describes this window's span.
    close = np.abs(ref_t[nearest] - centres) <= (STRIDE_SECONDS / 2.0)
    out[close] = ref_bpm[nearest][close]
    return out
