"""Sliding-window extraction and event-based window labelling.

The overlap machinery here exists for SisFall. A SisFall fall trial is roughly
15 s of recording of which the fall itself is about 1 s: the subject walks, falls,
and then lies still. Labelling the whole trial "fall" is the standard way
published fall-detection accuracy gets inflated, because the great majority of
those windows contain ordinary walking or ordinary lying.

`label_by_overlap` instead assigns a label per window from how much of the window
the event actually covers, with an explicit ambiguous band that the caller is
expected to *discard* rather than guess at.
"""

from __future__ import annotations

import numpy as np


def n_windows(n_samples: int, win: int, stride: int) -> int:
    """How many complete windows fit. Zero if the signal is shorter than a window."""
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")
    if n_samples < win:
        return 0
    return (n_samples - win) // stride + 1


def window_bounds(n_samples: int, win: int, stride: int) -> np.ndarray:
    """Half-open [start, end) sample bounds of each window, shape (m, 2)."""
    m = n_windows(n_samples, win, stride)
    starts = np.arange(m, dtype=np.int64) * stride
    return np.stack([starts, starts + win], axis=1)


def sliding_windows(sig: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Extract overlapping windows from a continuous (n_samples, n_channels) signal.

    Returns
    -------
    (m, win, n_channels) array. Empty with the right trailing dims if the signal
    is too short, so callers can concatenate unconditionally.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels); got shape {sig.shape}")
    m = n_windows(len(sig), win, stride)
    if m == 0:
        return np.empty((0, win, sig.shape[1]), dtype=sig.dtype)

    # as_strided would avoid the copy, but these arrays are handed to Keras and
    # written to .npz, both of which materialize anyway. A contiguous copy here
    # keeps every downstream consumer from tripping over a non-owning view.
    idx = np.arange(m)[:, None] * stride + np.arange(win)[None, :]
    return np.ascontiguousarray(sig[idx])


def overlap_fraction(
    bounds: np.ndarray, start: int, end: int, relative_to: str = "window"
) -> np.ndarray:
    """Overlap between each window in `bounds` and the event span [start, end).

    The normalizer matters, and getting it wrong makes event labelling
    impossible rather than merely inaccurate:

    ``relative_to="window"``
        overlap / window width -- "how much of this window is event". A 1 s
        impact can occupy at most 1.0/2.56 = 39% of a 2.56 s window, so any
        positive threshold above 0.39 is unreachable for short events.

    ``relative_to="event"``
        overlap / event width -- "how much of the event does this window
        capture". This is the right question for event detection: a window
        holding the entire fall impact plus surrounding motion *is* a fall
        window, which is exactly what the device sees at inference time.

    Returns
    -------
    (m,) float array in [0, 1].
    """
    bounds = np.asarray(bounds, dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be (m, 2); got {bounds.shape}")
    if end < start:
        raise ValueError(f"event span is inverted: [{start}, {end})")
    if relative_to not in ("window", "event"):
        raise ValueError(f"relative_to must be 'window' or 'event'; got {relative_to!r}")

    inter = np.clip(np.minimum(bounds[:, 1], end) - np.maximum(bounds[:, 0], start), 0, None)
    if relative_to == "window":
        denom = bounds[:, 1] - bounds[:, 0]
    else:
        denom = np.full(len(bounds), end - start, dtype=np.int64)
    return inter / np.maximum(denom, 1)


def label_by_overlap(
    bounds: np.ndarray,
    start: int,
    end: int,
    positive_min: float = 0.8,
    negative_max: float = 0.0,
    relative_to: str = "event",
) -> np.ndarray:
    """Label windows against an event span, with an explicit ambiguous band.

    Parameters
    ----------
    bounds : (m, 2)
        Window bounds from `window_bounds`.
    start, end : int
        Half-open sample span of the event.
    positive_min : float
        A window whose overlap fraction is at least this is labelled 1.
    negative_max : float
        A window whose overlap fraction is at most this is labelled 0. The
        default of 0.0 means "only windows that do not touch the event at all
        count as negative".
    relative_to : {"event", "window"}
        Normalizer for the overlap fraction; see `overlap_fraction`. Defaults to
        "event", which is what event detection wants -- with "window", short
        events make high `positive_min` values unreachable.

    Returns
    -------
    (m,) int8 array of 1 (positive), 0 (negative), -1 (ambiguous -- discard).

    Windows that partially overlap the event fall in the ambiguous band by
    design. A window holding half a fall impact is neither a clean positive nor
    a clean negative, and training on it as either teaches the model to fire
    (or not fire) on the transition, which is not what we want to measure.
    """
    if not 0.0 <= negative_max <= positive_min <= 1.0:
        raise ValueError(
            f"need 0 <= negative_max ({negative_max}) <= positive_min "
            f"({positive_min}) <= 1"
        )
    frac = overlap_fraction(bounds, start, end, relative_to=relative_to)
    out = np.full(len(frac), -1, dtype=np.int8)
    out[frac >= positive_min] = 1
    out[frac <= negative_max] = 0
    return out


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply frozen per-channel standardization.

    `mean` and `std` must come from the training split only, and must be frozen
    constants: the firmware computes this on device from two 6-element arrays
    and cannot recompute statistics over a dataset it does not have.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (X.shape[-1],) or std.shape != (X.shape[-1],):
        raise ValueError(
            f"mean/std must have one entry per channel ({X.shape[-1]}); "
            f"got {mean.shape} and {std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError("std has non-positive entries; a channel is constant")
    return ((X - mean) / std).astype(np.float32)


def channel_stats(X: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (mean, std) over all windows and timesteps. Train split only."""
    flat = np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1])
    return flat.mean(0).astype(np.float32), (flat.std(0) + eps).astype(np.float32)
